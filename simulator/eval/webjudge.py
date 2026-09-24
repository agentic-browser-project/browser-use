"""Online-Mind2Web task-success evaluation — the official WebJudge protocol.

WebJudge (OSU-NLP-Group/Online-Mind2Web, src/methods/webjudge_online_mind2web.py)
judges a trajectory in three stages, with NO reference answer:

  1. key-point identification — extract the task's explicit completion requirements;
  2. per-screenshot scoring    — rate every step screenshot 1-5 for task-relevance;
  3. outcome judgment          — task + key points + action history + the screenshots
                                 scoring >= threshold (default 3) -> success / failure.

The prompts below are verbatim from the official implementation. Deviations from it:
the judge is this harness's OpenAI-compatible client (the served model or DashScope)
instead of o4-mini, and each stage is grammar-constrained to a small JSON schema
(exactly like eval/success.py — the served 30B judge degenerates on free-form output)
with a lenient text-parse fallback. Reads only captured files; never opens a browser.

Works on any captured run, but is the intended judge for source=online_mind2web,
whose tasks ship no reference answers for the WebVoyager judge to use.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI

from simulator.config import DEFAULT_JUDGE_MODEL
from simulator.eval.common import client, find_task_dirs

MAX_IMAGE = 50  # official cap on screenshots forwarded to the final judgment

KEY_POINTS_SYSTEM = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""

JUDGE_IMAGE_SYSTEM = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.
- Does the image show actions, progress indicators, or critical information directly related to completing the task?
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.
- **Score**: Assign a score based on the reasoning, using the following scale:
    - **1**: The image does not contain any necessary steps or relevant information.
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:
1. **Reasoning**: [Your explanation]
2. **Score**: [1-5]"""

WEBJUDGE_SYSTEM = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the applied filter is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""


def _schema(name: str, props: dict, required: list[str]) -> dict:
	return {
		'type': 'json_schema',
		'json_schema': {'name': name, 'schema': {'type': 'object', 'properties': props, 'required': required}},
	}


def _image_part(path: Path) -> dict:
	mime = 'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
	b64 = base64.b64encode(path.read_bytes()).decode()
	return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}', 'detail': 'high'}}


async def _chat(judge: AsyncOpenAI, model: str, system: str, content: list[dict], response_format: dict) -> str:
	resp = await judge.chat.completions.create(
		model=model,
		messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
		temperature=0.0,
		max_completion_tokens=1024,
		response_format=response_format,
	)
	return resp.choices[0].message.content or ''


async def _identify_key_points(judge: AsyncOpenAI, model: str, task: str) -> str:
	raw = await _chat(
		judge, model, KEY_POINTS_SYSTEM,
		[{'type': 'text', 'text': f'Task: {task}'}],
		_schema('key_points', {'key_points': {'type': 'array', 'items': {'type': 'string'}}}, ['key_points']),
	)
	try:
		pts = json.loads(raw)['key_points']
		return '\n'.join(f'{i + 1}. {p}' for i, p in enumerate(pts))
	except Exception:  # noqa: BLE001 — free-text fallback: strip the official heading
		txt = raw.split('**Key Points**:')[-1].split('Key Points:')[-1]
		return '\n'.join(line.lstrip() for line in txt.splitlines() if line.strip())


async def _judge_image(judge: AsyncOpenAI, model: str, task: str, key_points: str, shot: Path,
                       sema: asyncio.Semaphore) -> dict:
	text = (f'**Task**: {task}\n\n**Key Points for Task Completion**: {key_points}\n\n'
	        'The snapshot of the web page is shown in the image.')
	# identify the screenshot by its step-relative path — the bare filename is
	# 'screenshot.jpg' for EVERY step and cannot address the file later.
	rel = str(shot.relative_to(shot.parent.parent))
	async with sema:
		try:
			raw = await _chat(
				judge, model, JUDGE_IMAGE_SYSTEM,
				[{'type': 'text', 'text': text}, _image_part(shot)],
				_schema('image_score', {'reasoning': {'type': 'string'},
				                        'score': {'type': 'integer', 'minimum': 1, 'maximum': 5}},
				        ['reasoning', 'score']),
			)
		except Exception as e:  # noqa: BLE001 — one bad image must not sink the task
			return {'screenshot': rel, 'score': 0, 'reasoning': '', 'error': str(e)[:200]}
	try:
		j = json.loads(raw)
		return {'screenshot': rel, 'score': int(j['score']), 'reasoning': str(j.get('reasoning', ''))}
	except Exception:  # noqa: BLE001 — free-text fallback: official regex parse
		m = re.findall(r'[1-5]', raw.split('Score')[-1])
		thought = raw.split('**Reasoning**:')[-1].strip().split('\n\n')[0].replace('\n', ' ')
		return {'screenshot': rel, 'score': int(m[0]) if m else 0, 'reasoning': thought}


def _action_history(task_dir: Path) -> list[str]:
	"""Per-step action strings from the recorded trajectory (history.json)."""
	try:
		hist = json.loads((task_dir / 'history.json').read_text()).get('history', [])
	except Exception:  # noqa: BLE001
		return []
	out = []
	for h in hist:
		acts = (h.get('model_output') or {}).get('action') or []
		for a in acts:
			out.append(json.dumps(a, ensure_ascii=False))
	return out


async def judge_webjudge(judge: AsyncOpenAI, model: str, task_dir: Path, score_threshold: int,
                         sema: asyncio.Semaphore, attempt: int = 1) -> dict:
	meta = json.loads((task_dir / 'meta.json').read_text())
	question = meta.get('question', '')
	shots = sorted(task_dir.glob('step_*/screenshot.*'))

	verdict, err, key_points, image_records, final_raw = None, None, '', [], ''
	try:
		key_points = await _identify_key_points(judge, model, question)
		image_records = list(await asyncio.gather(
			*(_judge_image(judge, model, question, key_points, s, sema) for s in shots)))

		kept = [r for r in image_records if r['score'] >= score_threshold][:MAX_IMAGE]
		thoughts = [r['reasoning'] for r in kept if r['reasoning']]
		actions = _action_history(task_dir)
		text = (f'User Task: {question}\n\nKey Points: {key_points}\n\nAction History:\n'
		        + '\n'.join(f'{i + 1}. {a}' for i, a in enumerate(actions)))
		if kept:
			text += ('\n\nThe potentially important snapshots of the webpage in the agent\'s trajectory and their reasons:\n'
			         + '\n'.join(f'{i + 1}. {t}' for i, t in enumerate(thoughts)))
		content: list[dict] = [{'type': 'text', 'text': text}]
		for r in kept:
			content.append(_image_part(task_dir / r['screenshot']))
		final_raw = await _chat(
			judge, model, WEBJUDGE_SYSTEM, content,
			_schema('webjudge', {'thoughts': {'type': 'string'},
			                     'status': {'type': 'string', 'enum': ['success', 'failure']}},
			        ['thoughts', 'status']),
		)
		try:
			verdict = json.loads(final_raw)['status'] == 'success'
		except Exception:  # noqa: BLE001 — free-text fallback: official two-line format
			m = re.search(r'Status:\s*"?(success|failure)', final_raw, re.IGNORECASE)
			verdict = m.group(1).lower() == 'success' if m else None
	except Exception as e:  # noqa: BLE001
		err = str(e)[:200]

	out = {
		'task': task_dir.name,
		'site': meta.get('site'),
		'source': meta.get('source'),
		'level': meta.get('level'),
		'question': question,
		'answer': meta.get('answer'),
		'key_points': key_points,
		'image_scores': image_records,
		'score_threshold': score_threshold,
		'screenshots_total': len(shots),
		'screenshots_used': sum(1 for r in image_records if r['score'] >= score_threshold),
		'success': verdict,
		'judge_reasoning': final_raw,
		'judge_model': model,
		'error': err,
		'attempt': attempt,
	}
	(task_dir / 'webjudge_eval.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
	mark = {True: 'SUCCESS', False: 'FAILURE', None: 'UNKNOWN'}[verdict]
	print(f'  [{mark:7s}] {task_dir.name:44s} {question[:60]}' + (f'  ERR={err}' if err else ''))
	return out


async def evaluate_webjudge(path: Path, model: str = DEFAULT_JUDGE_MODEL, score_threshold: int = 3) -> list[dict]:
	task_dirs = find_task_dirs(path)
	if not task_dirs:
		raise SystemExit(f'No task folders with step_* found under {path}')
	judge = client()
	sema = asyncio.Semaphore(int(os.environ.get('SIM_WEBJUDGE_CONCURRENCY', '4')))
	print(f'WebJudge (Online-Mind2Web protocol) judging {len(task_dirs)} task(s) | judge={model} '
	      f'| score_threshold={score_threshold} (no web)\n')
	results = []
	eval_cap = 5
	for td in task_dirs:
		ef = td / 'webjudge_eval.json'
		attempt = 1
		if ef.exists():  # resume: reuse a conclusive verdict, or give up after eval_cap failed attempts
			try:
				prev = json.loads(ef.read_text())
				if prev.get('success') is not None:
					results.append(prev)
					continue
				attempt = int(prev.get('attempt', 1)) + 1
				if attempt > eval_cap:
					results.append(prev)
					continue
			except Exception:  # noqa: BLE001
				pass
		results.append(await judge_webjudge(judge, model, td, score_threshold, sema, attempt))
	n_succ = sum(1 for r in results if r['success'] is True)
	print('\n' + '=' * 64)
	print(f'WEBJUDGE TASK SUCCESS: {n_succ}/{len(results)} ({n_succ / len(results):.0%})')
	levels = sorted({r.get('level') for r in results if r.get('level')})
	for lv in levels:
		sub = [r for r in results if r.get('level') == lv]
		s = sum(1 for r in sub if r['success'] is True)
		print(f'  {lv:8s}: {s}/{len(sub)} ({s / len(sub):.0%})')
	return results
