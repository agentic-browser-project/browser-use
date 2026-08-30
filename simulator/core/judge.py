"""Overlapped browser-use internal judge, via the raw Gemini REST API.

The WebVoyager e2e loop fires this the moment a task finishes, as a detached
asyncio task, so the grounding-aware evaluation runs CONCURRENTLY with the next
task's agent execution instead of blocking it. Each verdict is written to
``judge.json`` in the task folder.

Uses browser-use's real judge PROMPT (``construct_judge_messages``: ground-truth
precedence, screenshot verification, CAPTCHA / impossible-task flags) but sends it
through the **raw generativelanguage REST endpoint** — NOT the google-genai SDK.
The SDK (``ChatGoogle``) was found to stall intermittently (identical 10-image
payload: 10-64s and occasional >90s hangs); the same payload over raw REST with a
non-reasoning flash model returns in ~4-5s, consistently. The judge model is an
INDEPENDENT model from the agent under test.

Env:
  SIM_JUDGE=0            disable (default on)
  SIM_JUDGE_MODEL       judge model (default gemini-3.6-flash — fast, stable, vision)
  SIM_JUDGE_CONCURRENCY max in-flight judge calls (default 10)
  SIM_JUDGE_VISION=0    text-only judge (default: include last 10 screenshots)
  SIM_JUDGE_KEY         path to the API key file
  SIM_JUDGE_TIMEOUT     per-call HTTP timeout seconds (default 90)
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from types import SimpleNamespace

_DEFAULT_KEY = '/vast/projects/liuv/pennnetworks/jiaheng/BrowserSparseAttention/gemini-api-key.txt'

# JudgementResult schema for REST structured output (mirrors browser_use.agent.views.JudgementResult)
_SCHEMA = {
	'type': 'object',
	'properties': {
		'reasoning': {'type': 'string'},
		'verdict': {'type': 'boolean'},
		'failure_reason': {'type': 'string'},
		'impossible_task': {'type': 'boolean'},
		'reached_captcha': {'type': 'boolean'},
	},
	'required': ['verdict'],
}


def judge_enabled() -> bool:
	return os.environ.get('SIM_JUDGE', '1').lower() not in ('0', 'false', 'no', '')


def build_judge_llm():
	"""Return a small REST-judge config (model, api_key, url), or None if disabled/unconfigured.

	No SDK client — just the endpoint + key. Named build_judge_llm for the runner's call site.
	"""
	if not judge_enabled():
		return None
	key_path = os.environ.get('SIM_JUDGE_KEY', _DEFAULT_KEY)
	api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
	if not api_key and os.path.exists(key_path):
		api_key = open(key_path).read().strip()
	if not api_key:
		print('[judge] no API key found — evaluation disabled', flush=True)
		return None
	model = os.environ.get('SIM_JUDGE_MODEL', 'gemini-3.6-flash')
	print(f'[judge] overlapped browser-use judge (raw REST) ON: model={model} '
	      f'concurrency={os.environ.get("SIM_JUDGE_CONCURRENCY", "10")} '
	      f'vision={os.environ.get("SIM_JUDGE_VISION", "1")}', flush=True)
	return SimpleNamespace(
		model=model,
		api_key=api_key,
		url=f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
	)


def judge_semaphore() -> asyncio.Semaphore:
	return asyncio.Semaphore(int(os.environ.get('SIM_JUDGE_CONCURRENCY', '10')))


def _build_steps(history: list[dict]) -> list[str]:
	"""Mirror AgentHistoryList.agent_steps() from the serialized history."""
	steps = []
	for i, h in enumerate(history):
		t = f'Step {i + 1}:\n'
		acts = (h.get('model_output') or {}).get('action') or []
		if acts:
			t += f'Actions: {json.dumps(acts, indent=1)}\n'
		for j, r in enumerate(h.get('result') or []):
			if r.get('extracted_content'):
				t += f'Result {j + 1}: {r["extracted_content"]}\n'
			if r.get('error'):
				t += f'Error {j + 1}: {r["error"]}\n'
		steps.append(t)
	return steps


def _messages_to_rest(msgs) -> tuple[str, list[dict]]:
	"""Flatten browser-use judge messages into (system_text, content_parts) for the REST API."""
	sys_text, parts = '', []
	for m in msgs:
		c = m.content
		if getattr(m, 'role', '') == 'system':
			sys_text = c if isinstance(c, str) else ' '.join(
				p.text for p in c if getattr(p, 'type', '') == 'text')
			continue
		if isinstance(c, str):
			parts.append({'text': c})
		else:
			for p in c:
				if getattr(p, 'type', '') == 'text':
					parts.append({'text': p.text})
				elif getattr(p, 'type', '') == 'image_url':
					b64 = p.image_url.url.split(',', 1)[1]
					parts.append({'inline_data': {'mime_type': 'image/jpeg', 'data': b64}})
	return sys_text, parts


_POSSIBLE_REF_NOTE = """

IMPORTANT — how to use the reference for THIS task: the reference below is ONE VALID EXAMPLE answer, NOT the only correct answer. This task's answer may be time-varying (prices, availability, "latest"/"cheapest"/"top", counts that change over time), so the agent's answer may legitimately DIFFER from this example and still be fully correct. Judge SUCCESS by whether the agent's answer genuinely SATISFIES THE TASK'S CONSTRAINTS — verified against the screenshots and trajectory — NOT by whether it matches this example. Accept valid alternatives; real-time information may differ. (Still fail answers that violate an objective constraint, are fabricated/not on the page, or where the agent never actually completed the task.)
Reference example: {ref}
"""


def build_judge_payload(question, answer, steps, shots, reference, reference_type, use_vision):
	"""Build (sys_text, parts) for the REST judge with TWO-MODE ground-truth handling.

	golden reference   -> authoritative: passed as browser-use ground_truth (ABSOLUTE
	                      precedence / exact match).
	possible reference -> one valid EXAMPLE: NOT passed as ground_truth (so the strict
	                      "absolute precedence" block is omitted); instead appended as a
	                      non-binding example with the accept-alternatives / real-time note.
	This fixes the over-strict failures where a valid alternative or a time-drifted-correct
	answer was marked wrong only because it differed from the single annotated example.
	"""
	from browser_use.agent.judge import construct_judge_messages
	golden = (reference_type == 'golden')
	msgs = construct_judge_messages(
		task=question, final_result=answer or '', agent_steps=steps,
		screenshot_paths=shots, max_images=10,
		ground_truth=(reference if golden else None),
		use_vision=use_vision)
	sys_text, parts = _messages_to_rest(msgs)
	if not golden and reference:
		sys_text += _POSSIBLE_REF_NOTE.format(ref=reference)
	return sys_text, parts


def _rest_call(cfg, sys_text: str, parts: list[dict], timeout: float) -> dict:
	"""Blocking raw-REST Gemini call with structured output. Returns parsed JudgementResult dict."""
	body = json.dumps({
		'systemInstruction': {'parts': [{'text': sys_text}]},
		'contents': [{'parts': parts}],
		'generationConfig': {'temperature': 0, 'responseMimeType': 'application/json', 'responseSchema': _SCHEMA},
	}).encode()
	req = urllib.request.Request(cfg.url, data=body,
	                             headers={'x-goog-api-key': cfg.api_key, 'Content-Type': 'application/json'})
	r = json.load(urllib.request.urlopen(req, timeout=timeout))
	return json.loads(r['candidates'][0]['content']['parts'][0]['text'])


async def judge_task_dir(cfg, task, task_dir: Path, sema: asyncio.Semaphore) -> None:
	"""Read the finished trajectory from disk and write judge.json. Concurrency-safe.

	The blocking REST call runs in a thread (asyncio.to_thread) so it never blocks the
	event loop — later tasks keep executing while this evaluation is in flight.
	"""
	use_vision = os.environ.get('SIM_JUDGE_VISION', '1').lower() not in ('0', 'false', 'no', '')
	timeout = float(os.environ.get('SIM_JUDGE_TIMEOUT', '90'))
	out = task_dir / 'judge.json'
	if out.exists():  # resume: already judged
		try:
			if json.loads(out.read_text()).get('verdict') is not None:
				return
		except Exception:  # noqa: BLE001
			pass
	try:
		meta = json.loads((task_dir / 'meta.json').read_text())
		hist = json.loads((task_dir / 'history.json').read_text()).get('history', [])
	except Exception:  # noqa: BLE001
		return  # trajectory not on disk (task errored before recording) — nothing to judge

	steps = _build_steps(hist)
	shots = sorted(glob.glob(str(task_dir / 'step_*' / 'screenshot.jpg'))) if use_vision else []
	# Two-mode ground truth: golden = authoritative (exact); possible = one valid example.
	sys_text, parts = build_judge_payload(
		task.question, meta.get('answer') or '', steps, shots,
		meta.get('reference_answer'), meta.get('reference_type'), use_vision)

	async with sema:  # cap in-flight judge calls just below the API rate limit
		for attempt in range(4):
			try:
				j = await asyncio.wait_for(
					asyncio.to_thread(_rest_call, cfg, sys_text, parts, timeout), timeout=timeout + 15)
				out.write_text(json.dumps({
					'verdict': bool(j.get('verdict')),
					'reasoning': j.get('reasoning'),
					'failure_reason': j.get('failure_reason'),
					'impossible_task': bool(j.get('impossible_task', False)),
					'reached_captcha': bool(j.get('reached_captcha', False)),
					'judge_model': cfg.model,
					'reference_type': meta.get('reference_type'),
					'used_vision': use_vision,
				}, ensure_ascii=False, indent=2))
				return
			except urllib.error.HTTPError as e:  # 429/500/503 -> backoff and retry
				if e.code in (429, 500, 503) and attempt < 3:
					await asyncio.sleep(min(2 ** attempt, 20))
					continue
				j_err = f'HTTP {e.code}'
			except Exception as e:  # noqa: BLE001 — timeouts, transient network
				if attempt < 3:
					await asyncio.sleep(2 * (attempt + 1))
					continue
				j_err = f'{type(e).__name__}: {str(e)[:150]}'
			out.write_text(json.dumps({'verdict': None, 'error': j_err, 'judge_model': cfg.model}))
			return
