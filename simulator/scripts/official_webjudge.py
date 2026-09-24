"""Run the OFFICIAL Online-Mind2Web WebJudge over a captured simulator run.

Unlike ``eval --mode webjudge`` (this harness's re-implementation), this script
executes the benchmark authors' own evaluation code, vendored VERBATIM from
https://github.com/OSU-NLP-Group/Online-Mind2Web (src/run.py, src/utils.py,
src/methods/) under ``simulator/third_party/online_mind2web/``. Their prompts,
per-screenshot scoring, thresholding, and verdict parsing run untouched — the
ONLY substitution is the model backend: the judge calls go to the **Gemini API**
(raw REST, same key/env conventions as the in-run judge in core/judge.py)
instead of OpenAI, via a GeminiEngine exposing the same ``generate(messages)``
interface as their OpenaiEngine.

Three phases, all resumable:
  1. export   — convert the captured run into the official v1 trajectories
                layout: <run>/official_webjudge/trajectories/<task_id>/
                {result.json, trajectory/<i>_screenshot.jpg}
  2. judge    — the official auto_eval() over that directory (their output file:
                <run>/official_webjudge/WebJudge_Online_Mind2Web_eval_<model>_
                score_threshold_<t>_auto_eval_results.json, one JSON per line;
                already-judged task_ids are skipped on rerun)
  3. report   — write official_webjudge_eval.json back into each task folder
                and print the success rate, overall and per difficulty level.

Usage:
    python -m simulator.scripts.official_webjudge simulator/runs/<run> \
        [--model gemini-3.6-flash] [--score-threshold 3] [--workers 4] \
        [--source online_mind2web|all] [--export-only]

Env: GOOGLE_API_KEY / GEMINI_API_KEY / SIM_JUDGE_KEY (key file) — as core/judge.py;
     SIM_JUDGE_CONCURRENCY caps concurrent Gemini calls (default 10).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from simulator.eval.common import find_task_dirs

VENDOR_DIR = Path(__file__).parent.parent / 'third_party' / 'online_mind2web'
OFFICIAL_MODE = 'WebJudge_Online_Mind2Web_eval'


# --------------------------------------------------------------------------- #
# phase 1: export a captured run into the official v1 trajectories layout
# --------------------------------------------------------------------------- #

def _action_history(task_dir: Path) -> tuple[list[str], list[str]]:
	"""(action strings, per-step thoughts) from the recorded history.json."""
	try:
		hist = json.loads((task_dir / 'history.json').read_text()).get('history', [])
	except Exception:  # noqa: BLE001
		return [], []
	actions, thoughts = [], []
	for h in hist:
		mo = h.get('model_output') or {}
		for a in mo.get('action') or []:
			actions.append(json.dumps(a, ensure_ascii=False))
		if mo.get('thinking'):
			thoughts.append(str(mo['thinking']))
	return actions, thoughts


def export_run(run_dir: Path, export_dir: Path, source: str) -> dict[str, str]:
	"""Export task folders to the official layout; returns {task_id: sim folder name}."""
	mapping: dict[str, str] = {}
	skipped_src = 0
	for td in find_task_dirs(run_dir):
		try:
			meta = json.loads((td / 'meta.json').read_text())
		except Exception:  # noqa: BLE001
			print(f'  skip (no meta.json): {td.name}')
			continue
		if source != 'all' and meta.get('source') != source:
			skipped_src += 1
			continue
		task_id = meta['id']
		dest = export_dir / task_id
		traj = dest / 'trajectory'
		traj.mkdir(parents=True, exist_ok=True)
		shots = sorted(td.glob('step_*/screenshot.*'))
		for i, s in enumerate(shots):
			out = traj / f'{i}_screenshot{s.suffix.lower()}'
			if not out.exists():
				shutil.copyfile(s, out)
		actions, thoughts = _action_history(td)
		# v1 result.json — the official loader reads task/action_history/thoughts/
		# final_result_response and copies everything else through to its output.
		(dest / 'result.json').write_text(json.dumps({
			'task_id': task_id,
			'task': meta.get('question', ''),
			'action_history': actions,
			'thoughts': thoughts,
			'final_result_response': meta.get('answer'),
			'level': meta.get('level'),
			'reference_length': meta.get('reference_length'),
			'site': meta.get('site'),
			'sim_folder': td.name,
		}, ensure_ascii=False, indent=2))
		mapping[task_id] = td.name
	if skipped_src:
		print(f'  export: skipped {skipped_src} task(s) with source != {source!r} (use --source all to include)')
	print(f'  exported {len(mapping)} task(s) -> {export_dir}')
	return mapping


# --------------------------------------------------------------------------- #
# phase 2: the official auto_eval, with a Gemini backend
# --------------------------------------------------------------------------- #

class GeminiEngine:
	"""Drop-in for the official utils.OpenaiEngine, backed by the raw Gemini REST API.

	Same interface: ``generate(messages, max_new_tokens=512, temperature=0, ...)``
	-> [text]. OpenAI-style messages (system string + user text/image_url parts)
	are converted to a Gemini payload. Thread-safe (one urllib call per request);
	a process-wide semaphore caps concurrency, and 429/5xx retry with backoff —
	the official pipeline fans out one call per screenshot via asyncio.to_thread.
	"""

	def __init__(self, model: str, api_key: str, max_tokens_floor: int = 0):
		self.model = model
		self.api_key = api_key
		self.max_tokens_floor = max_tokens_floor  # 0 = honor the caller's max_new_tokens exactly
		self._sema = threading.Semaphore(int(os.environ.get('SIM_JUDGE_CONCURRENCY', '10')))
		self.url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

	@staticmethod
	def _convert(messages) -> tuple[str, list[dict]]:
		sys_text, parts = '', []
		for m in messages:
			c = m.get('content')
			if m.get('role') == 'system':
				sys_text = c if isinstance(c, str) else ' '.join(
					p.get('text', '') for p in c if p.get('type') == 'text')
				continue
			if isinstance(c, str):
				parts.append({'text': c})
				continue
			for p in c:
				if p.get('type') == 'text':
					parts.append({'text': p['text']})
				elif p.get('type') == 'image_url':
					url = p['image_url']['url']  # data:<mime>;base64,<b64>
					mime = url.split(':', 1)[1].split(';', 1)[0]
					parts.append({'inline_data': {'mime_type': mime, 'data': url.split(',', 1)[1]}})
		return sys_text, parts

	RETRIES = 8

	def generate(self, messages, max_new_tokens=512, temperature=0, model=None, **kwargs) -> list[str]:
		sys_text, parts = self._convert(messages)
		gen_cfg: dict = {
			'temperature': temperature,
			'maxOutputTokens': max(max_new_tokens, self.max_tokens_floor),
		}
		# Gemini bills thinking tokens against maxOutputTokens: at the official 512
		# the model spends ~490 on thoughts and emits ~17 visible tokens
		# (finishReason MAX_TOKENS), so the official "**Score**:" / "Status:" lines
		# never arrive and parse as score 0 / failure. Judge calls run without
		# thinking unless SIM_OFFICIAL_JUDGE_THINKING=1.
		if os.environ.get('SIM_OFFICIAL_JUDGE_THINKING', '0') != '1':
			gen_cfg['thinkingConfig'] = {'thinkingBudget': 0}
		body = json.dumps({
			'systemInstruction': {'parts': [{'text': sys_text}]},
			'contents': [{'parts': parts}],
			'generationConfig': gen_cfg,
		}).encode()
		url = self.url if model is None else self.url.replace(self.model, model)
		req = urllib.request.Request(
			url, data=body, headers={'x-goog-api-key': self.api_key, 'Content-Type': 'application/json'})
		last_err: Exception | None = None
		for attempt in range(self.RETRIES):
			try:
				with self._sema:
					r = json.load(urllib.request.urlopen(req, timeout=180))
				return [r['candidates'][0]['content']['parts'][0]['text']]
			except urllib.error.HTTPError as e:
				last_err = e
				# 503 "high demand" comes in bursts lasting minutes — back off up to 90s.
				if e.code in (429, 500, 503) and attempt < self.RETRIES - 1:
					time.sleep(min(5 * 2 ** attempt, 90) + random.uniform(0, 5))
					continue
				raise
			except (KeyError, IndexError) as e:  # safety block / empty candidate
				raise RuntimeError(f'Gemini returned no text: {e}') from e
			except Exception as e:  # noqa: BLE001 — timeouts, transient network
				last_err = e
				if attempt < self.RETRIES - 1:
					time.sleep(min(5 * 2 ** attempt, 90))
					continue
				raise
		raise last_err  # unreachable, keeps type-checkers happy


def _gemini_key() -> str:
	from simulator.core.judge import _DEFAULT_KEY
	key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
	key_path = os.environ.get('SIM_JUDGE_KEY', _DEFAULT_KEY)
	if not key and os.path.exists(key_path):
		key = open(key_path).read().strip()
	if not key:
		raise SystemExit('No Gemini API key: set GOOGLE_API_KEY/GEMINI_API_KEY or SIM_JUDGE_KEY (key file)')
	return key


def run_official(export_dir: Path, out_dir: Path, model_name: str, score_threshold: int, workers: int) -> Path:
	"""Invoke the vendored, unmodified official auto_eval() over the exported tasks."""
	sys.path.insert(0, str(VENDOR_DIR))
	import run as official_run  # noqa: PLC0415 — vendored official code

	# Floor on maxOutputTokens: the official code asks for 512, but Gemini counts
	# thinking tokens against the output budget, so 512 can truncate mid-answer
	# (a verdict cut before its "Status:" line parses as failure). Transport-level
	# only — prompts and parsing stay official. SIM_OFFICIAL_JUDGE_MAX_TOKENS=0
	# restores the exact official request.
	engine = GeminiEngine(model_name, _gemini_key(),
	                      max_tokens_floor=int(os.environ.get('SIM_OFFICIAL_JUDGE_MAX_TOKENS', '2048')))
	args = argparse.Namespace(
		mode=OFFICIAL_MODE, model=model_name, trajectories_dir=str(export_dir),
		output_path=str(out_dir), score_threshold=score_threshold, api_key='unused',
	)
	task_ids = sorted(d.name for d in export_dir.iterdir() if d.is_dir())
	out_file = out_dir / f'{OFFICIAL_MODE}_{model_name}_score_threshold_{score_threshold}_auto_eval_results.json'
	print(f'  official WebJudge over {len(task_ids)} task(s) | judge={model_name} (Gemini API) '
	      f'| threshold={score_threshold} | workers={workers}\n  output: {out_file}')

	# Their parallel_eval only chunks the ids across processes around auto_eval()
	# (and crashes when tasks < workers: chunk_size 0). The judging itself all lives
	# in auto_eval, which we run verbatim — here across threads, since the engine is
	# not picklable and the work is pure API I/O.
	workers = max(1, min(workers, len(task_ids)))
	lock = threading.Lock()
	labels: list[int] = []
	failed: list[str] = []

	# One auto_eval() call PER TASK: their loop has no try/except, so an API error
	# that outlives the engine's retries would otherwise kill the whole chunk and
	# silently skip every task after it. A failed task is left unjudged (no row
	# in the output file) and picked up by the next, resumable run.
	def worker(chunk: list[str]) -> None:
		for tid in chunk:
			try:
				official_run.auto_eval(args, [tid], labels, lock, engine)
			except Exception as e:  # noqa: BLE001
				with lock:
					failed.append(tid)
				print(f'  [FAILED] {tid}: {type(e).__name__}: {str(e)[:160]}', file=sys.stderr, flush=True)

	chunks = [task_ids[i::workers] for i in range(workers)]
	threads = [threading.Thread(target=worker, args=(chunk,)) for chunk in chunks if chunk]
	for t in threads:
		t.start()
	for t in threads:
		t.join()
	if failed:
		print(f'  {len(failed)} task(s) failed and were left unjudged (rerun to resume): {failed[:5]}...')
	return out_file


# --------------------------------------------------------------------------- #
# phase 3: import the verdicts back into the run + summary
# --------------------------------------------------------------------------- #

def report(run_dir: Path, out_file: Path, mapping: dict[str, str], model_name: str, score_threshold: int) -> None:
	if not out_file.exists():
		raise SystemExit(f'official output not found: {out_file}')
	rows = [json.loads(l) for l in out_file.read_text().splitlines() if l.strip()]
	by_id = {r['task_id']: r for r in rows}
	results = []
	for task_id, folder in sorted(mapping.items()):
		r = by_id.get(task_id)
		if r is None:
			print(f'  [MISSING] {task_id} — not judged (rerun to resume)')
			continue
		success = bool(r.get('predicted_label'))
		results.append({'task_id': task_id, 'level': r.get('level'), 'success': success})
		(run_dir / folder / 'official_webjudge_eval.json').write_text(json.dumps({
			'task': folder,
			'task_id': task_id,
			'level': r.get('level'),
			'success': success,
			'predicted_label': r.get('predicted_label'),
			'key_points': r.get('key_points'),
			'image_judge_record': r.get('image_judge_record'),
			'judge_response': (r.get('evaluation_details') or {}).get('response'),
			'judge_model': model_name,
			'score_threshold': score_threshold,
			'protocol': 'official WebJudge (OSU-NLP-Group/Online-Mind2Web), Gemini backend',
		}, ensure_ascii=False, indent=2))
	if not results:
		return
	n_succ = sum(r['success'] for r in results)
	print('\n' + '=' * 64)
	print(f'OFFICIAL WEBJUDGE TASK SUCCESS: {n_succ}/{len(results)} ({n_succ / len(results):.0%})'
	      f' | {len(mapping) - len(results)} unjudged')
	for lv in sorted({r['level'] for r in results if r['level']}):
		sub = [r for r in results if r['level'] == lv]
		s = sum(r['success'] for r in sub)
		print(f'  {lv:8s}: {s}/{len(sub)} ({s / len(sub):.0%})')


def main() -> None:
	ap = argparse.ArgumentParser(prog='python -m simulator.scripts.official_webjudge', description=__doc__)
	ap.add_argument('run_dir', help='A captured run folder (simulator/runs/<run>).')
	ap.add_argument('--model', default='gemini-3.6-flash', help='Gemini judge model.')
	ap.add_argument('--score-threshold', type=int, default=3, help='Official per-screenshot keep threshold (1-5).')
	ap.add_argument('--workers', type=int, default=4, help='Task-level parallelism.')
	ap.add_argument('--source', default='online_mind2web',
	                help="Only export tasks of this source ('all' = every task in the run).")
	ap.add_argument('--export-only', action='store_true', help='Only build the official trajectories dir.')
	a = ap.parse_args()

	run_dir = Path(a.run_dir)
	base = run_dir / 'official_webjudge'
	export_dir = base / 'trajectories'
	mapping = export_run(run_dir, export_dir, a.source)
	if not mapping:
		raise SystemExit('nothing to evaluate (no matching task folders)')
	(base / 'task_mapping.json').write_text(json.dumps(mapping, indent=2))
	if a.export_only:
		return
	out_file = run_official(export_dir, base, a.model, a.score_threshold, a.workers)
	report(run_dir, out_file, mapping, a.model, a.score_threshold)


if __name__ == '__main__':
	main()
