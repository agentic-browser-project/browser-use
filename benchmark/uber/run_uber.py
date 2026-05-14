#!/usr/bin/env python3
# ruff: noqa: ASYNC230, ASYNC240, ASYNC250
"""
Benchmark runner for Uber Benchmark (WebArena JSON format).
Uses browser_use.evaluators for task evaluation.
Compatible with any OpenAI-compatible LLM server (vLLM, SGLang, etc.).

Usage:
  python benchmark/uber/run_uber.py --yes
  python benchmark/uber/run_uber.py --task-ids 0,1,2 --yes
"""

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from browser_use import Agent, Browser, ChatOpenAI
from browser_use.evaluators import Evaluator

SCRIPT_DIR = Path(__file__).resolve().parent

load_dotenv()

# Headless server workarounds
os.environ['BROWSER_USE_DISABLE_EXTENSIONS'] = '1'
os.environ['TIMEOUT_BrowserStartEvent'] = '120'
os.environ['TIMEOUT_BrowserLaunchEvent'] = '120'

# Configuration — LLM_BASE_URL works with vLLM, SGLang, or any OpenAI-compatible server.
# Falls back to VLLM_URL for backward compatibility.
LLM_BASE_URL = os.getenv('LLM_BASE_URL') or os.getenv('VLLM_URL')
MODEL_NAME = os.getenv('MODEL_NAME')
LLM_API_KEY = os.getenv('LLM_API_KEY', 'EMPTY')
UBER_BASE_URL = os.getenv('UBER_BASE_URL', 'http://158.130.4.153:3002')
UBER_API_URL = os.getenv('UBER_API_URL', 'http://158.130.4.153:8000')
TEST_USERNAME = os.getenv('TEST_USERNAME', 'testuser1')
TEST_PASSWORD = os.getenv('TEST_PASSWORD', 'password123')
ENABLE_VISION_USAGE = 'auto'

# Results directory (may be overridden by --resume-dir in main)
BENCHMARK_DIR = Path('benchmark_results')
BENCHMARK_DIR.mkdir(exist_ok=True)
RUN_DIR: Path = Path()  # Set in main()
RESULTS_FILE: Path = Path()  # Set in main()
EXPECTED_ANSWERS_FILE: Path = Path()  # Set in main()
EVAL_REPORT_FILE: Path = Path()  # Set in main()

# Evaluator instance (initialized in main after config validation)
evaluator: Evaluator | None = None


# ---------------------------------------------------------------------------
# Task loading (WebArena JSON)
# ---------------------------------------------------------------------------
def load_tasks_webarena(filepath: str) -> list[dict]:
	"""Load tasks from WebArena-format JSON file."""
	with open(filepath, 'r') as f:
		tasks = json.load(f)
	for t in tasks:
		t['task_id'] = str(t['task_id'])
		t['description'] = t['intent']
		t['category'] = t.get('category', 'unknown')
	return tasks


def resolve_url(url_template: str) -> str:
	"""Replace __UBER__ placeholder with actual URL."""
	return url_template.replace('__UBER__', UBER_BASE_URL)


def build_task_prompt(task: dict) -> str:
	"""Build the full prompt including login and navigation."""
	start_url = resolve_url(task['start_url'])
	intent = task['intent']

	prompt = f"Go to {UBER_BASE_URL} and log in with username '{TEST_USERNAME}' and password '{TEST_PASSWORD}'."
	if start_url != UBER_BASE_URL:
		prompt += f' Then navigate to {start_url}.'
	prompt += f' Then complete this task: {intent}'
	return prompt


# ---------------------------------------------------------------------------
# Run single task
# ---------------------------------------------------------------------------
async def run_single_task(task: dict, llm: ChatOpenAI) -> dict:
	"""Run a single task and return results."""
	assert evaluator is not None

	start_time = datetime.now()
	benchmark_start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')

	task_id = task['task_id']
	task_desc = build_task_prompt(task)
	max_steps = task.get('max_steps', 15)

	result_data = {
		'task_id': task_id,
		'task_description': task['intent'],
		'category': task.get('category', ''),
		'difficulty': task.get('difficulty', ''),
		'prompt_sent': task_desc,
		'max_steps': max_steps,
		'completed': False,
		'result': None,
		'error': None,
		'log_dir': None,
		'duration_seconds': 0,
		'timestamp': start_time.isoformat(),
		'eval': None,
	}

	browser = None
	try:
		print(f'\n{"=" * 70}')
		print(f'Task {task_id}: {task["intent"]}')
		print(f'{"=" * 70}\n')

		browser = Browser(headless=True, args=['--no-sandbox'])
		agent = Agent(
			task=task_desc,
			llm=llm,
			browser=browser,
			use_vision=ENABLE_VISION_USAGE,
			llm_timeout=300,
			step_timeout=600,
		)

		history = await agent.run(max_steps=max_steps)

		# Extract final result text
		final_result = history.final_result() or ''
		if not final_result:
			for r in reversed(history.action_results()):
				if r.extracted_content:
					final_result = r.extracted_content
					break

		result_data['completed'] = True
		result_data['result'] = final_result

		# Get log directory (best-effort, must not contaminate result on failure)
		try:
			log_dirs = sorted(
				Path('agent_logs').glob('*'),
				key=lambda p: p.stat().st_mtime,
				reverse=True,
			)
			if log_dirs:
				result_data['log_dir'] = str(log_dirs[0])
		except Exception:
			pass

		print(f'\n  Completed! Result: {final_result[:200]}')

	except Exception as e:
		result_data['error'] = str(e)
		result_data['traceback'] = traceback.format_exc()
		print(f'\n  FAILED: {e}')

	finally:
		if browser:
			try:
				await browser.kill()
			except Exception:
				pass

	# Run evaluation
	agent_output = result_data.get('result', '') or ''
	try:
		eval_result = await evaluator.evaluate(task, agent_output, benchmark_start_time)
		result_data['eval'] = eval_result.model_dump()
		if eval_result.passed is True:
			print('  EVAL: PASS')
		elif eval_result.passed is False:
			print('  EVAL: FAIL')
			for er in eval_result.eval_results:
				print(f'    {er.method}: {er.details}')
		else:
			print('  EVAL: NEEDS MANUAL REVIEW')
	except Exception as e:
		result_data['eval'] = {'error': str(e)}
		print(f'  EVAL ERROR: {e}')

	end_time = datetime.now()
	result_data['duration_seconds'] = (end_time - start_time).total_seconds()

	return result_data


# ---------------------------------------------------------------------------
# Generate expected_answers.json
# ---------------------------------------------------------------------------
def generate_expected_answers_template(tasks: list[dict]) -> None:
	"""Generate template file for manual validation."""
	template = {
		'instructions': (
			'Fill in expected_answer for tasks where automatic evaluation returned '
			'needs_manual_review. Leave null if not applicable.'
		),
		'tasks': [],
	}
	for task in tasks:
		template['tasks'].append(
			{
				'task_id': task['task_id'],
				'category': task.get('category', ''),
				'description': task['intent'],
				'expected_answer': None,
				'validation_notes': '',
			}
		)

	with open(EXPECTED_ANSWERS_FILE, 'w') as f:
		json.dump(template, f, indent=2, ensure_ascii=False)
	print(f'  Expected answers template: {EXPECTED_ANSWERS_FILE}')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(results: list[dict]) -> None:
	total = len(results)
	completed = sum(1 for r in results if r['completed'])
	failed = total - completed

	auto_evaluated = [r for r in results if r.get('eval') and r['eval'].get('passed') is not None]
	passed = sum(1 for r in auto_evaluated if r['eval']['passed'])
	needs_review = sum(1 for r in results if r.get('eval') and r['eval'].get('needs_manual_review'))

	print(f'\n{"=" * 70}')
	print('BENCHMARK SUMMARY')
	print(f'{"=" * 70}')
	print(f'Total tasks:      {total}')
	print(f'Completed:        {completed} ({completed / total * 100:.0f}%)')
	print(f'Failed to run:    {failed}')
	print(f'Auto-evaluated:   {len(auto_evaluated)}')
	print(f'  Passed:         {passed}')
	print(f'  Failed:         {len(auto_evaluated) - passed}')
	print(f'Manual review:    {needs_review}')
	total_time = sum(r['duration_seconds'] for r in results)
	print(f'Total time:       {total_time:.0f}s ({total_time / 60:.1f}m)')
	print(f'\nResults: {RUN_DIR}')
	print(f'{"=" * 70}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
	global evaluator, RUN_DIR, RESULTS_FILE, EXPECTED_ANSWERS_FILE, EVAL_REPORT_FILE

	import argparse

	parser = argparse.ArgumentParser(description='Run Uber Benchmark (WebArena format)')
	parser.add_argument(
		'--tasks-file',
		default=str(SCRIPT_DIR / 'tasks_uber.json'),
		help='Path to WebArena JSON tasks file',
	)
	parser.add_argument('--task-ids', type=str, default=None, help='Comma-separated task IDs to run (default: all)')
	parser.add_argument('--no-reset', action='store_true', help='Skip benchmark reset between tasks')
	parser.add_argument('--start-from', type=int, default=0, help='Resume from task index N (results only include tasks from this run)')
	parser.add_argument('--resume-dir', type=str, default=None, help='Resume into an existing run directory (merges results)')
	parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
	args = parser.parse_args()

	# Set up results directory
	if args.resume_dir:
		RUN_DIR = Path(args.resume_dir)
		if not RUN_DIR.exists():
			print(f'ERROR: Resume directory not found: {RUN_DIR}')
			sys.exit(1)
	else:
		run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		RUN_DIR = BENCHMARK_DIR / f'run_{run_timestamp}'
		RUN_DIR.mkdir(exist_ok=True)
	RESULTS_FILE = RUN_DIR / 'results_summary.json'
	EXPECTED_ANSWERS_FILE = RUN_DIR / 'expected_answers.json'
	EVAL_REPORT_FILE = RUN_DIR / 'eval_report.json'

	# Validate config
	if not all([LLM_BASE_URL, MODEL_NAME]):
		print('ERROR: Missing LLM_BASE_URL (or VLLM_URL) and MODEL_NAME environment variables!')
		print('Please set them in .env or export them in your shell.')
		sys.exit(1)

	# Initialize evaluator
	evaluator = Evaluator(api_url=UBER_API_URL)

	# Load tasks
	all_tasks = load_tasks_webarena(args.tasks_file)

	if args.task_ids:
		target_ids = set(args.task_ids.split(','))
		tasks = [t for t in all_tasks if t['task_id'] in target_ids]
	else:
		tasks = all_tasks[args.start_from :]

	if not tasks:
		print('No tasks to run!')
		sys.exit(1)

	print(f'{"=" * 70}')
	print('UBER BENCHMARK (WebArena + DB Evaluation)')
	print(f'{"=" * 70}')
	print(f'Model:       {MODEL_NAME}')
	print(f'LLM URL:     {LLM_BASE_URL}')
	print(f'Benchmark:   {UBER_BASE_URL}')
	print(f'API:         {UBER_API_URL}')
	print(f'Tasks:       {len(tasks)} of {len(all_tasks)}')
	print(f'Results:     {RUN_DIR}')
	print(f'{"=" * 70}')

	# Pre-flight checks — works with vLLM, SGLang, or any OpenAI-compatible server
	try:
		async with httpx.AsyncClient(timeout=10) as client:
			resp = await client.get(f'{LLM_BASE_URL}/models')
			resp.raise_for_status()
			body = resp.json()
			# Both vLLM and SGLang return {"data": [{"id": ...}]}
			models = body.get('data', []) if isinstance(body, dict) else []
			if models:
				model_id = models[0].get('id', 'unknown')
				print(f'LLM OK:      {model_id}')
			else:
				print('LLM OK:      (no models listed, server responded)')
	except Exception as e:
		print(f'ERROR: LLM not reachable at {LLM_BASE_URL}: {e}')
		sys.exit(1)

	try:
		async with httpx.AsyncClient(timeout=10) as client:
			resp = await client.get(UBER_BASE_URL)
			print(f'Frontend OK: HTTP {resp.status_code}')
	except Exception as e:
		print(f'ERROR: Frontend not reachable at {UBER_BASE_URL}: {e}')
		sys.exit(1)

	try:
		qr = await evaluator.verify_query('SELECT 1')
		if 'error' in qr:
			raise RuntimeError(qr['error'])
		print('Verify API:  OK')
	except Exception as e:
		print(f'ERROR: Verify API not reachable: {e}')
		sys.exit(1)

	# Generate expected answers template
	generate_expected_answers_template(all_tasks)

	if not args.yes:
		response = input(f'\nRun {len(tasks)} tasks? (y/n): ').strip().lower()
		if response != 'y':
			print('Aborted.')
			return

	# Initialize LLM — compatible with vLLM, SGLang, and any OpenAI-compatible server.
	# ChatOpenAI defaults frequency_penalty=0.3 which both vLLM and SGLang support.
	llm = ChatOpenAI(
		model=MODEL_NAME,
		base_url=LLM_BASE_URL,
		api_key=LLM_API_KEY,
		temperature=0.7,
		max_completion_tokens=None,
	)

	# Load existing results if resuming into a prior run directory
	results = []
	if args.resume_dir and RESULTS_FILE.exists():
		with open(RESULTS_FILE, 'r') as f:
			results = json.load(f)
		print(f'  Loaded {len(results)} prior results from {RESULTS_FILE}')

	# Run tasks
	for i, task in enumerate(tasks):
		print(f'\n{"#" * 70}')
		print(f'TASK {i + 1}/{len(tasks)} (ID: {task["task_id"]}) [{task.get("difficulty", "?")}] [{task.get("category", "?")}]')
		print(f'{"#" * 70}')

		# Reset benchmark state
		if not args.no_reset:
			print('  Resetting benchmark state...')
			await evaluator.reset()

		result = await run_single_task(task, llm)
		results.append(result)

		# Save incrementally
		with open(RESULTS_FILE, 'w') as f:
			json.dump(results, f, indent=2, ensure_ascii=False)

		# Save eval report incrementally
		eval_summary = []
		for r in results:
			eval_summary.append(
				{
					'task_id': r['task_id'],
					'description': r['task_description'],
					'completed': r['completed'],
					'eval_passed': r.get('eval', {}).get('passed'),
					'needs_review': r.get('eval', {}).get('needs_manual_review', False),
					'eval_details': r.get('eval', {}).get('eval_results', []),
				}
			)
		with open(EVAL_REPORT_FILE, 'w') as f:
			json.dump(eval_summary, f, indent=2, ensure_ascii=False)

		await asyncio.sleep(2)

	print_summary(results)


if __name__ == '__main__':
	asyncio.run(main())
