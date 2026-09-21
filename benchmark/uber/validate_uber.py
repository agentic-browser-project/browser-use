#!/usr/bin/env python3
# ruff: noqa: ASYNC230, ASYNC240, ASYNC250
"""
Post-hoc validation script for Uber Benchmark results.
Re-evaluates saved results using browser_use.evaluators + string similarity fallback.

Usage:
  python benchmark/uber/validate_uber.py benchmark_results/run_<timestamp>/
  python benchmark/uber/validate_uber.py benchmark_results/run_<timestamp>/ --tasks-file benchmark/uber/tasks_uber.json
"""

import asyncio
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

from browser_use.evaluators import Evaluator

SCRIPT_DIR = Path(__file__).resolve().parent

UBER_API_URL = os.getenv('UBER_API_URL', 'http://158.130.4.153:8000')


def similarity_ratio(str1: str, str2: str) -> float:
	"""Calculate similarity ratio between two strings (0.0 to 1.0)."""
	if not str1 or not str2:
		return 0.0
	return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


async def evaluate_with_db(ev: Evaluator, task_spec: dict, agent_output: str) -> dict:
	"""Run DB-based evaluation, handling the post-hoc {benchmark_start_time} limitation."""
	eval_cfg = task_spec.get('eval', {})
	ref = eval_cfg.get('reference_answers', {})
	query = ref.get('postcondition_query', '')

	# For post-hoc validation we don't know when the task ran, so
	# time-dependent postcondition queries cannot be re-evaluated.
	if query and '{benchmark_start_time}' in query:
		eval_types = eval_cfg.get('eval_types', [])
		# We can still evaluate the string_match portion if present.
		if 'string_match' in eval_types:
			modified = dict(task_spec)
			modified['eval'] = {
				'eval_types': ['string_match'],
				'reference_answers': ref,
			}
			result = await ev.evaluate(modified, agent_output, '')
			return result.model_dump()
		return {
			'eval_results': [
				{
					'method': 'db_state_check',
					'passed': None,
					'details': 'Cannot re-evaluate: needs benchmark_start_time',
				}
			],
			'passed': None,
			'needs_manual_review': True,
		}

	result = await ev.evaluate(task_spec, agent_output, '')
	return result.model_dump()


async def validate_run(run_dir: Path, tasks_file: str | None = None) -> dict:
	"""Validate a benchmark run using DB evaluation + string similarity fallback."""
	results_file = run_dir / 'results_summary.json'
	expected_file = run_dir / 'expected_answers.json'

	if not results_file.exists():
		print(f'Error: {results_file} not found')
		sys.exit(1)

	with open(results_file, 'r') as f:
		results = json.load(f)

	# Load task specs for DB evaluation
	task_specs: dict[str, dict] = {}
	if tasks_file:
		with open(tasks_file, 'r') as f:
			specs = json.load(f)
			task_specs = {str(t['task_id']): t for t in specs}

	# Load expected answers for manual validation
	expected_answers: dict[str, dict] = {}
	if expected_file.exists():
		with open(expected_file, 'r') as f:
			data = json.load(f)
			expected_answers = {t['task_id']: t for t in data.get('tasks', [])}

	ev = Evaluator(api_url=UBER_API_URL)

	report: dict = {
		'total_tasks': len(results),
		'completed': 0,
		'failed_to_run': 0,
		'auto_eval_pass': 0,
		'auto_eval_fail': 0,
		'manual_eval_pass': 0,
		'manual_eval_fail': 0,
		'unevaluated': 0,
		'task_results': [],
	}

	for result in results:
		task_id = str(result['task_id'])
		agent_output = result.get('result', '') or ''
		task_report: dict = {
			'task_id': task_id,
			'description': result.get('task_description', ''),
			'completed': result.get('completed', False),
			'auto_eval': None,
			'manual_eval': None,
			'final_verdict': None,
		}

		if not result.get('completed'):
			report['failed_to_run'] += 1
			task_report['final_verdict'] = 'FAILED_TO_RUN'
			report['task_results'].append(task_report)
			continue

		report['completed'] += 1

		# 1. DB-based auto evaluation (if task spec available)
		if task_id in task_specs:
			db_eval = await evaluate_with_db(ev, task_specs[task_id], agent_output)
			task_report['auto_eval'] = db_eval

			if db_eval.get('passed') is True:
				report['auto_eval_pass'] += 1
				task_report['final_verdict'] = 'PASS'
				report['task_results'].append(task_report)
				continue
			elif db_eval.get('passed') is False:
				report['auto_eval_fail'] += 1
				task_report['final_verdict'] = 'FAIL'
				report['task_results'].append(task_report)
				continue

		# 2. Check inline eval (from the benchmark run itself)
		inline_eval = result.get('eval')
		if inline_eval and inline_eval.get('passed') is not None:
			if inline_eval['passed']:
				report['auto_eval_pass'] += 1
				task_report['final_verdict'] = 'PASS'
			else:
				report['auto_eval_fail'] += 1
				task_report['final_verdict'] = 'FAIL'
			task_report['auto_eval'] = inline_eval
			report['task_results'].append(task_report)
			continue

		# 3. Manual validation (string similarity fallback)
		expected = expected_answers.get(task_id, {})
		expected_answer = expected.get('expected_answer')
		if expected_answer:
			sim = similarity_ratio(agent_output, str(expected_answer))
			is_correct = sim >= 0.8 or str(expected_answer).lower() in agent_output.lower()
			task_report['manual_eval'] = {
				'expected': expected_answer,
				'similarity': round(sim, 3),
				'passed': is_correct,
			}
			if is_correct:
				report['manual_eval_pass'] += 1
				task_report['final_verdict'] = 'PASS'
			else:
				report['manual_eval_fail'] += 1
				task_report['final_verdict'] = 'FAIL'
		else:
			report['unevaluated'] += 1
			task_report['final_verdict'] = 'UNEVALUATED'

		report['task_results'].append(task_report)

	# Calculate accuracy
	total_evaluated = (
		report['auto_eval_pass'] + report['auto_eval_fail'] + report['manual_eval_pass'] + report['manual_eval_fail']
	)
	total_passed = report['auto_eval_pass'] + report['manual_eval_pass']
	report['accuracy'] = round(total_passed / total_evaluated, 3) if total_evaluated > 0 else None

	return report


def print_report(report: dict) -> None:
	print(f'\n{"=" * 70}')
	print('VALIDATION REPORT')
	print(f'{"=" * 70}\n')

	print(f'Total Tasks:     {report["total_tasks"]}')
	print(f'  Completed:     {report["completed"]}')
	print(f'  Failed to run: {report["failed_to_run"]}')
	print()
	print('Auto Evaluation (DB queries):')
	print(f'  Pass:          {report["auto_eval_pass"]}')
	print(f'  Fail:          {report["auto_eval_fail"]}')
	print()
	print('Manual Evaluation (string similarity):')
	print(f'  Pass:          {report["manual_eval_pass"]}')
	print(f'  Fail:          {report["manual_eval_fail"]}')
	print()
	print(f'Unevaluated:     {report["unevaluated"]}')
	if report['accuracy'] is not None:
		print(f'\nAccuracy:        {report["accuracy"] * 100:.1f}%')
	print()

	# Show failures
	failures = [t for t in report['task_results'] if t['final_verdict'] == 'FAIL']
	if failures:
		print(f'{"=" * 70}')
		print('FAILED TASKS')
		print(f'{"=" * 70}\n')
		for t in failures:
			print(f'  Task {t["task_id"]}: {t["description"][:60]}...')
			if t.get('auto_eval'):
				for r in t['auto_eval'].get('eval_results', []):
					print(f'    {r.get("method", "?")}: {r.get("details", "?")}')
			if t.get('manual_eval'):
				me = t['manual_eval']
				print(f"    Manual: sim={me.get('similarity')}, expected='{me.get('expected')}'")
			print()

	# Show unevaluated
	uneval = [t for t in report['task_results'] if t['final_verdict'] == 'UNEVALUATED']
	if uneval:
		print(f'{"=" * 70}')
		print('NEEDS MANUAL REVIEW')
		print(f'{"=" * 70}\n')
		for t in uneval:
			print(f'  Task {t["task_id"]}: {t["description"][:60]}...')
		print()


async def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description='Validate Uber Benchmark results')
	parser.add_argument('run_dir', type=str, help='Path to benchmark run directory')
	parser.add_argument(
		'--tasks-file',
		default=str(SCRIPT_DIR / 'tasks_uber.json'),
		help='Path to WebArena JSON tasks file (for DB evaluation)',
	)
	args = parser.parse_args()

	run_dir = Path(args.run_dir)
	if not run_dir.exists():
		print(f'Error: Directory not found: {run_dir}')
		sys.exit(1)

	tasks_file = args.tasks_file if Path(args.tasks_file).exists() else None

	print(f'Validating: {run_dir}')
	if tasks_file:
		print(f'Task specs: {tasks_file} (DB evaluation enabled)')
	else:
		print('No tasks file found (manual validation only)')

	report = await validate_run(run_dir, tasks_file)
	print_report(report)

	# Save report
	report_file = run_dir / 'validation_report.json'
	with open(report_file, 'w') as f:
		json.dump(report, f, indent=2, ensure_ascii=False)
	print(f'Report saved to: {report_file}')


if __name__ == '__main__':
	asyncio.run(main())
