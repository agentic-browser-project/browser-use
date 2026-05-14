"""Benchmark evaluator service for WebArena-format tasks.

Evaluates agent outputs against expected results using:
  - db_state_check: verify database postconditions via SQL
  - string_match: fuzzy string/number comparison against reference values

All database access goes through an HTTP verify endpoint, so no direct
database driver dependency is required.
"""

import logging
import re

import httpx

from browser_use.evaluators.views import EvalMethodResult, TaskEvaluation

logger = logging.getLogger(__name__)


class Evaluator:
	"""Benchmark evaluator that checks agent task results via a remote API.

	Args:
		api_url: Base URL for the benchmark API (e.g. ``http://host:8000``).
			Must expose ``/api/_benchmark/reset`` and ``/api/_benchmark/verify``.
		timeout: HTTP request timeout in seconds.
	"""

	def __init__(self, api_url: str, timeout: float = 30) -> None:
		assert api_url, 'api_url must not be empty'
		self.api_url = api_url.rstrip('/')
		self.timeout = timeout

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	async def reset(self) -> dict:
		"""Reset the benchmark database to its seed state.

		Returns:
			dict with keys *success*, *users_count*, *drivers_count*,
			*rides_count*, and optionally *error*.
		"""
		try:
			async with httpx.AsyncClient(timeout=self.timeout) as client:
				resp = await client.post(
					f'{self.api_url}/api/_benchmark/reset',
					json={'confirm': True},
				)
				if resp.status_code == 200:
					data = resp.json()
					self._log_reset_ok(data)
					return {'success': True, **data}
				else:
					self._log_reset_failed(f'HTTP {resp.status_code}')
					return {'success': False, 'error': f'HTTP {resp.status_code}'}
		except Exception as e:
			self._log_reset_failed(str(e))
			return {'success': False, 'error': str(e)}

	async def verify_query(self, query: str) -> dict:
		"""Execute a read-only SQL query via the benchmark verify endpoint.

		Returns:
			dict with keys *rows* (``list[dict]``), *count* (``int``), and
			optionally *error* (``str``).
		"""
		try:
			async with httpx.AsyncClient(timeout=self.timeout) as client:
				resp = await client.get(
					f'{self.api_url}/api/_benchmark/verify',
					params={'query': query},
				)
				if resp.status_code == 200:
					data = resp.json()
					if not isinstance(data, dict):
						return {'error': f'Unexpected response type: {type(data).__name__}', 'rows': [], 'count': 0}
					data.setdefault('rows', [])
					data.setdefault('count', 0)
					return data
				else:
					return {'error': f'HTTP {resp.status_code}: {resp.text}', 'rows': [], 'count': 0}
		except Exception as e:
			return {'error': str(e), 'rows': [], 'count': 0}

	async def evaluate(
		self,
		task: dict,
		agent_output: str,
		benchmark_start_time: str = '',
	) -> TaskEvaluation:
		"""Evaluate a single task using its eval spec from the WebArena JSON.

		Args:
			task: WebArena-format task dict with an ``eval`` key containing
				``eval_types`` and ``reference_answers``.
			agent_output: The text output produced by the agent.
			benchmark_start_time: Timestamp string substituted into SQL queries
				that use ``{benchmark_start_time}``.

		Returns:
			A :class:`TaskEvaluation` with per-method results and an overall
			pass/fail verdict.
		"""
		eval_cfg: dict = task.get('eval', {})
		eval_types: list[str] = eval_cfg.get('eval_types', [])
		ref: dict = eval_cfg.get('reference_answers', {})
		results: list[EvalMethodResult] = []

		for eval_type in eval_types:
			if eval_type == 'db_state_check':
				result = await self._eval_db_state(ref, benchmark_start_time)
				results.append(result)
			elif eval_type == 'string_match':
				result = await self._eval_string_match(ref, agent_output)
				results.append(result)

		# No eval criteria → cannot determine pass/fail, flag for manual review.
		if not results:
			return TaskEvaluation(
				eval_results=[],
				passed=None,
				needs_manual_review=True,
			)

		all_passed = all(r.passed for r in results if r.passed is not None)
		any_manual = any(r.passed is None for r in results)

		return TaskEvaluation(
			eval_results=results,
			passed=all_passed if not any_manual else None,
			needs_manual_review=any_manual,
		)

	# ------------------------------------------------------------------
	# Evaluation methods
	# ------------------------------------------------------------------

	async def _eval_db_state(
		self,
		ref: dict,
		benchmark_start_time: str,
	) -> EvalMethodResult:
		"""Evaluate by checking a database postcondition."""
		query: str = ref.get('postcondition_query', '')
		if not query:
			return EvalMethodResult(method='db_state_check', passed=False, details='No postcondition_query')

		query = query.replace('{benchmark_start_time}', benchmark_start_time)

		qr = await self.verify_query(query)
		if 'error' in qr:
			return EvalMethodResult(
				method='db_state_check',
				passed=False,
				details=f'Query error: {qr["error"]}',
			)

		check_type: str = ref.get('check_type', 'exists')

		if check_type == 'exists':
			passed = qr['count'] > 0
			details = f'Returned {qr["count"]} rows (need >= 1)'
		elif check_type == 'count':
			expected = ref.get('expected_result', 0)
			passed = qr['count'] == expected
			details = f'Count: {qr["count"]} (expected {expected})'
		elif check_type == 'value_match':
			if qr['rows']:
				first_val = list(qr['rows'][0].values())[0] if qr['rows'][0] else None
				expected = ref.get('expected_result')
				if first_val is None:
					passed = False
					details = 'Query returned NULL value'
				elif expected is not None:
					tol = ref.get('tolerance', 0.1)
					passed = self._values_match(first_val, expected, tol)
					details = f'Value: {first_val} (expected: {expected}, tol={tol})'
				else:
					# No expected_result configured — existence check only
					passed = True
					details = f'Value: {first_val} (no expected_result to compare)'
			else:
				passed = False
				details = 'No rows returned'
		else:
			passed = qr['count'] > 0
			details = f'Returned {qr["count"]} rows'

		return EvalMethodResult(
			method='db_state_check',
			passed=passed,
			details=details,
			query_result=qr,
		)

	async def _eval_string_match(self, ref: dict, agent_output: str) -> EvalMethodResult:
		"""Evaluate by matching the agent output against a reference value."""
		# 1. Check must_include list
		must_include: list[str] = ref.get('must_include', [])
		if must_include:
			missing = [s for s in must_include if s.lower() not in agent_output.lower()]
			passed = len(missing) == 0
			details = f'Missing: {missing}' if missing else 'All required strings found'
			return EvalMethodResult(method='string_match_must_include', passed=passed, details=details)

		# 2. Check reference_query (dynamic answer from DB)
		ref_query: str | None = ref.get('reference_query')
		if ref_query:
			qr = await self.verify_query(ref_query)
			if 'error' not in qr and qr['rows']:
				ref_value = str(list(qr['rows'][0].values())[0])
				tolerance: float = ref.get('tolerance', 0.3)
				passed = self._fuzzy_number_match(agent_output, ref_value, tolerance)
				details = f'Agent output vs reference {ref_value} (tol={tolerance})'
				return EvalMethodResult(
					method='string_match_query',
					passed=passed,
					details=details,
					reference=ref_value,
				)
			else:
				return EvalMethodResult(
					method='string_match_query',
					passed=False,
					details=f'Reference query failed: {qr.get("error", "no rows")}',
				)

		# 3. No auto-reference available
		return EvalMethodResult(
			method='string_match',
			passed=None,
			details='Manual validation needed (no auto-reference)',
		)

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------

	@staticmethod
	def _values_match(actual: object, expected: object, tolerance: float = 0.1) -> bool:
		"""Compare two values with numeric tolerance or exact string match."""
		if actual is None:
			return expected is None
		try:
			a, e = float(actual), float(expected)
			if e == 0:
				return abs(a) <= tolerance
			return abs(a - e) <= abs(e * tolerance)
		except (ValueError, TypeError):
			return str(actual).strip().lower() == str(expected).strip().lower()

	@staticmethod
	def _fuzzy_number_match(agent_output: str, reference: str, tolerance: float) -> bool:
		"""Check whether *agent_output* contains a number close to *reference*.

		Falls back to a case-insensitive substring check when *reference*
		is not numeric.
		"""
		if not reference.strip():
			return False
		try:
			ref_num = float(re.findall(r'[-+]?\d*\.?\d+', reference)[0])
		except (IndexError, ValueError):
			return reference.lower() in agent_output.lower()

		for match in re.findall(r'[-+]?\d*\.?\d+', agent_output):
			try:
				val = float(match)
				if ref_num == 0:
					# Use tolerance as absolute threshold when reference is zero
					# (relative comparison is undefined for zero).
					if abs(val) <= tolerance:
						return True
				elif abs(val - ref_num) / abs(ref_num) <= tolerance:
					return True
			except ValueError:
				continue
		return False

	# ------------------------------------------------------------------
	# Logging helpers
	# ------------------------------------------------------------------

	def _log_reset_ok(self, data: dict) -> None:
		logger.info(
			'Benchmark reset OK: %s users, %s drivers, %s rides',
			data.get('users_count'),
			data.get('drivers_count'),
			data.get('rides_count'),
		)

	def _log_reset_failed(self, reason: str) -> None:
		logger.warning('Benchmark reset FAILED: %s', reason)
