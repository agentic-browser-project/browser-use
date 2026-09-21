"""Tests for browser_use.evaluators module."""

import pytest

from browser_use.evaluators import EvalMethodResult, Evaluator, TaskEvaluation

# ---------------------------------------------------------------------------
# Model construction & serialization
# ---------------------------------------------------------------------------


class TestModels:
	def test_eval_method_result_basic(self):
		r = EvalMethodResult(method='db_state_check', passed=True, details='ok')
		assert r.method == 'db_state_check'
		assert r.passed is True
		assert r.details == 'ok'

	def test_eval_method_result_extra_fields(self):
		r = EvalMethodResult(method='db_state_check', passed=True, details='ok', query_result={'rows': [], 'count': 1})
		d = r.model_dump()
		assert d['query_result'] == {'rows': [], 'count': 1}

	def test_task_evaluation_serialization(self):
		r = EvalMethodResult(method='string_match', passed=True, details='found', reference='42')
		te = TaskEvaluation(eval_results=[r], passed=True, needs_manual_review=False)
		d = te.model_dump()
		assert d['passed'] is True
		assert len(d['eval_results']) == 1
		assert d['eval_results'][0]['reference'] == '42'


# ---------------------------------------------------------------------------
# Evaluator instantiation
# ---------------------------------------------------------------------------


class TestEvaluatorInit:
	def test_strips_trailing_slash(self):
		ev = Evaluator(api_url='http://localhost:8000/')
		assert ev.api_url == 'http://localhost:8000'

	def test_asserts_on_empty_url(self):
		with pytest.raises(AssertionError):
			Evaluator(api_url='')


# ---------------------------------------------------------------------------
# _fuzzy_number_match
# ---------------------------------------------------------------------------


class TestFuzzyNumberMatch:
	def test_exact_match(self):
		assert Evaluator._fuzzy_number_match('45', '45', 0.3) is True

	def test_within_tolerance(self):
		assert Evaluator._fuzzy_number_match('The ETA is 45 minutes', '43.5', 0.3) is True

	def test_outside_tolerance(self):
		assert Evaluator._fuzzy_number_match('The ETA is 45 minutes', '43.5', 0.01) is False

	def test_no_numbers_in_output(self):
		assert Evaluator._fuzzy_number_match('No numbers here', '42', 0.3) is False

	def test_non_numeric_reference_substring(self):
		assert Evaluator._fuzzy_number_match('The ride is UberX', 'UberX', 0.3) is True

	def test_non_numeric_reference_no_match(self):
		assert Evaluator._fuzzy_number_match('Some text', 'other', 0.3) is False

	def test_zero_reference_tight(self):
		# With tolerance=0.3, only values within 0.3 of zero should match.
		assert Evaluator._fuzzy_number_match('Value is 0.1', '0', 0.3) is True
		assert Evaluator._fuzzy_number_match('Value is 0.3', '0', 0.3) is True

	def test_zero_reference_rejects_large_values(self):
		# "version 0.5" should NOT match a reference of "0" with tolerance 0.3.
		assert Evaluator._fuzzy_number_match('version 0.5', '0', 0.3) is False
		assert Evaluator._fuzzy_number_match('Value is 5.0', '0', 0.3) is False

	def test_multiple_numbers_in_output(self):
		# Should match if ANY number in the output is close to reference.
		assert Evaluator._fuzzy_number_match('Items: 3, 25, 100', '25', 0.3) is True

	def test_empty_reference_returns_false(self):
		assert Evaluator._fuzzy_number_match('any output', '', 0.3) is False

	def test_whitespace_reference_returns_false(self):
		assert Evaluator._fuzzy_number_match('any output', '   ', 0.3) is False


# ---------------------------------------------------------------------------
# _values_match
# ---------------------------------------------------------------------------


class TestValuesMatch:
	def test_numeric_within_tolerance(self):
		assert Evaluator._values_match(25.5, 25, 0.1) is True

	def test_numeric_outside_tolerance(self):
		assert Evaluator._values_match(30, 25, 0.1) is False

	def test_string_match(self):
		assert Evaluator._values_match('UberX', 'uberx', 0.1) is True

	def test_string_mismatch(self):
		assert Evaluator._values_match('UberX', 'UberBlack', 0.1) is False

	def test_none_actual(self):
		assert Evaluator._values_match(None, None) is True
		assert Evaluator._values_match(None, 42) is False

	def test_zero_expected_within_tolerance(self):
		assert Evaluator._values_match(0.05, 0, 0.1) is True

	def test_zero_expected_outside_tolerance(self):
		assert Evaluator._values_match(0.5, 0, 0.1) is False

	def test_zero_expected_exact(self):
		assert Evaluator._values_match(0, 0, 0.1) is True


# ---------------------------------------------------------------------------
# evaluate — logic tests (no network)
# ---------------------------------------------------------------------------


class TestEvaluateLogic:
	@pytest.fixture
	def ev(self):
		return Evaluator(api_url='http://localhost:9999')

	async def test_empty_eval_types_returns_manual_review(self, ev):
		"""Bug fix: empty eval_types must NOT silently pass."""
		task = {'eval': {'eval_types': [], 'reference_answers': {}}}
		result = await ev.evaluate(task, 'any output')
		assert result.passed is None
		assert result.needs_manual_review is True

	async def test_no_eval_key_returns_manual_review(self, ev):
		"""Task with no eval key at all."""
		result = await ev.evaluate({}, 'any output')
		assert result.passed is None
		assert result.needs_manual_review is True

	async def test_db_state_no_query(self, ev):
		task = {'eval': {'eval_types': ['db_state_check'], 'reference_answers': {}}}
		result = await ev.evaluate(task, '')
		assert result.passed is False
		assert result.eval_results[0].details == 'No postcondition_query'

	async def test_db_state_network_error(self, ev):
		"""verify_query to unreachable host should fail gracefully."""
		task = {
			'eval': {
				'eval_types': ['db_state_check'],
				'reference_answers': {'postcondition_query': 'SELECT 1'},
			}
		}
		result = await ev.evaluate(task, '', '2025-01-01 00:00:00')
		assert result.passed is False
		assert 'error' in result.eval_results[0].details.lower()

	async def test_string_match_must_include_pass(self, ev):
		task = {
			'eval': {
				'eval_types': ['string_match'],
				'reference_answers': {'must_include': ['UberX', '$']},
			}
		}
		result = await ev.evaluate(task, 'The ride is UberX for $25')
		assert result.passed is True

	async def test_string_match_must_include_fail(self, ev):
		task = {
			'eval': {
				'eval_types': ['string_match'],
				'reference_answers': {'must_include': ['UberX', '$']},
			}
		}
		result = await ev.evaluate(task, 'The ride is UberX')
		assert result.passed is False

	async def test_string_match_no_reference_needs_manual(self, ev):
		task = {'eval': {'eval_types': ['string_match'], 'reference_answers': {}}}
		result = await ev.evaluate(task, 'anything')
		assert result.passed is None
		assert result.needs_manual_review is True

	async def test_serialization_roundtrip(self, ev):
		"""model_dump() output must be JSON-serializable."""
		import json

		task = {
			'eval': {
				'eval_types': ['string_match'],
				'reference_answers': {'must_include': ['UberX']},
			}
		}
		result = await ev.evaluate(task, 'UberX is great')
		d = result.model_dump()
		json_str = json.dumps(d)
		assert '"passed": true' in json_str or '"passed":true' in json_str
