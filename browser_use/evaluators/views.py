"""Pydantic models for benchmark evaluation results."""

from pydantic import BaseModel, ConfigDict


class EvalMethodResult(BaseModel):
	"""Result from a single evaluation method (db_state_check, string_match, etc.)."""

	model_config = ConfigDict(extra='allow')

	method: str
	passed: bool | None
	details: str


class TaskEvaluation(BaseModel):
	"""Aggregated evaluation result for a benchmark task."""

	model_config = ConfigDict(extra='allow')

	eval_results: list[EvalMethodResult]
	passed: bool | None
	needs_manual_review: bool
