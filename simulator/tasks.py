"""Task models and loaders for WebVoyager + GAIA-web + Online-Mind2Web.

Two datasets ship in the WebVoyager repo (both included here):
  - webvoyager_data.jsonl : 643 tasks across 15 live sites. Reference answers live
    separately in reference_answer.json, keyed by site -> answers[id].
  - gaia_web.jsonl        : 90 web tasks from GAIA. Each row carries its own
    ground-truth "Final answer" inline.
A third comes from OSU-NLP-Group/Online-Mind2Web (gated HF dataset):
  - online_mind2web.json  : 300 live-web tasks over ~147 sites. No reference
    answers — success is judged by WebJudge (``eval --mode webjudge``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel

from simulator.config import GAIA_JSONL, ONLINE_MIND2WEB_JSON, REFERENCE_JSON, WEBVOYAGER_JSONL


class WebVoyagerTask(BaseModel):
	id: str
	site: str
	question: str
	start_url: str
	source: str = 'webvoyager'  # 'webvoyager' | 'gaia' | 'online_mind2web'
	reference_answer: str | None = None
	reference_type: str | None = None  # 'golden' | 'possible' | 'exact' | 'gaia' | ...
	reference_notice: str | None = None
	level: str | None = None  # online_mind2web: 'easy' | 'medium' | 'hard'
	reference_length: int | None = None  # online_mind2web: human reference action count

	@property
	def folder_name(self) -> str:
		"""Filesystem-safe folder name for this task."""
		return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in f'{self.source}__{self.id}')


def _reference_index(path: Path = REFERENCE_JSON) -> dict[str, dict[int, tuple]]:
	"""site -> {answer_id: (ans, type, notice)} from reference_answer.json."""
	if not path.exists():
		return {}
	idx: dict[str, dict[int, tuple]] = {}
	for site, blk in json.loads(path.read_text()).items():
		notice = blk.get('notice')
		idx[site] = {a['id']: (a.get('ans'), a.get('type'), notice) for a in blk.get('answers', [])}
	return idx


def load_webvoyager_tasks(path: Path = WEBVOYAGER_JSONL) -> list[WebVoyagerTask]:
	ref = _reference_index()
	out = []
	for line in path.read_text().splitlines():
		if not line.strip():
			continue
		r = json.loads(line)
		site = r['web_name']
		try:
			aid = int(str(r['id']).split('--')[-1])
		except ValueError:
			aid = -1
		ans, typ, notice = ref.get(site, {}).get(aid, (None, None, None))
		out.append(
			WebVoyagerTask(
				id=r['id'],
				site=site,
				question=r['ques'],
				start_url=r['web'],
				source='webvoyager',
				reference_answer=ans,
				reference_type=typ,
				reference_notice=notice,
			)
		)
	return out


def load_gaia_tasks(path: Path = GAIA_JSONL) -> list[WebVoyagerTask]:
	out = []
	for line in path.read_text().splitlines():
		if not line.strip():
			continue
		r = json.loads(line)
		final = r.get('Final answer')
		out.append(
			WebVoyagerTask(
				id=r['id'],
				site=f'GAIA-L{r.get("Level", "?")}',
				question=r['ques'],
				start_url=r['web'],
				source='gaia',
				reference_answer=str(final) if final is not None else None,
				reference_type='gaia',
			)
		)
	return out


def load_online_mind2web_tasks(path: Path = ONLINE_MIND2WEB_JSON) -> list[WebVoyagerTask]:
	"""Online-Mind2Web (osunlp/Online-Mind2Web): 300 live-web tasks, no reference answers."""
	if not path.exists():
		raise SystemExit(
			f'{path} not found — the dataset is gated on HuggingFace; '
			'run: python -m simulator.scripts.download_data (needs an HF token, see its --help)'
		)
	out = []
	for r in json.loads(path.read_text()):
		host = urlparse(r['website']).netloc.removeprefix('www.')
		out.append(
			WebVoyagerTask(
				id=r['task_id'],
				site=host or r['website'],
				question=r['confirmed_task'],
				start_url=r['website'],
				source='online_mind2web',
				level=r.get('level'),
				reference_length=r.get('reference_length'),
			)
		)
	return out


def load_tasks(n: int, shuffle: bool = False, seed: int = 0, source: str = 'both',
               task_ids_file: str | None = None) -> list[WebVoyagerTask]:
	"""Load up to ``n`` tasks from the chosen source(s) (optionally shuffled first).

	``source``: 'webvoyager' | 'gaia' | 'both' (= webvoyager+gaia, the historical
	default) | 'online_mind2web' | 'all' (= every dataset).

	With ``task_ids_file`` (one task id per line, e.g. "Allrecipes--0"), the
	pool is restricted to EXACTLY those ids, in file order — pinning a run to
	a fixed subset regardless of pool ordering or shuffle seed."""
	tasks: list[WebVoyagerTask] = []
	if source in ('webvoyager', 'both', 'all'):
		tasks += load_webvoyager_tasks()
	if source in ('gaia', 'both', 'all'):
		tasks += load_gaia_tasks()
	if source in ('online_mind2web', 'all'):
		tasks += load_online_mind2web_tasks()
	if not tasks:
		raise SystemExit(f'no tasks loaded for source={source!r}')
	if task_ids_file:
		want = [l.strip() for l in open(task_ids_file) if l.strip()]
		by_id = {t.id: t for t in tasks}
		missing = [w for w in want if w not in by_id]
		if missing:
			raise SystemExit(f'task_ids_file: {len(missing)} ids not in pool, e.g. {missing[:3]}')
		tasks = [by_id[w] for w in want]
		return tasks[:n]
	if shuffle:
		random.Random(seed).shuffle(tasks)
	return tasks[:n]
