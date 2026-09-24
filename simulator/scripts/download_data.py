"""Download the task datasets the simulator needs into simulator/data/.

The raw datasets are third-party and gitignored (the repo ignores *.json/*.jsonl),
so fetch them once after cloning:

    python -m simulator.scripts.download_data          # skip files already present
    python -m simulator.scripts.download_data --force  # re-download everything

Online-Mind2Web (osunlp/Online-Mind2Web) is a GATED HuggingFace dataset: the
download needs an HF token (HF_TOKEN / HUGGING_FACE_HUB_TOKEN env, or the token
file under $HF_HOME / ~/.cache/huggingface). The gate is auto-approve — if the
account has not requested access yet, this script requests it once and retries.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx

from simulator.config import DATA_DIR

SOURCES = {
	'webvoyager_data.jsonl': 'https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/WebVoyager_data.jsonl',
	'gaia_web.jsonl': 'https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/GAIA_web.jsonl',
	'reference_answer.json': 'https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/reference_answer.json',
	'webarena_test.raw.json': 'https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json',
}

# name -> (resolve URL, ask-access URL): gated HF datasets, fetched with an auth token.
GATED_SOURCES = {
	'online_mind2web.json': (
		'https://huggingface.co/datasets/osunlp/Online-Mind2Web/resolve/main/Online_Mind2Web.json',
		'https://huggingface.co/datasets/osunlp/Online-Mind2Web/ask-access',
	),
}


def _hf_token() -> str | None:
	for var in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN'):
		if os.environ.get(var):
			return os.environ[var]
	hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))
	tok = hf_home / 'token'
	if tok.exists():
		return tok.read_text().strip() or None
	return None


def _fetch_gated(name: str, url: str, ask_url: str, dest: Path) -> None:
	token = _hf_token()
	if not token:
		print(f'  SKIP {name}: gated dataset and no HF token found (set HF_TOKEN or `huggingface-cli login`)')
		return
	headers = {'Authorization': f'Bearer {token}'}
	resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=60)
	if resp.status_code in (401, 403):
		# Auto-approve gate: request access for this account once, then retry.
		print(f'  {name}: no access yet — requesting access at {ask_url} ...')
		httpx.post(ask_url, headers=headers, follow_redirects=True, timeout=60)
		resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=60)
	if resp.status_code in (401, 403):
		print(f'  SKIP {name}: access denied — accept the terms in a browser at {ask_url.rsplit("/", 1)[0]}')
		return
	resp.raise_for_status()
	dest.write_bytes(resp.content)
	print(f'  wrote {name} ({len(resp.content):,} bytes)')


def main() -> None:
	ap = argparse.ArgumentParser(prog='python -m simulator.scripts.download_data', description=__doc__)
	ap.add_argument('--force', action='store_true', help='Re-download even if the file already exists.')
	args = ap.parse_args()

	DATA_DIR.mkdir(parents=True, exist_ok=True)
	for name, url in SOURCES.items():
		dest = DATA_DIR / name
		if dest.exists() and not args.force:
			print(f'  skip (exists): {name}')
			continue
		print(f'  downloading {name} ...', flush=True)
		resp = httpx.get(url, follow_redirects=True, timeout=60)
		resp.raise_for_status()
		dest.write_bytes(resp.content)
		print(f'  wrote {name} ({len(resp.content):,} bytes)')
	for name, (url, ask_url) in GATED_SOURCES.items():
		dest = DATA_DIR / name
		if dest.exists() and not args.force:
			print(f'  skip (exists): {name}')
			continue
		print(f'  downloading {name} (gated) ...', flush=True)
		_fetch_gated(name, url, ask_url, dest)
	print(f'datasets in {DATA_DIR}')


if __name__ == '__main__':
	main()
