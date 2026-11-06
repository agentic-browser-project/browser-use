"""Revisit recorded WebVoyager URLs and measure paired page representations.

No model requests or recorded actions are executed. See protocol.json for the
preselected population/sample. Saved HTML and text permit offline recounting.
Run with browser-use's uv environment from its repository root.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import json
import random
import re
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'simulator/runs/WebVoyager-GAIA-topk32-30b-fixed-20260628'
OUT = ROOT / 'simulator/runs/raw_html_context_20260922'
TOKENIZER_PATH = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137/tokenizer.json'
SEED = 20260922


class Sample(BaseModel):
    """A preselected original prompt and the URL in its current-tab metadata."""

    site: str
    task: str
    step: int
    relative_path: str
    url: str
    displayed_url: str
    url_resolution: str
    original_title: str
    message_index: int
    part_index: int | None
    payload_start: int
    payload_end: int
    source_sha256: str
    rank: int = 0

    @property
    def key(self) -> str:
        return f'{self.task}__step_{self.step:03d}'


class Capture(BaseModel):
    """Browser-returned page data. Raw HTML is the rendered top-level DOM."""

    url: str
    title: str
    html: str
    text: str
    ready_state: str
    response_status: int | None = None
    iframe_count: int
    open_shadow_roots: int
    scroll_x: float
    scroll_y: float
    document_height: float
    viewport_height: float
    browser_version: str


CAPTURE_JS = '''JSON.stringify({
  url:location.href, title:document.title,
  html:document.documentElement.outerHTML,
  text:document.body ? document.body.innerText : '',
  ready_state:document.readyState,
  response_status:performance.getEntriesByType('navigation')[0]?.responseStatus ?? null,
  iframe_count:document.querySelectorAll('iframe,frame').length,
  open_shadow_roots:Array.from(document.querySelectorAll('*')).filter(e=>e.shadowRoot).length,
  scroll_x:scrollX, scroll_y:scrollY,
  document_height:document.documentElement.scrollHeight,
  viewport_height:innerHeight, browser_version:navigator.userAgent
})'''


def text_parts(messages: list[dict]):
    """Yield each textual content part separately, excluding image payloads."""
    for mi, m in enumerate(messages):
        c = m['content']
        if isinstance(c, str):
            yield mi, None, c
        elif isinstance(c, list):
            for pi, p in enumerate(c):
                if p.get('type') == 'text':
                    yield mi, pi, p['text']


def shortened_url(url: str) -> str:
    """Reproduce browser-use's 25-character query/fragment alias exactly."""
    starts = [i for i in (url.find('?'), url.find('#')) if i >= 0]
    start = min(starts) if starts else len(url)
    base, tail = url[:start], url[start:]
    if len(tail) <= 25:
        return url
    alias = base + tail[:25] + '...' + hashlib.md5(tail.encode()).hexdigest()[:7]
    return alias if len(alias) < len(url) else url


def full_url_candidates(task_dir: Path) -> set[str]:
    """Read only structured URL fields from history and per-step states."""
    found = set()

    def visit(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in ('url', 'start_url') and isinstance(v, str) and v.startswith(('http://', 'https://')):
                    found.add(v)
                elif isinstance(v, (dict, list)):
                    visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)

    for p in [task_dir / 'history.json', *task_dir.glob('step_*/state.json')]:
        if p.exists():
            visit(json.loads(p.read_text()))
    return found


def build_sample(n: int) -> None:
    """Sample uniformly without replacement within each task website."""
    if (OUT / 'protocol.json').exists():
        raise ValueError('Protocol exists. Reuse its immutable sample with --run.')
    OUT.mkdir(parents=True, exist_ok=True)
    by_site = defaultdict(list)
    exclusions = []
    total = 0
    for td in sorted(SOURCE.glob('webvoyager__*')):
        meta = json.loads((td / 'meta.json').read_text())
        candidates = full_url_candidates(td)
        for mp in sorted(td.glob('step_*/messages.json')):
            total += 1
            data = mp.read_bytes()
            messages = json.loads(data)['messages']
            matches = []
            for mi, pi, text in text_parts(messages):
                if messages[mi]['role'] != 'user':
                    continue
                for bs in re.finditer(r'<browser_state>\n(.*?)</browser_state>', text, re.S):
                    match = re.search(r'\nInteractive elements(?: \(truncated to \d+ characters\))?:\n', bs[1])
                    if match:
                        matches.append((mi, pi, text, bs, match))
            reason = None
            if len(matches) != 1:
                reason = 'missing_or_ambiguous_browser_payload'
            else:
                mi, pi, text, bs, match = matches[0]
                tab = re.search(r'^Current tab: (\S+)\s*$', bs[1], re.M)
                url = re.search(r'^Tab ' + re.escape(tab[1]) + r': (https?://\S+) - (.*)$', bs[1], re.M) if tab else None
                if not url:
                    reason = 'missing_unambiguous_http_current_tab'
                elif urlparse(url[1]).path.lower().endswith('.pdf') or '/pdf/' in urlparse(url[1]).path:
                    reason = 'pdf_viewer_url'
                elif re.search(r'\.\.\.[0-9a-f]{7}$', url[1]):
                    originals = [v for v in candidates if shortened_url(v) == url[1] and v != url[1]]
                    if len(originals) != 1:
                        reason = 'unresolved_or_ambiguous_shortened_url'
                    else:
                        resolved_url, resolution = originals[0], 'validated_browser_use_md5_alias'
                else:
                    resolved_url, resolution = url[1], 'literal_message_url'
            if reason:
                exclusions.append({'path': str(mp.relative_to(SOURCE)), 'reason': reason})
                continue
            item = Sample(site=meta['site'], task=td.name, step=int(mp.parent.name.split('_')[1]),
                          relative_path=str(mp.relative_to(SOURCE)), url=resolved_url, displayed_url=url[1],
                          url_resolution=resolution, original_title=url[2],
                          message_index=mi, part_index=pi, payload_start=bs.start(1)+match.end(),
                          payload_end=bs.end(1), source_sha256=hashlib.sha256(data).hexdigest())
            by_site[meta['site']].append(item)
    sample = []
    counts = {}
    for site, rows in sorted(by_site.items()):
        rng = random.Random(f'{SEED}:{site}')
        rng.shuffle(rows)
        selected = rows[:n]
        for rank, item in enumerate(selected):
            item.rank = rank
        sample.extend(selected)
        counts[site] = {'eligible_steps': len(rows), 'selected': len(selected),
                        'selected_tasks': len({r.task for r in selected})}
    sample.sort(key=lambda x: (x.rank, x.site))
    protocol = {
        'source': str(SOURCE), 'source_step_count': total, 'seed': SEED,
        'sampling': 'uniform recorded steps without replacement within task website; no replacement after capture failure',
        'measurement': 'live URL revisit, not action replay or historical raw HTML recovery',
        'url_resolution': 'message current-tab URL; shortened aliases resolved against structured original URLs by exact browser-use MD5 alias verification',
        'raw_html': 'untruncated rendered top-level documentElement.outerHTML including script/style/head/hidden markup; excludes iframe documents, shadow-root interiors, external resource bodies',
        'baseline': 'fresh browser-use llm_representation, default 40000-character cap plus page boundary markers',
        'tokens': 'Qwen3-32B sum of independently encoded text content parts, no special tokens, no images/chat-template/tool-schema/current output',
        'tokenizer': str(TOKENIZER_PATH), 'tokenizer_sha256': hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest(),
        'settle': 'minimum seven seconds after navigation, then wait up to 20s for document.readyState=complete; no claim of network idleness',
        'counts': counts, 'exclusions': exclusions,
        'samples': [s.model_dump() for s in sample],
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps({'source_steps': total, 'exclusions': len(exclusions), 'by_site': counts}), flush=True)


def count_variants(sample: Sample, tokenizer: Tokenizer, payloads: dict[str, str]) -> dict:
    """Retokenize whole modified text parts, preserving BPE boundary effects."""
    data = (SOURCE / sample.relative_path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sample.source_sha256
    messages = json.loads(data)['messages']
    tokens = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
    result = {'recorded_tokens': 0, **{key + '_tokens': 0 for key in payloads}}
    changed_parts = 0
    for mi, pi, text in text_parts(messages):
        original_count = tokens(text)
        result['recorded_tokens'] += original_count
        target = mi == sample.message_index and pi == sample.part_index
        if target:
            changed_parts += 1
            # Retain the original heading identically in both paired variants.
            # A historical "truncated" heading is changed identically in BOTH.
            prefix = re.sub(r'Interactive elements \(truncated to \d+ characters\):\n$',
                            'Interactive elements:\n', text[:sample.payload_start])
            suffix = text[sample.payload_end:]
            result['recorded_payload_tokens'] = tokens(text[sample.payload_start:sample.payload_end])
            result['frozen_text_tokens'] = result.get('frozen_text_tokens', 0) + tokens(prefix + suffix)
        else:
            result['frozen_text_tokens'] = result.get('frozen_text_tokens', 0) + original_count
        for key, payload in payloads.items():
            variant = prefix + payload + '\n' + suffix if target else text
            result[key + '_tokens'] += tokens(variant) if target else original_count
    assert changed_parts == 1
    return result


async def evaluate(cdp) -> Capture:
    response = await cdp.cdp_client.send.Runtime.evaluate(
        params={'expression': CAPTURE_JS, 'returnByValue': True}, session_id=cdp.session_id)
    if response.get('exceptionDetails'):
        raise RuntimeError(str(response['exceptionDetails']))
    return Capture.model_validate_json(response['result']['value'])


def classify(capture: Capture) -> str:
    """Flag obvious failures without filtering merely because HTML is short."""
    title = capture.title.lower().strip()
    body = capture.text[:5000].lower()
    if capture.url.startswith(('chrome-error:', 'about:', 'data:')):
        return 'browser_error'
    if capture.response_status and capture.response_status >= 400:
        return 'http_error'
    if re.search(r'^(just a moment|access denied|robot check|attention required|verify|captcha|sorry|403|404|error)', title):
        return 'challenge_or_error'
    patterns = ['verify you are human', 'verify that you are human', 'unusual traffic from your computer',
                'not a robot', 'enter the characters you see below', 'performing security verification',
                'please enable js and disable any ad blocker', 'pardon our interruption']
    if len(capture.text) < 6000 and any(p in body for p in patterns):
        return 'challenge_or_error'
    if not capture.html or not capture.text.strip():
        return 'empty_document'
    return 'captured'


async def measure_one(sample: Sample, tokenizer: Tokenizer) -> dict:
    """Capture a selected URL in a fresh isolated headed browser profile."""
    from browser_use import Browser

    dest = OUT / 'captures' / sample.key
    dest.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix='browsa_html_')
    browser = Browser(headless=False, enable_default_extensions=False, user_data_dir=profile,
                      max_iframes=15, viewport={'width': 1280, 'height': 720},
                      window_size={'width': 1280, 'height': 720})
    result = {'sample': sample.model_dump(), 'capture_started_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    start = time.monotonic()
    try:
        async with asyncio.timeout(95):
            await browser.start()
            cdp = await browser.get_or_create_cdp_session()
            nav = await cdp.cdp_client.send.Page.navigate(params={'url': sample.url}, session_id=cdp.session_id)
            result['navigation_error'] = nav.get('errorText')
            await asyncio.sleep(7)
            before = await evaluate(cdp)
            deadline = time.monotonic() + 20
            while before.ready_state != 'complete' and time.monotonic() < deadline:
                await asyncio.sleep(1)
                before = await evaluate(cdp)
            result['status'] = classify(before)
            if nav.get('errorText'):
                result['status'] = 'navigation_error'
            result['page'] = before.model_dump(exclude={'html', 'text'})
            result['body_chars'] = len(before.text)
            gzip.open(dest / 'raw_before.html.gz', 'wt', encoding='utf-8').write(before.html)
            (dest / 'visible_text.txt').write_text(before.text)
            try:
                shot = await cdp.cdp_client.send.Page.captureScreenshot(params={'format':'jpeg','quality':60}, session_id=cdp.session_id)
                (dest / 'screenshot.jpg').write_bytes(base64.b64decode(shot['data']))
            except Exception as e:
                result['screenshot_error'] = str(e)[:200]
            dom_started = time.monotonic()
            state = await browser.get_browser_state_summary(include_screenshot=False)
            dom_full = state.dom_state.llm_representation()
            if not state.dom_state._root or dom_full.startswith('Empty DOM tree'):
                result['status'] = 'dom_extraction_failed'
            result['serializer_seconds'] = time.monotonic() - dom_started
            dom = dom_full[:40000]
            pi = state.page_info
            if pi and pi.pixels_above <= 0:
                dom = '[Start of page]\n' + dom
            if pi and pi.pixels_below <= 0:
                dom += '\n[End of page]'
            result['dom_chars_before_cap'] = len(dom_full)
            result['dom_truncated'] = len(dom_full) > 40000
            (dest / 'dom_full.txt').write_text(dom_full)
            (dest / 'dom_payload.txt').write_text(dom)
            after = await evaluate(cdp)
            after_status = classify(after)
            if result['status'] == 'captured' and after_status != 'captured':
                result['status'] = 'after_' + after_status
            gzip.open(dest / 'raw_after.html.gz', 'wt', encoding='utf-8').write(after.html)
            result['url_changed_during_pair'] = before.url != after.url
            result['redirected'] = sample.url != before.url
            if result['status'] == 'captured' and result['url_changed_during_pair']:
                result['status'] = 'url_changed_during_pair'
            # Re-tokenize every complete modified part, never approximate by subtraction.
            result.update(count_variants(sample, tokenizer, {'raw': before.html, 'raw_after': after.html, 'dom': dom}))
            result['raw_drift_fraction'] = abs(result['raw_tokens'] - result['raw_after_tokens']) / result['raw_tokens']
            result['raw_sha256'] = hashlib.sha256(before.html.encode()).hexdigest()
            result['dom_sha256'] = hashlib.sha256(dom.encode()).hexdigest()
    except Exception as e:
        result['status'] = 'capture_exception'
        result['error'] = f'{type(e).__name__}: {str(e)[:500]}'
    finally:
        try:
            await asyncio.wait_for(browser.kill(), timeout=15)
        except BaseException as e:
            result['cleanup_error'] = f'{type(e).__name__}: {str(e)[:200]}'
        shutil.rmtree(profile, ignore_errors=True)
    result['elapsed_seconds'] = time.monotonic() - start
    temp_result = dest / 'result.json.tmp'
    temp_result.write_text(json.dumps(result, indent=2))
    temp_result.replace(dest / 'result.json')
    print(json.dumps({'sample':sample.key, 'site':sample.site, 'status':result['status'],
                      'raw':result.get('raw_tokens'), 'dom':result.get('dom_tokens'),
                      'elapsed':round(result['elapsed_seconds'], 1)}), flush=True)
    return result


async def run(workers: int, max_rank: int) -> None:
    protocol = json.loads((OUT / 'protocol.json').read_text())
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    queue = asyncio.Queue()
    for value in protocol['samples']:
        s = Sample.model_validate(value)
        previous = OUT / 'captures' / s.key / 'result.json'
        if previous.exists():
            assert json.loads(previous.read_text())['protocol_sha256'] == hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest()
        if s.rank < max_rank and not (OUT / 'captures' / s.key / 'result.json').exists():
            queue.put_nowait(s)

    async def worker():
        while not queue.empty():
            try:
                s = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await measure_one(s, tokenizer)

    await asyncio.gather(*(worker() for _ in range(workers)))
    results = [json.loads(p.read_text()) for p in sorted((OUT / 'captures').glob('*/result.json'))]
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    print('FINISHED', dict(Counter(r['status'] for r in results)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample', type=int, help='Create fixed N per-website sample, without visiting pages')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--max-rank', type=int, default=30)
    args = ap.parse_args()
    if args.sample:
        build_sample(args.sample)
    if args.run:
        asyncio.run(run(args.workers, args.max_rank))
