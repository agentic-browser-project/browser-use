"""Independent saved-artifact recount, without importing the measurement code."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'simulator/runs/raw_html_context_20260922'
REPORT = Path(__file__).with_name('independent_audit.json')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parts(messages: list[dict]) -> list[tuple[str, str]]:
    out = []
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            out.append((message['role'], content))
        else:
            out.extend((message['role'], part['text']) for part in content if part['type'] == 'text')
    return out


def main() -> None:
    protocol_bytes = (DATA / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    source = Path(protocol['source'])
    tokenizer_path = Path(protocol['tokenizer'])
    assert sha(tokenizer_path.read_bytes()) == protocol['tokenizer_sha256']
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    assert tokenizer.truncation is None and tokenizer.padding is None
    records = [json.loads(p.read_text()) for p in sorted(DATA.glob('captures/*/result.json'))]
    manifest = {(s['task'], s['step']): s for s in protocol['samples']}
    for record in records:
        sample = record['sample']
        assert record['protocol_sha256'] == sha(protocol_bytes)
        assert sample == manifest[(sample['task'], sample['step'])]
        assert sha((source / sample['relative_path']).read_bytes()) == sample['source_sha256']
    alias_count = 0
    for sample in protocol['samples']:
        original, alias = sample['url'], sample['displayed_url']
        if original == alias:
            continue
        alias_count += 1
        base = re.split(r'[?#]', original, maxsplit=1)[0]
        rest = original[len(base):]
        expected = base + rest[:25] + '...' + hashlib.md5(rest.encode()).hexdigest()[:7]
        assert expected == alias
        assert sample['url_resolution'] == 'validated_browser_use_md5_alias'

    complete = [r for r in records if all(k in r for k in ('recorded_tokens', 'raw_tokens', 'dom_tokens'))]
    # Deterministic independent selection: smallest task/step SHA-256 in each site.
    ordered = sorted(complete, key=lambda r: sha((r['sample']['task'] + str(r['sample']['step'])).encode()))
    picked = {}
    for record in ordered:
        picked.setdefault(record['sample']['site'], record)
    selected = list(picked.values())
    # Also exercise each special substitution path when a completed example exists.
    for predicate in (
        lambda r: r['dom_truncated'],
        lambda r: r['sample']['url_resolution'] == 'validated_browser_use_md5_alias',
        lambda r: '(truncated to ' in (source / r['sample']['relative_path']).read_text(),
    ):
        extra = next((r for r in ordered if predicate(r) and r not in selected), None)
        if extra:
            selected.append(extra)

    recounts = []
    pattern = re.compile(r'(<browser_state>[\s\S]*?\nInteractive elements(?: \(truncated to [0-9]+ characters\))?:\n)([\s\S]*?)(</browser_state>)')
    for record in selected:
        sample = record['sample']
        key = f"{sample['task']}__step_{sample['step']:03d}"
        directory = DATA / 'captures' / key
        raw = gzip.decompress((directory / 'raw_before.html.gz').read_bytes()).decode()
        dom = (directory / 'dom_payload.txt').read_text()
        assert sha(raw.encode()) == record['raw_sha256']
        assert sha(dom.encode()) == record['dom_sha256']
        text_parts = parts(json.loads((source / sample['relative_path']).read_text())['messages'])
        modified = {'recorded': [], 'raw': [], 'dom': []}
        changed = 0
        for role, text in text_parts:
            modified['recorded'].append(text)
            matches = list(pattern.finditer(text)) if role == 'user' else []
            if not matches:
                modified['raw'].append(text)
                modified['dom'].append(text)
                continue
            assert len(matches) == 1
            changed += 1
            match = matches[0]
            header = re.sub(r' \(truncated to [0-9]+ characters\)(?=:\n$)', '', match[1])
            for name, payload in (('raw', raw), ('dom', dom)):
                modified[name].append(text[:match.start()] + header + payload + '\n' + match[3] + text[match.end():])
        assert changed == 1
        counts = {name: sum(len(tokenizer.encode(text, add_special_tokens=False).ids) for text in values) for name, values in modified.items()}
        for name, value in counts.items():
            assert value == record[name + '_tokens'], (key, name, value, record[name + '_tokens'])
        recounts.append({'key': key, 'site': sample['site'], **counts})
    report = {
        'protocol_sha256': sha(protocol_bytes),
        'available_results_checked': len(records),
        'all_source_and_protocol_hashes_valid': True,
        'all_selected_aliases_valid': alias_count,
        'independent_recounts': recounts,
        'recount_count': len(recounts),
        'sites_recounted': len({r['site'] for r in recounts}),
        'all_recounts_exact_match': True,
        'available_status_counts': dict(Counter(r['status'] for r in records)),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
