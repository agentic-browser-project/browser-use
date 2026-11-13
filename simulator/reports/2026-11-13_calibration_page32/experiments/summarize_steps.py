"""Rank all eligible completed steps, retaining raw and normalized ID scores."""
import argparse
import hashlib
import json
from pathlib import Path

from action_metrics import METRIC_VERSION, target_indices_equal


def summarize(root, limit=None):
    settings = json.loads((root / 'settings.json').read_text())
    configs = [(a, b) for a in settings['anchors'] for b in settings['alphas']]
    cap = settings['maximum_chunk_size']
    if cap == 16:
        manifest = json.loads((root / 'inputs/manifest.json').read_text())
        records = [r for r in manifest['records'] if r['included']]
        paths = [(r, root / 'replay' / (r['capture'].replace('/', '_') + '.json')) for r in records]
    elif cap == 32:
        manifest = json.loads((root / 'inputs/candidate_manifest.json').read_text())
        paths = [(r, root / 'replay' / f"webvoyager_new_req_{r['request']:03d}.json") for r in manifest['records']]
    else:
        raise ValueError('Expected the page32/max16 or page64/max32 experiment')
    paths = [(r, p) for r, p in paths if p.exists()]
    if limit is not None:
        if limit <= 0 or limit > len(paths):
            raise ValueError('The diagnostic prefix must be within completed steps')
        paths = paths[:limit]
    totals = {c: {'matches': 0, 'raw_matches': 0} for c in configs}
    counts = {'eligible_steps': 0, 'dense_unfinished': 0, 'dense_unparsed': 0, 'dense_no_target': 0}
    corrections = []
    hashes = {}
    for record, path in paths:
        raw = path.read_bytes()
        r = json.loads(raw)
        assert r['complete'] and r['task'] == record['task'] and r['settings'] == settings
        assert r['max_new_tokens'] == settings['max_new_tokens']
        assert [(v['cap'], v['A'], v['alpha']) for v in r['variants']] == [(cap, a, b) for a, b in configs] + [(0, 0, 0)]
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        dense = r['variants'][-1]
        if not dense['finished']:
            counts['dense_unfinished'] += 1
            continue
        if not dense['parsed']:
            counts['dense_unparsed'] += 1
            continue
        if not dense['indices']:
            counts['dense_no_target'] += 1
            continue
        counts['eligible_steps'] += 1
        for v in r['variants'][:-1]:
            same_type = bool(v['finished'] and v['parsed'] and v['types'] == dense['types'])
            old = bool(same_type and v['indices'] == dense['indices'])
            new = bool(same_type and target_indices_equal(v['indices'], dense['indices']))
            total = totals[(v['A'], v['alpha'])]
            total['raw_matches'] += int(old)
            total['matches'] += int(new)
            if old != new:
                corrections.append(dict(task=r['task'], capture=r['capture'], A=v['A'], alpha=v['alpha'],
                    raw_match=old, normalized_match=new, dense_indices=dense['indices'], sparse_indices=v['indices']))
    rows = []
    denominator = counts['eligible_steps']
    for (a, b), t in totals.items():
        rows.append(dict(A=a, alpha=b, **t, denominator=denominator,
            rate=t['matches'] / denominator if denominator else None,
            raw_rate=t['raw_matches'] / denominator if denominator else None,
            rank=1 + sum(x['matches'] > t['matches'] for x in totals.values()) if denominator else None,
            raw_rank=1 + sum(x['raw_matches'] > t['raw_matches'] for x in totals.values()) if denominator else None))
    rows.sort(key=lambda r: (r['rank'] or 999, r['A'], r['alpha']))
    return dict(metric_version=METRIC_VERSION, aggregation='equal weight per eligible completed step',
        diagnostic_prefix_limit=limit, completed_steps=len(paths), **counts, rankings=rows,
        corrections=corrections, replay_hashes=hashes,
        metric_sha256=hashlib.sha256(Path(__file__).with_name('action_metrics.py').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--limit', type=int, help='Diagnostic completed-step prefix only; omit for the full pool')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.limit)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
