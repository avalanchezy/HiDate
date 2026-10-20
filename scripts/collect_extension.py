"""Collect nested1000-law sources and40 new expert-family revision boundaries."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

from hidate.io import write_json
from hidate.extension import nested_catalogue, latest_boundary
from hidate.sources import extract_provisions


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def collect(base, output, total=1000, expert_count=40, workers=4, acquire_only=False):
    base, output = Path(base).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = read(base / 'selected-laws.json')
    selected = nested_catalogue(read(base / 'catalogue.json'), original,
                                read(base / 'source-exclusions.json'), total=total)
    write_json(output / 'selected-laws.json', selected)
    write_json(output / 'nested-corpora.json', {str(n): [r['law_info']['law_id'] for r in selected[:n]]
                                              for n in (len(original), 500, total) if n <= total})
    original_ids = {r['law_info']['law_id'] for r in original}
    extras = [r for r in selected if r['law_info']['law_id'] not in original_ids]
    manifest = {'complete': False, 'job_id': os.environ.get('PBS_JOBID'), 'seed': 42,
                'original_source_dir': str(base), 'original_laws': len(original),
                'selected_laws': len(selected), 'new_laws': len(extras), 'expert_families': expert_count,
                'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'anchor_dates': ['2018-09-30', '2021-09-30', '2024-09-30', '2026-09-30']}
    write_json(output / 'manifest.json', manifest)

    def get(endpoint, params, path):
        if path.exists():
            try:
                return read(path)
            except json.JSONDecodeError:
                # The first collection hit the server quota and left partial cache files.
                path.unlink()
        response = requests.get('https://laws.e-gov.go.jp/api/2/' + endpoint,
                                params=params, timeout=90)
        response.raise_for_status()
        payload = response.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.part')
        temporary.write_bytes(response.content)
        temporary.replace(path)
        return payload

    # Discover boundaries first so the expert packet can be produced immediately.
    experts, boundaries, errors = [], [], []
    for row in extras:
        if row['law_info']['law_type'] != 'Act' or len(experts) >= expert_count:
            continue
        law_id = row['law_info']['law_id']
        payload = get('law_revisions/' + law_id, {'response_format': 'json'},
                      output / 'revisions' / (law_id + '.json'))
        boundary = latest_boundary(row, payload['revisions'])
        if boundary:
            experts.append(row)
            boundaries.append(boundary)
    if len(experts) != expert_count:
        raise ValueError(f'Only {len(experts)} new expert families have a valid enforcement boundary')
    write_json(output / 'expert-selected-laws.json', experts)
    write_json(output / 'expert-boundaries.json', boundaries)
    expert_dates = {r['law_id']: [r['before'], r['after'], r['day_after']] for r in boundaries}
    dates = {r['law_info']['law_id']: list(dict.fromkeys(manifest['anchor_dates'] +
             expert_dates.get(r['law_info']['law_id'], []))) for r in extras}
    by_id = {r['law_info']['law_id']: r for r in extras}

    def snapshot(task):
        law_id, asof = task
        path = output / 'snapshots' / law_id / (asof + '.json')
        result = {'law_id': law_id, 'family_id': by_id[law_id]['family_id'],
                  'source_role': 'full', 'requested_asof': asof,
                  'raw_path': path.relative_to(output).as_posix()}
        try:
            payload = get('law_data/' + law_id, {'asof': asof, 'law_full_text_format': 'json', 'json_format': 'full',
                                                'response_format': 'json'}, path)
            if payload['law_info']['law_id'] != law_id:
                raise ValueError('Returned law ID differs')
            parsed = [] if acquire_only else extract_provisions(payload)
            result.update(status='collected', revision_id=payload['revision_info']['law_revision_id'],
                          law_info=payload['law_info'], revision_info=payload['revision_info'],
                          provision_count=len(parsed))
            return result, parsed
        except (requests.RequestException, KeyError, ValueError) as error:
            result.update(status='collection_failure', error=str(error))
            return result, []

    tasks = [(law_id, day) for law_id, days in dates.items() for day in days]
    # Prioritize expert before/after evidence before the larger distractor census.
    tasks.sort(key=lambda t: (t[0] not in expert_dates, t[0], t[1]))
    bindings_path = output / 'asof-bindings.jsonl'
    count = 0
    with bindings_path.open('w', encoding='utf8') as stream, \
            (output / 'provisions.jsonl').open('w', encoding='utf8') as provisions, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        for binding, parsed in pool.map(snapshot, tasks):
            stream.write(json.dumps(binding, ensure_ascii=False) + '\n')
            stream.flush()
            for record in parsed:
                provisions.write(json.dumps({**record, **{k: binding[k] for k in
                    ('requested_asof', 'raw_path', 'source_role', 'family_id')}}, ensure_ascii=False) + '\n')
            if binding['status'] != 'collected':
                errors.append(binding)
            count += 1
            if count % 20 == 0:
                write_json(output / 'progress.json', {'completed': count, 'requested': len(tasks),
                                                     'failures': len(errors)})
                print(json.dumps({'completed': count, 'requested': len(tasks), 'failures': len(errors)}), flush=True)
    manifest.update(complete=not errors, requested_snapshots=len(tasks), successful_snapshots=count - len(errors),
                    errors=errors, provisions_materialized=not acquire_only,
                    finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('data/raw/hidate/full'))
    parser.add_argument('--output', type=Path, default=Path('data/raw/hidate/extended-benchmark/extension'))
    parser.add_argument('--total', type=int, default=1000)
    parser.add_argument('--expert-count', type=int, default=40)
    parser.add_argument('--acquire-only-local', action='store_true',
                        help='Download source snapshots without materializing provisions')
    args = parser.parse_args()
    result = collect(args.base, args.output, args.total, args.expert_count,
                     workers=2 if args.acquire_only_local else 4, acquire_only=args.acquire_only_local)
    raise SystemExit(0 if result['complete'] else 1)
