"""Nested law distractors and actual enforcement boundaries, seed42 only."""

from collections import defaultdict
from datetime import date, timedelta

from .sources import catalogue_selection


def nested_catalogue(catalogue, original, exclusions, total=1000):
    eligible = catalogue_selection(catalogue, exclusions['law_ids'], exclusions['family_titles'],
                                   count_per_type=len(catalogue), seed=42)
    original_ids = {row['law_info']['law_id'] for row in original}
    original_families = {row['family_id'] for row in original}
    groups = defaultdict(list)
    for row in eligible:
        if row['law_info']['law_id'] not in original_ids and row['family_id'] not in original_families:
            groups[row['law_info']['law_type']].append(row)
    result, seen = list(original), set(original_families)
    while len(result) < total and any(groups.values()):
        for kind in ('Act', 'CabinetOrder', 'MinisterialOrdinance'):
            while groups[kind] and groups[kind][0]['family_id'] in seen:
                groups[kind].pop(0)
            if groups[kind] and len(result) < total:
                row = groups[kind].pop(0)
                result.append(row)
                seen.add(row['family_id'])
    if len(result) != total:
        raise ValueError(f'Only {len(result)} eligible laws for requested {total}')
    return result


def latest_boundary(law, revisions, start='2018-09-30', end='2026-09-30'):
    candidates = [r for r in revisions if r.get('amendment_enforcement_date')
                  and start < r['amendment_enforcement_date'] <= end
                  and r.get('current_revision_status') in ('CurrentEnforced', 'PreviousEnforced')
                  and not r.get('amendment_scheduled_enforcement_date')]
    if not candidates:
        return None
    revision = max(candidates, key=lambda r: (r['amendment_enforcement_date'], r['law_revision_id']))
    boundary = date.fromisoformat(revision['amendment_enforcement_date'])
    return {'law_id': law['law_info']['law_id'], 'family_id': law['family_id'],
            'law_title': law['revision_info']['law_title'],
            'before': (boundary - timedelta(days=1)).isoformat(), 'after': boundary.isoformat(),
            'day_after': (boundary + timedelta(days=1)).isoformat(),
            'revision_id': revision['law_revision_id'], 'source': 'official e-Gov enforcement revision'}
