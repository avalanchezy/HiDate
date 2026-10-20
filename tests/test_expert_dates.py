"""Date-schedule selection and revision-level acquisition, without web access."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import prepare_expert_dates as worker


def rev(identity, day, status='PreviousEnforced', scheduled=None):
    return {'law_revision_id':identity,'amendment_enforcement_date':day,
        'current_revision_status':status,'amendment_scheduled_enforcement_date':scheduled}


def test_schedule_preserves_official_same_day_order_and_excludes_future_or_scheduled():
    schedule=[rev('future','2027-01-01','UnEnforced'),
        rev('official-first','2022-04-01'),rev('z-lexicographically-later','2022-04-01'),
        rev('scheduled','2022-03-01','CurrentEnforced','2022-03-01'),rev('old','2020-01-01')]
    selected=worker.choose_revision(schedule,'2022-04-01')
    assert selected['revision']['law_revision_id']=='official-first'
    assert selected['same_day_revision_ids']==['official-first','z-lexicographically-later']
    assert worker.choose_revision(schedule,'2022-03-31')['revision']['law_revision_id']=='old'
    assert worker.choose_revision(schedule,'2019-12-31') is None


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture_args(tmp_path):
    original=tmp_path/'original';extension=tmp_path/'extension';out=tmp_path/'new'
    args=SimpleNamespace(project_root=tmp_path,original=original,extension=extension,output=out,
        dates=tmp_path/'dates.json',workers=2)
    save(args.dates,['2022-03-31','2022-04-01','2023-01-01'])
    save(extension/'selected-laws.json',[{'family_id':lid,'law_info':{'law_id':lid}} for lid in ('A','B')])
    for lid in ('A','B'):
        save(original/'revisions'/f'{lid}.json',{'law_info':{'law_id':lid},'revisions':
            [rev(lid+'-new','2022-04-01'),rev(lid+'-old','2020-01-01')] if lid=='A' else [rev('B-old','2020-01-01')]})
    bindings=[]
    for lid,day,rid in [('A','2021-09-30','A-old'),('B','2021-09-30','B-old')]:
        raw=f'snapshots/{lid}/{day}.json'
        save(original/raw,{'law_info':{'law_id':lid},'revision_info':{'law_revision_id':rid}})
        bindings.append({'law_id':lid,'requested_asof':day,'revision_id':rid,'status':'collected','raw_path':raw})
    (original/'asof-bindings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bindings))
    return args


def test_census_keeps_cartesian_bank_and_downloads_each_missing_revision_once(tmp_path):
    args=fixture_args(tmp_path)
    result=worker.census(args,expected_laws=2)
    assert result['schedule_census_complete'] and not result['ready_for_index']
    assert result['expected_bindings']==result['resolved_bindings']==6
    plan=json.loads((args.output/'download-plan.json').read_text())
    assert len(plan)==1 and plan[0]['revision_id']=='A-new'
    assert plan[0]['target_dates']==['2022-04-01','2023-01-01']
    calls=[]
    def fetch(url,params):
        calls.append((url,params))
        return {'law_info':{'law_id':'A'},'revision_info':{'law_revision_id':'A-new'}}
    done=worker.acquire(args,fetch=fetch)
    assert done['complete'] and len(calls)==1
    bindings=list(worker.rows(args.output/'bindings.jsonl'))
    assert len(bindings)==6 and all(r['status']=='collected' for r in bindings)
    assert all((tmp_path/r['raw_path']).exists() for r in bindings)
    assert worker.read(args.output/'census.json')['ready_for_index']
    assert worker.acquire(args,fetch=lambda *a:pytest.fail('cache hit requested HTTP'))['new_http_requests']==0


def test_acquisition_rejects_asof_response_with_another_revision(tmp_path):
    args=fixture_args(tmp_path);worker.census(args,expected_laws=2)
    wrong=lambda *a:{'law_info':{'law_id':'A'},'revision_info':{'law_revision_id':'wrong'}}
    result=worker.acquire(args,fetch=wrong)
    assert not result['complete'] and result['errors']
    assert not worker.read(args.output/'census.json')['ready_for_index']
    assert not (args.output/'snapshots/A/A-new.json').exists()


def test_census_reports_disagreement_with_saved_actual_asof_response(tmp_path):
    args=fixture_args(tmp_path)
    path=args.original/'asof-bindings.jsonl'
    items=list(worker.rows(path));items[0]['revision_id']='another-actual-version'
    worker.save_rows(path,items)
    result=worker.census(args,expected_laws=2)
    assert not result['schedule_census_complete'] and not result['complete']
    assert result['source_known_asof_mismatches'][0]['actual_revision_id']=='another-actual-version'
    with pytest.raises(ValueError,match='Resolve'):
        worker.acquire(args,fetch=lambda *args:pytest.fail('download before date resolution'))


def test_legacy_pre_enforcement_api_response_is_separate_from_new_valid_date_domain(tmp_path):
    args=fixture_args(tmp_path)
    path=args.original/'asof-bindings.jsonl'
    items=list(worker.rows(path));items[0]['requested_asof']='2019-12-31'
    worker.save_rows(path,items)
    result=worker.census(args,expected_laws=2)
    assert result['schedule_census_complete'] and not result['ready_for_index']
    assert result['source_known_asof_mismatches']==[]
    anomaly=result['legacy_pre_enforcement_source_anomalies'][0]
    assert anomaly['law_id']=='A' and anomaly['actual_enforcement_date']=='2020-01-01'
    assert all(r['revision_id'] for r in worker.rows(args.output/'bindings.jsonl'))


def test_actual_asof_override_applies_only_to_its_confirmed_date(tmp_path):
    args=fixture_args(tmp_path)
    p=args.original/'revisions/A.json';schedule=worker.read(p)
    schedule['revisions'].insert(1,rev('A-alt','2022-04-01'));save(p,schedule)
    raw=tmp_path/'actual/A-alt.json'
    save(raw,{'law_info':{'law_id':'A'},'revision_info':{'law_revision_id':'A-alt'}})
    save(args.output/'actual-asof-overrides.json',[{'law_id':'A','requested_asof':'2022-04-01',
        'revision_id':'A-alt','raw_path':'actual/A-alt.json','snapshot_requested_asof':'2022-04-01'}])
    result=worker.census(args,expected_laws=2)
    records={(r['law_id'],r['requested_asof']):r for r in worker.rows(args.output/'bindings.jsonl')}
    assert result['actual_asof_override_count']==1
    assert records['A','2022-04-01']['revision_id']=='A-alt'
    assert records['A','2022-04-01']['binding_basis']=='official_actual_asof_response'
    assert records['A','2023-01-01']['revision_id']=='A-new'
