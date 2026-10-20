"""Official date coverage for the unchanged1000-law expert evaluation bank.

Only catalogue identities and target dates enter this source-acquisition worker.
No expert gold, questions, model output, fitting or retrieval is read here.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode

import requests


API='https://laws.e-gov.go.jp/api/2/'
ANCHORS=('2018-09-30','2021-09-30','2024-09-30','2026-09-30')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def save_rows(path,values):
    with Path(path).open('w',encoding='utf-8') as stream:
        for value in values:stream.write(json.dumps(value,ensure_ascii=False)+'\n')


def receipt(**values):
    return {'at_utc':datetime.now(timezone.utc).isoformat(),'job_id':os.environ.get('PBS_JOBID'),
        'expert_gold_read':False,'model_calls':0,'training_runs':0,**values}


def relative(args,path):return Path(path).resolve().relative_to(args.project_root.resolve()).as_posix()


def choose_revision(revisions,target_date):
    """Latest actually enforced date, retaining official list order for ties.

    This rule is checked against every existing actual-asof response before the
    census can become ready. A retrieval timestamp is never an effective date.
    """
    eligible=[r for r in revisions if r.get('amendment_enforcement_date')
        and r['amendment_enforcement_date']<=target_date
        and r.get('current_revision_status') in ('CurrentEnforced','PreviousEnforced')
        and not r.get('amendment_scheduled_enforcement_date')]
    if not eligible:return None
    latest=max(r['amendment_enforcement_date'] for r in eligible)
    same=[r for r in eligible if r['amendment_enforcement_date']==latest]
    return {'revision':same[0],'same_day_revision_ids':[r['law_revision_id'] for r in same]}


def catalogue(args,expected_laws):
    selected=read(args.extension/'selected-laws.json')
    if len(selected)!=expected_laws or len({r['law_info']['law_id'] for r in selected})!=expected_laws:
        raise ValueError('Preserve the exact selected1000-law catalogue')
    return selected


def schedule_path(args,law_id):
    for root in (args.original,args.extension,args.output):
        path=root/'revisions'/(law_id+'.json')
        if path.exists():return path
    return None


def http_json(url,params):
    response=requests.get(url,params=params,timeout=90)
    response.raise_for_status()
    return response.json()


def snapshot_identity(payload,law_id,revision_id):
    if (payload['law_info']['law_id']!=law_id
            or payload['revision_info']['law_revision_id']!=revision_id):
        raise ValueError('Official asof response differs: expected '+law_id+'/'+revision_id+
            '; returned '+payload['law_info']['law_id']+'/'+payload['revision_info']['law_revision_id'])


def fetch_schedules(args,*,fetch=None,expected_laws=1000):
    selected=catalogue(args,expected_laws);fetch=fetch or http_json
    args.output.mkdir(parents=True,exist_ok=True)
    missing=[r['law_info']['law_id'] for r in selected if schedule_path(args,r['law_info']['law_id']) is None]
    successes=[];errors=[]
    def one(law_id):
        try:
            payload=fetch(API+'law_revisions/'+law_id,{'response_format':'json'})
            if payload['law_info']['law_id']!=law_id or not isinstance(payload['revisions'],list):
                raise ValueError('Official revision schedule returned another law')
            path=args.output/'revisions'/(law_id+'.json');save(path,payload)
            return {'law_id':law_id,'raw_path':relative(args,path),'revision_count':len(payload['revisions'])},None
        except (requests.RequestException,ValueError,KeyError) as error:
            return None,{'law_id':law_id,'error':str(error),'endpoint':API+'law_revisions/'+law_id}
    # One actual request establishes this host's official route before the batch.
    pending=missing
    if pending:
        ok,error=one(pending[0])
        if error:errors.append(error);pending=[]
        else:successes.append(ok);pending=pending[1:]
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for ok,error in pool.map(one,pending):
                (errors if error else successes).append(error or ok)
                if (len(successes)+len(errors))%50==0:
                    print(json.dumps({'phase':'schedules','new_complete':len(successes),'new_errors':len(errors)}),flush=True)
    available=sum(schedule_path(args,r['law_info']['law_id']) is not None for r in selected)
    result=receipt(complete=available==expected_laws,selected_laws=expected_laws,
        schedules_available=available,previously_cached=expected_laws-len(missing),
        new_http_requests=len(successes)+len(errors),new_schedules=len(successes),errors=errors,
        sources=[str(args.original/'revisions'),str(args.extension/'revisions'),str(args.output/'revisions')])
    save(args.output/'schedules.json',result);return result


def known_sources(args,law_ids):
    bindings={};cached={};indexed=set()
    for root in (args.original,args.extension):
        path=root/'asof-bindings.jsonl'
        if not path.exists():continue
        for row in rows(path):
            if row['law_id'] not in law_ids or row.get('status')!='collected':continue
            bindings[row['law_id'],row['requested_asof']]=row
            raw=root/row['raw_path']
            key=(row['law_id'],row['revision_id'])
            if row['requested_asof'] in ANCHORS:indexed.add(key)
            if raw.exists():cached.setdefault(key,{'path':raw,'requested_asof':row['requested_asof']})
    return bindings,cached,indexed


def census(args,*,expected_laws=1000):
    selected=catalogue(args,expected_laws);args.output.mkdir(parents=True,exist_ok=True)
    target_dates=sorted(set(read(args.dates)))
    for day in target_dates:date.fromisoformat(day)
    if not target_dates:raise ValueError('Provide the actual expert target dates')
    law_ids=[r['law_info']['law_id'] for r in selected]
    known,cached,indexed=known_sources(args,set(law_ids))
    override_path=args.output/'actual-asof-overrides.json'
    overrides={(r['law_id'],r['requested_asof']):r for r in read(override_path)} if override_path.exists() else {}
    for (law_id,day),row in overrides.items():
        if law_id not in law_ids or day not in target_dates:
            raise ValueError('Actual-asof overrides must belong to the frozen bank/date domain')
        path=args.project_root/row['raw_path']
        snapshot_identity(read(path),law_id,row['revision_id'])
        cached.setdefault((law_id,row['revision_id']),{'path':path,
            'requested_asof':row['snapshot_requested_asof']})
    schedules={};missing_schedules=[];mismatches=[];legacy_pre_enforcement=[]
    for law_id in law_ids:
        path=schedule_path(args,law_id)
        if path is None:missing_schedules.append(law_id);continue
        payload=read(path)
        if payload['law_info']['law_id']!=law_id:raise ValueError('Schedule law identity differs')
        schedules[law_id]=(payload['revisions'],path)
    for (law_id,day),actual in known.items():
        if law_id not in schedules:continue
        chosen=choose_revision(schedules[law_id][0],day)
        actual_revision=next((r for r in schedules[law_id][0]
            if r['law_revision_id']==actual['revision_id']),None)
        if (chosen is None and actual_revision is not None
                and actual_revision.get('amendment_enforcement_date','')>day):
            # e-Gov can return its first future revision before initial enactment.
            # Keep that old source record, but never alias its future text to a
            # new target date. New cells still require choose_revision success.
            legacy_pre_enforcement.append({'law_id':law_id,'requested_asof':day,
                'actual_revision_id':actual['revision_id'],
                'actual_enforcement_date':actual_revision['amendment_enforcement_date']})
            continue
        if chosen is None or chosen['revision']['law_revision_id']!=actual['revision_id']:
            mismatches.append({'law_id':law_id,'requested_asof':day,'actual_revision_id':actual['revision_id'],
                'selected_revision_id':chosen['revision']['law_revision_id'] if chosen else None})
    bindings=[];required=defaultdict(list);unresolved=[]
    for law in selected:
        law_id=law['law_info']['law_id']
        for day in target_dates:
            chosen=choose_revision(schedules[law_id][0],day) if law_id in schedules else None
            override=overrides.get((law_id,day))
            if override is not None:
                actual=next((r for r in schedules[law_id][0]
                    if r['law_revision_id']==override['revision_id']),None)
                if actual is None or actual['amendment_enforcement_date']>day:
                    raise ValueError('Actual-asof override revision is absent or not yet effective')
                chosen={'revision':actual,'same_day_revision_ids':chosen['same_day_revision_ids'] if chosen else []}
            if chosen is None:
                row={'law_id':law_id,'family_id':law['family_id'],'source_role':'full','requested_asof':day,
                    'revision_id':None,'status':'unresolved_no_enforced_revision' if law_id in schedules else 'unresolved_missing_schedule',
                    'raw_path':None}
                bindings.append(row);unresolved.append(row);continue
            revision=chosen['revision'];rid=revision['law_revision_id'];required[law_id,rid].append(day)
            cached_row=cached.get((law_id,rid));new_path=args.output/'snapshots'/law_id/(rid+'.json')
            if cached_row is None and new_path.exists():
                snapshot_identity(read(new_path),law_id,rid)
                cached_row={'path':new_path,'requested_asof':day};cached[law_id,rid]=cached_row
            path=cached_row['path'] if cached_row else new_path
            bindings.append({'law_id':law_id,'family_id':law['family_id'],'source_role':'full',
                'requested_asof':day,'revision_id':rid,'status':'collected' if cached_row else 'pending_snapshot',
                'raw_path':relative(args,path),'raw_source_root':'.',
                'snapshot_requested_asof':cached_row['requested_asof'] if cached_row else None,
                'binding_basis':'official_actual_asof_response' if override else 'official_enforcement_schedule_and_same_revision_snapshot',
                'schedule_path':relative(args,schedules[law_id][1]),
                'amendment_enforcement_date':revision['amendment_enforcement_date'],
                'amendment_promulgate_date':revision.get('amendment_promulgate_date'),
                'repeal_status':revision.get('repeal_status'),'repeal_date':revision.get('repeal_date'),
                'same_day_revision_ids':chosen['same_day_revision_ids'],
                'official_url':API+'law_data/'+law_id+'?'+urlencode({'asof':day,
                    'law_full_text_format':'json','json_format':'full','response_format':'json'})})
    plan=[]
    for (law_id,rid),days in sorted(required.items()):
        if (law_id,rid) not in cached:
            plan.append({'law_id':law_id,'revision_id':rid,'representative_asof':min(days),'target_dates':days,
                'raw_path':relative(args,args.output/'snapshots'/law_id/(rid+'.json'))})
    metadata_complete=not(missing_schedules or unresolved or mismatches)
    ready=metadata_complete and not plan
    result=receipt(complete=ready,ready_for_index=ready,schedule_census_complete=metadata_complete,
        target_dates=target_dates,law_ids=law_ids,selected_laws=len(law_ids),expected_bindings=len(law_ids)*len(target_dates),
        resolved_bindings=sum(r['revision_id'] is not None for r in bindings),
        collected_bindings=sum(r['status']=='collected' for r in bindings),
        required_unique_revisions=len(required),already_indexed_unique_revisions=len(set(required)&indexed),
        not_in_base_index_unique_revisions=len(set(required)-indexed),
        cached_snapshot_unique_revisions=len(set(required)&set(cached)),missing_snapshot_unique_revisions=len(plan),
        source_known_asof_checks=sum(law in schedules for law,day in known),
        source_known_asof_comparable_checks=sum(law in schedules for law,day in known)-len(legacy_pre_enforcement),
        source_known_asof_mismatches=mismatches,legacy_pre_enforcement_source_anomalies=legacy_pre_enforcement,
        actual_asof_override_count=len(overrides),actual_asof_override_law_ids=sorted({law for law,day in overrides}),
        actual_asof_override_path=relative(args,override_path) if overrides else None,
        missing_schedules=missing_schedules,unresolved=unresolved,
        same_day_tie_binding_cells=sum(len(r.get('same_day_revision_ids',[]))>1 for r in bindings),
        selection_rule='latest enforced date <= T; official schedule order within same-day ties; all old actual asof bindings checked',
        original_anchors=list(ANCHORS),catalogue_path=str(args.extension/'selected-laws.json'),
        date_list_path=str(args.dates),binding_file='bindings.jsonl',download_plan='download-plan.json')
    save_rows(args.output/'bindings.jsonl',bindings);save(args.output/'download-plan.json',plan)
    save(args.output/'census.json',result);return result


def acquire(args,*,fetch=None):
    meta=read(args.output/'census.json');fetch=fetch or http_json
    if not meta['schedule_census_complete']:raise ValueError('Resolve the full date-schedule census before downloading')
    plan=read(getattr(args,'download_plan',None) or args.output/'download-plan.json');errors=[];records=[]
    def one(task):
        path=args.project_root/task['raw_path']
        params={'asof':task['representative_asof'],'law_full_text_format':'json','json_format':'full','response_format':'json'}
        try:
            hit=path.exists()
            if not hit and getattr(args,'delay',0):time.sleep(args.delay)
            payload=read(path) if hit else fetch(API+'law_data/'+task['law_id'],params)
            snapshot_identity(payload,task['law_id'],task['revision_id'])
            if not hit:
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.json.part')
                temporary.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
                temporary.replace(path)
            return {**task,'status':'collected','requested_asof':task['representative_asof'],
                'new_http_request':not hit,'bytes':path.stat().st_size},None
        except (requests.RequestException,ValueError,KeyError) as error:
            return None,{**task,'error':str(error)}
    pending=plan
    if pending:
        ok,error=one(pending[0])
        if error:errors.append(error);pending=[]
        else:records.append(ok);pending=pending[1:]
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for ok,error in pool.map(one,pending):
                (errors if error else records).append(error or ok)
                if (len(records)+len(errors))%25==0:
                    print(json.dumps({'phase':'snapshot_acquisition','revisions':len(records),'errors':len(errors)}),flush=True)
    acquired={(r['law_id'],r['revision_id']):r for r in records}
    bindings=list(rows(args.output/'bindings.jsonl'))
    for row in bindings:
        key=(row['law_id'],row['revision_id'])
        if key in acquired:
            row.update(status='collected',snapshot_requested_asof=acquired[key]['representative_asof'])
    complete=not errors and all(r['status']=='collected' for r in bindings)
    save_rows(args.output/'bindings.jsonl',bindings)
    result=receipt(complete=complete,unique_revision_tasks=len(plan),collected_unique_revisions=len(records),
        new_http_requests=sum(r['new_http_request'] for r in records)+len(errors),
        cache_hits=sum(not r['new_http_request'] for r in records),errors=errors,records=records,
        fulltext_request_policy='one representative actual-asof request per missing revision, then share exact revision across its target dates')
    save(args.output/'acquisition.json',result)
    meta.update(complete=complete,ready_for_index=complete,
        collected_bindings=sum(r['status']=='collected' for r in bindings),
        acquisition_complete=complete,acquisition_receipt='acquisition.json')
    save(args.output/'census.json',meta);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('schedules','census','acquire','all'))
    parser.add_argument('--project-root',type=Path,default=Path('.'))
    parser.add_argument('--original',type=Path,default=Path('data/raw/hidate/full'))
    parser.add_argument('--extension',type=Path,default=Path('data/raw/hidate/extended-benchmark/extension'))
    parser.add_argument('--output',type=Path,default=Path('data/raw/hidate/expert-dates'))
    parser.add_argument('--dates',type=Path,default=Path('data/raw/hidate/expert-dates/target-dates.json'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--delay',type=float,default=0.0,help='Seconds before each missing HTTP request')
    parser.add_argument('--download-plan',type=Path,default=None,help='Explicit remaining-source subset; full original plan stays preserved')
    parser.add_argument('--acquire-only-local',action='store_true')
    args=parser.parse_args()
    if args.acquire_only_local and args.phase not in ('schedules','acquire'):
        parser.error('--acquire-only-local supports the schedules and acquire phases')
    for phase,function in (('schedules',fetch_schedules),('census',census),('acquire',acquire)):
        if args.phase in (phase,'all'):
            result=function(args)
            print(json.dumps({k:result[k] for k in ('complete','ready_for_index','schedule_census_complete',
                'schedules_available','new_schedules','new_http_requests','expected_bindings','resolved_bindings',
                'not_in_base_index_unique_revisions','missing_snapshot_unique_revisions') if k in result}),flush=True)
            if phase=='schedules' and not result['complete']:return 1
            if phase=='census' and not result['schedule_census_complete']:return 1
            if phase=='acquire' and not result['complete']:return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
