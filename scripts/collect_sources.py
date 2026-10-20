"""Capture official dated e-Gov trees and the frozen LawQA source documents."""

import argparse
import json
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests

from hidate.io import write_json
from hidate.sources import catalogue_selection, extract_provisions, source_exclusions


API = "https://laws.e-gov.go.jp/api/2"
DEFAULT_PROTOCOL = Path("configs/source-protocol.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def http_get(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def pdftotext_layout(pdf_path, text_path):
    subprocess.run(["pdftotext", "-layout", str(pdf_path), str(text_path)],
                   check=True, capture_output=True, text=True)


def collect_sources(mode, output_dir, protocol_path=DEFAULT_PROTOCOL, *,
                    transport=None, pdf_to_text=None, sleep=time.sleep):
    """Collect one mode; injected external byte/text boundaries support small fixtures."""
    if mode not in ("probe", "full"):
        raise ValueError("mode must be probe or full")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    transport = transport or http_get
    pdf_to_text = pdf_to_text or pdftotext_layout
    failures, bindings, guidance, selected, laws = [], [], [], [], []
    revision_lists = []
    manifest = {"mode": mode, "complete": False, "started_at_utc": utc_now(),
                "source_output_dir": str(output.resolve()), "protocol": protocol,
                "training_runs": 0, "model_inference_runs": 0,
                "requested_snapshots": 0, "successful_snapshots": 0,
                "requested_guidance": 0, "successful_guidance": 0,
                "successful_revision_lists": 0, "law_ids": [], "catalogue_laws": 0}
    write_json(output / "manifest.json", manifest)
    (output / "requests.jsonl").touch(exist_ok=True)
    (output / "failures.jsonl").touch(exist_ok=True)

    def relative(path):
        return path.relative_to(output).as_posix()

    def append(path, record):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fail(stage, error, **detail):
        record = {"stage": stage, "status": "collection_failure", "error": str(error),
                  "error_type": type(error).__name__, "recorded_at_utc": utc_now(), **detail}
        failures.append(record)
        append(output / "failures.jsonl", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return record

    def progress(stage):
        write_json(output / "progress.json", {"mode": mode, "stage": stage,
            "updated_at_utc": utc_now(), "requested_snapshots": manifest["requested_snapshots"],
            "successful_snapshots": sum(row["status"] == "collected" for row in bindings),
            "requested_guidance": manifest["requested_guidance"],
            "successful_guidance": sum(row["status"] == "collected" for row in guidance),
            "failed_tasks_this_run": len(failures), "complete": False})

    def cached_bytes(url, path):
        if path.exists():
            return path.read_bytes()
        for attempt in range(3):
            try:
                content = transport(url)
                break
            except (requests.RequestException, OSError):
                if attempt == 2:
                    raise
                sleep(attempt + 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        append(output / "requests.jsonl", {"url": url, "raw_path": relative(path),
            "retrieved_at_utc": utc_now(), "bytes": len(content)})
        return content

    def cached_json(endpoint, params, path):
        url = f"{API}/{endpoint}?{urlencode(params)}"
        return json.loads(cached_bytes(url, path).decode("utf-8-sig"))

    source_path, questions_path = Path(protocol["lawqa_sources"]), Path(protocol["lawqa_questions"])
    source_list = json.loads(source_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    (output / "lawqa").mkdir(exist_ok=True)
    shutil.copyfile(source_path, output / "lawqa/law_list.json")
    shutil.copyfile(questions_path, output / "lawqa/selection.json")
    exclusions = source_exclusions(source_list)
    write_json(output / "source-exclusions.json", exclusions)
    manifest.update(lawqa_revision=protocol["lawqa_revision"], lawqa_source_documents=len(source_list),
                    lawqa_questions=len(questions["samples"]), external_law_ids=exclusions["law_ids"],
                    requested_guidance=len(exclusions["guidance"]),
                    lawqa_sources_path="lawqa/law_list.json", lawqa_questions_path="lawqa/selection.json")

    catalogue_complete = mode == "probe"
    if mode == "full":
        offset = 0
        try:
            while True:
                path = output / "catalogue" / protocol["catalogue_asof"] / f"offset-{offset:06d}.json"
                page = cached_json("laws", {"asof": protocol["catalogue_asof"], "limit": 100,
                    "offset": offset, "law_type": ",".join(protocol["law_types"]),
                    "response_format": "json"}, path)
                if not isinstance(page.get("laws"), list):
                    raise ValueError("catalogue response has no laws list")
                laws.extend(page["laws"])
                progress("catalogue")
                next_offset = page.get("next_offset")
                if next_offset is None:
                    if "total_count" in page and len(laws) != page["total_count"]:
                        raise ValueError("catalogue ended before its reported total_count")
                    break
                if not isinstance(next_offset, int) or next_offset <= offset:
                    raise ValueError("catalogue next_offset does not advance")
                offset = next_offset
            selected = catalogue_selection(laws, exclusions["law_ids"], exclusions["family_titles"],
                count_per_type=protocol["count_per_type"], seed=protocol["seed"], protocol=protocol)
            catalogue_complete = True
        except (requests.RequestException, OSError, ValueError, KeyError, TypeError) as error:
            fail("catalogue", error, requested_asof=protocol["catalogue_asof"], offset=offset)
        write_json(output / "catalogue.json", laws)
        write_json(output / "selected-laws.json", selected)

    selected_ids = [row["law_info"]["law_id"] for row in selected]
    selected_families = {row["law_info"]["law_id"]: row["family_id"] for row in selected}
    if mode == "probe":
        law_ids = list(protocol["probe_law_ids"])
        dates_by_id = {law_id: list(protocol["probe_dates"]) for law_id in law_ids}
    else:
        law_ids = list(dict.fromkeys([*selected_ids, *exclusions["law_ids"]]))
        dates_by_id = {law_id: list(protocol["anchor_dates"]) for law_id in law_ids}
        for law_id in exclusions["law_ids"]:
            dates_by_id[law_id] = list(dict.fromkeys([*dates_by_id[law_id], *protocol["external_extra_dates"]]))
    manifest.update(law_ids=law_ids, catalogue_complete=catalogue_complete, catalogue_laws=len(laws),
                    selected_law_ids=selected_ids,
                    selected_by_type=dict(Counter(row["law_info"]["law_type"] for row in selected)),
                    requested_revision_lists=len(law_ids),
                    requested_snapshots=sum(map(len, dates_by_id.values())))
    write_json(output / "collection-plan.json", {"mode": mode, "dates_by_law_id": dates_by_id,
               "guidance": exclusions["guidance"], "catalogue_complete": catalogue_complete})
    progress("revisions")

    for law_id in law_ids:
        path = output / "revisions" / f"{law_id}.json"
        try:
            revisions = cached_json(f"law_revisions/{law_id}", {"response_format": "json"}, path)
            if revisions["law_info"]["law_id"] != law_id or not isinstance(revisions["revisions"], list):
                raise ValueError("revision-list law ID or revisions list is invalid")
            revision_lists.append({"law_id": law_id, "raw_path": relative(path),
                                   "revision_count": len(revisions["revisions"]), "status": "collected"})
        except (requests.RequestException, OSError, ValueError, KeyError, TypeError) as error:
            fail("revision_list", error, law_id=law_id, raw_path=relative(path))

    with (output / "provisions.jsonl").open("w", encoding="utf-8") as provision_stream, \
            (output / "asof-bindings.jsonl").open("w", encoding="utf-8") as binding_stream:
        for law_id in law_ids:
            for date in dates_by_id[law_id]:
                path = output / "snapshots" / law_id / f"{date}.json"
                binding = {"law_id": law_id, "requested_asof": date, "raw_path": relative(path),
                           "source_role": "external" if law_id in exclusions["law_ids"] else mode,
                           "family_id": selected_families.get(law_id, law_id)}
                try:
                    snapshot = cached_json(f"law_data/{law_id}", {"asof": date,
                        "law_full_text_format": "json", "json_format": "full",
                        "response_format": "json"}, path)
                    if snapshot["law_info"]["law_id"] != law_id:
                        raise ValueError("snapshot returned a different law ID")
                    tree = snapshot["law_full_text"]
                    if not isinstance(tree, dict) or tree.get("tag") != "Law" or not isinstance(tree.get("children"), list):
                        raise ValueError("snapshot law_full_text is not a detailed Law JSON tree")
                    revision_id = snapshot["revision_info"]["law_revision_id"]
                    if not revision_id:
                        raise ValueError("snapshot revision_info has no actual revision ID")
                    provisions = extract_provisions(snapshot)
                    if not provisions:
                        raise ValueError("snapshot parsed no provisions")
                    binding.update(status="collected", revision_id=revision_id,
                        law_info=snapshot["law_info"], revision_info=snapshot["revision_info"],
                        provision_count=len(provisions), scopes=sorted({row["scope"] for row in provisions}))
                    for provision in provisions:
                        provision_stream.write(json.dumps({**provision, "requested_asof": date,
                            "raw_path": relative(path), "source_role": binding["source_role"],
                            "family_id": binding["family_id"]}, ensure_ascii=False) + "\n")
                except (requests.RequestException, OSError, ValueError, KeyError, TypeError) as error:
                    failure = fail("snapshot", error, law_id=law_id, requested_asof=date, raw_path=relative(path))
                    binding.update(status="collection_failure", error=failure["error"],
                                   error_type=failure["error_type"])
                bindings.append(binding)
                binding_stream.write(json.dumps(binding, ensure_ascii=False) + "\n")
                binding_stream.flush()
                provision_stream.flush()
                progress("snapshots")

    for index, source in enumerate(exclusions["guidance"], 1):
        basename = Path(urlparse(source["url"]).path).name
        path = output / "guidance" / f"{index:02d}-{basename}"
        text_path = path.with_suffix(".txt")
        archive = protocol.get("guidance_archive_urls", {}).get(source["url"])
        acquisition_url = archive["url"] if archive else source["url"]
        record = {**source, "source_id": f"guidance-{index:02d}", "pdf_path": relative(path),
                  "text_path": relative(text_path), "text_command": "pdftotext -layout",
                  "acquisition_url": acquisition_url}
        if archive:
            record.update(archive_capture=archive["archive_capture"],
                acquisition_provenance="National Diet Library WARP capture of the original source URL")
        try:
            content = cached_bytes(acquisition_url, path)
            if not content.startswith(b"%PDF-"):
                raise ValueError("guidance response is not PDF bytes")
            if not text_path.exists() or not text_path.read_text(encoding="utf-8").strip():
                pdf_to_text(path, text_path)
            text = text_path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError("pdftotext produced no nonempty guidance text")
            record.update(status="collected", text_characters=len(text))
        except (requests.RequestException, OSError, ValueError, subprocess.CalledProcessError) as error:
            failure = fail("guidance", error, url=source["url"], acquisition_url=acquisition_url, pdf_path=relative(path),
                           text_path=relative(text_path))
            record.update(status="collection_failure", error=failure["error"], error_type=failure["error_type"])
        guidance.append(record)
        progress("guidance")
    write_json(output / "guidance.json", guidance)
    write_json(output / "revision-lists.json", revision_lists)

    manifest.update(successful_snapshots=sum(row["status"] == "collected" for row in bindings),
        successful_guidance=sum(row["status"] == "collected" for row in guidance),
        successful_revision_lists=len(revision_lists), failed_tasks_this_run=len(failures),
        current_failures=failures, provision_count=sum(row.get("provision_count", 0) for row in bindings),
        finished_at_utc=utc_now(), bindings_path="asof-bindings.jsonl", provisions_path="provisions.jsonl",
        guidance_path="guidance.json", revision_lists_path="revision-lists.json",
        failures_path="failures.jsonl", requests_path="requests.jsonl")
    manifest["complete"] = (catalogue_complete and not failures
        and manifest["successful_snapshots"] == manifest["requested_snapshots"]
        and manifest["successful_guidance"] == manifest["requested_guidance"]
        and manifest["successful_revision_lists"] == manifest["requested_revision_lists"])
    write_json(output / "manifest.json", manifest)
    write_json(output / "progress.json", {"mode": mode, "stage": "finished", "complete": manifest["complete"],
        "updated_at_utc": manifest["finished_at_utc"], "successful_snapshots": manifest["successful_snapshots"],
        "successful_guidance": manifest["successful_guidance"], "failed_tasks_this_run": len(failures)})
    print(json.dumps({key: manifest[key] for key in ("mode", "complete", "successful_snapshots",
        "requested_snapshots", "successful_guidance", "failed_tasks_this_run")}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "full"), default="probe")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    manifest = collect_sources(args.mode, args.output_dir, args.protocol)
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
