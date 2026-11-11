"""Validate measured performance evidence at the actual train/inference entry.

No model is loaded and no benchmark is launched by this module. Formal callers
must supply their *effective* parameters, including any hard-coded overrides.
Training reuse must trace back to an accepted capacity/throughput selection.
A successful 2--3 step smoke test checks compatibility, not selection quality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "wam_performance_preflight_v1"
REUSE_SCHEMA = "wam_performance_reuse_v1"
REPORT_ENV = "CARE_DRIVE_PERFORMANCE_PREFLIGHT"
# These are compared separately, or do not change the measured execution shape.
TUNING_KEYS = frozenset({
    "microbatch", "microbatch_size", "base_microbatch", "cached_microbatch",
    "batch_size", "gradient_accumulation_steps", "accumulation", "world_size",
    "gradient_checkpointing", "checkpointing", "workers_per_rank", "loader_workers",
    "epochs", "updates", "seed", "experiment", "checkpoint_epochs",
    "checkpoint_every_epochs", "checkpoint_every_steps", "performance_preflight",
    "effective_batch", "effective_batch_size",
})


class PerformancePreflightError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise PerformancePreflightError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _shape_config(value):
    if isinstance(value, dict):
        return {k: _shape_config(v) for k, v in value.items() if k not in TUNING_KEYS}
    if isinstance(value, (list, tuple)):
        return [_shape_config(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def code_identity(root=ROOT):
    """Measurement provenance, not a decision about whether to repeat a search."""
    files = []
    for directory in (root / "care_drive",):
        files.extend(directory.rglob("*.py"))
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(files)})


def observe_resources(backend="npu"):
    """Read each worker's real device before model allocation; no device changes."""
    import torch
    count = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "cpu":
        return {"backend": "cpu", "device_model": platform.machine(), "world_size": count,
                "device_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
                "software": {"torch": torch.__version__}, "threads": torch.get_num_threads()}
    if backend == "cuda":
        props = torch.cuda.get_device_properties(local)
        return {"backend": "cuda", "device_model": props.name, "world_size": count,
                "device_memory_bytes": props.total_memory,
                "software": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                             "transformers": importlib.metadata.version("transformers")}}
    _require(backend == "npu", "Unsupported device backend")
    import torch_npu  # noqa: F401 - registers the NPU API
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        ids = [v.strip() for v in visible.split(",") if v.strip()]
        _require(len(set(ids)) == len(ids) and len(ids) >= count,
                 "Visible NPU allocation is smaller than WORLD_SIZE or contains duplicates")
    props = torch.npu.get_device_properties(local)
    versions = {}
    for name in ("torch", "torch-npu", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {"backend": "npu", "device_model": str(torch.npu.get_device_name(local)),
            "device_memory_bytes": int(props.total_memory), "world_size": count,
            "software": versions}


def describe(stage, config, parameters, *, workload):
    _require(stage in ("training", "inference"), "Unknown performance stage")
    return {"stage": stage, "workload": workload,
            "config_sha256": digest(_shape_config(config)),
            "code_sha256": code_identity(), "resources": observe_resources(config.get("device", "npu"))}


def training_parameters(config, *, microbatch=None, effective_batch=None,
                        checkpointing=None, workers=None, precision="bf16-forward/fp32-state"):
    tr = config.get("training", config)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    expected = config.get("world_size", tr.get("world_size",
               config.get("local_execution", {}).get("world_size", world)))
    _require(world == expected, "Actual WORLD_SIZE differs from the training configuration")
    micro = microbatch if microbatch is not None else tr.get("microbatch_size", tr.get("microbatch"))
    batch = effective_batch if effective_batch is not None else tr.get("effective_batch_size", tr.get("effective_batch"))
    _require(type(micro) is int and micro > 0 and type(batch) is int and batch > 0,
             "The entry must declare its actual microbatch and effective batch")
    _require(batch % (world * micro) == 0, "Effective batch must divide across ranks and microbatches")
    gc = (checkpointing if checkpointing is not None else
          config.get("model", {}).get("gradient_checkpointing", config.get("gradient_checkpointing", False)))
    return {"microbatch": micro, "accumulation": batch // (world * micro),
            "effective_batch": batch, "gradient_checkpointing": bool(gc),
            "precision": precision, "parallelism": "ddp" if world > 1 else "single",
            "workers_per_rank": workers if workers is not None else tr.get("workers_per_rank", config.get("loader_workers", 2))}


def inference_parameters(batch, *, workers=2, precision="bf16-forward", parallelism="rank-sharded"):
    return {"microbatch": batch, "accumulation": 1, "effective_batch": None,
            "gradient_checkpointing": False, "precision": precision,
            "parallelism": parallelism, "workers_per_rank": workers}


def bounded_probe(steps):
    """Only call when the native worker actually stops at this update bound."""
    _require(type(steps) is int and 0 <= steps <= 32,
             "A preflight worker must have an enforced limit of 1..32 updates")
    return steps > 0


def _artifact(reference, base):
    _require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
             "Evidence needs a path and SHA256, not a bare PASS flag")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = base / path
    _require(path.is_file(), f"Missing performance evidence: {path}")
    _require(file_hash(path) == reference["sha256"], f"Performance evidence changed: {path}")
    return path


def _json_artifact(reference, base):
    return json.loads(_artifact(reference, base).read_text())


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _parameters(params, context):
    required = {"microbatch", "accumulation", "effective_batch", "gradient_checkpointing",
                "precision", "parallelism", "workers_per_rank"}
    _require(isinstance(params, dict) and set(params) == required, "Incomplete effective parameters")
    for key in ("microbatch", "accumulation"):
        _require(type(params[key]) is int and params[key] > 0, f"Invalid {key}")
    _require(type(params["gradient_checkpointing"]) is bool, "Checkpointing must be explicit")
    _require(type(params["workers_per_rank"]) is int and params["workers_per_rank"] >= 0,
             "Invalid worker count")
    _require(bool(params["precision"]) and bool(params["parallelism"]), "Missing precision/parallelism")
    if context["stage"] == "training":
        _require(params["effective_batch"] == context["resources"]["world_size"] *
                 params["microbatch"] * params["accumulation"], "Effective batch arithmetic mismatch")
    else:
        _require(params["accumulation"] == 1 and params["effective_batch"] is None,
                 "Inference must not pretend to use gradient accumulation")


def _validate_reuse(path, report, context, actual_parameters, chain):
    """Check the agent's reuse decision and real smoke evidence, not code similarity."""
    current = report.get("context", {})
    # Source/config hashes remain provenance. The agent reviews their differences.
    for key in ("stage", "workload", "resources"):
        _require(current.get(key) == context.get(key),
                 f"Reuse report does not cover current {key}")
    _parameters(actual_parameters, context)
    _require(report.get("parameters") == actual_parameters,
             "Actual launch parameters differ from the reviewed reuse configuration")
    assessment = report.get("assessment", {})
    _require(assessment.get("compatible") is True and
             all(isinstance(assessment.get(k), str) and assessment[k].strip()
                 for k in ("reason", "differences")),
             "Reuse needs the agent's compatibility decision, differences and reason")
    history = report.get("historical", {})
    _artifact(history.get("config"), path.parent)
    logs = history.get("run_evidence", [])
    _require(bool(logs), "Historical reuse needs evidence of an actual run")
    for ref in logs:
        _artifact(ref, path.parent)
    previous = history.get("context", {})
    _require(previous.get("stage") == context["stage"],
             "Training and inference cannot share a reuse decision")
    for key in ("backend", "device_model", "device_memory_bytes", "world_size"):
        _require(key in context["resources"] and
                 previous.get("resources", {}).get(key) == context["resources"][key],
                 f"Historical {key} differs; explore parameters for the current resources")
    _require(history.get("parameters") == actual_parameters,
             "Historical budget parameters differ; explore the changed configuration")
    selection = None
    if context["stage"] == "training":
        _require(history.get("selection_report"),
                 "Training reuse needs historical.selection_report tracing to accepted capacity/throughput "
                 "selection; a short probe or successful run alone is not selection evidence")
        selection_path = _artifact(history["selection_report"], path.parent)
        selection = validate_report(selection_path, previous, actual_parameters, _chain=chain)
    smoke = _json_artifact(report.get("smoke_test"), path.parent)
    for key in ("stage", "workload", "resources"):
        _require(smoke.get("context", {}).get(key) == context.get(key),
                 f"Smoke test does not cover current {key}")
    _require(smoke.get("parameters") == actual_parameters,
             "Smoke test used different budget parameters")
    _require(smoke.get("status") == "PASS" and smoke.get("oom") is False and
             smoke.get("errors") == [],
             "Smoke test failed: OOM requires parameter exploration; diagnose other errors")
    steps = smoke.get("completed_steps_by_rank", [])
    world = context["resources"]["world_size"]
    _require(len(steps) == world and all(type(n) is int and n in (2, 3) for n in steps)
             and len(set(steps)) == 1,
             "Reuse needs 2--3 completed steps on every rank")
    checks = {"finite_values", "representative_inputs"}
    if context["stage"] == "training":
        checks |= {"full_optimizer_steps", "optimizer_initialized"}
    else:
        checks |= {"complete_batches"}
    _require(all(smoke.get("checks", {}).get(k) is True for k in checks),
             "Smoke test must cover actual inputs and complete execution steps")
    raw_logs = smoke.get("raw_logs", [])
    _require(bool(raw_logs), "Smoke test needs original run logs")
    for ref in raw_logs:
        _artifact(ref, path.parent)
    return {"status": "PASS", "mode": "reuse", "report": str(path.resolve()),
            "report_sha256": file_hash(path), "completed_steps": steps[0],
            "reason": assessment["reason"],
            "selection_report": selection["report"] if selection else None}


def _memory_records(records, world, capacity, maximum_samples):
    """Validate all-rank measurements shared by capacity and throughput probes."""
    _require(isinstance(records, list) and records, "Missing measured capacity records")
    for row in records:
        count = row.get("real_samples")
        _require(type(count) is int and 0 < count <= maximum_samples,
                 "Invalid measured capacity sample count")
        for key in ("seconds_by_rank", "peak_allocated_bytes_by_rank", "free_bytes_by_rank"):
            values = row.get(key, [])
            _require(len(values) == world and all(_positive(v) for v in values),
                     f"Incomplete/invalid capacity all-rank measurement: {key}")
        _require(max(row["peak_allocated_bytes_by_rank"]) < capacity and
                 max(row["free_bytes_by_rank"]) <= capacity, "Invalid capacity memory measurement")


def _validate_capacity(path, report, context, actual_parameters, throughput):
    """Capacity is measured at accumulation=1, independently of the formal global batch."""
    search = report.get("capacity_search")
    _require(isinstance(search, dict) and search.get("candidates"),
             "Training selection needs independent capacity_search evidence; two small throughput "
             "candidates do not establish capacity")
    world = context["resources"]["world_size"]
    capacity = context["resources"]["device_memory_bytes"]
    results = {}
    micros = set()
    for item in search["candidates"]:
        name = item.get("id")
        _require(isinstance(name, str) and name and name not in results,
                 "Duplicate/missing capacity candidate ID")
        evidence = _json_artifact(item.get("evidence"), path.parent)
        _require(evidence.get("context") == context, "Capacity probe used different workload/resources")
        params = evidence.get("parameters")
        _parameters(params, context)
        _require(params["accumulation"] == 1, "Capacity probes must use accumulation=1")
        for key in set(params) - {"microbatch", "accumulation", "effective_batch"}:
            _require(params[key] == actual_parameters[key], f"Capacity probe changed {key}")
        _require(params["microbatch"] not in micros, "Duplicate capacity microbatch is not exploration")
        micros.add(params["microbatch"])
        _require(evidence.get("raw_logs"), "Capacity probe needs original logs")
        for ref in evidence["raw_logs"]:
            _artifact(ref, path.parent)
        status = evidence.get("status")
        result = {"status": status, "microbatch": params["microbatch"]}
        if status == "CAPACITY_FAILURE":
            _require(evidence.get("failure_kind") == "OOM" and evidence.get("reason"),
                     "Only diagnosed OOM is capacity failure")
        else:
            _require(status == "PASS" and evidence.get("oom") is False and evidence.get("errors") == [],
                     "Capacity probe did not pass")
            steps = evidence.get("completed_steps_by_rank", [])
            _require(len(steps) == world and all(type(n) is int and n in (1, 2) for n in steps)
                     and len(set(steps)) == 1, "Capacity needs 1--2 complete optimizer steps on every rank")
            checks = {"finite_values", "representative_inputs", "full_optimizer_steps", "optimizer_initialized"}
            _require(all(evidence.get("checks", {}).get(k) is True for k in checks),
                     "Capacity probe lacks full optimizer/input/numerical checks")
            records = evidence.get("records", [])
            _require(len(records) == steps[0], "Capacity step records disagree with completed steps")
            _memory_records(records, world, capacity, params["effective_batch"])
            result["minimum_free_bytes"] = min(min(r["free_bytes_by_rank"]) for r in records)
        results[name] = result
    viable = [r for r in results.values() if r["status"] == "PASS"]
    _require(viable, "Capacity search has no successful candidate")
    largest = max(r["microbatch"] for r in viable)
    _require(actual_parameters["microbatch"] <= largest, "Selected microbatch exceeds measured capacity")
    stop = search.get("stop", {})
    _require(isinstance(stop, dict) and isinstance(stop.get("reason"), str) and stop["reason"].strip(),
             "Capacity search needs an evidenced stopping reason")
    kind = stop.get("kind")
    if kind == "OOM":
        upper = results.get(stop.get("candidate"), {})
        _require(upper.get("status") == "CAPACITY_FAILURE" and upper["microbatch"] > largest,
                 "Capacity OOM stop needs a measured failed microbatch above the viable range")
    elif kind == "MEMORY_MARGIN":
        boundary = results.get(stop.get("candidate"), {})
        reserve = stop.get("minimum_free_bytes")
        _require(boundary.get("status") == "PASS" and boundary["microbatch"] == largest and
                 _positive(reserve) and reserve < capacity and boundary["minimum_free_bytes"] <= reserve,
                 "Memory stop is unsupported by measured all-rank free memory and declared reserve")
    elif kind == "THROUGHPUT_PLATEAU":
        larger = throughput.get(stop.get("larger_candidate"), {})
        smaller = throughput.get(stop.get("smaller_candidate"), {})
        _require(larger.get("status") == smaller.get("status") == "PASS" and
                 larger["parameters"]["microbatch"] == largest and
                 smaller["parameters"]["microbatch"] < largest and
                 larger["samples_per_second"] <= smaller["samples_per_second"],
                 "Throughput stop needs stable measurements showing no gain at the largest viable batch")
    else:
        raise PerformancePreflightError("Capacity stop must be evidenced OOM, MEMORY_MARGIN or THROUGHPUT_PLATEAU")
    return {"largest_passed_microbatch": largest, "stop_kind": kind, "candidates": results}


def validate_report(path, context, actual_parameters, *, _chain=()):
    """Accept reviewed reuse plus a smoke test, or validate a measured search."""
    path = Path(path).resolve()
    _require(path not in _chain and len(_chain) < 16, "Cyclic or excessive performance selection lineage")
    chain = (*_chain, path)
    report = json.loads(path.read_text())
    if report.get("schema") == REUSE_SCHEMA:
        return _validate_reuse(path, report, context, actual_parameters, chain)
    _require(report.get("schema") == SCHEMA, "Unsupported performance report schema")
    _require(report.get("context") == context,
             "Performance evidence does not match current workload/resources/software/source; "
             "assess historical reuse with a 2--3 step smoke test, or run bounded exploration")
    _parameters(actual_parameters, context)
    candidates = report.get("candidates", [])
    _require(len(candidates) >= 2, "Need comparative candidates, including evidenced capacity failures")
    world = context["resources"]["world_size"]
    capacity = context["resources"]["device_memory_bytes"]
    _require(type(world) is int and world > 0 and _positive(capacity), "Invalid resource identity")
    results = {}
    parameter_ids = set()
    for item in candidates:
        name = item.get("id")
        _require(isinstance(name, str) and name and name not in results, "Duplicate/missing candidate ID")
        evidence = _json_artifact(item.get("evidence"), path.parent)
        _require(evidence.get("context") == context, f"Candidate {name} was measured under different conditions")
        params = evidence.get("parameters")
        _parameters(params, context)
        _require(digest(params) not in parameter_ids, "Duplicate configuration is not a comparison")
        parameter_ids.add(digest(params))
        if context["stage"] == "training":
            _require(params["effective_batch"] == actual_parameters["effective_batch"],
                     "Candidate changed the agreed training batch")
        raw_logs = evidence.get("raw_logs", [])
        _require(bool(raw_logs), f"Candidate {name} has no original logs")
        for ref in raw_logs:
            _artifact(ref, path.parent)
        if evidence.get("status") == "CAPACITY_FAILURE":
            _require(evidence.get("reason") and evidence.get("failure_kind") == "OOM",
                     "Only diagnosed capacity failure can eliminate a candidate")
            results[name] = {"parameters": params, "status": "CAPACITY_FAILURE"}
            continue
        _require(evidence.get("status") == "PASS", f"Candidate {name} has not passed")
        records = evidence.get("records", [])
        warmups = [r for r in records if r.get("phase") == "warmup"]
        measured = [r for r in records if r.get("phase") == "measure"]
        _require(len(warmups) >= 2 and len(measured) >= 10,
                 f"Candidate {name} requires >=2 warmup and >=10 measured updates/batches")
        _require(len(warmups) + len(measured) == len(records), "Unknown timing phase")
        _require(all(r.get("phase") == "warmup" for r in records[:len(warmups)]),
                 "Warmup must precede measured work")
        seconds = []; samples = 0; peak = 0
        for row in records:
            _require(type(row.get("real_samples")) is int and row["real_samples"] > 0,
                     "Missing actual sample count")
            maximum_samples = params["effective_batch"] if context["stage"] == "training" else world * params["microbatch"]
            _require(row["real_samples"] <= maximum_samples,
                     "Measured sample count exceeds one actual update/batch")
            for key in ("seconds_by_rank", "peak_allocated_bytes_by_rank", "free_bytes_by_rank"):
                values = row.get(key, [])
                _require(len(values) == world and all(_positive(v) for v in values),
                         f"Incomplete/invalid all-rank measurement: {key}")
            _require(max(row["peak_allocated_bytes_by_rank"]) < capacity and
                     max(row["free_bytes_by_rank"]) <= capacity, "Invalid device memory measurement")
            if row["phase"] == "measure":
                seconds.append(max(row["seconds_by_rank"]))
                samples += row["real_samples"]
                peak = max(peak, max(row["peak_allocated_bytes_by_rank"]))
        results[name] = {"parameters": params, "status": "PASS",
                         "samples_per_second": samples / sum(seconds),
                         "median_seconds": statistics.median(seconds), "peak_allocated_bytes": peak}
    selected = report.get("selected_candidate")
    _require(selected in results and results[selected]["status"] == "PASS", "Selected candidate did not pass")
    _require(results[selected]["parameters"] == actual_parameters,
             "Actual launch parameters differ from the selected measured candidate")
    capacity_search = (_validate_capacity(path, report, context, actual_parameters, results)
                       if context["stage"] == "training" else None)
    _require(bool(report.get("selection_reason")), "Missing selection rationale")
    best = max(v["samples_per_second"] for v in results.values() if v["status"] == "PASS")
    verification = _json_artifact(report.get("verification"), path.parent)
    _require(verification.get("context") == context and verification.get("parameters") == actual_parameters,
             "Correctness/stability verification does not cover the selected configuration")
    checks = {"finite_values", "sample_coverage", "equivalent_output_or_gradient", "continuous_run"}
    if context["stage"] == "training":
        checks |= {"tail_weighting", "frozen_parameter_scope", "optimizer_initialized"}
    _require(verification.get("status") == "PASS" and
             all(verification.get("checks", {}).get(k) is True for k in checks),
             "Selected configuration lacks correctness/stability acceptance")
    _require(bool(verification.get("raw_logs")), "Missing original verification logs")
    for ref in verification["raw_logs"]:
        _artifact(ref, path.parent)
    return {"status": "PASS", "report": str(path.resolve()), "report_sha256": file_hash(path),
            "selected_candidate": selected, "measurements": results,
            "capacity_search": capacity_search,
            "selected_throughput_fraction_of_best": results[selected]["samples_per_second"] / best}


def require_preflight(stage, config, parameters, *, workload):
    report = os.environ.get(REPORT_ENV) or config.get("performance_preflight")
    _require(report, f"Missing {REPORT_ENV}: formal {stage} requires current performance evidence; "
             "assess historical configuration reuse and test 2--3 steps, or run bounded exploration")
    context = describe(stage, config, parameters, workload=workload)
    result = validate_report(report, context, parameters)
    if int(os.environ.get("RANK", "0")) == 0:
        print("PERFORMANCE_PREFLIGHT " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_report(args.report, json.loads(args.context.read_text()),
                                    json.loads(args.parameters.read_text())), indent=2))


if __name__ == "__main__":
    main()
