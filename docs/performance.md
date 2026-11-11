# Performance setup

The training, prediction, and model-feature-cache entry points retain the project's performance-evidence check. Set `CARE_DRIVE_PERFORMANCE_PREFLIGHT` to the report for the actual stage, hardware allocation, and batch settings before a formal run. The validator does not load models, launch jobs, or choose hardware automatically.

Use bounded runs to prepare measurements, for example:

```bash
torchrun --standalone --nproc_per_node=8 -m care_drive.train expert \
  --config configs/care_drive.yaml --stage B1-H-SFT --probe-steps 3
```

The limit is 1–32 optimizer steps/batches. Training probes save under `<stage>/probe/`, separate from completed-stage checkpoints. Probe results are diagnostics; a successful short run is not a throughput-selection report. Formal entry points cannot be run by setting an unlimited probe budget.

## Report interfaces

The implementation is [care_drive/performance_gate.py](../care_drive/performance_gate.py). `describe(stage, config, parameters, workload=...)` records configuration/source hashes and actual backend, device model, capacity, device count, and software. `training_parameters` / `inference_parameters` construct the effective batch, accumulation, precision, checkpointing, and worker settings used for validation. Device count comes from the launcher; changing it requires corresponding measurements.

| Entry | Workload |
| --- | --- |
| Backbone training | `care-drive-backbone-l1` |
| Heading SFT / GRPO | `care-drive-heading-SFT` / `care-drive-heading-GRPO` |
| Position SFT / GRPO | `care-drive-position-SFT` / `care-drive-position-GRPO` |
| Greedy inference | `care-drive-greedy-inference` |
| Teacher / backbone feature cache | `care-drive-cache-teacher` / `care-drive-cache-features` |

Use the exact context constructed in the entry point. Expert stages include the stage recipe and feature-cache path; cache generation includes the manifest, sensor root, cache kind, and backbone checkpoint. Greedy inference includes the selected input and checkpoint paths. Different stages need their own accepted context. Paths in the configuration must resolve before hashing.

The validator supports two report schemas:

* `wam_performance_preflight_v1`: measured capacity/throughput selection, with at least two distinct parameter candidates (an evidenced OOM may be one candidate), a selected candidate, rationale, and correctness verification.
* `wam_performance_reuse_v1`: a reviewed compatibility decision, historical configuration and run evidence, and a current two- or three-step smoke test. Training reuse must also trace to an accepted capacity/throughput report.

All artifact references use `{"path": "relative/path.json", "sha256": "actual file digest"}`. Reports and logs must describe actual measurements.

## Measuring a new setup

For training, first measure capacity with one or two complete optimizer updates per candidate, at accumulation 1. Record every rank's completed-step count, synchronized elapsed time, peak allocated memory, and free device memory. Stop for an evidenced OOM, a declared minimum memory margin reached by the largest successful candidate, or a measured throughput plateau. Keep this evidence in `capacity_search.candidates` and describe the observed stop in `capacity_search.stop`.

Then compare viable microbatches at the intended effective batch. Each throughput candidate needs at least two warmup and ten measured complete updates (or complete inference batches). Each record contains:

```text
phase: warmup | measure
real_samples: global count excluding padding
seconds_by_rank: one synchronized elapsed time per rank
peak_allocated_bytes_by_rank: one real peak measurement per rank
free_bytes_by_rank: one real free-memory measurement per rank
```

Candidate evidence contains `context`, `parameters`, `status`, `records`, and `raw_logs`. An OOM candidate uses `status: CAPACITY_FAILURE`, `failure_kind: OOM`, and a diagnostic `reason`. Training capacity candidates additionally contain `oom`, `errors`, `completed_steps_by_rank`, and checks for finite values, representative inputs, complete optimizer updates, and initialized optimizer state.

The selected configuration also needs independent verification with `context`, `parameters`, `status`, `checks`, and raw logs. Check finite values, coverage, equivalent outputs/gradients, and stable continuous execution; training additionally checks tail weighting, the frozen parameter scope, and optimizer initialization. The validator checks structure, identities, hashes, and measured batch arithmetic. It does not replace the correctness measurements.

## Reusing accepted measurements

Keep the accepted historical report and raw evidence. Record a new `assessment` with `compatible: true`, an explanation in `reason`, and reviewed `differences`; identify the historical configuration, parameters, context, and run logs. The current smoke test must record two or three completed steps on every rank with no OOM/errors and the same parameters. Hardware count/model/capacity must match; otherwise measure the new setup.

No accepted reports are shipped: they depend on the user's hardware and input cohort. The release's CPU unit tests are not substitutes for these reports.
