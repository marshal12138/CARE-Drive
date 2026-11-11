# Data preparation

Use [NAVSIM v1.1](https://github.com/autonomousvision/navsim/tree/v1.1), OpenScene logs/sensor blobs, and nuPlan maps. The official [download instructions](https://github.com/autonomousvision/navsim/blob/v1.1/docs/install.md) and [split definitions](https://github.com/autonomousvision/navsim/blob/v1.1/docs/splits.md) explain which downloads cover navtrain and navtest. Set these variables after installing the external devkit:

```bash
export CARE_DRIVE_DATA=/absolute/path/to/data
export OPENSCENE_DATA_ROOT="$CARE_DRIVE_DATA"
export NAVSIM_DEVKIT_ROOT="$PWD/third_party/navsim"
export NAVSIM_EXP_ROOT="$CARE_DRIVE_DATA/navsim_exp"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="$CARE_DRIVE_DATA/maps"
```

Generate the official metric caches before creating the manifests. These caches supply map/vehicle/reference information to PDMS; they are distinct from teacher features and frozen backbone features.

```bash
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtrain \
  cache.cache_path="$CARE_DRIVE_DATA/metric_cache/navtrain"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest \
  cache.cache_path="$CARE_DRIVE_DATA/metric_cache/navtest"
```

Export training observations and targets. The scene filter is passed unchanged to NAVSIM, including its log/token lists and future-frame availability filter; the target itself consists of the next eight poses.

```bash
python -m care_drive.prepare manifest \
  --logs "$CARE_DRIVE_DATA/navsim_logs/trainval" \
  --sensor-root "$CARE_DRIVE_DATA/sensor_blobs/trainval" \
  --scene-filter "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml" \
  --metric-cache "$CARE_DRIVE_DATA/metric_cache/navtrain" \
  --output "$CARE_DRIVE_DATA/manifests/navtrain.jsonl"
```

For inference, omit training targets and future-image paths:

```bash
python -m care_drive.prepare manifest \
  --logs "$CARE_DRIVE_DATA/navsim_logs/test" \
  --sensor-root "$CARE_DRIVE_DATA/sensor_blobs/test" \
  --scene-filter "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml" \
  --metric-cache "$CARE_DRIVE_DATA/metric_cache/navtest" \
  --observations-only --output "$CARE_DRIVE_DATA/manifests/navtest.jsonl"
```

The exporter creates a neighboring summary with scene counts, scoreable counts, and the scene-filter hash. It validates observation structure and timestamps. Missing/corrupt images fail at loading rather than being silently replaced. Keep held-out validation logs separate from training by supplying your own scene-filter YAML with disjoint `log_names` / `tokens`.

The paper used **92,772 backbone training scenes**, **92,614 scoreable expert training scenes**, and **12,146 navtest scenes**. A plain official-navtrain export is not the paper's internal filtered training/validation split. Exact cohort manifests and trained paper checkpoints are not bundled with this code release. Record your filters, manifests, and resulting counts when running new experiments; changing that cohort changes the reproduction setting.

After installing the V-JEPA2.1 source and checkpoint, cache the teacher features:

```bash
export CARE_DRIVE_PERFORMANCE_PREFLIGHT=/absolute/path/to/performance/teacher.json
python -m care_drive.prepare cache teacher \
  --config configs/care_drive.yaml \
  --manifest "$CARE_DRIVE_DATA/manifests/navtrain.jsonl" \
  --sensor-root "$CARE_DRIVE_DATA/sensor_blobs/trainval" \
  --output "$CARE_DRIVE_DATA/cache/teacher"
```

The teacher cache stores `teacher_features` with shape `[96,1664]` in `<scene-prefix>/<scene-id>.safetensors`. Every backbone training row needs these targets. Frozen expert features are generated **after A0 training**, using the README command: `scene` is `[N,8,2048]`, `anchor` and `gt` are `[N,8,3]`, and `tokens`/`metric_cache_paths` align with those tensors. The feature exporter selects the scoreable subset and records counts plus manifest/backbone hashes. Use a single process for the feature command shown in the README, producing one cache file for expert training.

Observation-only inference reads no future images or teacher features. The manifest can retain metric-cache references for subsequent evaluation; the model does not use those caches when predicting.
