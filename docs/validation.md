# Release validation

Checked on 2026-11-11. This is a portable extraction of the paper's training and inference components. It uses official NAVSIM v1 CPU scoring; the original accelerator scoring implementation is not included.

## Completed checks

* Fifteen CPU tests cover position quantization, adjacent-angle decoding and wrapping, teacher-forced versus cached autoregressive probabilities, frozen input boundaries, conditional heading generation, checkpoint round trips, sample weighting, masked advantage normalization, safety-gated reward shaping, GRPO gradients, and complete epoch sharding.
* The tests execute a two-step heading SFT worker, a two-step position GRPO worker with explicitly synthetic rewards, and the frozen-feature inference CLI on small models.
* Twelve core model/codec/weighting definitions were compared structurally against the source implementation and match exactly. New portable orchestration and dependency adapters are separate changes.
* Training, inference, evaluation, and preparation command parsers and model imports were checked. Editable package installation passed. The official NAVSIM v1 scorer constructor and data-loader imports passed after installing their runtime dependencies in an isolated test environment.

The CPU checks used Python 3.11.15, PyTorch 2.12.0+cpu, Transformers 4.57.6, and NumPy 1.26.4. They used synthetic inputs, with no benchmark data or trained research weights loaded.

## Scope remaining

Full backbone training, V-JEPA target generation with the large checkpoint, real-data GRPO, distributed accelerator execution, and full navtest evaluation have not been rerun through these portable entry points. CUDA support has not been validated end to end. The paper's **88.8 PDMS** is an existing experimental result, not a score measured by these release checks.

The repository contains code and stage recipes. It does not include the trained A0/C3/D1 checkpoints or the exact paper cohort manifests. Pretrained Qwen weights alone do not contain the trained CARE-Drive backbone or experts. Generate stage checkpoints with the documented chain; do not interpret a probe checkpoint as a completed stage. The current CLI starts each stage from its declared predecessor; it does not resume an interrupted stage's optimizer state.

Run local checks with:

```bash
python -m unittest discover -s tests -v
```
