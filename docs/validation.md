# Release validation

Checked on 2026-11-11. This is a portable extraction of the paper's training and inference components. It uses official NAVSIM v1 CPU scoring; the original accelerator scoring implementation is not included.

## Completed checks

* Fifteen CPU tests cover position quantization, adjacent-angle decoding and wrapping, teacher-forced versus cached autoregressive probabilities, frozen input boundaries, conditional heading generation, checkpoint round trips, sample weighting, masked advantage normalization, safety-gated reward shaping, GRPO gradients, and complete epoch sharding.
* The tests execute a two-step heading SFT worker, a two-step position GRPO worker with explicitly synthetic rewards, and the frozen-feature inference CLI on small models.
* Twelve core model/codec/weighting definitions were compared structurally against the source implementation and match exactly. New portable orchestration and dependency adapters are separate changes.
* Training, inference, evaluation, and preparation command parsers and model imports were checked. Editable package installation passed. The official NAVSIM v1 scorer constructor and data-loader imports passed after installing their runtime dependencies in an isolated test environment.

The CPU checks used Python 3.11.15, PyTorch 2.12.0+cpu, Transformers 4.57.6, and NumPy 1.26.4. They used synthetic inputs, with no benchmark data or trained research weights loaded.

## Checkpoint release validation

The final A0/C3/D1 checkpoints are published on [Hugging Face](https://huggingface.co/maxiao123/CARE-Drive). Each source file was checked against the SHA256 recorded for the accepted final evaluation. The export retains all **1,086 state-dict tensors** with identical names, shapes, values, and dtypes, while omitting optimizer, scheduler, RNG, and internal run metadata.

The actual released weights passed strict loading through the published backbone and expert loaders on CPU. The backbone tokenizer contract matched, loaded tensors matched the exported states exactly, and one synthetic feature sample passed coupled expert generation. These checks establish export and loading compatibility; they do not rerun benchmark evaluation. File sizes, source identities, and exported SHA256 hashes are listed in the [checkpoint manifest](https://huggingface.co/maxiao123/CARE-Drive/blob/main/checkpoint_manifest.json).

## Scope remaining

Full backbone training, V-JEPA target generation with the large checkpoint, real-data GRPO, distributed accelerator execution, and full navtest evaluation have not been rerun through these portable entry points. CUDA support has not been validated end to end. The paper's **88.8 PDMS** is an existing experimental result, not a score measured by these release checks.

The repository contains code and stage recipes. Final A0/C3/D1 weights are published in [maxiao123/CARE-Drive](https://huggingface.co/maxiao123/CARE-Drive); exact paper cohort manifests remain unbundled. Pretrained Qwen weights alone do not contain the trained CARE-Drive backbone or experts. Download the released checkpoints or generate new ones with the documented chain; probe checkpoints represent diagnostic runs. The current CLI starts each stage from its declared predecessor; it does not resume an interrupted stage's optimizer state.

Run local checks with:

```bash
python -m unittest discover -s tests -v
```
