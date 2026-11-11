# Environment setup

Use Linux and Python 3.11. Create an isolated environment, install an appropriate PyTorch / torchvision pair for your hardware, and install this repository with `python -m pip install -e .`. The model interface is pinned to Transformers 4.57.6. Ascend additionally requires a compatible torch-npu/CANN stack. Set `device: npu` or `device: cuda` in the YAML; CPU is intended for the small tests.

NAVSIM v1's original requirements pin an older Torch/Transformers-era stack. Keep the model runtime above and install the external source packages without applying their old complete dependency lock:

```bash
mkdir -p third_party
git clone --branch v1.1 --depth 1 https://github.com/autonomousvision/navsim.git third_party/navsim
git clone --branch nuplan-devkit-v1.2 --depth 1 https://github.com/motional/nuplan-devkit.git third_party/nuplan-devkit
python -m pip install -r requirements-navsim.txt
python -m pip install --no-deps -e third_party/nuplan-devkit -e third_party/navsim
git clone https://github.com/facebookresearch/vjepa2.git third_party/vjepa2
git -C third_party/vjepa2 apply ../../scripts/vjepa2_compat.patch
```

The small V-JEPA compatibility patch keeps the drop-path schedule on CPU during meta-device construction and uses the CUDA attention context only on CUDA devices. It preserves the encoder parameters and attention operation.

Download [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) into the configured `model.vlm_path`. Download the **V-JEPA2.1 ViT-G/16 384** checkpoint from the [official model table](https://github.com/facebookresearch/vjepa2#models), and set `model.teacher_source` and `model.teacher_path`. Record the V-JEPA source commit; an optional `teacher_sha256` checks the checkpoint before loading. The teacher expects the original checkpoint's `encoder` state dictionary, not a Transformers-converted checkpoint.

Check that the official scorer imports before launching training:

```bash
python -c 'from care_drive.integrations.navsim import OfficialScorer; OfficialScorer(); print("NAVSIM v1 scorer ready")'
python -m care_drive.train --help
python -m care_drive.infer --help
python -m unittest discover -s tests -v
```

The NAVSIM dependency list covers the CLI data/scoring path. Data downloads, model downloads, accelerator drivers, and metric-cache generation are separate setup steps. This release does not supply a fully validated cross-platform environment lock; see [validation](validation.md) for the environment actually tested.
