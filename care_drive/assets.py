"""Model configuration and processor files bundled beside a backbone checkpoint."""
import hashlib
import json
from pathlib import Path

SCHEMA = "care-drive-backbone-assets-v1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _portable(value):
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()
                if k not in ("_name_or_path", "name_or_path")}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    return value


def save_backbone_assets(model, directory):
    """Save the final vocabulary, chat template, processors, and architecture."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.vlm.processor.save_pretrained(directory)
    model.vlm.model.config.save_pretrained(directory)
    if model.vlm.model.generation_config is not None:
        model.vlm.model.generation_config.save_pretrained(directory)
    # Machine-local origin paths do not belong in a portable model bundle.
    for path in directory.glob("*config.json"):
        if path.is_file():
            path.write_text(json.dumps(_portable(json.loads(path.read_text())),
                                       indent=2, ensure_ascii=False) + "\n")
    files = {str(p.relative_to(directory)): {"sha256": _sha256(p), "bytes": p.stat().st_size}
             for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and not p.name.startswith(".")}
    manifest = {"schema": SCHEMA, "files": files,
                "tokenizer_vocab_sha256": model.tokenizer_identity()["vocab_sha256"]}
    temporary = directory / ".manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(directory / "manifest.json")


def backbone_assets(checkpoint):
    """Resolve and verify the local bundle, with no pretrained-model fallback."""
    directory = Path(checkpoint).resolve().parent / "assets"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing backbone assets at {directory}. Download the complete "
            "A0-DYNAMIC-L1 directory, including assets/, from maxiao123/CARE-Drive.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("files"):
        raise ValueError("Invalid backbone asset manifest")
    for name, expected in manifest["files"].items():
        path = directory / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file():
            raise ValueError(f"Missing or invalid backbone asset: {name}")
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"Backbone asset checksum mismatch: {name}")
    return directory
