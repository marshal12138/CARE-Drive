import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from care_drive.assets import backbone_assets
from care_drive.checkpoints import save, load_backbone


class Processor:
    def save_pretrained(self, directory):
        (Path(directory) / 'tokenizer.json').write_text('{"vocab": {"query": 0, "name_or_path": 1}}')
        (Path(directory) / 'preprocessor_config.json').write_text('{"size": 32}')
        (Path(directory) / 'chat_template.jinja').write_text('{{ messages }}')


class Configuration:
    def save_pretrained(self, directory):
        (Path(directory) / 'config.json').write_text(json.dumps({
            'model_type': 'test', '_name_or_path': '/private/training/origin'}))


class SmallBackbone(torch.nn.Module):
    def __init__(self, config=None, *, load_pretrained=True):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.config = config
        self.load_pretrained = load_pretrained
        self.vlm = SimpleNamespace(processor=Processor(), model=SimpleNamespace(
            config=Configuration(), generation_config=None, tie_weights=lambda: None))

    def tokenizer_identity(self):
        return {'vocab_sha256': 'test-vocab', 'token_ids': {'query': 0}}

    def forward(self, x):
        return self.linear(x)


class CheckpointAssetTests(unittest.TestCase):
    def test_saved_backbone_is_self_contained_and_preserves_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'epoch_005.pt'
            source = SmallBackbone().double()
            save(checkpoint, source, {'model': {}})
            assets = backbone_assets(checkpoint)
            self.assertEqual(assets, checkpoint.parent / 'assets')
            self.assertNotIn('_name_or_path', json.loads((assets / 'config.json').read_text()))
            self.assertIn('name_or_path', json.loads((assets / 'tokenizer.json').read_text())['vocab'])
            config = {'model': {'vlm_path': '/nonexistent/original/model'}}
            with patch('care_drive.training.losses.L1Model', SmallBackbone):
                restored = load_backbone(checkpoint, config, 'cpu')
            self.assertFalse(restored.load_pretrained)
            self.assertEqual(restored.config['model']['vlm_path'], str(assets))
            self.assertEqual(config['model']['vlm_path'], '/nonexistent/original/model')
            self.assertEqual(restored.linear.weight.dtype, torch.float64)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            x = torch.randn(2, 3, dtype=torch.float64)
            torch.testing.assert_close(restored(x), source(x), rtol=0, atol=0)

    def test_missing_assets_fail_without_constructing_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'epoch_005.pt'
            with patch('care_drive.training.losses.L1Model') as model:
                with self.assertRaisesRegex(FileNotFoundError, 'complete A0-DYNAMIC-L1'):
                    load_backbone(checkpoint, {'model': {'vlm_path': 'unused'}}, 'cpu')
                model.assert_not_called()

    def test_changed_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'epoch_005.pt'
            save(checkpoint, SmallBackbone(), {'model': {}})
            (checkpoint.parent / 'assets/config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                backbone_assets(checkpoint)

    def test_asset_symlinks_support_hub_snapshot_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'snapshot/epoch_005.pt'
            save(checkpoint, SmallBackbone(), {'model': {}})
            file = checkpoint.parent / 'assets/tokenizer.json'
            blob = root / 'blob'
            file.replace(blob)
            file.symlink_to(blob)
            self.assertTrue(backbone_assets(checkpoint).is_dir())

    def test_checkpoint_tokenizer_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'epoch_005.pt'
            save(checkpoint, SmallBackbone(), {'model': {}})
            value = torch.load(checkpoint, weights_only=True)
            value['contract']['tokenizer']['vocab_sha256'] = 'different'
            torch.save(value, checkpoint)
            with patch('care_drive.training.losses.L1Model', SmallBackbone):
                with self.assertRaisesRegex(ValueError, 'tokenizer identity differs'):
                    load_backbone(checkpoint, {'model': {}}, 'cpu')

    def test_incomplete_weight_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'epoch_005.pt'
            save(checkpoint, SmallBackbone(), {'model': {}})
            value = torch.load(checkpoint, weights_only=True)
            del value['model']['linear.bias']
            torch.save(value, checkpoint)
            with patch('care_drive.training.losses.L1Model', SmallBackbone):
                with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                    load_backbone(checkpoint, {'model': {}}, 'cpu')


if __name__ == '__main__':
    unittest.main()
