import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from care_drive.models.codec import position_labels, position_decode, uniforms
from care_drive.models.heading import HeadingExpert, encode_headings, decode, statistics, grpo_objective
from care_drive.models.refinement import ExpertPolicy, sample_heading, CAREDriveRefiner, load_expert
from care_drive.training.advantages import build_advantages
from care_drive.training.rewards import training_reward
from care_drive.training.weighting import build_weights, weighted_terms
from care_drive.training.sampler import epoch_windows
from care_drive.checkpoints import save
from care_drive.performance_gate import bounded_probe, require_preflight, PerformancePreflightError


ARCH = {'width': 16, 'heads': 4, 'ffn': 32, 'layers': 1}


def recipe(component):
    return {'component': component, 'architecture': dict(ARCH, **({'heading_state_features': True} if component == 'heading' else {})),
            'quantizer': {'xy_range_m': [5., 3.]}, 'task': 'SFT', 'source': None,
            'seed': 42, 'epochs': 5, 'lr': 1e-4, 'weight_decay': .01,
            'effective_batch': 2, 'microbatch': 1, 'group_size': 8,
            'advantage_mode': 'scene_std', 'advantage_std_floor': .01,
            'reward_mode': 'official_pdms', 'epsilon': .2, 'beta': .04}


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.scene = torch.randn(2, 8, 2048)
        self.base = torch.randn(2, 8, 3)

    def test_metric_residual_quantization(self):
        base = torch.zeros(1, 8, 3)
        gt = torch.ones_like(base) * 100
        ids = position_labels(base, gt)
        self.assertTrue((ids == 400).all())
        expected = torch.tensor([5., 3.]).expand(1, 8, 2)
        torch.testing.assert_close(position_decode(base, ids), expected)
        gt[..., :2] = torch.tensor([.025, -.015])
        torch.testing.assert_close(position_decode(base, position_labels(base, gt)), gt[..., :2])

    def test_heading_increments_and_wrap(self):
        degrees = np.array([[10, 20, 30, 40, 50, 60, 70, 80.]])
        ids, saturated = encode_headings(np.deg2rad(degrees))
        self.assertTrue((ids == 300).all())
        self.assertFalse(saturated.any())
        result = decode(np.zeros((1, 8, 2)), ids)
        np.testing.assert_allclose(result[..., 2], np.deg2rad(degrees), atol=1e-6)
        boundary = np.deg2rad([[179., -179., -178., -177., -176., -175., -174., -173.]])
        ids, saturated = encode_headings(boundary)
        self.assertEqual(ids[0, 1], 220)
        self.assertTrue(saturated[0, 0])

    def test_heading_cached_sampling_replays_exact_prefix(self):
        model = HeadingExpert(dict(ARCH, heading_state_features=True))
        # Exercise the heading-state branch after it has learned nonzero weights.
        torch.nn.init.normal_(model.heading_state_projection.weight, std=.1)
        xy = self.base[..., :2]
        tokens, logp = sample_heading(model, self.scene, xy, uniforms([1, 2], 8, 1))
        full = statistics(model(self.scene, xy, tokens), tokens)[0]
        torch.testing.assert_close(full, logp, atol=2e-6, rtol=2e-6)
        greedy, greedy_logp = sample_heading(model, self.scene, xy)
        torch.testing.assert_close(greedy, model.generate(self.scene, xy))
        torch.testing.assert_close(statistics(model(self.scene, xy, greedy), greedy)[0], greedy_logp, atol=2e-6, rtol=2e-6)

    def test_position_cached_sampling_and_frozen_inputs(self):
        policy = ExpertPolicy(recipe('position'))
        scene = self.scene.requires_grad_()
        base = self.base.requires_grad_()
        tokens, logp = policy.expert.generate(scene, base, uniforms=uniforms([1, 2], 16, 1))
        full = statistics(policy(scene, base, tokens), tokens)[0]
        torch.testing.assert_close(full, logp, atol=2e-6, rtol=2e-6)
        (-full.mean()).backward()
        self.assertIsNone(scene.grad)
        self.assertIsNone(base.grad)
        self.assertTrue(any(p.grad is not None for p in policy.parameters()))

    def test_heading_does_not_use_base_heading(self):
        policy = ExpertPolicy(recipe('heading'))
        ids = torch.full((2, 8), 200)
        xy = self.base[..., :2].clone()
        first = policy(self.scene, self.base, ids, xy)
        changed = self.base.clone(); changed[..., 2] += 5
        torch.testing.assert_close(first, policy(self.scene, changed, ids, xy), rtol=0, atol=0)

    def test_coupled_generation_and_checkpoint_roundtrip(self):
        pos = ExpertPolicy(recipe('position')).eval().requires_grad_(False)
        heading = ExpertPolicy(recipe('heading')).eval().requires_grad_(False)
        result = CAREDriveRefiner(pos, heading)(self.scene, self.base)
        expected = heading.expert.generate(self.scene, result['xy'])
        torch.testing.assert_close(result['heading_ids'], expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'position.pt'
            save(path, pos, pos.config)
            restored = load_expert(path, pos.config)
            actual = CAREDriveRefiner(restored, heading)(self.scene, self.base)
            np.testing.assert_array_equal(actual['trajectory'], result['trajectory'])

    def test_dynamic_weights_and_unweighted_future_loss(self):
        weights, summary = build_weights(['hard', 'easy', 'missing'], {'hard': .2, 'easy': .5}, {'hard': .5, 'easy': .51})
        self.assertAlmostEqual(weights['hard'], 4/3)
        self.assertAlmostEqual(weights['easy'], 2/3)
        self.assertEqual(weights['missing'], 1)
        self.assertAlmostEqual(summary['full_weight_mean'], 1)
        traj, future = torch.tensor([3., 3., 99.]), torch.tensor([4., 4., 99.])
        total, _, aux = weighted_terms(traj, future, torch.tensor([2., 1., 1.]), torch.tensor([1., 1., 0.]))
        self.assertEqual(float(aux), 8)
        self.assertEqual(float(total), 17)

    def test_advantage_mask_rms_and_greedy_baseline(self):
        reward = torch.tensor([[.1, .4, .6, .8], [100., 300., 400., 600.]])
        valid = torch.tensor([1., 0.])
        standard, _ = build_advantages(reward, valid, 'scene_std')
        matched, _ = build_advantages(reward, valid, 'global_rms_matched')
        torch.testing.assert_close(matched, standard)
        greedy, _ = build_advantages(reward, valid, 'greedy_baseline_rms_matched', greedy_reward=torch.tensor([.2, 100.]))
        self.assertTrue((greedy[1] == 0).all())
        self.assertGreater(float(greedy[0].mean()), 0)
        torch.testing.assert_close(greedy.square().sum(), standard.square().sum())

    def test_reward_bonus_requires_all_four_safety_components(self):
        metrics = torch.ones(5, 7); metrics[:, 0] = .5; metrics[:, 3] = .5
        for i, column in enumerate([1, 2, 4, 5], 1):
            metrics[i, column] = 0
        reward = training_reward(metrics, {'reward_mode': 'safe_ep_bonus', 'safe_ep_coefficient': 1})
        torch.testing.assert_close(reward, torch.tensor([1.25, .5, .5, .5, .5]))

    def test_grpo_one_update_gradients(self):
        logits = torch.randn(3, 8, 401, requires_grad=True)
        reference = logits.detach().clone()
        ids = logits.detach().argmax(-1)
        old = statistics(reference, ids)[0]
        loss, diagnostic = grpo_objective(logits, reference, ids, old, torch.tensor([1., 0., -1.]))
        self.assertEqual(float(diagnostic['ratio_error'].detach()), 0.)
        self.assertAlmostEqual(float(diagnostic['kl'].detach()), 0., places=5)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_epoch_shards_cover_every_real_sample_once(self):
        seen = []
        for rank in range(3):
            for count, entries in epoch_windows(17, 42, 0, 12, rank, 3):
                seen.extend(i for i, valid in entries if valid)
                self.assertGreater(count, 0)
        self.assertEqual(sorted(seen), list(range(17)))

    def test_position_grpo_worker_with_synthetic_reward(self):
        from care_drive.training.experts import train
        class SyntheticScorer:
            def score(self, paths, trajectories):
                metric = np.ones((*trajectories.shape[:2], 7), dtype=np.float32)
                progress = .5 + .2 * np.tanh(trajectories[..., -1, 0] / 10)
                metric[..., 3] = progress
                metric[..., 0] = (5 * progress + 7) / 12
                return metric
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cache = directory / 'features.pt'
            torch.save({'tokens': ['a', 'b'], 'scene': self.scene, 'anchor': self.base,
                        'gt': self.base, 'metric_cache_paths': ['synthetic-a', 'synthetic-b']}, cache)
            heading = recipe('heading')
            save(directory / 'B3-H-GRPO2/epoch_005.pt', ExpertPolicy(heading), heading)
            position = recipe('position')
            position.update(task='GRPO', heading_source='B3-H-GRPO2', advantage_mode='global_rms_matched')
            cfg = {'device': 'cpu', 'expert_cache': str(cache), 'output_dir': str(directory)}
            with patch('care_drive.integrations.navsim.OfficialScorer', SyntheticScorer):
                train(cfg, 'C2-XY-P01', {'C2-XY-P01': position, 'B3-H-GRPO2': heading}, probe_steps=2)
            self.assertTrue((directory / 'C2-XY-P01/probe/probe.pt').exists())

    def test_feature_inference_cli(self):
        import sys
        from care_drive.infer import main
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pc, hc = recipe('position'), recipe('heading')
            save(directory / 'position.pt', ExpertPolicy(pc), pc)
            save(directory / 'heading.pt', ExpertPolicy(hc), hc)
            torch.save({'tokens': ['a', 'b'], 'scene': self.scene, 'anchor': self.base, 'gt': self.base}, directory / 'features.pt')
            (directory / 'stages.json').write_text(json.dumps({'C3-XY-EP1': pc, 'D1-H-ADAPT': hc}))
            (directory / 'config.json').write_text(json.dumps({'device': 'cpu', 'stage_config': str(directory / 'stages.json')}))
            argv = ['infer', '--config', str(directory / 'config.json'), '--position', str(directory / 'position.pt'),
                    '--heading', str(directory / 'heading.pt'), '--feature-cache', str(directory / 'features.pt'),
                    '--output', str(directory / 'out.jsonl'), '--probe-steps', '2']
            with patch.object(sys, 'argv', argv):
                main()
            predictions = [json.loads(line) for line in (directory / 'out.jsonl').read_text().splitlines()]
            self.assertEqual([row['sample_id'] for row in predictions], ['a', 'b'])
            self.assertEqual(np.asarray(predictions[0]['trajectory']).shape, (8, 3))

    def test_performance_gate_rejects_unbounded_probe_or_missing_evidence(self):
        self.assertTrue(bounded_probe(3))
        self.assertFalse(bounded_probe(0))
        with self.assertRaises(PerformancePreflightError):
            bounded_probe(33)
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(PerformancePreflightError):
                require_preflight('training', {}, {}, workload='test')

    def test_bounded_sft_worker_writes_probe_only(self):
        from care_drive.training.experts import train
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cache = directory / 'features.pt'
            torch.save({'tokens': ['a', 'b'], 'scene': self.scene, 'anchor': self.base,
                        'gt': self.base, 'metric_cache_paths': [None, None]}, cache)
            cfg = {'device': 'cpu', 'expert_cache': str(cache), 'output_dir': str(directory / 'runs')}
            train(cfg, 'B1-H-SFT', {'B1-H-SFT': recipe('heading')}, probe_steps=2)
            output = directory / 'runs/B1-H-SFT'
            self.assertTrue((output / 'probe/probe.pt').exists())
            self.assertFalse((output / 'epoch_005.pt').exists())
            info = json.loads((output / 'probe/probe.rank000.json').read_text())
            self.assertEqual(info['completed_steps'], 2)


if __name__ == '__main__':
    unittest.main()
