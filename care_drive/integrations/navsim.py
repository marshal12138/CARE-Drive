"""Official NAVSIM v1 CPU scoring; no substitute reward implementation."""
from dataclasses import asdict
from functools import lru_cache
import lzma
from pathlib import Path
import pickle
import numpy as np
from ..training.rewards import PDM_METRICS


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


@lru_cache(maxsize=64)
def load_metric_cache(path):
    # NAVSIM metric caches are trusted, locally generated pickle artifacts.
    with lzma.open(Path(path), "rb") as stream:
        return NumpyCompatUnpickler(stream).load()


class OfficialScorer:
    def __init__(self):
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        self.sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        config = PDMScorerConfig()
        actual = (config.progress_weight, config.ttc_weight, config.comfortable_weight, config.driving_direction_weight)
        if actual != (5, 5, 2, 0):
            raise ValueError("Install NAVSIM v1: the configured PDMS weights differ from the paper")
        self.simulator, self.scorer = PDMSimulator(self.sampling), PDMScorer(self.sampling, config)

    def score(self, metric_paths, trajectories):
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score
        values = np.asarray(trajectories, dtype=np.float32)
        if values.ndim != 4 or values.shape[0] != len(metric_paths) or values.shape[-2:] != (8, 3):
            raise ValueError("Expected candidates [scenes,candidates,8,3]")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite candidate trajectories")
        results = []
        for path, candidates in zip(metric_paths, values):
            if not path:
                raise ValueError("A metric cache is required for every scored training scene")
            cache = load_metric_cache(str(Path(path).resolve()))
            row = []
            for candidate in candidates:
                result = asdict(pdm_score(cache, Trajectory(candidate, TrajectorySampling(time_horizon=4.0, interval_length=0.5)),
                                         self.sampling, self.simulator, self.scorer))
                metrics = [float(result[name]) for name in PDM_METRICS]
                if not np.isfinite(metrics).all():
                    raise ValueError("Official scorer returned nonfinite values")
                expected = metrics[1] * metrics[2] * (5 * metrics[3] + 5 * metrics[4] + 2 * metrics[5]) / 12
                if abs(metrics[0] - expected) > 1e-10:
                    raise ValueError("NAVSIM scoring protocol mismatch")
                row.append(metrics)
            results.append(row)
        return np.asarray(results, dtype=np.float32)
