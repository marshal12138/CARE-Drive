"""Score prediction JSONL files with official NAVSIM v1 PDMS."""
import argparse
import json
from pathlib import Path
import numpy as np
from .datasets import read_manifest
from .integrations.navsim import OfficialScorer
from .training.rewards import PDM_METRICS
from .io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    predictions = {}
    for path in args.predictions:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            token = record["sample_id"]
            if token in predictions:
                raise ValueError("Duplicate prediction: " + token)
            predictions[token] = record["trajectory"]
    if set(predictions) != {r["sample_id"] for r in rows}:
        raise ValueError("Prediction coverage differs from the evaluation manifest")
    scorer = OfficialScorer()
    results = []
    for row in rows:
        value = scorer.score([row["evaluation"]["metric_cache_path"]],
                             np.asarray(predictions[row["sample_id"]])[None, None])[0, 0]
        results.append({"sample_id": row["sample_id"], **dict(zip(PDM_METRICS, value.astype(float).tolist()))})
    summary = {name: float(np.mean([r[name] for r in results])) for name in PDM_METRICS}
    write_json(Path(args.output), {"scenes": len(rows), "pdms": 100 * summary["score"],
                                 "normalized_metrics": summary, "per_scene": results}, replace=True)
    print(summary)


if __name__ == "__main__":
    main()
