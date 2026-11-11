"""Train the predictive backbone or one stage of the coupled expert chain."""
import argparse
from .config import load, recipes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["backbone", "expert"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--probe-steps", type=int, default=0)
    args = parser.parse_args()
    config = load(args.config)
    if args.component == "backbone":
        from .training.backbone import train
        train(config, probe_steps=args.probe_steps)
    else:
        stages = recipes(config["stage_config"])
        if args.stage not in stages:
            parser.error("--stage must be one of " + ", ".join(stages))
        from .training.experts import train
        train(config, args.stage, stages, probe_steps=args.probe_steps)


if __name__ == "__main__":
    main()
