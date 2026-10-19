"""Distributed training CLI; help and membership validation need no training stack."""
import argparse
from runtime import main


def cli(argv=None):
    parser = argparse.ArgumentParser(description="Launch one explicitly identified training process")
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--node-id', required=True)
    args = parser.parse_args(argv)
    from lifecycle import RunFailed
    try:
        return main(args.config, node_id=args.node_id)
    except RunFailed as outcome:
        return 130 if outcome.interrupted else 1
    except TimeoutError as error:
        print(f"Run failed: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
