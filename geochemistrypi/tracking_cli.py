"""Private JSON bridge used by the separately installed MCP runtime."""

import argparse
import json
import sys
from typing import List, Optional

from .tracking import TrackingStoreError, get_experiment, list_experiments, resolve_tracking_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m geochemistrypi.tracking_cli")
    subcommands = parser.add_subparsers(dest="action", required=True)
    list_command = subcommands.add_parser("list")
    list_command.add_argument("--tracking-root", required=True)
    list_command.add_argument("--maximum-experiments", type=int, default=100)
    get_command = subcommands.add_parser("get")
    get_command.add_argument("--tracking-root", required=True)
    get_command.add_argument("--experiment-id", required=True)
    get_command.add_argument("--maximum-runs", type=int, default=50)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = resolve_tracking_root(args.tracking_root)
        value = (
            list_experiments(root, args.maximum_experiments)
            if args.action == "list"
            else get_experiment(root, args.experiment_id, args.maximum_runs)
        )
    except (OSError, TrackingStoreError) as exc:
        print(json.dumps({"error": " ".join(str(exc).split())[:1000]}), file=sys.stderr)
        return 2
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
