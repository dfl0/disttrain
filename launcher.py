"""Validate a topology and produce one deployment/launch entry per node."""

import argparse
import sys


def launch_entries(topology):
    """Return root, edges, evaluator, then workers, preserving configured order."""
    nodes = topology["node_configs"]
    root_id = topology["root_id"]
    eval_id = topology["eval_id"]
    ordered = [root_id]
    ordered.extend(node_id for node_id, node in nodes.items() if node["role"] == "edge_ps")
    ordered.append(eval_id)
    ordered.extend(node_id for node_id, node in nodes.items() if node["role"] == "worker")
    return [(nodes[node_id]["host"], node_id, nodes[node_id]["role"]) for node_id in ordered]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args(argv)
    try:
        from topology import load_topology

        entries = launch_entries(load_topology(args.config))
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"invalid topology: {exc}\n")
    for host, node_id, role in entries:
        print(f"{host}\t{node_id}\t{role}")


if __name__ == "__main__":
    main()
