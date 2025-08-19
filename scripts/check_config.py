#!/usr/bin/env python3
"""Script to check CRISP configuration paths and files."""

import argparse
import os
from pathlib import Path

try:
    from rich import print
except ImportError:
    print = print

from crisp_gym.config.path import CRISP_CONFIG_PATH, CRISP_CONFIG_PATHS, find_config

parser = argparse.ArgumentParser(description="Check CRISP configuration paths and files")
parser.add_argument(
    "--no-tree",
    action="store_true",
    help="Disable tree view of configuration directories",
)

args = parser.parse_args()


def print_tree(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """Print directory tree structure."""
    if current_depth >= max_depth:
        return

    try:
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1

            # Tree symbols
            if is_last_item:
                current_prefix = "└── "
                next_prefix = prefix + "    "
            else:
                current_prefix = "├── "
                next_prefix = prefix + "│   "

            # Item name with type indicator
            if item.is_dir():
                print(f"{prefix}{current_prefix}{item.name}/")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
            else:
                print(f"{prefix}{current_prefix}{item.name}")

    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")


def main():
    """Check configuration paths and report status."""
    print("Checking CRISP configuration setup...")

    env_var = os.environ.get("CRISP_CONFIG_PATH")
    if env_var:
        print(f"CRISP_CONFIG_PATH environment variable: {env_var}")
    else:
        print("CRISP_CONFIG_PATH environment variable: Not set (using default)")

    print(f"Number of valid configuration paths: {len(CRISP_CONFIG_PATHS)}")
    print(f"Primary configuration path: {CRISP_CONFIG_PATH}")

    for i, path in enumerate(CRISP_CONFIG_PATHS, 1):
        print(f"Path {i}: {path}")
        if path.exists():
            if path.is_dir():
                print("  ✓ Directory exists and is accessible")

                if not args.no_tree:
                    print("  Tree structure:")
                    print_tree(path, "    ")
            else:
                print("  ✗ Path exists but is not a directory")
        else:
            print("  ✗ Path does not exist")

    print("Testing configuration file search...")

    test_files = ["envs", "control"]

    for filename in test_files:
        result = find_config(filename)
        if result:
            print(f"  ✓ Found '{filename}' at: {result}")

    print("Configuration check complete!")


if __name__ == "__main__":
    main()
