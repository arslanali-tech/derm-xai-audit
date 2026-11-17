"""Thin wrapper: same as `python -m dermxai preprocess` (use --limit 20 for a timing run first)."""

import sys

from dermxai.utils import use_project_local_caches

if __name__ == "__main__":
    use_project_local_caches()
    from dermxai.cli import main

    sys.exit(main(["preprocess", *sys.argv[1:]]))
