"""Thin wrapper: same as `python -m dermxai verify-data`."""

import sys

from dermxai.utils import use_project_local_caches

if __name__ == "__main__":
    use_project_local_caches()
    from dermxai.cli import main

    sys.exit(main(["verify-data", *sys.argv[1:]]))
