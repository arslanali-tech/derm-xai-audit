"""Thin wrapper: same as `python -m dermxai download` (add --yes to download, --extract to unzip)."""

import sys

from dermxai.utils import use_project_local_caches

if __name__ == "__main__":
    use_project_local_caches()
    from dermxai.cli import main

    sys.exit(main(["download", *sys.argv[1:]]))
