"""Allow `python -m dermxai ...`."""

import sys

from dermxai.utils import use_project_local_caches

if __name__ == "__main__":
    use_project_local_caches()  # must run before torch / matplotlib are imported
    from dermxai.cli import main

    sys.exit(main())
