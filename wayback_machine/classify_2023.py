"""Historical classification entrypoint.

The March-2023 run used a batch classifier that has been removed. This command
exits 2. Production classification is ``python -m two_pass_classifier``.
"""

from __future__ import annotations

import sys

RETIRED_MESSAGE = (
    "Historical classification ran on the retired batch classifier. "
    "Production classification is python -m two_pass_classifier."
)


def main() -> None:
    """Exit 2. There is no batch runner left to classify archive evidence."""
    print(RETIRED_MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
