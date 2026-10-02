#!/usr/bin/env python3
"""
run_benchmark.py — DEPRECATED.  Use ``run_full_simulation.py`` instead.

This is a forwarding shim: it takes the same arguments and produces the same
output as ``run_full_simulation.py``, which is where the sweep, plotting, and
output logic — and the configuration — actually live. It carries no sweep
constants of its own, so it cannot drift out of sync with the published
methodology. Kept only so that existing references and scripts that invoke
``run_benchmark.py`` keep working; it will be removed in a future release —
update scripts and documentation to call ``run_full_simulation.py`` directly.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_full_simulation import main as _full_main


def main(argv=None) -> int:
    print(
        "\n"
        "  " + "!" * 68 + "\n"
        "  DEPRECATED: run_benchmark.py has been replaced by run_full_simulation.py.\n"
        "              Forwarding to run_full_simulation.py — please update your\n"
        "              scripts and documentation.\n"
        "  " + "!" * 68 + "\n",
        file=sys.stderr, flush=True,
    )
    return _full_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
