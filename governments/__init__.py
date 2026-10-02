# Governments package
from typing import Dict, Tuple, Type

from .anarchy import AnarchyGovernment
from .democracy import DemocracyGovernment
from .republic import RepublicGovernment
from .autocracy import AutocracyGovernment
from .autocracy_lookahead import AutocracyLookaheadGovernment
from .oligarchy import OligarchyGovernment
from .federated import FederatedGovernment
from .ads import AdsGovernment

__all__ = [
    "AnarchyGovernment",
    "DemocracyGovernment",
    "RepublicGovernment",
    "AutocracyGovernment",
    "AutocracyLookaheadGovernment",
    "OligarchyGovernment",
    "FederatedGovernment",
    "AdsGovernment",
    "GOVERNMENT_REGISTRY",
    "ABLATION_GOVERNMENTS",
    "DEFAULT_GOVERNMENTS",
]

#: Every government that can be constructed by name.  This is the *validation*
#: set — what ``--governments`` will accept.  It is allowed to be wider than the
#: set that runs by default; today the two coincide.  See
#: ``DEFAULT_GOVERNMENTS``.
GOVERNMENT_REGISTRY: Dict[str, Type] = {
    "anarchy":              AnarchyGovernment,
    "democracy":            DemocracyGovernment,
    "republic":             RepublicGovernment,
    "autocracy":            AutocracyGovernment,
    "autocracy_lookahead":  AutocracyLookaheadGovernment,
    "oligarchy":            OligarchyGovernment,
    "federated":            FederatedGovernment,
    "ads":                  AdsGovernment,
}

#: Regimes excluded from a default sweep.  **Currently empty, on purpose.**
#:
#: ``autocracy_lookahead`` is a foresight-vs-coordination control and a full
#: primary regime at once: it appears in the same sweep and the same
#: comparison figures as the other seven, so it is not excluded here. Its
#: role as a clean control is documented in its own module — being a control
#: and being a primary regime are not in conflict.
#:
#: The *seam* is retained rather than deleted so that a future ablation is
#: still a one-line change here instead of a four-file edit.  An empty
#: exclusion set keeps ``DEFAULT_GOVERNMENTS`` derived rather than hand-listed,
#: which is the property that keeps such a change safe.
ABLATION_GOVERNMENTS: frozenset = frozenset()

#: The paper's primary regimes, in registry order — the set every entry point
#: sweeps when ``--governments`` is not given.
#:
#: Derived, never hand-listed: adding a name to ``ABLATION_GOVERNMENTS`` keeps
#: the default set correct with no second edit.  Today the exclusion set is
#: empty, so this is every registered government — but the two names are NOT
#: interchangeable and should not be collapsed.  ``GOVERNMENT_REGISTRY`` is the
#: *validation* set (what ``--governments`` accepts) and ``DEFAULT_GOVERNMENTS``
#: is the *default sweep* set; they coincide only until someone registers
#: something that should not sweep by default.  Callers that mean "the default
#: sweep" must say ``DEFAULT_GOVERNMENTS``, not ``tuple(GOVERNMENT_REGISTRY)``.
DEFAULT_GOVERNMENTS: Tuple[str, ...] = tuple(
    name for name in GOVERNMENT_REGISTRY if name not in ABLATION_GOVERNMENTS
)

# The published methodology is stated as eight regimes in README.md,
# run_full_simulation.py's docstring, and the manuscript.  Fail at import rather
# than let a sweep silently run a different number of cells than the paper
# claims: 8 governments x 21 difficulties x 100 seeds = 16,800 runs.
#
# NOT an `assert`.  `python -O` strips asserts, which would silently vanish
# this guard under exactly the invocation someone is most likely to use for a
# multi-hour production sweep.  An explicit `raise` costs one comparison at
# import and cannot be optimised away.
if len(DEFAULT_GOVERNMENTS) != 8:
    raise RuntimeError(
        f"expected 8 default governments — the paper's stated design, all eight "
        f"primary regimes including autocracy_lookahead — "
        f"got {len(DEFAULT_GOVERNMENTS)}: {DEFAULT_GOVERNMENTS}"
    )
