#!/usr/bin/env python3
"""
Regression tests for sweep logging and per-run detail output.

What these lock in, and why each one is here rather than left to inspection:

  1. **The serialization contract.**  ``json.dumps`` emits a bare ``Infinity``
     for a non-finite float, which is not valid JSON.  One such value anywhere
     in the 16,800-run production sweep makes that run's file unreadable by
     ``jq`` and by every strict parser — and it fails at *read* time, months
     later, not at write time.  So: non-finite floats must become ``null``,
     and the dump must use ``allow_nan=False`` so an escapee is loud rather
     than silent.

  2. **The record schema.**  A real run must produce parseable JSONL with a
     header first, a footer last, and one cycle record per cycle.

  3. **The failure path.**  A run that raises must still leave a footer carrying
     the exception — a failed run has to leave *more* evidence than a
     successful one, not less — and the exception must still propagate.

  4. **The decision record.**  It must appear only on cycles where a round
     actually ran (no stale duplication), and must carry all six evidence
     fields the decision round computes.

  5. **Logger level isolation.**  The design depends on non-obvious Python
     behaviour: ``sim`` at WARNING must not suppress ``sim.bench`` at INFO.
     If that were wrong, the sweep would run with no progress reporting at all.

  6. **The --detailed-agents ceiling.**  A hard limit, not advice: the flag at
     full scale writes ~2.4 GB.

  7. **Output stability.**  health_stats.csv and final_stats.json must be
     byte-identical regardless of whether per-run detail logging is enabled.

Usage (from the Simulation project root):
    python3 tests/system/test_stage2_logging_detail.py
"""

import argparse
import contextlib
import inspect
import io
import json
import logging
import logging.handlers
import multiprocessing
import os
import queue as _q
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root: tests/<subpkg>/<file>.py -> tests/<subpkg> -> tests/ -> root
_PARENT = _HERE  # sys.path target for the flat top-level packages (engine, governments, ...)
sys.path.insert(0, _PARENT)

import benchmark_core as benchmark_core
import benchmark_logging as benchmark_logging
from benchmark_core import (
    BenchmarkConfig,
    ConfigError,
    build_plan,
    run_gov_difficulty,
)
from benchmark_logging import (
    BENCH_LOGGER,
    DETAILED_AGENTS_MAX_RUNS,
    ROOT_LOGGER,
    default_log_path,
    resolve_log_path,
    setup_parent_logging,
    utc_now,
)
from engine.run_recorder import (
    AGENT_FIELDS,
    RunRecorder,
    _jsonable,
    dumps_record,
    summarise_params,
)
from governments.ads import (
    GROUP_DISTRIBUTION_COUNTS,
    FoodEvidence,
)
from governments.base import Government


# ==========================================================================
# Harness (mirrors the other suites in this directory)
# ==========================================================================

class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name, detail=""):
        self.passed.append((name, detail))
        print(f"  PASS: {name}")

    def fail(self, name, detail=""):
        self.failed.append((name, detail))
        print(f"  FAIL: {name} — {detail}")

    def check(self, condition, name, detail=""):
        if condition:
            self.ok(name)
        else:
            self.fail(name, detail)

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"  RESULTS: {len(self.passed)}/{total} passed, {len(self.failed)} failed")
        if self.failed:
            print("\n  FAILURES:")
            for name, detail in self.failed:
                print(f"    - {name}: {detail}")
        print(f"{'='*60}\n")
        return len(self.failed) == 0


results = TestResults()


def _tiny_config(output_root, **overrides):
    """A configuration small enough to run in about a second."""
    kwargs = dict(
        governments=("ads",),
        difficulties=(50,),
        k_runs=1,
        grid_size=8,
        n_agents=12,
        max_cycles=14,
        max_steps=2,
        output_root=output_root,
        label="logging-detail test",
        log_file=os.path.join(output_root, "test.log"),
        heartbeat_every=5,
    )
    kwargs.update(overrides)
    return BenchmarkConfig(**kwargs)


def _run_cell(config, gov_name="ads", difficulty=50):
    """Run one cell quietly and return the run_01 directory."""
    # Plans are derived per run inside the cell, not passed in.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_gov_difficulty(config, gov_name, difficulty)
    return os.path.join(config.output_root, gov_name, str(difficulty), "run_01")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ==========================================================================
# TEST 1: the serialization contract
# ==========================================================================

def test_serialization_contract():
    """
    Non-finite floats, sets, tuples and dataclasses must all survive
    ``allow_nan=False``.

    This is the rule whose absence is a latent production bug rather than a
    style problem: Python happily writes ``Infinity`` into a .jsonl file and
    nothing complains until someone tries to read it.
    """
    print("\n--- Test 1: JSON serialization contract ---")

    for name, value in (
        ("+inf", float("inf")),
        ("-inf", float("-inf")),
        ("nan", float("nan")),
    ):
        results.check(
            _jsonable(value) is None,
            f"non-finite float ({name}) is sanitised to null",
            f"got {_jsonable(value)!r}",
        )

    results.check(
        _jsonable({"b", "a", "c"}) == ["a", "b", "c"],
        "a set becomes a sorted list (so records are deterministic)",
        f"got {_jsonable({'b', 'a', 'c'})!r}",
    )
    results.check(
        _jsonable((1, 2, 3)) == [1, 2, 3],
        "a tuple becomes a list",
        f"got {_jsonable((1, 2, 3))!r}",
    )

    try:
        import numpy as np
        results.check(
            _jsonable(np.float64(1.5)) == 1.5 and _jsonable(np.int64(3)) == 3,
            "numpy scalars are unwrapped without importing numpy in the recorder",
            f"got {_jsonable(np.float64(1.5))!r} / {_jsonable(np.int64(3))!r}",
        )
    except ImportError:                              # pragma: no cover
        pass

    # An unserialisable object must degrade to a repr, never raise: a bad value
    # in one field must not cost the whole record.
    class Opaque:
        def __repr__(self):
            return "<opaque>"

    results.check(
        _jsonable(Opaque()) == "<opaque>",
        "an unknown object degrades to its repr instead of raising",
        f"got {_jsonable(Opaque())!r}",
    )

    # The end-to-end guarantee, on the exact dataclass the design named: the
    # denominator of cycles_until_empty is a division that can go non-finite.
    evidence = FoodEvidence(
        mean_stock=0.0,
        pct_critical=1.0,
        pct_starving=float("nan"),
        grid_mean=0.0,
        grid_total=0.0,
        cycles_until_empty=float("inf"),
        drought_active=True,
        warning_drought=False,
    )
    try:
        line = dumps_record({"rec": "decision", "evidence": {"food": evidence}})
    except ValueError as exc:
        results.fail(
            "an evidence dataclass carrying inf/nan serialises to valid JSON",
            f"dumps_record raised {exc}",
        )
        return

    parsed = json.loads(line)
    results.check(
        parsed["evidence"]["food"]["cycles_until_empty"] is None
        and parsed["evidence"]["food"]["pct_starving"] is None,
        "cycles_until_empty=inf and pct_starving=nan both serialise as null",
        f"got {parsed['evidence']['food']}",
    )
    results.check(
        "Infinity" not in line and "NaN" not in line,
        "the emitted line contains no bare Infinity/NaN token",
        f"line: {line[:200]}",
    )
    results.check(
        "\n" not in line,
        "a record is exactly one line",
        "record contains an embedded newline",
    )


# ==========================================================================
# TEST 2: law parameter summarising
# ==========================================================================

def test_law_param_summarising():
    """
    Big collections in ``Law.params`` collapse to a count.

    ``donor_amounts`` holds hundreds of agent ids; dumping it raw multiplies the
    file size by an order of magnitude and tells a reader nothing they can act
    on.  A small numeric tuple (a region bbox) must survive intact, though —
    that one *is* actionable.
    """
    print("\n--- Test 2: law params are summarised, not dumped ---")

    out = summarise_params({
        "max_per_cycle": 4.0,
        "region": (1, 2, 3, 4),
        "recipient_ids": [f"A-{i:04d}" for i in range(120)],
        "donor_amounts": {f"A-{i:04d}": 1.5 for i in range(80)},
        "law_directed": True,
    })
    results.check(out.get("max_per_cycle") == 4.0, "scalars pass through")
    results.check(out.get("region") == [1, 2, 3, 4],
                  "a small numeric tuple (region bbox) is kept whole",
                  f"got {out.get('region')!r}")
    results.check(out.get("recipient_ids_count") == 120
                  and "recipient_ids" not in out,
                  "a long list collapses to <key>_count",
                  f"got {out!r}")
    results.check(out.get("donor_amounts_count") == 80
                  and "donor_amounts" not in out,
                  "a dict collapses to <key>_count",
                  f"got {out!r}")


# ==========================================================================
# TEST 3: a real run produces well-formed JSONL
# ==========================================================================

def test_run_detail_schema():
    print("\n--- Test 3: run_detail.jsonl schema ---")

    tmp = tempfile.mkdtemp(prefix="stage2_schema_")
    try:
        cfg = _tiny_config(tmp)
        run_dir = _run_cell(cfg)
        path = os.path.join(run_dir, "run_detail.jsonl")

        if not os.path.isfile(path):
            results.fail("run_detail.jsonl is written by default", f"missing {path}")
            return
        results.ok("run_detail.jsonl is written by default (no flag required)")

        # Every line must parse. Reported per-line so a failure names the line.
        records = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    results.fail("every line of run_detail.jsonl parses as JSON",
                                 f"line {lineno}: {exc}")
                    return
        results.ok("every line of run_detail.jsonl parses as JSON")

        kinds = [r["rec"] for r in records]
        results.check(kinds[0] == "header", "the first record is the header",
                      f"got {kinds[0]!r}")
        results.check(kinds[-1] == "footer", "the last record is the footer",
                      f"got {kinds[-1]!r}")
        results.check(kinds.count("header") == 1, "exactly one header",
                      f"got {kinds.count('header')}")
        results.check(kinds.count("footer") == 1, "exactly one footer",
                      f"got {kinds.count('footer')}")
        results.check(
            kinds.count("cycle") == cfg.max_cycles,
            "one cycle record per simulated cycle",
            f"expected {cfg.max_cycles}, got {kinds.count('cycle')}",
        )
        results.check(
            set(kinds) <= {"header", "cycle", "decision", "footer"},
            "no unexpected record types",
            f"got {sorted(set(kinds))}",
        )

        header = records[0]
        results.check(header.get("schema_version") == 1,
                      "the header declares a schema version")
        run = header.get("run", {})
        # The header records the full seed lattice ({env, pop, gov} seeds
        # derived from base_seed), not just base_seed itself. The check that
        # matters is that the recorded seeds are the ones the derivation
        # actually produces — a header that reports seeds a rerun would not
        # reproduce is worse than no header.
        from engine.scenario_plan import derive_seed
        expected_env = derive_seed(cfg.base_seed, "env", 50, 0)
        expected_pop = derive_seed(cfg.base_seed, "pop", 50, 0)
        expected_gov = derive_seed(cfg.base_seed, "gov", "ads", 50, 0)
        results.check(
            run.get("base_seed") == cfg.base_seed
            and run.get("env_seed") == expected_env
            and run.get("pop_seed") == expected_pop
            and run.get("gov_seed") == expected_gov,
            "the header records the full seed lattice, each correctly derived",
            f"got {run}",
        )
        results.check(
            isinstance(run.get("env_fingerprint"), str)
            and len(run["env_fingerprint"]) == 16,
            "the header records the environment fingerprint",
            f"got {run.get('env_fingerprint')!r}",
        )
        results.check(
            header.get("government_params", {}).get("group_distribution_counts")
            == list(GROUP_DISTRIBUTION_COUNTS),
            "the header records the government's tunables",
            f"got {header.get('government_params')}",
        )

        cycles = [r for r in records if r["rec"] == "cycle"]
        first = cycles[0]
        for section in ("pop", "env", "events", "laws", "government"):
            results.check(section in first, f"a cycle record carries its `{section}` block")
        results.check(
            {"starving", "dehydrated", "hungry", "thirsty", "infected"}
            <= set(first["pop"]),
            "population counts include the starving/dehydrated thresholds",
            f"got {sorted(first['pop'])}",
        )
        results.check(
            first["cycle"] == 0 and cycles[-1]["cycle"] == cfg.max_cycles - 1,
            "cycle records are indexed 0..max_cycles-1",
            f"got {first['cycle']}..{cycles[-1]['cycle']}",
        )

        footer = records[-1]
        results.check(footer.get("status") == "ok",
                      "a clean run reports status=ok", f"got {footer.get('status')}")
        results.check(footer.get("recorder_errors") == 0,
                      "a clean run reports zero recorder errors",
                      f"got {footer.get('recorder_errors')}")
        results.check(
            footer.get("cycles_completed") == cfg.max_cycles,
            "the footer counts every completed cycle",
            f"got {footer.get('cycles_completed')}",
        )
        results.check(
            footer.get("summary", {}).get("normalized_health_score") is not None,
            "the footer embeds the run summary (self-contained record)",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 4: the ADS decision record
# ==========================================================================

def test_decision_record():
    print("\n--- Test 4: ADS decision records ---")

    tmp = tempfile.mkdtemp(prefix="stage2_decision_")
    try:
        cfg = _tiny_config(tmp, max_cycles=20)
        run_dir = _run_cell(cfg)
        records = _read_jsonl(os.path.join(run_dir, "run_detail.jsonl"))

        decisions = [r for r in records if r["rec"] == "decision"]
        cycles = {r["cycle"]: r for r in records if r["rec"] == "cycle"}
        results.check(decisions, "ADS produces at least one decision record",
                      "no decision records found")
        if not decisions:
            return

        # No stale duplication: a decision record may only exist on a cycle
        # whose sibling cycle record agrees a round happened.
        mismatched = [
            d["cycle"] for d in decisions
            if not cycles.get(d["cycle"], {}).get("government", {})
            .get("decision_this_cycle")
        ]
        results.check(
            not mismatched,
            "every decision record's cycle agrees with decision_this_cycle",
            f"mismatched cycles: {mismatched}",
        )
        flagged = {c for c, r in cycles.items()
                   if r["government"]["decision_this_cycle"]}
        results.check(
            flagged == {d["cycle"] for d in decisions},
            "no cycle is flagged as deciding without a matching decision record",
            f"flagged={sorted(flagged)} "
            f"records={sorted(d['cycle'] for d in decisions)}",
        )

        first = decisions[0]
        # The six evidence fields the decision record must always carry.
        preserved = {
            ("food", "pct_starving"),
            ("water", "pct_dehydrated"),
            ("food", "cycles_until_empty"),
            ("water", "cycles_until_empty"),
            ("terrain", "mean_hazard"),
            ("terrain", "avg_dist_to_shelter"),
            ("terrain", "shelter_cell_count"),
        }
        for category, field in sorted(preserved):
            value = first.get("evidence", {}).get(category, {}).get(field, "MISSING")
            results.check(
                isinstance(value, (int, float)) and not isinstance(value, bool),
                f"evidence.{category}.{field} is present and numeric",
                f"got {value!r}",
            )

        groupings = first.get("groupings", [])
        results.check(
            isinstance(groupings, list),
            "groupings is an array (integer keys and order preserved)",
            f"got {type(groupings).__name__}",
        )
        results.check(
            [g["n_groups_target"] for g in groupings] == list(GROUP_DISTRIBUTION_COUNTS),
            "every grouping scheme in GROUP_DISTRIBUTION_COUNTS is recorded, in order",
            f"got {[g.get('n_groups_target') for g in groupings]}",
        )
        results.check(
            all("n_groups_actual" in g for g in groupings),
            "each grouping records the ACTUAL group count, not just the target",
        )
        results.check(
            isinstance(first.get("rejected_reasons"), dict)
            and "law_type_already_active" in first["rejected_reasons"],
            "rejected_reasons explains why candidates did not become laws",
            f"got {first.get('rejected_reasons')}",
        )
        results.check(
            isinstance(first.get("enacted"), list),
            "enacted is the complete list of laws from the round, not just the winner",
        )
        results.check(
            first.get("trigger", {}).get("kind") in
            {"scheduled_interval", "crisis_interval", "warning_preempt"},
            "the trigger records why the round ran",
            f"got {first.get('trigger')}",
        )

        # Law provenance: reflex vs evaluated round is a structured field, not
        # a free-text prefix inside the description.
        sources = {
            law["source"]
            for r in records if r["rec"] == "cycle"
            for law in r["laws"]["enacted"]
        }
        results.check(
            sources and sources <= {"fast_response", "decision_round",
                                    "preempt_warning", "eco_ration", None},
            "enacted laws record which mechanism produced them",
            f"got {sources}",
        )
        reasons = {
            law["reason"]
            for r in records if r["rec"] == "cycle"
            for law in r["laws"]["expired"]
        }
        results.check(
            reasons <= {"duration_elapsed", "event_type_gone", "event_id_cleared"},
            "expired laws record WHY they lifted",
            f"got {reasons}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 5: non-ADS governments still get per-cycle governance detail
# ==========================================================================

def test_non_ads_governments_recorded():
    print("\n--- Test 5: non-ADS governments produce audit detail ---")

    tmp = tempfile.mkdtemp(prefix="stage2_nonads_")
    try:
        for gov in ("democracy", "anarchy"):
            cfg = _tiny_config(tmp, governments=(gov,), max_cycles=10)
            run_dir = _run_cell(cfg, gov_name=gov)
            records = _read_jsonl(os.path.join(run_dir, "run_detail.jsonl"))
            cycles = [r for r in records if r["rec"] == "cycle"]
            results.check(
                cycles and isinstance(cycles[0]["government"]["audit"], dict)
                and cycles[0]["government"]["audit"],
                f"{gov} contributes get_audit_info() to every cycle record",
                f"got {cycles[0]['government'] if cycles else 'no cycles'}",
            )
            results.check(
                not any(r["rec"] == "decision" for r in records),
                f"{gov} produces no decision records (default hook returns None)",
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_default_decision_hook():
    print("\n--- Test 6: Government.get_decision_record default ---")
    results.check(
        Government.get_decision_record(object(), 0) is None,
        "the base hook returns None so non-deliberating governments opt out",
    )
    results.check(
        Government.get_params(object()) == {},
        "the base params hook returns an empty dict",
    )


# ==========================================================================
# TEST 7: the failure path leaves a forensic footer
# ==========================================================================

def test_failed_run_writes_footer():
    """
    A run that dies must leave a footer with the exception, and the exception
    must still reach the caller.  This is the whole premise of writing JSONL
    incrementally rather than one document at the end.
    """
    print("\n--- Test 7: a failed run still writes a forensic footer ---")

    tmp = tempfile.mkdtemp(prefix="stage2_fail_")
    try:
        detail = os.path.join(tmp, "run_detail.jsonl")
        recorder = RunRecorder(detail, {"run": {"government": "test"}})
        boom = RuntimeError("injected mid-run failure")

        propagated = False
        try:
            with recorder:
                recorder._cycles_written = 3     # pretend three cycles ran
                raise boom
        except RuntimeError as exc:
            propagated = exc is boom

        results.check(
            propagated,
            "the exception still propagates (__exit__ returns False)",
            "the recorder swallowed the exception",
        )

        records = _read_jsonl(detail)
        footer = records[-1]
        results.check(footer["rec"] == "footer" and footer["status"] == "failed",
                      "the footer records status=failed", f"got {footer}")
        results.check(footer.get("cycles_completed") == 3,
                      "the footer records how far the run got",
                      f"got {footer.get('cycles_completed')}")
        results.check(
            footer.get("error", {}).get("type") == "RuntimeError"
            and "injected mid-run failure" in footer["error"]["message"]
            and "Traceback" in footer["error"]["traceback"],
            "the footer carries the exception type, message and traceback",
            f"got {footer.get('error')}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_recorder_never_raises_on_bad_output_path():
    """An unwritable output path must degrade, not kill the run."""
    print("\n--- Test 8: the recorder degrades rather than failing the science ---")

    # A path whose parent is a FILE, so makedirs/open cannot succeed.
    tmp = tempfile.mkdtemp(prefix="stage2_badpath_")
    try:
        blocker = os.path.join(tmp, "not_a_dir")
        with open(blocker, "w", encoding="utf-8") as f:
            f.write("x")
        try:
            with RunRecorder(os.path.join(blocker, "run_detail.jsonl"), {}) as rec:
                rec.record_cycle(None, 0, ())     # would explode if unguarded
            results.ok("an unopenable detail file is reported, not raised")
            results.check(rec.errors > 0,
                          "the failure is counted rather than swallowed silently",
                          f"errors={rec.errors}")
        except Exception as exc:
            results.fail("an unopenable detail file is reported, not raised",
                         f"raised {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 9: --detailed-agents
# ==========================================================================

def test_detailed_agents_ceiling():
    print("\n--- Test 9: --detailed-agents is hard-capped ---")

    base = dict(
        governments=("ads",),
        difficulties=(50,),
        grid_size=8,
        n_agents=12,
        max_cycles=10,
        max_steps=2,
        output_root="/tmp/does-not-need-to-exist",
    )

    # At the limit: allowed.
    try:
        BenchmarkConfig(k_runs=DETAILED_AGENTS_MAX_RUNS, agent_states_every=10, **base)
        results.ok(f"--detailed-agents is accepted at exactly "
                   f"{DETAILED_AGENTS_MAX_RUNS} runs")
    except ConfigError as exc:
        results.fail(f"--detailed-agents is accepted at exactly "
                     f"{DETAILED_AGENTS_MAX_RUNS} runs", str(exc))

    # One over: refused, with an actionable message.
    try:
        BenchmarkConfig(
            k_runs=DETAILED_AGENTS_MAX_RUNS + 1, agent_states_every=10, **base
        )
    except ConfigError as exc:
        message = str(exc)
        results.ok("--detailed-agents is refused above the ceiling")
        results.check(
            "--governments" in message and "--difficulties" in message,
            "the refusal tells the user how to narrow the sweep",
            f"message was: {message}",
        )
        results.check(
            "MB" in message or "GB" in message,
            "the refusal quantifies what it prevented",
            f"message was: {message}",
        )
        # At small sweeps the estimate must not render as the
        # self-contradicting "about 0.0 GB" — a refusal that looks like it
        # prevented nothing. Below 1 GB the message reports MB instead.
        results.check(
            "0.0 GB" not in message,
            "the refusal never renders a self-contradicting '0.0 GB'",
            f"message was: {message}",
        )
    else:
        results.fail("--detailed-agents is refused above the ceiling",
                     "an oversized sweep was accepted")

    # A cadence below 1 would emit on every cycle forever.
    try:
        BenchmarkConfig(k_runs=1, agent_states_every=0, **base)
    except ConfigError:
        results.ok("a --detailed-agents cadence below 1 is rejected")
    else:
        results.fail("a --detailed-agents cadence below 1 is rejected",
                     "cadence 0 was accepted")


def test_detailed_agents_output():
    print("\n--- Test 10: --detailed-agents output format ---")

    tmp = tempfile.mkdtemp(prefix="stage2_agents_")
    try:
        cfg = _tiny_config(tmp, max_cycles=12, agent_states_every=5)
        run_dir = _run_cell(cfg)
        path = os.path.join(run_dir, "agent_states.jsonl")

        if not os.path.isfile(path):
            results.fail("agent_states.jsonl is written when the flag is set",
                         f"missing {path}")
            return
        results.ok("agent_states.jsonl is written when the flag is set")

        records = _read_jsonl(path)
        results.check(all(r["rec"] == "agents" for r in records),
                      "every agent_states record is of type `agents`")
        results.check(
            records[0]["cycle"] == 0
            and records[-1]["cycle"] == cfg.max_cycles - 1,
            "the first and last cycles are always captured, whatever the cadence",
            f"cycles={[r['cycle'] for r in records]}",
        )
        results.check(
            len(records) < cfg.max_cycles,
            "the cadence actually bounds the number of emissions",
            f"{len(records)} emissions for {cfg.max_cycles} cycles",
        )
        first = records[0]
        results.check(
            first["fields"] == list(AGENT_FIELDS),
            "the record is columnar and self-describing (schema travels with data)",
            f"got {first.get('fields')}",
        )
        results.check(
            all(len(row) == len(AGENT_FIELDS) for row in first["rows"]),
            "every row matches the declared field list",
        )
        results.check(
            len(first["rows"]) == cfg.n_agents,
            "every agent appears, alive or dead",
            f"expected {cfg.n_agents}, got {len(first['rows'])}",
        )

        # Under this flag law params are emitted raw — it is the targeted
        # debugging mode, so the size tradeoff flips.
        detail = _read_jsonl(os.path.join(run_dir, "run_detail.jsonl"))
        raw_seen = any(
            isinstance(law["params"].get("donor_amounts"), dict)
            or isinstance(law["params"].get("recipient_ids"), list)
            for r in detail if r["rec"] == "cycle"
            for law in r["laws"]["enacted"]
        )
        summarised_seen = any(
            "donor_amounts_count" in law["params"]
            or "recipient_ids_count" in law["params"]
            for r in detail if r["rec"] == "cycle"
            for law in r["laws"]["enacted"]
        )
        results.check(
            raw_seen and not summarised_seen,
            "law params are emitted RAW under --detailed-agents",
            f"raw={raw_seen} summarised={summarised_seen}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 11: logging
# ==========================================================================

def test_log_path_resolution():
    print("\n--- Test 11: log file naming and placement ---")

    when = utc_now()
    path = default_log_path("sim_full", when, "/tmp")
    name = os.path.basename(path)
    results.check(
        name.startswith("sim_full_") and name.endswith("Z.log")
        and len(name) == len("sim_full_20260901T143022Z.log"),
        "the generated name is sim_full_<UTC basic timestamp>.log",
        f"got {name}",
    )
    results.check(
        os.path.isabs(path),
        "the log path is absolute (so the banner can be copy-pasted)",
        f"got {path}",
    )

    tmp = tempfile.mkdtemp(prefix="stage2_logpath_")
    try:
        # A directory override places the generated name inside it.
        in_dir = resolve_log_path("sim_quick", tmp, when)
        results.check(
            os.path.dirname(in_dir) == os.path.abspath(tmp)
            and os.path.basename(in_dir).startswith("sim_quick_"),
            "a directory override receives the generated filename",
            f"got {in_dir}",
        )
        # An explicit file path is used verbatim.
        explicit = os.path.join(tmp, "custom.log")
        results.check(
            resolve_log_path("sim_quick", explicit, when) == explicit,
            "an explicit --log-file path is used verbatim",
            f"got {resolve_log_path('sim_quick', explicit, when)}",
        )
        # Default: the CWD, NOT the output directory — the log must survive
        # someone clearing benchmark_results/.
        results.check(
            os.path.dirname(resolve_log_path("sim_full", None, when, os.getcwd()))
            == os.path.abspath(os.getcwd()),
            "with no override the log lands in the current working directory",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_logger_level_isolation():
    """
    The load-bearing, non-obvious behaviour the whole logging design rests on.

    Python checks the *originating* logger's effective level and then walks to
    ancestor HANDLERS without re-checking ancestor levels.  So `sim` pinned at
    WARNING does not suppress `sim.bench` at INFO — which is why the harness can
    report progress while the engine's per-cycle flood stays off.  If this were
    the other way round, the default configuration would produce a silent sweep.
    """
    print("\n--- Test 12: sim=WARNING does not mute sim.bench=INFO ---")

    tmp = tempfile.mkdtemp(prefix="stage2_levels_")
    try:
        log_path = os.path.join(tmp, "levels.log")
        with setup_parent_logging(log_path, engine_level="WARNING", console=False):
            logging.getLogger(BENCH_LOGGER).info("BENCH_INFO_MARKER")
            logging.getLogger("sim.ADS").info("ENGINE_INFO_MARKER")
            logging.getLogger("sim.ADS").warning("ENGINE_WARNING_MARKER")
            logging.getLogger("sim.events").debug("ENGINE_DEBUG_MARKER")

        with open(log_path, encoding="utf-8") as f:
            text = f.read()

        results.check("BENCH_INFO_MARKER" in text,
                      "a sim.bench INFO record reaches the file",
                      "harness progress reporting was suppressed by the engine level")
        results.check("ENGINE_INFO_MARKER" not in text,
                      "a sim.<Gov> INFO record is suppressed at the default level",
                      "the per-cycle engine flood is not being filtered")
        results.check("ENGINE_WARNING_MARKER" in text,
                      "a sim.<Gov> WARNING record reaches the file",
                      "engine warnings are still being thrown away")
        results.check("ENGINE_DEBUG_MARKER" not in text,
                      "a DEBUG record is suppressed at the default level")

        # The session must leave the logging tree as it found it: its own file
        # handler gone (and closed), any pre-existing handler restored.  A
        # leaked FileHandler would keep writing into a finished sweep's log.
        leaked = [
            h for h in logging.getLogger(ROOT_LOGGER).handlers
            if isinstance(h, logging.FileHandler)
        ]
        results.check(
            not leaked,
            "the session removes (and closes) its file handler on exit",
            f"leaked file handler(s): {leaked}",
        )
        before = list(logging.getLogger(ROOT_LOGGER).handlers)
        with setup_parent_logging(
            os.path.join(tmp, "nested.log"), console=False
        ):
            pass
        results.check(
            list(logging.getLogger(ROOT_LOGGER).handlers) == before,
            "pre-existing handlers are restored exactly as they were",
            f"before={before} after={logging.getLogger(ROOT_LOGGER).handlers}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_log_level_raises_engine_verbosity():
    print("\n--- Test 13: --log-level raises engine verbosity ---")

    tmp = tempfile.mkdtemp(prefix="stage2_verbose_")
    try:
        log_path = os.path.join(tmp, "verbose.log")
        with setup_parent_logging(log_path, engine_level="INFO", console=False):
            logging.getLogger("sim.ADS").info("ENGINE_INFO_MARKER")
        with open(log_path, encoding="utf-8") as f:
            text = f.read()
        results.check(
            "ENGINE_INFO_MARKER" in text,
            "engine INFO records appear once the level is raised",
            "the --log-level override had no effect",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 14: backward compatibility
# ==========================================================================

def test_existing_outputs_unchanged():
    """
    The new files are purely additive: the numeric dataset must be identical
    byte for byte with and without them.  The paper's analysis path and its
    embedded figures read these files.
    """
    print("\n--- Test 14: existing outputs are byte-identical ---")

    tmp_with = tempfile.mkdtemp(prefix="stage2_compat_on_")
    tmp_without = tempfile.mkdtemp(prefix="stage2_compat_off_")
    try:
        run_with = _run_cell(_tiny_config(tmp_with))
        run_without = _run_cell(
            _tiny_config(tmp_without, write_run_detail=False)
        )

        for name in ("health_stats.csv", "final_stats.json"):
            a = os.path.join(run_with, name)
            b = os.path.join(run_without, name)
            if not (os.path.isfile(a) and os.path.isfile(b)):
                results.fail(f"{name} exists in both runs",
                             f"missing {a if not os.path.isfile(a) else b}")
                continue
            with open(a, "rb") as fa, open(b, "rb") as fb:
                same = fa.read() == fb.read()
            results.check(
                same,
                f"{name} is byte-identical with and without run detail",
                "run-detail logging changed an existing artifact",
            )

        results.check(
            not os.path.isfile(os.path.join(run_without, "run_detail.jsonl")),
            "--no-run-detail suppresses the detail file entirely",
        )
        results.check(
            not os.path.isfile(os.path.join(run_with, "agent_states.jsonl")),
            "agent_states.jsonl is NOT written without --detailed-agents",
        )
    finally:
        shutil.rmtree(tmp_with, ignore_errors=True)
        shutil.rmtree(tmp_without, ignore_errors=True)


def test_no_decision_detail_drops_only_groupings():
    print("\n--- Test 15: --no-decision-detail drops only the groupings ---")

    tmp = tempfile.mkdtemp(prefix="stage2_nodetail_")
    try:
        cfg = _tiny_config(tmp, max_cycles=20, decision_detail=False)
        run_dir = _run_cell(cfg)
        decisions = [r for r in _read_jsonl(os.path.join(run_dir, "run_detail.jsonl"))
                     if r["rec"] == "decision"]
        if not decisions:
            results.fail("decision records are still written", "none found")
            return
        first = decisions[0]
        results.check("groupings" not in first,
                      "the groupings array is dropped (~95% of the bytes)")
        for field in ("evidence", "summary", "winner", "enacted", "rejected_reasons"):
            results.check(field in first,
                          f"`{field}` is still written (what was decided, not the workings)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 16: the audit trail is off on the benchmark path only
# ==========================================================================

def test_audit_trail_disabled_for_benchmark():
    """
    ``AuditTrail`` builds a text log the benchmark never saves, at the cost of a
    full pass over every living agent every cycle.  The benchmark opts out;
    ``run_simulation.py`` must not.
    """
    print("\n--- Test 16: AuditTrail is skipped on the benchmark path ---")

    from engine.scenario_plan import build_simulation_from_plan
    from scenarios.scenario_base import _make_citizen_population
    from governments import GOVERNMENT_REGISTRY

    tmp = tempfile.mkdtemp(prefix="stage2_audit_")
    try:
        cfg = _tiny_config(tmp)
        plan = build_plan(cfg, 50, 0)
        agents = _make_citizen_population(cfg.n_agents, seed=42)
        gov = GOVERNMENT_REGISTRY["democracy"](seed=42)
        sim = build_simulation_from_plan(plan, gov, agents, record_audit_trail=False)
        results.check(sim.audit is None,
                      "the benchmark path builds no AuditTrail")
        sim._step(0)
        results.ok("a cycle runs cleanly with the audit trail disabled")

        agents2 = _make_citizen_population(cfg.n_agents, seed=42)
        gov2 = GOVERNMENT_REGISTRY["democracy"](seed=42)
        sim2 = build_simulation_from_plan(plan, gov2, agents2)
        results.check(sim2.audit is not None,
                      "the default (run_simulation.py) still builds one")
        sim2._step(0)
        results.check(bool(sim2.audit._entries),
                      "and still records into it")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# TEST 17: manifest.json
# ==========================================================================

def test_manifest_contents():
    print("\n--- Test 17: manifest.json provenance ---")

    tmp = tempfile.mkdtemp(prefix="stage2_manifest_")
    try:
        cfg = _tiny_config(tmp, governments=("democracy",), max_cycles=8)
        manifest = benchmark_core.build_manifest(
            cfg, "/tmp/x.log", "complete", "2026-09-01T00:00:00.000Z",
            ended_at="2026-09-01T00:01:00.000Z", wall_seconds=60.0,
            results={"total_runs_expected": 1, "total_runs_recorded": 1,
                     "short_cells": [], "failed_cells": [], "exit_code": 0},
        )
        json.dumps(manifest)          # must be serialisable
        results.ok("the manifest is JSON-serialisable")
        for key in ("schema_version", "status", "entry_point", "argv", "started_at",
                    "host", "user", "cwd", "log_file", "software", "config",
                    "results"):
            results.check(key in manifest, f"the manifest records `{key}`")
        results.check(
            manifest["config"]["difficulties"] == list(cfg.difficulties)
            and manifest["config"]["k_runs"] == cfg.k_runs,
            "the manifest records the resolved sweep configuration",
        )
        results.check(
            "short_cells" in manifest["results"],
            "the manifest carries per-cell sample sizes so n travels with the data",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# Fork/logging teardown deadlock guard
#
# A worker initializer that calls `handler.close()` on logging handlers
# inherited across `fork` can deadlock: `close()` flushes, the flush blocks on
# the stream's C-level buffer lock, and if the parent's QueueListener thread
# holds that lock at the instant of the fork, the child waits on a lock no
# living thread can release. This can hang a sweep indefinitely at
# `--log-level DEBUG` *after* all the science has completed, with
# `combined_final_stats.csv` and the plots never written.
#
# Starting children with `spawn` eliminates the hazard: there is no inherited
# parent thread state for a child to deadlock on. The stress test at the
# bottom of test_stage1_refactor.py samples the race empirically; the three
# tests here pin the invariants deterministically, which makes a regression
# fail in a second rather than in one run out of many.
# ==========================================================================

def test_children_are_never_forked():
    """The whole guarantee in one assertion: no child of this harness is forked."""
    print("\n--- every child process is started with spawn ---")

    results.check(
        benchmark_logging.WORKER_START_METHOD == "spawn",
        "WORKER_START_METHOD is 'spawn'",
        f"got {benchmark_logging.WORKER_START_METHOD!r}; 'fork' reintroduces the "
        f"handler-close deadlock hazard at DEBUG-level logging",
    )
    results.check(
        benchmark_logging.worker_mp_context().get_start_method() == "spawn",
        "worker_mp_context() hands out a spawn context",
    )

    benchmark_core.configure_multiprocessing()
    results.check(
        multiprocessing.get_start_method() == "spawn",
        "configure_multiprocessing() pins the process-global start method too",
    )


def test_sweep_pool_pins_the_spawn_context():
    """
    The *sweep's own* pool must pin `mp_context`, not inherit the global default.

    `configure_multiprocessing()` also pins the global, which would mask a
    missing `mp_context=` argument in every normal run — and then a caller who
    reaches `run_benchmark()` directly (the test suites do) would silently get a
    forking pool. So intercept the real construction and read the argument back,
    rather than reading the source or trusting the global.
    """
    print("\n--- the sweep's pool pins its own spawn context ---")

    seen: list = []
    real_pool = benchmark_core.ProcessPoolExecutor

    class _RecordingPool(real_pool):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("mp_context"))
            super().__init__(*args, **kwargs)

    # Pin the global to fork for the duration: if _run_sweep were relying on the
    # global rather than passing its own context, this test would catch it.
    saved_global = multiprocessing.get_start_method(allow_none=True)
    tmp = tempfile.mkdtemp(prefix="stage2_mpctx_")
    try:
        try:
            multiprocessing.set_start_method("fork", force=True)
        except (RuntimeError, ValueError):        # pragma: no cover - non-POSIX
            pass
        benchmark_core.ProcessPoolExecutor = _RecordingPool
        cfg = _tiny_config(os.path.join(tmp, "out"), write_run_detail=False)
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = benchmark_core.run_benchmark(cfg)

        results.check(exit_code == 0, "the intercepted sweep still succeeds",
                      f"exit {exit_code}")
        results.check(
            len(seen) >= 1,
            "the sweep constructed at least one worker pool",
            f"recorded {len(seen)} constructions",
        )
        results.check(
            bool(seen) and all(
                ctx is not None and ctx.get_start_method() == "spawn"
                for ctx in seen
            ),
            "every worker pool is constructed with an explicit spawn context",
            f"recorded contexts: "
            f"{[None if c is None else c.get_start_method() for c in seen]} — "
            f"None means the pool fell back to the process-global default, "
            f"which is fork on Linux",
        )
    finally:
        benchmark_core.ProcessPoolExecutor = real_pool
        if saved_global is not None:
            multiprocessing.set_start_method(saved_global, force=True)
        shutil.rmtree(tmp, ignore_errors=True)


class _ExplodingHandler(logging.Handler):
    """
    A handler that records — and forbids — the calls that caused the deadlock.

    Standing in for a ``FileHandler`` whose stream lock is held: in the real
    failure ``flush()`` never returns.  A test cannot wait forever, so instead
    of hanging it records the call, which turns an unbounded hang into an
    assertion.
    """

    def __init__(self):
        super().__init__()
        self.flushed = False
        self.closed_ = False

    def emit(self, record):            # pragma: no cover - never called here
        pass

    def flush(self):
        self.flushed = True            # in production this would block forever

    def close(self):
        self.closed_ = True
        super().close()


def test_worker_init_never_flushes_an_inherited_handler():
    """
    `init_worker_logging` must detach inherited handlers without touching them.

    Calling flush() or close() on an inherited handler is the exact operation
    that can deadlock (see the fork/logging teardown deadlock guard above).
    Under `spawn` there is normally nothing attached, so this test installs a
    handler by hand to prove the *code path* is safe rather than merely
    unreachable — the two are very different when someone later adds a
    handler at import time.
    """
    print("\n--- init_worker_logging detaches without flushing ---")

    root = logging.getLogger(ROOT_LOGGER)
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_propagate = root.propagate
    trap = _ExplodingHandler()
    queue = _q.Queue()                 # a plain queue is enough; nothing reads it

    try:
        root.handlers[:] = [trap]
        benchmark_logging.init_worker_logging(queue, "WARNING")

        results.check(
            not trap.flushed,
            "init_worker_logging does not flush an inherited handler",
            "flush() acquires the stream buffer lock — the deadlock itself",
        )
        results.check(
            not trap.closed_,
            "init_worker_logging does not close an inherited handler",
            "close() calls flush(), which risks the deadlock this test guards against",
        )
        results.check(
            trap not in root.handlers,
            "the inherited handler is still detached",
            "not closing it must not mean leaving it attached — two processes "
            "writing one file is the corruption this design exists to prevent",
        )
        results.check(
            len(root.handlers) == 1
            and isinstance(root.handlers[0], logging.handlers.QueueHandler),
            "the worker is left with exactly one QueueHandler",
            f"got {root.handlers!r}",
        )
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        root.propagate = saved_propagate


def _worker_probe(_):
    """Report state that can only be inherited, never re-imported."""
    return (os.getpid(), getattr(benchmark_logging, "_FORK_TELLTALE", None))


def test_workers_start_from_a_fresh_interpreter():
    """
    Prove `spawn` semantics end-to-end rather than trusting the context object.

    A module attribute set in the parent *after* import is visible in a forked
    child and absent in a spawned one.  That single bit distinguishes the two
    start methods from inside a real pool — which is what actually matters,
    since the deadlock hazard above is entirely about what a child inherits.
    """
    print("\n--- pool workers do not inherit parent memory ---")

    benchmark_logging._FORK_TELLTALE = "inherited-from-parent"
    try:
        with ProcessPoolExecutor(
            max_workers=2, mp_context=benchmark_logging.worker_mp_context()
        ) as pool:
            seen = list(pool.map(_worker_probe, range(2)))
    finally:
        del benchmark_logging._FORK_TELLTALE

    parent_pid = os.getpid()
    results.check(
        all(pid != parent_pid for pid, _ in seen),
        "tasks really did run in child processes",
        f"pids {[p for p, _ in seen]} vs parent {parent_pid}",
    )
    results.check(
        all(telltale is None for _, telltale in seen),
        "workers do not inherit the parent's post-import module state",
        f"a worker saw {seen!r} — that is fork, and fork can inherit a held "
        f"stream lock along with it",
    )


# ==========================================================================
# --help must render cleanly on every entry point
#
# argparse renders help through `help % dict(...)`, so *any* literal percent
# sign in a help string raises at format time and takes the entire help
# output with it — both entry points, every flag. A literal `%` in a help
# string must always be written `%%`.
# ==========================================================================

def _entry_point_parsers():
    """(name, parser) for every shipped command-line entry point."""
    import run_full_simulation
    import run_quick_test
    return [
        ("run_full_simulation.py", run_full_simulation.parse_args),
        ("run_quick_test.py", run_quick_test.parse_args),
    ]


def test_entry_point_help_renders():
    """
    `--help` must print and exit 0 on every entry point.

    Driven through `parse_args(["--help"])` rather than the parser object, so it
    exercises the same call the user makes — including argparse's help action
    and its SystemExit — instead of a lookalike.
    """
    print("\n--- --help renders on every entry point ---")

    for name, parse_args in _entry_point_parsers():
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                parse_args(["--help"])
        except SystemExit as exc:
            code = exc.code
        except Exception as exc:
            results.fail(
                f"{name} --help renders without raising",
                f"{type(exc).__name__}: {exc}",
            )
            continue
        else:
            results.fail(f"{name} --help exits", "no SystemExit was raised")
            continue

        text = buf.getvalue()
        results.check(code == 0, f"{name} --help exits 0", f"exit code {code!r}")
        results.check(
            len(text) > 500 and "--log-level" in text,
            f"{name} --help prints the full option list",
            f"{len(text)} chars",
        )
        # The specific string that broke it: `%%` in the source must survive
        # argparse's interpolation as a single literal `%`.
        results.check(
            "95% of the bytes" in text,
            f"{name} --help renders a literal percent sign correctly",
            "expected '95% of the bytes' in --no-decision-detail's help",
        )


def test_no_argparse_help_string_has_a_bare_percent():
    """
    Formatting every parser must not raise, for any argument.

    `format_help()` walks every action's help text, so this catches an unescaped
    `%` in a flag added later, rather than testing only one known instance.
    """
    print("\n--- no help string carries an unescaped % ---")

    for name, parse_args in _entry_point_parsers():
        parser = argparse.ArgumentParser(prog=name)
        # Rebuild the shared block on a bare parser: this is where every
        # common flag (and therefore every help string) is defined.
        benchmark_core.add_common_arguments(parser, "/tmp/does-not-matter")
        try:
            rendered = parser.format_help()
        except Exception as exc:
            results.fail(
                f"add_common_arguments' help formats for {name}",
                f"{type(exc).__name__}: {exc} — a literal '%' in a help string "
                f"must be written '%%'",
            )
            continue
        results.check(
            "--no-decision-detail" in rendered,
            f"add_common_arguments' help formats for {name}",
        )


# ==========================================================================
# Main
# ==========================================================================

def run_all():
    print("=" * 60)
    print("  SWEEP LOGGING & PER-RUN DETAIL — SYSTEM REGRESSION TESTS")
    print("  Sweep logging, per-run JSONL detail, serialization contract")
    print("=" * 60)

    test_serialization_contract()
    test_law_param_summarising()
    test_run_detail_schema()
    test_decision_record()
    test_non_ads_governments_recorded()
    test_default_decision_hook()
    test_failed_run_writes_footer()
    test_recorder_never_raises_on_bad_output_path()
    test_detailed_agents_ceiling()
    test_detailed_agents_output()
    test_log_path_resolution()
    test_logger_level_isolation()
    test_log_level_raises_engine_verbosity()
    test_existing_outputs_unchanged()
    test_no_decision_detail_drops_only_groupings()
    test_audit_trail_disabled_for_benchmark()
    test_manifest_contents()

    # The multiprocess fork/logging teardown deadlock guard
    test_children_are_never_forked()
    test_sweep_pool_pins_the_spawn_context()
    test_worker_init_never_flushes_an_inherited_handler()
    test_workers_start_from_a_fresh_interpreter()

    # --help must render cleanly on every entry point
    test_entry_point_help_renders()
    test_no_argparse_help_string_has_a_bare_percent()

    return results.summary()


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
