#!/usr/bin/env python3
"""
Validate Lasair object API + forced-photometry light-curve path (no ALeRCE).

Discovers ``diaObjectId`` values with a broad SQL query, then fetches each object until
one returns non-empty forced photometry (``diaForcedSourcesList``).

Usage (from repo root, conda env presn-alerts active):

    export LASAIR_API_TOKEN=...
    python scripts/validate_lasair_forced.py

Optional: pin a specific object (skips discovery):

    python scripts/validate_lasair_forced.py 170028526468071460
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from plotly_scanner import _forced_photometry_traces  # noqa: E402
from pre_sn_alerts import (  # noqa: E402
    DEFAULT_LASAIR_ENDPOINT,
    fetch_lasair_object_full,
    lasair_forced_photometry_dataframe,
    make_lasair_client,
)


def _query_variants(client: Any) -> list[int]:
    """
    Run progressively simpler Lasair queries until we get at least one diaObjectId.

    Uses only the public ``client.query`` API (same as the main pipeline).
    """
    specs: list[tuple[str, str, str, int]] = [
        (
            "objects.diaObjectId",
            "objects",
            "objects.nDiaSources > 0 AND mjdnow() - objects.lastDiaSourceMjdTai < 120",
            150,
        ),
        (
            "objects.diaObjectId",
            "objects",
            "objects.nDiaSources > 0",
            200,
        ),
        (
            "objects.diaObjectId",
            "objects",
            "objects.nDiaSources >= 1",
            200,
        ),
        (
            "diaObjectId",
            "objects",
            "nDiaSources > 0",
            200,
        ),
        (
            "objects.diaObjectId",
            "objects,sherlock_classifications",
            "objects.diaObjectId = sherlock_classifications.diaObjectId "
            'AND sherlock_classifications.classification = "SN"',
            200,
        ),
        (
            "objects.diaObjectId",
            "objects,sherlock_classifications",
            "objects.diaObjectId = sherlock_classifications.diaObjectId",
            100,
        ),
    ]
    oids: list[int] = []
    for selected, tables, conditions, limit in specs:
        try:
            rows = client.query(selected, tables, conditions, limit=limit)
        except Exception as exc:
            print(f"Query skipped ({conditions[:60]}...): {exc}", file=sys.stderr)
            continue
        if not rows:
            continue
        for row in rows:
            raw = row.get("diaObjectId")
            if raw is None:
                continue
            try:
                oids.append(int(raw))
            except (TypeError, ValueError):
                continue
        if oids:
            print(
                f"Discovery query matched {len(oids)} row(s): "
                f"tables={tables!r} limit={limit}",
                file=sys.stderr,
            )
            break
    seen: set[int] = set()
    out: list[int] = []
    for x in oids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main() -> int:
    if not os.environ.get("LASAIR_API_TOKEN", "").strip():
        print("Set LASAIR_API_TOKEN in the environment.", file=sys.stderr)
        return 2

    client = make_lasair_client(
        os.environ["LASAIR_API_TOKEN"].strip(),
        DEFAULT_LASAIR_ENDPOINT,
    )

    if len(sys.argv) > 1:
        candidates = [int(sys.argv[1])]
        print(f"Using single diaObjectId from CLI: {candidates[0]}", file=sys.stderr)
    else:
        candidates = _query_variants(client)
        if not candidates:
            print(
                "FAIL: no diaObjectId values returned from relaxed discovery queries.",
                file=sys.stderr,
            )
            return 1
        print(
            f"Trying up to {min(60, len(candidates))} object(s) for non-empty forced photometry…",
            file=sys.stderr,
        )

    max_try = min(60, len(candidates))
    for idx, oid in enumerate(candidates[:max_try], start=1):
        print(f"[{idx}/{max_try}] diaObjectId={oid} …", file=sys.stderr)
        try:
            obj = fetch_lasair_object_full(client, oid, early_exit="forced_ok")
        except Exception as exc:
            print(f"  object() failed: {exc}", file=sys.stderr)
            continue
        top_keys = list(obj.keys())[:20] if isinstance(obj, dict) else []
        forced = lasair_forced_photometry_dataframe(obj)
        n_forced = len(forced)
        print(f"  top-level keys (sample): {top_keys}", file=sys.stderr)
        print(f"  diaForcedSourcesList rows (after merge): {n_forced}")
        if forced.empty:
            continue
        traces, warn = _forced_photometry_traces(forced, "lsst")
        print(f"  Plotly traces: {len(traces)}  warning: {warn!r}")
        if not traces:
            print(f"  skip: could not build traces ({warn})", file=sys.stderr)
            continue
        print(f"OK — validated diaObjectId={oid} with {n_forced} forced rows and {len(traces)} LC trace(s).")
        return 0

    print(
        "FAIL: no object in the candidate list had non-empty forced photometry "
        f"(tried {max_try} id(s)).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
