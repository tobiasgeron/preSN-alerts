#!/usr/bin/env python3
"""
Lasair LSST API + ALeRCE cutout experiment (refactored from notebook/Lasair_and_alerce_experiment.ipynb).

This module queries the Lasair API for Sherlock-classified transients, filters recent
``diaSource`` epochs, fetches science/template/difference stamps from ALeRCE, and
displays a four-panel summary figure. Configuration is centralized in
:class:`ExperimentConfig` and can be overridden from the command line.

Notes
-----
Requires a Lasair API token (environment variable ``LASAIR_API_TOKEN`` or ``--lasair-token``).
Optional ``LASAIR_MAX_CALLS_PER_HOUR`` (or ``--lasair-max-calls-per-hour``) throttles API usage
to stay within Lasair's per-account hourly limits. Pair discovery defaults to one ``object()`` call per candidate; opt in to a single SQL query
with ``LASAIR_SQL_PAIRS=1`` or ``--lasair-sql-pairs`` when your Lasair deployment supports it;
set ``LASAIR_CLIENT_CACHE_DIR`` to enable the Lasair client's on-disk response cache.

Diagnostic output uses the standard library :mod:`logging` module. Configure a log file
with ``--log-file``; messages are written to that file and echoed to stderr at the
console log level (see ``--console-log-level`` and ``--file-log-level``).
"""

from __future__ import annotations

import argparse
import gzip
import inspect
import io
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import types
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urljoin

import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

from alerce.core import Alerce
from alerce.ms_stamp_utils import create_stamp_parameters
from lasair import LasairError, lasair_client as lasair

from utils.utilities import (
    configure_application_logging,
    get_app_logger,
    parse_loglevel_name,
    tqdm_log_stream,
)

# -----------------------------------------------------------------------------
# Defaults (CLI and :class:`ExperimentConfig` use these as factory defaults)
# -----------------------------------------------------------------------------

DEFAULT_LASAIR_ENDPOINT = "https://api.lasair.lsst.ac.uk/api"
DEFAULT_LASAIR_TOKEN_ENV = "LASAIR_API_TOKEN"
# Lasair REST API: standard account tokens are limited to ~100 calls per rolling hour;
# power users ~10,000/h (https://lasair.readthedocs.io/en/main/core_functions/rest-api.html).
DEFAULT_LASAIR_MAX_CALLS_PER_HOUR_ENV = "LASAIR_MAX_CALLS_PER_HOUR"
DEFAULT_LASAIR_MAX_CALLS_PER_HOUR = 90
DEFAULT_LASAIR_CLIENT_CACHE_ENV = "LASAIR_CLIENT_CACHE_DIR"
DEFAULT_LASAIR_DIASOURCES_TABLE_ENV = "LASAIR_DIASOURCES_TABLE"
DEFAULT_LASAIR_DIASOURCES_TABLE = "diasources"
DEFAULT_LASAIR_SQL_PAIRS_ENV = "LASAIR_SQL_PAIRS"

DEFAULT_OBJECTS_SHERLOCK_TABLES = "objects,sherlock_classifications"
DEFAULT_SHERLOCK_CLASSIFICATION = "SN"
DEFAULT_ALERCE_SURVEY = "lsst"

DEFAULT_LOG_FILE = Path("logs") / "pre_sn_alerts.log"

# Sherlock distance-related columns (Lasair LSST schema browser:
# https://lasair.lsst.ac.uk/schema/#sherlock_classifications-schema).
# - distance: luminosity distance from spectral redshift (Mpc).
# - direct_distance: distance from non-redshift methods, e.g. standard candles (Mpc).
# - physical_separation_kpc: projected separation transient–host (kpc).
ALLOWED_SHERLOCK_DISTANCE_COLUMNS: dict[str, str] = {
    "luminosity": "sherlock_classifications.distance",
    "direct": "sherlock_classifications.direct_distance",
    "separation_kpc": "sherlock_classifications.physical_separation_kpc",
}

# Candidate-query rows may include these keys for merging into the pair table / logs.
CANDIDATE_MERGE_KEYS: tuple[str, ...] = (
    "ra",
    "decl",
    "distance",
    "direct_distance",
    "physical_separation_kpc",
    "z",
    "separationArcsec",
    "association_type",
    "classification",
    "classificationReliability",
)

# LSST/Rubin-style AB magnitude from ``objects.{band}_psfFlux`` in nanojanskys (Lasair schema).
# m_AB ≈ ZP - 2.5*log10(flux_nJy); see Rubin / alert packet conventions (ZP ≈ 31.4).
LSST_AB_MAG_ZP_NJY: float = 31.4

# Per-band latest PSF flux (nJy) and MJD of that measurement (Lasair objects table).
ALLOWED_BRIGHTNESS_BAND_FLUX: dict[str, str] = {
    "u": "objects.u_psfFlux",
    "g": "objects.g_psfFlux",
    "r": "objects.r_psfFlux",
    "i": "objects.i_psfFlux",
    "z": "objects.z_psfFlux",
    "y": "objects.y_psfFlux",
}
ALLOWED_BRIGHTNESS_BAND_LATEST_MJD: dict[str, str] = {
    "u": "objects.u_latestMJD",
    "g": "objects.g_latestMJD",
    "r": "objects.r_latestMJD",
    "i": "objects.i_latestMJD",
    "z": "objects.z_latestMJD",
    "y": "objects.y_latestMJD",
}


def default_sn_candidate_select_list() -> str:
    """
    Return the default comma-separated Lasair SELECT list for SN candidate queries.

    Returns
    -------
    str
        Fields joined for use in ``LasairClient.query``.
    """
    fields = [
        "objects.diaObjectId",
        "objects.ra",
        "objects.decl",
        "mjdnow()-objects.lastDiaSourceMjdTai AS since",
        "objects.lastDiaSourceMjdTai",
        "objects.latestR",
        "objects.nDiaSources",
        "sherlock_classifications.classification",
        "sherlock_classifications.association_type",
        "sherlock_classifications.distance",
        "sherlock_classifications.z",
        "sherlock_classifications.classificationReliability",
        "sherlock_classifications.major_axis_arcsec",
        "sherlock_classifications.separationArcsec",
        "sherlock_classifications.direct_distance",
        "sherlock_classifications.physical_separation_kpc",
    ]
    return ",\n       ".join(fields)


@dataclass
class PlotStyle:
    """
    Matplotlib styling for :func:`plot_main_figure`.

    Attributes
    ----------
    ncol, nrow : int
        Subplot grid shape.
    crosshair_color : str
        Matplotlib color for the crosshair.
    crosshair_linewidth : float
        Line width for crosshair segments.
    crosshair_arm_fraction : float
        Half-length of each crosshair arm as a fraction of stamp width (pixels).
    crosshair_gap_fraction : float
        Gap from center to the start of each arm, as a fraction of stamp width.
    titles_fontsize : float
        Font size for subplot titles.
    subplot_figsize : tuple of float
        Base ``(width, height)`` in inches for one subplot cell; the full figure
        scales by ``ncol`` and ``nrow``.
    science_norm, template_norm : str
        ``imshow`` norm for science and template panels (e.g. ``\"asinh\"``).
    diff_norm, snr_norm : str
        ``imshow`` norm for difference and SNR panels (typically ``\"linear\"``).
    science_cmap, template_cmap, diff_cmap, snr_cmap : str
        Colormap names.
    vmin_percentile, vmax_percentile : float
        Percentiles for scaling science/template from the template image.
    diff_vmin_percentile, diff_vmax_percentile : float
        Percentiles for difference image scaling.
    snr_vmin, snr_vmax : float
        Fixed color scale for the SNR panel.
    panel_titles : tuple of str
        Titles for science, template, difference, and SNR panels (length 4).
    """

    ncol: int = 4
    nrow: int = 1
    crosshair_color: str = "red"
    crosshair_linewidth: float = 2.0
    crosshair_arm_fraction: float = 0.1
    crosshair_gap_fraction: float = 0.05
    titles_fontsize: float = 22.0
    subplot_figsize: tuple[float, float] = (4.0, 4.3)
    science_norm: str = "asinh"
    template_norm: str = "asinh"
    diff_norm: str = "linear"
    snr_norm: str = "linear"
    science_cmap: str = "grey"
    template_cmap: str = "grey"
    diff_cmap: str = "grey"
    snr_cmap: str = "viridis"
    vmin_percentile: float = 30.0
    vmax_percentile: float = 99.0
    diff_vmin_percentile: float = 1.0
    diff_vmax_percentile: float = 99.0
    snr_vmin: float = 0.0
    snr_vmax: float = 5.0
    panel_titles: tuple[str, str, str, str] = (
        "Science Image",
        "Template Image",
        "Difference Image",
        "SNR Map",
    )


@dataclass
class ExperimentConfig:
    """
    Runtime parameters for Lasair queries, ALeRCE stamp requests, and CLI workflows.

    Attributes
    ----------
    lasair_endpoint : str
        Lasair API base URL.
    delta_t : float
        Maximum allowed ``mjd_now - midpointMjdTai`` for a diaSource to be kept (days).
    query_limit : int
        SQL ``LIMIT`` on the SN candidate query.
    max_plots : int
        Maximum stamp figures to show in the full pipeline.
    simple_query_limit : int
        SQL ``LIMIT`` for :func:`run_simple_query`.
    preview_table_rows : int
        Number of rows to print when previewing the pair table.
    lasair_objects_sherlock_tables : str
        Comma-separated table list for object + Sherlock joins.
    sherlock_classification : str
        Sherlock classification string to filter (e.g. ``\"SN\"``).
    sn_candidate_select : str
        Full SELECT clause string for detailed SN queries.
    simple_query_select : str
        SELECT list for the minimal exploratory query.
    alerce_survey : str
        Survey name passed to ALeRCE (e.g. ``\"lsst\"``).
    alerce_include_variance_and_mask : bool
        Passed to ``Alerce.get_stamps``.
    lasair_object_lasair_added : bool
        Passed to ``LasairClient.object`` (pipeline uses ``False`` by default).
    shuffle_pairs : bool
        If True, shuffle object/source pairs before plotting.
    random_state : int or None
        Seed for shuffling when ``shuffle_pairs`` is True.
    plot_style : PlotStyle
        Figure styling for :func:`plot_main_figure`.
    sherlock_distance_column : str
        Fully qualified Sherlock column used for SQL distance bounds (see
        ``ALLOWED_SHERLOCK_DISTANCE_COLUMNS`` / Lasair schema).
    distance_min : float or None
        Optional lower bound (same units as the chosen column: Mpc or kpc).
    distance_max : float or None
        Optional upper bound.
    apply_distance_filter_to_simple_query : bool
        If True, apply the same distance predicates to :func:`run_simple_query`.
    z_min, z_max : float or None
        Optional redshift bounds on ``sherlock_classifications.z`` (``z_max`` is strict).
    apply_delta_t_to_simple_query : bool
        If True, :func:`run_simple_query` also requires
        ``mjdnow() - objects.lastDiaSourceMjdTai <= delta_t`` (same ``delta_t`` as
        the detailed query).
    bright_max_mag : float or None
        If set, require AB magnitude **<** this in ``brightness_band`` using
        ``objects.{band}_psfFlux`` (nJy) and :data:`LSST_AB_MAG_ZP_NJY`.
    brightness_band : str
        One of ``u``, ``g``, ``r``, ``i``, ``z``, ``y`` (Lasair ``objects`` table).
    brightness_recency_days : float or None
        ``mjdnow() - {band}_latestMJD`` must be ≤ this; if None, use ``delta_t``.
    lasair_max_calls_per_hour : int or None
        Client-side cap on Lasair HTTP calls per rolling hour (``None`` = env
        ``LASAIR_MAX_CALLS_PER_HOUR`` or 90; ``0`` disables throttling).
    lasair_client_cache_dir : str or None
        Directory for the Lasair Python client's on-disk cache (``None`` = env
        ``LASAIR_CLIENT_CACHE_DIR`` if set).
    lasair_use_sql_for_pairs : bool
        If True, build (diaObjectId, diaSourceId) pairs with one SQL query when possible
        instead of one ``object()`` call per object.
    lasair_diasources_table : str
        Lasair SQL table name joined to ``objects`` for per-detection rows (env
        ``LASAIR_DIASOURCES_TABLE`` overrides when non-empty).
    lasair_pair_query_row_limit : int or None
        ``LIMIT`` on the SQL pair query; ``None`` picks a value from candidate count
        (capped at 10_000 for standard Lasair tokens).
    lasair_one_row_per_source : bool
        If True (default), emit one scan/plot row per ``diaObjectId``: the latest
        in-window ``diaSource`` (``midpointMjdTai``). Candidate SQL already restricts
        sources by redshift, recent alert (``lastDiaSourceMjdTai``), and magnitude cuts.
        If False, keep every in-window detection as its own row (legacy behaviour).
    """

    lasair_endpoint: str = DEFAULT_LASAIR_ENDPOINT
    delta_t: float = 1.0
    query_limit: int = 10
    max_plots: int = 10
    simple_query_limit: int = 8
    preview_table_rows: int = 10
    lasair_objects_sherlock_tables: str = DEFAULT_OBJECTS_SHERLOCK_TABLES
    sherlock_classification: str = DEFAULT_SHERLOCK_CLASSIFICATION
    sn_candidate_select: str = field(default_factory=default_sn_candidate_select_list)
    simple_query_select: str = (
        "objects.diaObjectId, objects.ra, objects.decl, "
        "sherlock_classifications.distance, sherlock_classifications.direct_distance, "
        "sherlock_classifications.physical_separation_kpc, sherlock_classifications.z"
    )
    alerce_survey: str = DEFAULT_ALERCE_SURVEY
    alerce_include_variance_and_mask: bool = True
    lasair_object_lasair_added: bool = False
    shuffle_pairs: bool = True
    random_state: int | None = None
    plot_style: PlotStyle = field(default_factory=PlotStyle)
    sherlock_distance_column: str = ALLOWED_SHERLOCK_DISTANCE_COLUMNS["luminosity"]
    distance_min: float | None = None
    distance_max: float | None = None
    apply_distance_filter_to_simple_query: bool = False
    z_min: float | None = None
    z_max: float | None = None
    apply_delta_t_to_simple_query: bool = False
    bright_max_mag: float | None = None
    brightness_band: str = "r"
    brightness_recency_days: float | None = None
    lasair_max_calls_per_hour: int | None = None
    lasair_client_cache_dir: str | None = None
    lasair_use_sql_for_pairs: bool = False
    lasair_diasources_table: str = DEFAULT_LASAIR_DIASOURCES_TABLE
    lasair_pair_query_row_limit: int | None = None
    lasair_one_row_per_source: bool = True


LasairObjectEarlyExit = Literal["plotly", "forced_ok"] | None


# -----------------------------------------------------------------------------
# Lasair client and authentication
# -----------------------------------------------------------------------------


class LasairApiThrottle:
    """
    Sliding-window limiter aligned with Lasair's per-user hourly API caps.

    Each successful reservation records one "call" toward the limit. Use the same
    instance for ``client.fetch``, direct ``/api/object/`` POSTs, and authenticated
    cutout GETs so totals stay within budget.
    """

    def __init__(self, max_calls_per_hour: int, window_seconds: float = 3600.0) -> None:
        self.max_calls = max(1, int(max_calls_per_hour))
        self.window = float(window_seconds)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        log = get_app_logger()
        with self._lock:
            while True:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self.window:
                    self._times.popleft()
                if len(self._times) < self.max_calls:
                    self._times.append(time.monotonic())
                    return
                wait = self.window - (now - self._times[0]) + 0.05
                if wait > 1.0:
                    log.info(
                        "Lasair API throttle: at %d call(s) per %.0fs; waiting %.1fs",
                        self.max_calls,
                        self.window,
                        wait,
                    )
                self._lock.release()
                try:
                    time.sleep(wait)
                finally:
                    self._lock.acquire()


def resolve_lasair_max_calls_per_hour(override: int | None) -> int | None:
    """
    Effective max Lasair API calls per rolling hour, or None to disable throttling.

    Precedence: ``override`` (from CLI / config), then ``LASAIR_MAX_CALLS_PER_HOUR``,
    then :data:`DEFAULT_LASAIR_MAX_CALLS_PER_HOUR`. Non-positive values disable.
    """
    log = get_app_logger()
    if override is not None:
        if override <= 0:
            return None
        return max(1, int(override))
    raw = os.environ.get(DEFAULT_LASAIR_MAX_CALLS_PER_HOUR_ENV, "").strip()
    if raw:
        try:
            v = int(raw, 10)
            if v <= 0:
                return None
            return max(1, v)
        except ValueError:
            log.warning(
                "Ignoring invalid %s=%r; using default %s",
                DEFAULT_LASAIR_MAX_CALLS_PER_HOUR_ENV,
                raw,
                DEFAULT_LASAIR_MAX_CALLS_PER_HOUR,
            )
    return DEFAULT_LASAIR_MAX_CALLS_PER_HOUR


def resolve_lasair_client_cache_dir(override: str | None) -> str | None:
    """Return a cache directory path, or None to disable the Lasair client's disk cache."""
    if override is not None and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return str(p.resolve())
    raw = os.environ.get(DEFAULT_LASAIR_CLIENT_CACHE_ENV, "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


def resolve_lasair_use_sql_for_pairs(override: bool | None) -> bool:
    """Whether to prefer a single SQL query for diaObject/diaSource pairing."""
    if override is not None:
        return bool(override)
    raw = os.environ.get(DEFAULT_LASAIR_SQL_PAIRS_ENV, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return False


def resolve_lasair_diasources_table(config_table: str) -> str:
    """Effective per-detection table name (identifier-safe)."""
    env_t = os.environ.get(DEFAULT_LASAIR_DIASOURCES_TABLE_ENV, "").strip()
    t = env_t or (config_table or DEFAULT_LASAIR_DIASOURCES_TABLE)
    return _validate_lasair_sql_identifier(t, "diasources table")


def _validate_lasair_sql_identifier(name: str, kind: str = "identifier") -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,62}", name):
        raise ValueError(f"Invalid {kind} {name!r}: use letters, digits, underscore only.")
    return name


def _install_lasair_throttle(client: Any, throttle: LasairApiThrottle) -> None:
    """Monkey-patch ``client.fetch`` and stash ``_presn_lasair_throttle`` for POST/GET helpers."""
    orig_fetch = client.fetch

    def fetch_wrapped(_self: Any, method: str, input: Any) -> Any:
        log = get_app_logger()
        throttle.acquire()
        try:
            return orig_fetch(method, input)
        except LasairError as exc:
            msg = str(getattr(exc, "message", exc))
            if "limit exceeded" not in msg.lower():
                raise
            log.warning(
                "Lasair rate limit (429) despite client throttle; sleeping 90s then retrying once"
            )
            time.sleep(90.0)
            throttle.acquire()
            return orig_fetch(method, input)

    client._presn_lasair_throttle = throttle
    client.fetch = types.MethodType(fetch_wrapped, client)


def get_lasair_token(cli_token: str | None, env_var: str = DEFAULT_LASAIR_TOKEN_ENV) -> str:
    """
    Resolve the Lasair API token from the CLI or environment.

    Parameters
    ----------
    cli_token : str or None
        Token from ``--lasair-token``; if set, takes precedence.
    env_var : str, optional
        Environment variable name to read when ``cli_token`` is absent.

    Returns
    -------
    str
        Non-empty trimmed token.

    Raises
    ------
    SystemExit
        If no token is available (message logged and process exits with code 1).
    """
    log = get_app_logger()
    token = cli_token or os.environ.get(env_var)
    if not token or not str(token).strip():
        log.error(
            "Missing Lasair API token. Set %s or pass --lasair-token.",
            env_var,
        )
        sys.exit(1)
    log.debug("Lasair API token loaded from %s", "CLI" if cli_token else env_var)
    return str(token).strip()


def make_lasair_client(
    token: str,
    endpoint: str,
    max_calls_per_hour: int | None = None,
    *,
    cache_dir: str | None = None,
) -> Any:
    """
    Construct a Lasair API client with optional sliding-window request throttling.

    Parameters
    ----------
    token : str
        Lasair API token.
    endpoint : str
        API base URL.
    max_calls_per_hour : int or None, optional
        Rolling-hour cap on Lasair HTTP calls made through this process. ``None`` uses
        :func:`resolve_lasair_max_calls_per_hour`; ``0`` disables throttling.
    cache_dir : str or None, optional
        Writable directory for the Lasair client's JSON cache. ``None`` uses
        :func:`resolve_lasair_client_cache_dir`.

    Returns
    -------
    object
        Client instance returned by ``lasair_client.lasair`` (typed as ``Any``).
    """
    log = get_app_logger()
    log.info("Lasair client created for endpoint %r", endpoint)
    cache_path = resolve_lasair_client_cache_dir(cache_dir)
    if cache_path:
        log.info("Lasair client disk cache enabled at %r", cache_path)
        client = lasair(token, endpoint=endpoint, cache=cache_path)
    else:
        client = lasair(token, endpoint=endpoint)
    limit = resolve_lasair_max_calls_per_hour(max_calls_per_hour)
    if limit is not None:
        throttle = LasairApiThrottle(limit)
        _install_lasair_throttle(client, throttle)
        log.info(
            "Lasair API throttle active: max %d HTTP call(s) per rolling %.0fs",
            limit,
            throttle.window,
        )
    else:
        log.info("Lasair API throttle disabled")
    return client


# -----------------------------------------------------------------------------
# SQL fragments
# -----------------------------------------------------------------------------


def _validate_sherlock_distance_column(column: str) -> str:
    """
    Ensure ``column`` is an allowed Sherlock distance field (avoid SQL injection).

    Parameters
    ----------
    column : str
        Fully qualified name, e.g. ``sherlock_classifications.distance``.

    Returns
    -------
    str
        The same string if allowed.

    Raises
    ------
    ValueError
        If not in :data:`ALLOWED_SHERLOCK_DISTANCE_COLUMNS` values.
    """
    allowed = frozenset(ALLOWED_SHERLOCK_DISTANCE_COLUMNS.values())
    if column not in allowed:
        raise ValueError(
            f"distance column must be one of {sorted(allowed)!r}; got {column!r}."
        )
    return column


def build_distance_range_sql_fragments(
    column: str,
    distance_min: float | None,
    distance_max: float | None,
) -> str:
    """
    Build ``AND ...`` SQL fragments for a numeric range on a Sherlock distance column.

    When either bound is set, ``column IS NOT NULL`` is required so NULL hosts
    do not pass comparison filters.

    Parameters
    ----------
    column : str
        Whitelisted ``table.column`` (see :func:`_validate_sherlock_distance_column`).
    distance_min, distance_max : float or None
        Inclusive bounds in the column's native units (Mpc for ``distance`` /
        ``direct_distance``, kpc for ``physical_separation_kpc``).

    Returns
    -------
    str
        Empty string if both bounds are None; otherwise `` AND ...`` clauses.
    """
    if distance_min is None and distance_max is None:
        return ""
    _validate_sherlock_distance_column(column)
    parts = [f"{column} IS NOT NULL"]
    if distance_min is not None:
        parts.append(f"{column} >= {float(distance_min)}")
    if distance_max is not None:
        parts.append(f"{column} <= {float(distance_max)}")
    return " AND " + " AND ".join(parts)


def build_redshift_sql_fragments(
    sherlock_table: str,
    z_min: float | None,
    z_max: float | None,
) -> str:
    """
    Build ``AND ...`` SQL fragments for Sherlock host redshift ``z``.

    Uses ``sherlock_classifications.z`` (spectroscopic redshift of the top-ranked
    host match; see Lasair schema). Rows with NULL ``z`` are excluded when any
    bound is set.

    Parameters
    ----------
    sherlock_table : str
        Table alias (typically ``sherlock_classifications``).
    z_min, z_max : float or None
        Lower bound is inclusive (``>=``); upper bound is strict (``<``) so
        ``z_max=0.03`` implements *z < 0.03*.

    Returns
    -------
    str
        Empty if both bounds are None; otherwise `` AND ...`` clauses.
    """
    if z_min is None and z_max is None:
        return ""
    col = f"{sherlock_table}.z"
    parts = [f"{col} IS NOT NULL"]
    if z_min is not None:
        parts.append(f"{col} >= {float(z_min)}")
    if z_max is not None:
        parts.append(f"{col} < {float(z_max)}")
    return " AND " + " AND ".join(parts)


def nanojansky_flux_floor_for_mag_brighter_than(
    mag_limit: float,
    zp_njy: float = LSST_AB_MAG_ZP_NJY,
) -> float:
    """
    Minimum PSF flux (nJy) so that AB magnitude is **brighter** than ``mag_limit``.

    Uses m ≈ ``zp_njy - 2.5*log10(flux_nJy)``; brighter means **lower** m, so require
    m < mag_limit ⇔ flux > 10**((zp_njy - mag_limit) / 2.5).

    Parameters
    ----------
    mag_limit : float
        Upper magnitude cutoff (e.g. 22 for “brighter than 22nd mag”).
    zp_njy : float, optional
        AB zeropoint for flux expressed in nJy (Rubin-style ~31.4).

    Returns
    -------
    float
        Strict minimum flux in nJy (SQL should use ``flux > floor``).
    """
    return float(10.0 ** ((zp_njy - float(mag_limit)) / 2.5))


def build_brightness_sql_fragments(
    objects_table: str,
    band: str,
    bright_max_mag: float | None,
    brightness_recency_days: float,
    zp_njy: float = LSST_AB_MAG_ZP_NJY,
) -> str:
    """
    SQL ``AND`` clauses: latest band flux implies AB mag brighter than limit, and
    that flux is no older than ``brightness_recency_days`` (``mjdnow - latestMJD``).

    Parameters
    ----------
    objects_table : str
        Objects table name (usually ``objects``).
    band : str
        One of ``u``, ``g``, ``r``, ``i``, ``z``, ``y``.
    bright_max_mag : float or None
        If set, require AB magnitude **<** this value (brighter than this limit).
    brightness_recency_days : float
        ``mjdnow() - {band}_latestMJD`` must be ≤ this value.
    zp_njy : float, optional
        AB zeropoint for nJy fluxes.

    Returns
    -------
    str
        Empty if ``bright_max_mag`` is None; otherwise `` AND ...`` fragments.
    """
    if bright_max_mag is None:
        return ""
    if band not in ALLOWED_BRIGHTNESS_BAND_FLUX:
        raise ValueError(
            f"brightness band must be one of {sorted(ALLOWED_BRIGHTNESS_BAND_FLUX)!r}; "
            f"got {band!r}."
        )
    flux_ref = ALLOWED_BRIGHTNESS_BAND_FLUX[band]
    mjd_ref = ALLOWED_BRIGHTNESS_BAND_LATEST_MJD[band]
    flux_col = flux_ref.replace("objects.", f"{objects_table}.", 1)
    mjd_col = mjd_ref.replace("objects.", f"{objects_table}.", 1)
    f_floor = nanojansky_flux_floor_for_mag_brighter_than(bright_max_mag, zp_njy)
    return (
        f" AND {flux_col} IS NOT NULL AND {flux_col} > 0"
        f" AND {flux_col} > {f_floor}"
        f" AND mjdnow() - {mjd_col} <= {float(brightness_recency_days)}"
    )


def augment_select_with_brightness_columns(select: str, band: str) -> str:
    """
    Append ``objects.{band}_psfFlux`` and ``objects.{band}_latestMJD`` if missing.

    Parameters
    ----------
    select : str
        Existing comma-separated SELECT list.
    band : str
        Band letter in ``ALLOWED_BRIGHTNESS_BAND_FLUX``.

    Returns
    -------
    str
        Extended SELECT list.
    """
    if band not in ALLOWED_BRIGHTNESS_BAND_FLUX:
        raise ValueError(f"unknown brightness band {band!r}")
    flux = ALLOWED_BRIGHTNESS_BAND_FLUX[band]
    mjd = ALLOWED_BRIGHTNESS_BAND_LATEST_MJD[band]
    if flux in select and mjd in select:
        return select
    return f"{select},\n       {flux},\n       {mjd}"


def brightness_merge_keys(config: ExperimentConfig) -> tuple[str, ...]:
    """Return extra Lasair row keys to merge when a brightness filter is active."""
    if config.bright_max_mag is None:
        return ()
    b = config.brightness_band
    return (f"{b}_psfFlux", f"{b}_latestMJD")


def brightness_recency_days_resolved(config: ExperimentConfig) -> float:
    """Days for ``mjdnow - latestMJD`` brightness recency (defaults to ``delta_t``)."""
    if config.brightness_recency_days is not None:
        return float(config.brightness_recency_days)
    return float(config.delta_t)


def _assert_safe_sql_token(value: str, name: str = "value") -> str:
    """
    Reject characters that could break out of a quoted SQL string literal.

    Parameters
    ----------
    value : str
        User- or config-supplied token.
    name : str, optional
        Name for error messages.

    Returns
    -------
    str
        The same string if valid.

    Raises
    ------
    ValueError
        If ``value`` contains a double quote.
    """
    if '"' in value:
        raise ValueError(f'{name} must not contain double quotes; got {value!r}.')
    return value


def build_conditions_sn_recent(
    delta_t: float,
    classification: str,
    objects_table: str = "objects",
    sherlock_table: str = "sherlock_classifications",
    *,
    distance_column: str | None = None,
    distance_min: float | None = None,
    distance_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    bright_max_mag: float | None = None,
    brightness_band: str = "r",
    brightness_recency_days: float | None = None,
) -> str:
    """
    Build a WHERE clause for recent Sherlock-classified transients.

    Optional bounds use Sherlock distance columns documented at
    https://lasair.lsst.ac.uk/schema/ (e.g. ``distance`` = luminosity distance in Mpc).

    Parameters
    ----------
    delta_t : float
        Upper bound on ``mjdnow() - lastDiaSourceMjdTai`` (Lasair SQL units).
    classification : str
        Sherlock classification to match (quoted in SQL).
    objects_table, sherlock_table : str, optional
        Table names used in the join.
    distance_column : str or None, optional
        Column for ``distance_min`` / ``distance_max``; defaults to luminosity distance.
    distance_min, distance_max : float or None, optional
        Inclusive filter on ``distance_column`` (native units for that column).
    z_min, z_max : float or None, optional
        Redshift bounds on ``sherlock_classifications.z`` (``z_max`` is strict ``<``).
    bright_max_mag : float or None, optional
        If set, require AB magnitude **<** this in ``brightness_band`` (see Lasair
        ``objects.{band}_psfFlux`` in nJy).
    brightness_band : str, optional
        Filter band for brightness (default ``r``).
    brightness_recency_days : float or None, optional
        ``mjdnow() - {band}_latestMJD`` ≤ this; defaults to ``delta_t`` when None.

    Returns
    -------
    str
        SQL condition string.
    """
    classification = _assert_safe_sql_token(classification, "classification")
    col = distance_column or ALLOWED_SHERLOCK_DISTANCE_COLUMNS["luminosity"]
    extra = build_distance_range_sql_fragments(col, distance_min, distance_max)
    rz = build_redshift_sql_fragments(sherlock_table, z_min, z_max)
    brd = (
        float(brightness_recency_days)
        if brightness_recency_days is not None
        else float(delta_t)
    )
    bx = build_brightness_sql_fragments(
        objects_table,
        brightness_band,
        bright_max_mag,
        brd,
    )
    return f"""
    {objects_table}.diaObjectId={sherlock_table}.diaObjectId
  AND {sherlock_table}.classification IN ("{classification}")
  AND mjdnow() - {objects_table}.lastDiaSourceMjdTai <= {delta_t}{extra}{rz}{bx}
  """


def build_conditions_simple_sn(classification: str) -> str:
    """
    Build a minimal WHERE clause for Sherlock classification only.

    Parameters
    ----------
    classification : str
        Sherlock class to match.

    Returns
    -------
    str
        SQL fragment for ``classification="..."``.
    """
    classification = _assert_safe_sql_token(classification, "classification")
    return f'classification="{classification}"'


def build_simple_query_conditions(config: ExperimentConfig) -> str:
    """
    Build the full WHERE clause for :func:`run_simple_query`.

    Parameters
    ----------
    config : ExperimentConfig
        Classification, optional recency (``apply_delta_t_to_simple_query``),
        optional distance bounds, and optional redshift bounds.

    Returns
    -------
    str
        SQL condition string for ``client.query``.
    """
    cond = build_conditions_simple_sn(config.sherlock_classification)
    if config.apply_delta_t_to_simple_query:
        cond = (
            f"({cond}) AND mjdnow() - objects.lastDiaSourceMjdTai "
            f"<= {float(config.delta_t)}"
        )
    if config.apply_distance_filter_to_simple_query:
        extra = build_distance_range_sql_fragments(
            config.sherlock_distance_column,
            config.distance_min,
            config.distance_max,
        )
        if extra:
            cond = f"({cond}){extra}"
    rz = build_redshift_sql_fragments(
        "sherlock_classifications",
        config.z_min,
        config.z_max,
    )
    if rz:
        cond = f"({cond}){rz}"
    bx = build_brightness_sql_fragments(
        "objects",
        config.brightness_band,
        config.bright_max_mag,
        brightness_recency_days_resolved(config),
    )
    if bx:
        cond = f"({cond}){bx}"
    return cond


# -----------------------------------------------------------------------------
# Object / diaSource helpers
# -----------------------------------------------------------------------------


def merge_candidate_metadata_into_pairs(
    data: pd.DataFrame,
    candidate_rows: Sequence[Mapping[str, Any]] | None,
    extra_keys: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Add Sherlock (and optional brightness) fields from Lasair rows onto a pair table.

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain a ``diaObjectId`` column.
    candidate_rows : sequence of mapping or None
        Rows returned by :func:`query_sn_candidates` (or compatible).
    extra_keys : sequence of str, optional
        Additional row keys to copy (e.g. ``r_psfFlux``, ``r_latestMJD``).

    Returns
    -------
    pandas.DataFrame
        ``data`` with extra columns where keys exist on each candidate row.
    """
    if candidate_rows is None or data.empty:
        return data
    meta = {int(r["diaObjectId"]): r for r in candidate_rows}
    out = data.copy()
    for key in (*CANDIDATE_MERGE_KEYS, *extra_keys):
        out[key] = [meta.get(int(oid), {}).get(key) for oid in out["diaObjectId"]]
    return out


def pick_latest_dia_source_in_window(
    obj_result: Mapping[str, Any],
    mjd_now: float,
    delta_t: float,
) -> int | None:
    """
    Return the ``diaSourceId`` with the latest ``midpointMjdTai`` inside the window.

    Window: ``mjd_now - midpointMjdTai <= delta_t`` (same rule as
    :func:`dia_source_ids_within_delta_t`).
    """
    src_list: Sequence[Mapping[str, Any]] = obj_result.get("diaSourcesList") or ()
    if not src_list:
        return None
    best_sid: int | None = None
    best_mjd: float | None = None
    for s in src_list:
        try:
            mjd = float(s["midpointMjdTai"])
            sid = int(s["diaSourceId"])
        except (KeyError, TypeError, ValueError):
            continue
        if mjd_now - mjd > delta_t:
            continue
        if best_mjd is None or mjd > best_mjd:
            best_mjd = mjd
            best_sid = sid
    return best_sid


def collapse_sql_pair_rows_to_latest_per_object(
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep one (diaObjectId, diaSourceId) per object: latest ``midpointMjdTai``.

    Expects each row to include a midpoint-MJD field (e.g. ``midpointMjdTai``).
    """
    if not pair_rows:
        return []
    df = pd.DataFrame(pair_rows)
    if "diaObjectId" not in df.columns or "diaSourceId" not in df.columns:
        return []
    mj_col: str | None = None
    for c in df.columns:
        cl = c.lower().replace(".", "")
        if "midpoint" in cl and "mjd" in cl:
            mj_col = c
            break
    if mj_col is None:
        get_app_logger().warning(
            "SQL pair rows have no midpoint MJD column; using first row per diaObjectId"
        )
        df = df.drop_duplicates(subset=["diaObjectId"], keep="first")
    else:
        df = df.copy()
        df[mj_col] = pd.to_numeric(df[mj_col], errors="coerce")
        df = df.sort_values(mj_col, ascending=False, na_position="last")
        df = df.drop_duplicates(subset=["diaObjectId"], keep="first")
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            out.append(
                {
                    "diaObjectId": int(row["diaObjectId"]),
                    "diaSourceId": int(row["diaSourceId"]),
                }
            )
        except (TypeError, ValueError):
            continue
    return out


def dia_source_ids_within_delta_t(
    obj_result: Mapping[str, Any],
    mjd_now: float,
    delta_t: float,
) -> np.ndarray:
    """
    Return diaSource IDs from a Lasair ``object`` response within a time window.

    Parameters
    ----------
    obj_result : mapping
        Lasair ``L.object(...)`` dict containing ``diaSourcesList``.
    mjd_now : float
        Current MJD (e.g. from :class:`astropy.time.Time`).
    delta_t : float
        Keep sources with ``mjd_now - midpointMjdTai <= delta_t``.

    Returns
    -------
    ndarray
        Integer diaSource IDs, possibly empty (shape ``(0,)``).
    """
    src_list: Sequence[Mapping[str, Any]] = obj_result["diaSourcesList"]
    if not src_list:
        return np.array([], dtype=np.int64)
    dia_source_ids = np.array([s["diaSourceId"] for s in src_list], dtype=np.int64)
    mjds = np.array([s["midpointMjdTai"] for s in src_list], dtype=float)
    time_since_mjd = mjd_now - mjds
    keep = time_since_mjd <= delta_t
    return dia_source_ids[keep]


def query_dia_object_source_pairs_sql(
    client: Any,
    dia_object_ids: Sequence[int],
    delta_t: float,
    *,
    diasources_table: str,
    row_limit: int,
) -> list[dict[str, Any]]:
    """
    One ``client.query`` returning recent (diaObjectId, diaSourceId) rows.

    Joins ``objects`` to the per-detection table (default ``diasources``). Falls back
    to per-object ``object()`` calls if this query fails (unknown table name, etc.).
    """
    tbl = _validate_lasair_sql_identifier(diasources_table, "diasources table")
    ids_sql = ",".join(str(int(x)) for x in dia_object_ids)
    selected = f"{tbl}.diaObjectId, {tbl}.diaSourceId, {tbl}.midpointMjdTai"
    tables = f"objects,{tbl}"
    conditions = f"""
    objects.diaObjectId = {tbl}.diaObjectId
  AND objects.diaObjectId IN ({ids_sql})
  AND mjdnow() - {tbl}.midpointMjdTai <= {float(delta_t)}
  """
    lim = max(1, min(int(row_limit), 1_000_000))
    return client.query(selected, tables, conditions, limit=lim)


def query_sn_candidates(
    client: Any,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """
    Run the detailed Lasair SN candidate query using ``config``.

    Parameters
    ----------
    client : object
        Authenticated Lasair client.
    config : ExperimentConfig
        Supplies SELECT list, tables, ``delta_t``, ``query_limit``, and classification.

    Returns
    -------
    list of dict
        Rows returned by ``client.query``.
    """
    log = get_app_logger()
    br_day = brightness_recency_days_resolved(config)
    conditions = build_conditions_sn_recent(
        config.delta_t,
        config.sherlock_classification,
        distance_column=config.sherlock_distance_column,
        distance_min=config.distance_min,
        distance_max=config.distance_max,
        z_min=config.z_min,
        z_max=config.z_max,
        bright_max_mag=config.bright_max_mag,
        brightness_band=config.brightness_band,
        brightness_recency_days=config.brightness_recency_days,
    )
    log.info(
        "Lasair detailed query: tables=%r classification=%r delta_t=%s limit=%s "
        "distance_col=%r distance_min=%s distance_max=%s z_min=%s z_max=%s "
        "bright_max_mag=%s brightness_band=%s brightness_recency_days=%s",
        config.lasair_objects_sherlock_tables,
        config.sherlock_classification,
        config.delta_t,
        config.query_limit,
        config.sherlock_distance_column,
        config.distance_min,
        config.distance_max,
        config.z_min,
        config.z_max,
        config.bright_max_mag,
        config.brightness_band,
        br_day,
    )
    if config.bright_max_mag is not None:
        f_floor = nanojansky_flux_floor_for_mag_brighter_than(config.bright_max_mag)
        log.info(
            "Brightness cut: %s-band PSF flux > %.6g nJy (AB m < %s, ZP=%s nJy); "
            "mjdnow - %s_latestMJD <= %s d",
            config.brightness_band,
            f_floor,
            config.bright_max_mag,
            LSST_AB_MAG_ZP_NJY,
            config.brightness_band,
            br_day,
        )
    log.debug("Lasair WHERE clause: %s", " ".join(conditions.split()))
    select = config.sn_candidate_select
    if config.bright_max_mag is not None:
        select = augment_select_with_brightness_columns(select, config.brightness_band)
    rows = client.query(
        select,
        config.lasair_objects_sherlock_tables,
        conditions,
        limit=config.query_limit,
    )
    log.info("Lasair detailed query returned %d row(s)", len(rows))
    if rows and log.isEnabledFor(logging.DEBUG):
        sample = rows[0]
        log.debug("First row keys: %s", list(sample.keys()))
    return rows


def collect_dia_object_source_pairs(
    client: Any,
    dia_object_ids: Sequence[int],
    mjd_now: float,
    config: ExperimentConfig,
    candidate_rows: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Build a table of ``diaObjectId`` with a representative ``diaSourceId``.

    By default (:attr:`ExperimentConfig.lasair_one_row_per_source` True) there is one
    row per diaObject: the **latest** in-window detection (``midpointMjdTai`` within
    ``delta_t``). The Lasair candidate query already applies redshift, object-level
    alert recency (``lastDiaSourceMjdTai``), and magnitude cuts on ``objects``.

    Set ``lasair_one_row_per_source=False`` to emit one row per in-window diaSource
    (legacy behaviour).

    When :attr:`ExperimentConfig.lasair_use_sql_for_pairs` is True, uses a single
    ``client.query`` (see :func:`query_dia_object_source_pairs_sql`) instead of one
    ``object()`` per ID. On failure, falls back to per-object API calls and stores
    each payload on ``client._presn_pair_object_cache`` for reuse by
    :func:`fetch_lasair_object_full`.

    Parameters
    ----------
    client : object
        Lasair API client.
    dia_object_ids : sequence of int
        Candidate object IDs from :func:`query_sn_candidates`.
    mjd_now : float
        Reference MJD for recency filtering (object-API fallback only; SQL uses ``mjdnow()``).
    config : ExperimentConfig
        ``delta_t`` and ``lasair_object_lasair_added``; optional shuffle settings.
    candidate_rows : sequence of mapping or None, optional
        Original Lasair candidate rows; merged into the frame via
        :func:`merge_candidate_metadata_into_pairs` (distance, ``z``, etc.).

    Returns
    -------
    pandas.DataFrame
        Columns ``diaObjectId`` and ``diaSourceId``, plus Sherlock fields when
        ``candidate_rows`` is provided. Empty if no rows qualify.
    """
    log = get_app_logger()
    n_obj = len(dia_object_ids)
    client._presn_pair_object_cache = {}
    pair_cache: dict[int, dict[str, Any]] = client._presn_pair_object_cache
    rows: list[dict[str, int]] = []
    used_sql = False

    if config.lasair_use_sql_for_pairs and n_obj > 0:
        try:
            tbl = resolve_lasair_diasources_table(config.lasair_diasources_table)
            row_lim = config.lasair_pair_query_row_limit
            if row_lim is None:
                row_lim = min(10_000, max(200, n_obj * 50))
            pair_rows = query_dia_object_source_pairs_sql(
                client,
                dia_object_ids,
                config.delta_t,
                diasources_table=tbl,
                row_limit=int(row_lim),
            )
            used_sql = True
            n_sql_raw = len(pair_rows)
            if config.lasair_one_row_per_source:
                pair_rows = collapse_sql_pair_rows_to_latest_per_object(pair_rows)
            for pr in pair_rows:
                try:
                    rows.append(
                        {
                            "diaObjectId": int(pr["diaObjectId"]),
                            "diaSourceId": int(pr["diaSourceId"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            extra_sql = ""
            if config.lasair_one_row_per_source and n_sql_raw > len(rows):
                extra_sql = f" (collapsed from {n_sql_raw} detection row(s))"
            log.info(
                "Lasair SQL pair query (table=%r): %d row(s) for %d object(s); limit=%s%s",
                tbl,
                len(rows),
                n_obj,
                row_lim,
                extra_sql,
            )
        except (LasairError, ValueError) as exc:
            log.warning(
                "Lasair SQL pair query failed (%s); falling back to per-object object() calls",
                exc,
            )
            used_sql = False
            rows = []

    if not used_sql:
        log.info(
            "Fetching Lasair object records for %d diaObject(s); mjd_now=%.6f delta_t=%s "
            "lasair_added=%s",
            n_obj,
            mjd_now,
            config.delta_t,
            config.lasair_object_lasair_added,
        )
        for i in tqdm(
            range(n_obj),
            desc="Lasair objects",
            file=tqdm_log_stream(logging.DEBUG),
            mininterval=2.0,
        ):
            oid = int(dia_object_ids[i])
            obj_result = client.object(oid, lasair_added=config.lasair_object_lasair_added)
            if isinstance(obj_result, dict):
                pair_cache[oid] = dict(obj_result)
            kept = dia_source_ids_within_delta_t(obj_result, mjd_now, config.delta_t)
            log.debug(
                "diaObjectId=%s: %d diaSource(s) in window (index %d/%d)",
                oid,
                len(kept),
                i + 1,
                n_obj,
            )
            if config.lasair_one_row_per_source:
                sid_pick = pick_latest_dia_source_in_window(
                    obj_result, mjd_now, config.delta_t
                )
                if sid_pick is not None:
                    rows.append({"diaObjectId": oid, "diaSourceId": int(sid_pick)})
            else:
                for sid in kept:
                    rows.append({"diaObjectId": oid, "diaSourceId": int(sid)})
    if config.lasair_one_row_per_source:
        log.info(
            "Built %d source row(s) (latest in-window diaSource per diaObject) "
            "before optional shuffle",
            len(rows),
        )
    else:
        log.info(
            "Built %d (diaObjectId, diaSourceId) pair(s) before optional shuffle",
            len(rows),
        )
    data = pd.DataFrame(rows)
    data = merge_candidate_metadata_into_pairs(
        data,
        candidate_rows,
        extra_keys=brightness_merge_keys(config),
    )
    if data.empty:
        return data
    if config.shuffle_pairs:
        log.info(
            "Shuffling pair table (random_state=%s)",
            config.random_state,
        )
        return data.sample(frac=1.0, random_state=config.random_state).reset_index(
            drop=True
        )
    log.info("Pair shuffle disabled; row order is deterministic")
    return data


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def draw_crosshair(
    ax: matplotlib.axes.Axes,
    x: float,
    y: float,
    arm_length: float,
    gap: float,
    **line_kwargs: Any,
) -> None:
    """
    Draw a plus-shaped crosshair with a central gap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x, y : float
        Center in data coordinates.
    arm_length : float
        Length of each arm segment (from gap edge outward).
    gap : float
        Half-gap along each axis from center before the arm starts.
    **line_kwargs
        Forwarded to ``Axes.plot`` (e.g. ``color``, ``linewidth``).
    """
    ax.plot([x - arm_length - gap, x - gap], [y, y], **line_kwargs)
    ax.plot([x + gap, x + arm_length + gap], [y, y], **line_kwargs)
    ax.plot([x, x], [y - arm_length - gap, y - gap], **line_kwargs)
    ax.plot([x, x], [y + gap, y + arm_length + gap], **line_kwargs)


def plot_main_figure(
    calexp: Any,
    template: Any,
    diff: Any,
    style: PlotStyle | None = None,
    show: bool = True,
) -> Figure:
    """
    Plot science, template, difference, and SNR panels for ALeRCE/Lasair stamp HDUs.

    Each of ``calexp``, ``template``, and ``diff`` should behave like Astropy FITS
    HDU lists: index ``0`` is data, index ``1`` variance (for SNR), index ``2`` unused.

    Parameters
    ----------
    calexp, template, diff : HDUList-like
        Stamp data for science, template, and difference images.
    style : PlotStyle or None, optional
        Visual parameters; default is a new :class:`PlotStyle`.
    show : bool, optional
        If True, call ``plt.show()`` after drawing.

    Returns
    -------
    Figure
        The figure containing the four panels.
    """
    log = get_app_logger()
    st = style or PlotStyle()
    calexp_cutout = np.asarray(calexp[0].data)
    template_cutout = np.asarray(template[0].data)
    diff_cutout = np.asarray(diff[0].data)
    if len(diff) > 1 and getattr(diff[1], "data", None) is not None:
        variance = np.asarray(diff[1].data, dtype=float)
    else:
        sigma = float(np.nanstd(diff_cutout)) or 1.0
        variance = np.full_like(diff_cutout, sigma**2, dtype=float)
    snr = diff_cutout / np.sqrt(np.maximum(variance, 1e-24))

    log.debug(
        "plot_main_figure: science shape=%s template=%s diff=%s",
        getattr(calexp_cutout, "shape", None),
        getattr(template_cutout, "shape", None),
        getattr(diff_cutout, "shape", None),
    )

    stamp_px = float(calexp_cutout.shape[0])
    cx = stamp_px / 2.0
    cy = stamp_px / 2.0
    arm = stamp_px * st.crosshair_arm_fraction
    gap = stamp_px * st.crosshair_gap_fraction
    ch_kw = {"color": st.crosshair_color, "linewidth": st.crosshair_linewidth}

    fig = plt.figure(
        figsize=(st.subplot_figsize[0] * st.ncol, st.subplot_figsize[1] * st.nrow)
    )

    vmin_st = np.percentile(template_cutout, st.vmin_percentile)
    vmax_st = np.percentile(template_cutout, st.vmax_percentile)
    vmin_df = np.percentile(diff_cutout, st.diff_vmin_percentile)
    vmax_df = np.percentile(diff_cutout, st.diff_vmax_percentile)

    panels: list[dict[str, Any]] = [
        {
            "data": calexp_cutout,
            "norm": st.science_norm,
            "cmap": st.science_cmap,
            "vmin": vmin_st,
            "vmax": vmax_st,
            "title": st.panel_titles[0],
        },
        {
            "data": template_cutout,
            "norm": st.template_norm,
            "cmap": st.template_cmap,
            "vmin": vmin_st,
            "vmax": vmax_st,
            "title": st.panel_titles[1],
        },
        {
            "data": diff_cutout,
            "norm": st.diff_norm,
            "cmap": st.diff_cmap,
            "vmin": vmin_df,
            "vmax": vmax_df,
            "title": st.panel_titles[2],
        },
        {
            "data": snr,
            "norm": st.snr_norm,
            "cmap": st.snr_cmap,
            "vmin": st.snr_vmin,
            "vmax": st.snr_vmax,
            "title": st.panel_titles[3],
        },
    ]

    for idx, spec in enumerate(panels, start=1):
        ax = fig.add_subplot(st.nrow, st.ncol, idx)
        ax.imshow(
            spec["data"],
            origin="lower",
            norm=spec["norm"],
            cmap=spec["cmap"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
        )
        draw_crosshair(ax, cx, cy, arm, gap, **ch_kw)
        ax.set_title(spec["title"], fontsize=st.titles_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    log.debug("plot_main_figure: layout complete; show=%s", show)
    if show:
        plt.show()
    return fig


# -----------------------------------------------------------------------------
# Lasair-hosted stamp FITS (object API cutout URLs)
# -----------------------------------------------------------------------------


def _lasair_diasources_list(obj: Mapping[str, Any]) -> list[dict[str, Any]]:
    sl = obj.get("diaSourcesList")
    if sl is not None:
        return list(sl)
    ld = obj.get("lasairData")
    if isinstance(ld, dict) and ld.get("diaSourcesList") is not None:
        return list(ld["diaSourcesList"])
    return []


def _deep_find_dia_forced_sources_list(
    node: Any,
    *,
    max_depth: int = 12,
    _depth: int = 0,
) -> list[dict[str, Any]] | None:
    """Locate the first non-empty ``diaForcedSourcesList`` anywhere under ``node``."""
    if _depth > max_depth or node is None:
        return None
    if isinstance(node, dict):
        fl = node.get("diaForcedSourcesList")
        if isinstance(fl, list) and fl:
            dict_rows = [x for x in fl if isinstance(x, dict)]
            if dict_rows:
                return dict_rows
        for v in node.values():
            hit = _deep_find_dia_forced_sources_list(
                v, max_depth=max_depth, _depth=_depth + 1
            )
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for item in node:
            hit = _deep_find_dia_forced_sources_list(
                item, max_depth=max_depth, _depth=_depth + 1
            )
            if hit is not None:
                return hit
    return None


def _lasair_forced_sources_list(obj: Mapping[str, Any]) -> list[dict[str, Any]]:
    def _dict_rows(seq: Any) -> list[dict[str, Any]]:
        if not isinstance(seq, list) or not seq:
            return []
        return [x for x in seq if isinstance(x, dict)]

    fl = obj.get("diaForcedSourcesList")
    dr = _dict_rows(fl)
    if dr:
        return dr
    ld = obj.get("lasairData")
    if isinstance(ld, dict):
        dr = _dict_rows(ld.get("diaForcedSourcesList"))
        if dr:
            return dr
    dia_obj = obj.get("diaObject")
    if isinstance(dia_obj, dict):
        dr = _dict_rows(dia_obj.get("diaForcedSourcesList"))
        if dr:
            return dr
    od = obj.get("objectData")
    if isinstance(od, dict):
        dr = _dict_rows(od.get("diaForcedSourcesList"))
        if dr:
            return dr
    deep = _deep_find_dia_forced_sources_list(obj)
    return deep if deep is not None else []


def find_lasair_diasource_row(
    obj: Mapping[str, Any],
    dia_source_id: int,
) -> dict[str, Any] | None:
    """Return the ``diaSourcesList`` entry matching ``dia_source_id``."""
    target = int(dia_source_id)
    for row in _lasair_diasources_list(obj):
        try:
            if int(row.get("diaSourceId", -1)) == target:
                return row
        except (TypeError, ValueError):
            continue
    return None


def cutout_urls_from_lasair_diasource(cand: Mapping[str, Any]) -> dict[str, str] | None:
    """
    Extract science / template / difference FITS URLs from a Lasair diaSource dict.

    Supports ZTF-style ``image_urls`` and several flat / LSST naming conventions.
    """
    out: dict[str, str] = {}

    iu = cand.get("image_urls")
    if isinstance(iu, dict):
        candidates_map = {
            "cutoutScience": ("Science", "science", "ScienceImage", "cutoutScience"),
            "cutoutTemplate": ("Template", "template", "TemplateImage", "cutoutTemplate"),
            "cutoutDifference": (
                "Difference",
                "difference",
                "DifferenceImage",
                "cutoutDifference",
            ),
        }
        for dest, keys in candidates_map.items():
            for k in keys:
                v = iu.get(k)
                if isinstance(v, str) and v.startswith("http"):
                    out[dest] = v
                    break

    flat_triplets: tuple[tuple[str, str, str], ...] = (
        ("cutoutScience", "cutoutTemplate", "cutoutDifference"),
        ("scienceUrl", "templateUrl", "differenceUrl"),
        ("science_stamp", "template_stamp", "difference_stamp"),
    )
    if len(out) < 3:
        for sci_k, tmp_k, diff_k in flat_triplets:
            a, b, c = cand.get(sci_k), cand.get(tmp_k), cand.get(diff_k)
            if (
                isinstance(a, str)
                and isinstance(b, str)
                and isinstance(c, str)
                and a.startswith("http")
            ):
                out = {
                    "cutoutScience": a,
                    "cutoutTemplate": b,
                    "cutoutDifference": c,
                }
                break

    if len(out) < 3:
        for _k, v in cand.items():
            if not isinstance(v, str) or not v.startswith("http"):
                continue
            low = v.lower()
            if "cutoutscience" in low or (
                "science" in low and ".fits" in low and "template" not in low and "diff" not in low
            ):
                out.setdefault("cutoutScience", v)
            elif "cutouttemplate" in low or ("template" in low and ".fits" in low):
                out.setdefault("cutoutTemplate", v)
            elif "cutoutdifference" in low or ("difference" in low and ".fits" in low):
                out.setdefault("cutoutDifference", v)

    if all(k in out for k in ("cutoutScience", "cutoutTemplate", "cutoutDifference")):
        return out
    return None


def _resolve_lasair_static_url(url: str, lasair_client: Any) -> str:
    if isinstance(url, str) and url.startswith("http"):
        return url.strip()
    base = str(getattr(lasair_client, "endpoint", "") or "").rsplit("/api", 1)[0]
    if base and isinstance(url, str):
        return urljoin(base + "/", url.lstrip("/"))
    return str(url)


def open_fits_from_bytes(raw: bytes) -> Any:
    """Open FITS from bytes; handle gzip-compressed responses."""
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    try:
        return fits.open(io.BytesIO(raw), memmap=False)
    except Exception as exc:
        preview = raw[:120]
        raise ValueError(
            f"Response is not valid FITS after gzip check (starts with {preview!r})"
        ) from exc


def download_lasair_cutout_fits(url: str, lasair_client: Any) -> Any:
    """GET a cutout URL using the Lasair token when needed."""
    log = get_app_logger()
    resolved = _resolve_lasair_static_url(url, lasair_client)
    headers = dict(getattr(lasair_client, "headers", {}) or {})
    throttle = getattr(lasair_client, "_presn_lasair_throttle", None)
    r: requests.Response | None = None
    for attempt in range(2):
        if throttle is not None and headers.get("Authorization"):
            throttle.acquire()
        r = requests.get(resolved, headers=headers, timeout=120)
        if r.status_code == 401:
            r = requests.get(resolved, timeout=120)
        if r.status_code == 429 and attempt == 0:
            log.warning("Lasair cutout GET returned 429; sleeping 90s then retrying once")
            time.sleep(90.0)
            continue
        break
    assert r is not None
    r.raise_for_status()
    return open_fits_from_bytes(r.content)


def _diasource_row_sort_key(row: Mapping[str, Any]) -> tuple[int, int]:
    """Prefer rows with cutout URLs, then richer key sets (for merging API variants)."""
    urls = cutout_urls_from_lasair_diasource(row)
    return (1 if urls else 0, len(row))


def _merge_two_diasource_rows(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Combine two Lasair diaSource dicts, keeping non-empty fields and unioning ``image_urls``."""
    hi, lo = (
        (a, b)
        if _diasource_row_sort_key(a) >= _diasource_row_sort_key(b)
        else (b, a)
    )
    out = dict(hi)
    for k, v in lo.items():
        if k not in out or out[k] in (None, "", [], {}):
            out[k] = v
    iu_a = out.get("image_urls")
    iu_b = lo.get("image_urls")
    if isinstance(iu_a, dict) and isinstance(iu_b, dict):
        merged_iu = dict(iu_b)
        merged_iu.update(iu_a)
        out["image_urls"] = merged_iu
    return out


def merge_lasair_object_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Merge several successful ``/api/object/`` JSON blobs for the same ``diaObjectId``.

    Lasair LSST often returns *different* shapes for ``lite=True`` vs ``lite=False`` (or POST
    variants): one variant may carry ``diaForcedSourcesList`` while another carries cutout URLs
    on ``diaSourcesList`` rows. This merge matches :func:`fetch_lasair_object_full` behaviour used
    by the Plotly scanner and validation script.
    """
    if not payloads:
        raise ValueError("merge_lasair_object_payloads: empty payloads")
    if len(payloads) == 1:
        return dict(payloads[0])

    merged = dict(max(payloads, key=len))

    seen_f: set[tuple[Any, ...]] = set()
    merged_forced: list[dict[str, Any]] = []
    for p in payloads:
        for row in _lasair_forced_sources_list(p):
            if not isinstance(row, dict):
                continue
            key = (row.get("midpointMjdTai"), row.get("mjd"), row.get("band"), row.get("psfFlux"))
            if all(x is None for x in key):
                merged_forced.append(row)
                continue
            if key in seen_f:
                continue
            seen_f.add(key)
            merged_forced.append(row)
    merged["diaForcedSourcesList"] = merged_forced

    by_sid: dict[int, dict[str, Any]] = {}
    for p in payloads:
        for row in _lasair_diasources_list(p):
            if not isinstance(row, dict):
                continue
            try:
                sid = int(row["diaSourceId"])
            except (KeyError, TypeError, ValueError):
                continue
            if sid not in by_sid:
                by_sid[sid] = dict(row)
            else:
                by_sid[sid] = _merge_two_diasource_rows(by_sid[sid], row)
    merged["diaSourcesList"] = sorted(
        by_sid.values(),
        key=lambda r: int(r.get("diaSourceId", 0)),
    )
    return merged


def _post_lasair_object_api(
    lasair_client: Any,
    form: dict[str, Any],
) -> dict[str, Any]:
    """
    POST ``/api/object/`` with explicit form fields (LSST may expect ``diaObjectId``).
    """
    log = get_app_logger()
    base = str(getattr(lasair_client, "endpoint", "") or "").rstrip("/")
    url = f"{base}/object/"
    headers = dict(getattr(lasair_client, "headers", {}) or {})
    throttle = getattr(lasair_client, "_presn_lasair_throttle", None)
    r: requests.Response | None = None
    for attempt in range(2):
        if throttle is not None:
            throttle.acquire()
        r = requests.post(url, headers=headers, data=form, timeout=120)
        if r.status_code == 429 and attempt == 0:
            log.warning("Lasair object POST returned 429; sleeping 90s then retrying once")
            time.sleep(90.0)
            continue
        break
    assert r is not None
    if r.status_code != 200:
        raise LasairError(
            f"HTTP return code {r.status_code} for\n{url}\n{r.text[:500]!r}"
        )
    try:
        result = r.json()
    except Exception as exc:
        raise LasairError(f"Cannot parse JSON from object API: {exc!s}") from exc
    if isinstance(result, dict) and result.get("error"):
        raise LasairError(str(result["error"]))
    if not isinstance(result, dict):
        raise TypeError(f"Lasair object API expected dict, got {type(result)}")
    return result


def _merge_lasair_fetch_successes(successes: list[dict[str, Any]]) -> dict[str, Any]:
    if not successes:
        raise ValueError("_merge_lasair_fetch_successes: empty list")
    if len(successes) == 1:
        return dict(successes[0])
    return merge_lasair_object_payloads(successes)


def _lasair_fetch_early_exit_payload(
    successes: list[dict[str, Any]],
    *,
    early_exit: LasairObjectEarlyExit,
    dia_source_id: int | None,
) -> dict[str, Any] | None:
    if early_exit is None or not successes:
        return None
    try:
        m = _merge_lasair_fetch_successes(successes)
    except Exception:
        return None
    if early_exit == "forced_ok" and _lasair_forced_sources_list(m):
        return m
    if early_exit == "plotly" and dia_source_id is not None:
        if not _lasair_forced_sources_list(m):
            return None
        row = find_lasair_diasource_row(m, dia_source_id)
        if row is not None and cutout_urls_from_lasair_diasource(row) is not None:
            return m
    return None


def fetch_lasair_object_full(
    lasair_client: Any,
    dia_object_id: int,
    *,
    dia_source_id: int | None = None,
    early_exit: LasairObjectEarlyExit = None,
) -> dict[str, Any]:
    """
    Lasair object payload for stamps + ``diaForcedSourcesList``.

    LSST sometimes returns HTTP 404 for ``lite=False`` or certain flag combinations; we
    try several variants so forced photometry (usually present for ``lite=True``) still loads.
    Cutout URL fields, when present, are most complete for ``lite=False``.

    Seeding: if ``lasair_client._presn_pair_object_cache`` contains this ``diaObjectId``
    (from :func:`collect_dia_object_source_pairs`), it is merged first to avoid redundant
    ``object()`` variants when possible.

    Parameters
    ----------
    early_exit : optional
        ``\"plotly\"`` stops once forced photometry and stamp URLs exist for ``dia_source_id``.
        ``\"forced_ok\"`` stops once ``diaForcedSourcesList`` is non-empty.
        ``None`` tries all variants and merges (maximum completeness).
    """
    log = get_app_logger()
    oid_int = int(dia_object_id)
    oid_str = str(oid_int)

    client_attempts: list[tuple[bool, bool]] = [
        (True, False),
        (True, True),
        (False, True),
        (False, False),
    ]

    successes: list[dict[str, Any]] = []
    last_err: Exception | None = None

    pair_cache = getattr(lasair_client, "_presn_pair_object_cache", None)
    seed = pair_cache.get(oid_int) if isinstance(pair_cache, dict) else None
    if isinstance(seed, dict) and "error" not in seed:
        successes.append(dict(seed))
        hit = _lasair_fetch_early_exit_payload(
            successes,
            early_exit=early_exit,
            dia_source_id=dia_source_id,
        )
        if hit is not None:
            log.info(
                "Lasair object early exit from pair-cache seed: diaObjectId=%s early_exit=%r",
                oid_str,
                early_exit,
            )
            return hit

    for lasair_added, lite in client_attempts:
        hit = _lasair_fetch_early_exit_payload(
            successes,
            early_exit=early_exit,
            dia_source_id=dia_source_id,
        )
        if hit is not None:
            log.info(
                "Lasair object early exit after %d variant(s): diaObjectId=%s early_exit=%r",
                len(successes),
                oid_str,
                early_exit,
            )
            return hit
        try:
            result = lasair_client.object(
                oid_str, lasair_added=lasair_added, lite=lite
            )
            if isinstance(result, dict) and "error" not in result:
                successes.append(result)
                log.info(
                    "Lasair object variant OK: diaObjectId=%s lasair_added=%s lite=%s",
                    oid_str,
                    lasair_added,
                    lite,
                )
        except LasairError as exc:
            last_err = exc
            log.debug(
                "Lasair object() try lasair_added=%s lite=%s failed: %s",
                lasair_added,
                lite,
                exc,
            )
            continue

    hit = _lasair_fetch_early_exit_payload(
        successes,
        early_exit=early_exit,
        dia_source_id=dia_source_id,
    )
    if hit is not None:
        log.info(
            "Lasair object early exit after GET variants: diaObjectId=%s early_exit=%r",
            oid_str,
            early_exit,
        )
        return hit

    form_attempts: list[dict[str, Any]] = [
        {
            "diaObjectId": oid_str,
            "objectId": oid_str,
            "lasair_added": "true",
            "lite": "false",
            "reliabilityThreshold": "0",
        },
        {
            "diaObjectId": oid_str,
            "objectId": oid_str,
            "lasair_added": "true",
            "lite": "true",
            "reliabilityThreshold": "0",
        },
        {"diaObjectId": oid_str, "lasair_added": "false", "lite": "true"},
    ]
    for form in form_attempts:
        hit = _lasair_fetch_early_exit_payload(
            successes,
            early_exit=early_exit,
            dia_source_id=dia_source_id,
        )
        if hit is not None:
            log.info(
                "Lasair object early exit before POST: diaObjectId=%s early_exit=%r",
                oid_str,
                early_exit,
            )
            return hit
        try:
            result = _post_lasair_object_api(lasair_client, form)
            successes.append(result)
            log.info(
                "Lasair object variant OK (POST): diaObjectId=%s form_keys=%s",
                oid_str,
                list(form.keys()),
            )
        except LasairError as exc:
            last_err = exc
            log.debug("Lasair object POST try failed: %s", exc)
            continue

    hit = _lasair_fetch_early_exit_payload(
        successes,
        early_exit=early_exit,
        dia_source_id=dia_source_id,
    )
    if hit is not None:
        return hit

    if not successes:
        if last_err is not None:
            raise last_err
        raise LasairError(f"Lasair object fetch failed for diaObjectId={oid_str}")

    if len(successes) == 1:
        return successes[0]
    merged = merge_lasair_object_payloads(successes)
    log.info(
        "Lasair object merged %d API variant(s) for diaObjectId=%s "
        "(forced rows=%d diaSources=%d)",
        len(successes),
        oid_str,
        len(merged.get("diaForcedSourcesList") or []),
        len(merged.get("diaSourcesList") or []),
    )
    return merged


def lasair_forced_photometry_dataframe(obj: Mapping[str, Any]) -> pd.DataFrame:
    """Build a table from ``diaForcedSourcesList`` for light-curve plotting."""
    rows = _lasair_forced_sources_list(obj)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "band" in df.columns and "band_name" not in df.columns:
        df = df.copy()
        df["band_name"] = df["band"]
    return df


def get_lasair_stamp_cutouts(
    lasair_client: Any,
    obj: Mapping[str, Any],
    dia_source_id: int,
) -> dict[str, Any] | None:
    """
    Download science/template/difference stamp FITS from Lasair cutout URLs.

    Parameters
    ----------
    lasair_client : object
        Lasair client (``headers`` and ``endpoint`` used for HTTP).
    obj : mapping
        Return value of :func:`fetch_lasair_object_full` (or equivalent).
    dia_source_id : int
        LSST ``diaSourceId`` for the epoch to display.

    Returns
    -------
    dict or None
        Keys ``cutoutScience``, ``cutoutTemplate``, ``cutoutDifference`` mapping to
        ``HDUList``-like objects, or None if URLs are missing or download/parse fails.
    """
    log = get_app_logger()
    row = find_lasair_diasource_row(obj, dia_source_id)
    if row is None:
        log.warning("Lasair object has no diaSourceId=%s in diaSourcesList", dia_source_id)
        return None
    urls = cutout_urls_from_lasair_diasource(row)
    if not urls:
        log.warning(
            "Lasair diaSourceId=%s has no cutout URLs in object payload "
            "(cutouts may have expired or schema may omit links for this source).",
            dia_source_id,
        )
        log.debug("Lasair diaSource key sample: %s", sorted(row.keys())[:50])
        return None

    out: dict[str, Any] = {}
    try:
        for k in ("cutoutScience", "cutoutTemplate", "cutoutDifference"):
            out[k] = download_lasair_cutout_fits(urls[k], lasair_client)
    except Exception:
        log.exception("Lasair stamp download/parse failed for diaSourceId=%s", dia_source_id)
        return None
    return out


def _alerce_stamp_http_body_to_fits_bytes(content: bytes) -> bytes:
    """
    ALeRCE LSST stamp responses may be gzip-wrapped FITS or raw FITS; error bodies are plain text.

    The stock ``alerce`` multisurvey client always uses ``gzip.open`` on the body, which raises
    ``BadGzipFile`` when the server returns JSON/HTML/text (often starting with ``b'In'`` as in
    ``Internal server error`` or ``Invalid``).
    """
    if len(content) >= 2 and content[:2] == b"\x1f\x8b":
        return gzip.decompress(content)
    return content


def _get_alerce_lsst_stamps_robust(
    alerce_client: Any,
    dia_object_id: int,
    dia_source_id: int,
    include_variance: bool,
    *,
    request_timeout: float = 120.0,
) -> dict[str, Any] | None:
    """
    Fetch LSST stamp FITS via the multisurvey stamp API with correct HTTP / gzip handling.

    Returns the same ``dict`` shape as ``multisurvey_get_stamps`` (``cutoutScience``, etc.) or
    ``None`` if any panel fails.
    """
    log = get_app_logger()
    msc = alerce_client.multisurvey_stamps_client
    survey = "lsst"
    oid = int(dia_object_id)
    measurement_id = int(dia_source_id)
    avro_url = msc.config["STAMP_URL"] + msc.config["AVRO_ROUTES"]["get_stamp"]
    stamp_types: tuple[str, ...] = ("cutoutScience", "cutoutTemplate", "cutoutDifference")
    stamp_list: list[Any] = []

    for stamp_type in stamp_types:
        url = create_stamp_parameters(
            oid, survey, measurement_id, stamp_type, avro_url, "get"
        )
        try:
            r = msc.session.request("GET", url, timeout=request_timeout)
        except Exception as exc:
            log.warning(
                "ALeRCE stamp request failed %s oid=%s measurement_id=%s: %s",
                stamp_type,
                oid,
                measurement_id,
                exc,
            )
            return None
        if r.status_code != 200:
            preview = r.content[:200].decode("utf-8", errors="replace").replace("\n", " ")
            log.warning(
                "ALeRCE stamp HTTP %s for %s oid=%s measurement_id=%s — %s",
                r.status_code,
                stamp_type,
                oid,
                measurement_id,
                preview,
            )
            return None
        try:
            raw_fits = _alerce_stamp_http_body_to_fits_bytes(r.content)
        except OSError as exc:
            log.warning(
                "ALeRCE stamp gzip decode failed %s oid=%s measurement_id=%s: %s (body starts %r)",
                stamp_type,
                oid,
                measurement_id,
                exc,
                r.content[:40],
            )
            return None
        try:
            tmp_hdulist = fits.open(io.BytesIO(raw_fits), ignore_missing_simple=True)
        except Exception as exc:
            log.warning(
                "ALeRCE stamp FITS open failed %s oid=%s measurement_id=%s: %s (body starts %r)",
                stamp_type,
                oid,
                measurement_id,
                exc,
                r.content[:40],
            )
            return None
        if include_variance:
            stamp_list.append(tmp_hdulist)
        else:
            stamp_list.append(tmp_hdulist[0])

    return {
        stamp_types[0]: stamp_list[0],
        stamp_types[1]: stamp_list[1],
        stamp_types[2]: stamp_list[2],
    }


def get_alerce_stamps(
    alerce_client: Any,
    dia_object_id: int,
    dia_source_id: int,
    survey: str,
    include_variance: bool,
) -> Any:
    """
    Download science/template/difference stamps from ALeRCE.

    For ``survey='lsst'`` we bypass ``alerce.multisurvey_get_stamps``: that helper always
    gzip-decompresses the HTTP body and crashes on plain-text API errors or uncompressed FITS.

    Other surveys use the installed client's ``get_stamps`` (variance keyword chosen by signature).
    """
    if str(survey).lower() == "lsst":
        return _get_alerce_lsst_stamps_robust(
            alerce_client,
            dia_object_id,
            dia_source_id,
            include_variance,
        )
    sig = inspect.signature(alerce_client.get_stamps)
    names = set(sig.parameters)
    kwargs: dict[str, Any] = {
        "oid": dia_object_id,
        "measurement_id": dia_source_id,
        "survey": survey,
    }
    if "include_variance_and_mask" in names:
        kwargs["include_variance_and_mask"] = include_variance
    elif "include_variance_and_psf" in names:
        kwargs["include_variance_and_psf"] = include_variance
    return alerce_client.get_stamps(**kwargs)


@dataclass
class AlercePlotlyPrefetch:
    """
    Bulk ALeRCE data for the Plotly transient scan: stamp FITS per (object, detection)
    and forced-photometry tables per object (same sources the web client uses).
    """

    stamps: dict[tuple[int, int], dict[str, Any] | None]
    forced_by_oid: dict[int, pd.DataFrame]


def query_alerce_forced_photometry_dataframe(
    alerce_client: Any,
    dia_object_id: int,
    survey: str,
) -> pd.DataFrame:
    """Return ALeRCE forced photometry for one ``diaObjectId`` as a DataFrame."""
    out = alerce_client.query_forced_photometry(
        str(int(dia_object_id)),
        format="pandas",
        survey=survey,
    )
    if out is None:
        return pd.DataFrame()
    if hasattr(out, "empty") and out.empty:
        return pd.DataFrame()
    return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)


def alerce_stamp_allowed_measurement_ids(
    alerce_client: Any,
    dia_object_id: int,
    survey: str,
) -> frozenset[int] | None:
    """
    Return measurement_ids that have ALeRCE stamps, or None if the filter cannot be applied.

    When None, callers should keep all (object, detection) pairs for that object so a
    transient list still works if ``query_detections`` fails or lacks ``has_stamp``.
    """
    log = get_app_logger()
    try:
        d = alerce_client.query_detections(
            int(dia_object_id), format="pandas", survey=survey
        )
    except Exception:
        log.debug(
            "ALeRCE query_detections failed for oid=%s; not filtering pairs by has_stamp",
            dia_object_id,
            exc_info=True,
        )
        return None
    if d is None or (hasattr(d, "empty") and d.empty):
        return frozenset()
    if not isinstance(d, pd.DataFrame):
        d = pd.DataFrame(d)
    if "has_stamp" not in d.columns or "measurement_id" not in d.columns:
        log.warning(
            "ALeRCE detections for oid=%s lack has_stamp/measurement_id columns; "
            "not filtering pairs for that object",
            dia_object_id,
        )
        return None
    mask = d["has_stamp"].fillna(False).astype(bool)
    sub = d.loc[mask, "measurement_id"]
    return frozenset(int(x) for x in sub if pd.notna(x))


def filter_scan_records_for_alerce_stamps(
    alerce_client: Any,
    records: Sequence[Mapping[str, Any]],
    survey: str,
) -> list[dict[str, Any]]:
    """
    Drop scan rows whose ``diaSourceId`` is not a stamp-backed ALeRCE detection.

    Lasair can list diaSources for which ALeRCE returns HTTP 500 on ``get_stamps`` when
    ``has_stamp`` is false; those pairs are removed so prefetch stays reliable.
    """
    log = get_app_logger()
    by_oid: dict[int, list[Mapping[str, Any]]] = {}
    for r in records:
        oid = int(r["diaObjectId"])
        by_oid.setdefault(oid, []).append(r)
    cache: dict[int, frozenset[int] | None] = {}
    out: list[dict[str, Any]] = []
    dropped = 0
    for oid, group in by_oid.items():
        if oid not in cache:
            cache[oid] = alerce_stamp_allowed_measurement_ids(
                alerce_client, oid, survey
            )
        allowed = cache[oid]
        for r in group:
            row = dict(r)
            sid = int(row["diaSourceId"])
            if allowed is None or sid in allowed:
                out.append(row)
            else:
                dropped += 1
    n = len(records)
    if dropped:
        log.info(
            "ALeRCE stamp filter: dropped %d/%d pair(s) (has_stamp=False or missing in ALeRCE)",
            dropped,
            n,
        )
    return out


def alerce_best_stamp_measurement_id(
    alerce_client: Any,
    dia_object_id: int,
    survey: str,
    mjd_now: float,
    delta_t: float,
) -> int | None:
    """
    ``measurement_id`` of the stamp-backed detection with the largest ``|snr|`` in the window.

    Keeps rows with ``has_stamp``, ``mjd <= mjd_now``, and ``mjd_now - mjd <= delta_t``.
    Tie-break: latest ``mjd``.
    """
    try:
        d = alerce_client.query_detections(
            int(dia_object_id), format="pandas", survey=survey
        )
    except Exception:
        return None
    if d is None or (hasattr(d, "empty") and d.empty):
        return None
    if not isinstance(d, pd.DataFrame):
        d = pd.DataFrame(d)
    need = ("has_stamp", "measurement_id", "mjd", "snr")
    if not set(need).issubset(d.columns):
        return None
    d = d.loc[d["has_stamp"].fillna(False).astype(bool)].copy()
    if d.empty:
        return None
    mjd = pd.to_numeric(d["mjd"], errors="coerce")
    dt = float(mjd_now) - mjd
    ok = (dt >= 0) & (dt <= float(delta_t)) & mjd.notna()
    d = d.loc[ok]
    if d.empty:
        return None
    d["_abs_snr"] = pd.to_numeric(d["snr"], errors="coerce").abs().fillna(0.0)
    d["_mjd"] = mjd.loc[d.index]
    d = d.sort_values(["_abs_snr", "_mjd"], ascending=[False, False])
    mid = d.iloc[0]["measurement_id"]
    if pd.isna(mid):
        return None
    return int(mid)


def assign_best_alerce_stamp_detection_for_scan(
    alerce_client: Any,
    records: Sequence[Mapping[str, Any]],
    survey: str,
    mjd_now: float,
    delta_t: float,
) -> list[dict[str, Any]]:
    """
    Set each row's ``diaSourceId`` to ALeRCE's best stamp detection for that object.

    Objects with no qualifying detection are dropped (see
    :func:`alerce_best_stamp_measurement_id`).
    """
    log = get_app_logger()
    by_oid: dict[int, list[Mapping[str, Any]]] = {}
    for r in records:
        oid = int(r["diaObjectId"])
        by_oid.setdefault(oid, []).append(r)
    out: list[dict[str, Any]] = []
    skipped: list[int] = []
    for oid, group in by_oid.items():
        mid = alerce_best_stamp_measurement_id(
            alerce_client, oid, survey, mjd_now, delta_t
        )
        if mid is None:
            skipped.append(oid)
            continue
        for r in group:
            row = dict(r)
            row["diaSourceId"] = mid
            out.append(row)
    if skipped:
        log.warning(
            "ALeRCE: skipped %d object(s) with no stamp detection in %.1f d window "
            "(has_stamp, mjd_now-mjd<=delta_t); sample diaObjectId(s)=%s",
            len(skipped),
            float(delta_t),
            skipped[:8],
        )
    log.info(
        "ALeRCE scan rows: %d source(s) after choosing highest-|snr| stamp detection "
        "per diaObject (window=%.1f d)",
        len(out),
        float(delta_t),
    )
    return out


def prefetch_alerce_plotly_scan_data(
    alerce_client: Any,
    records: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
    *,
    max_workers: int = 4,
) -> AlercePlotlyPrefetch:
    """
    Download ALeRCE stamp cutouts and forced photometry for all scan records up front.

    Lasair is only used earlier for candidate selection; this function performs the
    follow-up data load from ALeRCE (parallel HTTP for stamps, one query per unique
    object for forced photometry).
    """
    log = get_app_logger()
    survey = config.alerce_survey
    inc = config.alerce_include_variance_and_mask

    unique_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in records:
        p = (int(r["diaObjectId"]), int(r["diaSourceId"]))
        if p not in seen:
            seen.add(p)
            unique_pairs.append(p)

    unique_oids = sorted({p[0] for p in unique_pairs})
    forced_by_oid: dict[int, pd.DataFrame] = {}
    log.info(
        "ALeRCE prefetch: query_forced_photometry for %d unique diaObjectId(s), survey=%r",
        len(unique_oids),
        survey,
    )
    for oid in unique_oids:
        try:
            forced_by_oid[oid] = query_alerce_forced_photometry_dataframe(
                alerce_client, oid, survey
            )
        except Exception:
            log.exception("ALeRCE query_forced_photometry failed for oid=%s", oid)
            forced_by_oid[oid] = pd.DataFrame()

    def _fetch_stamps(pair: tuple[int, int]) -> tuple[tuple[int, int], dict[str, Any] | None]:
        oid, sid = pair
        try:
            raw = get_alerce_stamps(alerce_client, oid, sid, survey, inc)
            if raw is None:
                return pair, None
            if isinstance(raw, dict) and all(
                k in raw for k in ("cutoutScience", "cutoutTemplate", "cutoutDifference")
            ):
                return pair, raw
            log.warning(
                "ALeRCE get_stamps returned unexpected type for oid=%s sid=%s: %s",
                oid,
                sid,
                type(raw).__name__,
            )
            return pair, None
        except Exception as exc:
            log.warning("ALeRCE get_stamps failed for oid=%s sid=%s: %s", oid, sid, exc)
            return pair, None

    stamps: dict[tuple[int, int], dict[str, Any] | None] = {}
    n_pairs = len(unique_pairs)
    log.info(
        "ALeRCE prefetch: get_stamps for %d (diaObjectId, diaSourceId) pair(s)",
        n_pairs,
    )
    workers = max(1, min(int(max_workers), n_pairs or 1))
    if workers <= 1 or n_pairs <= 1:
        for pair in unique_pairs:
            k, v = _fetch_stamps(pair)
            stamps[k] = v
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_fetch_stamps, p) for p in unique_pairs]
            for fut in as_completed(futures):
                k, v = fut.result()
                stamps[k] = v

    n_ok = sum(1 for v in stamps.values() if v is not None)
    log.info(
        "ALeRCE prefetch finished: %d/%d stamp bundle(s) retrieved successfully",
        n_ok,
        n_pairs,
    )
    return AlercePlotlyPrefetch(stamps=stamps, forced_by_oid=forced_by_oid)


def fetch_and_plot_stamps(
    lasair_client: Any,
    alerce_client: Alerce,
    dia_object_id: int,
    dia_source_id: int,
    config: ExperimentConfig,
) -> Figure:
    """
    Request stamps from Lasair (object API cutout URLs) when available, else ALeRCE.

    Parameters
    ----------
    lasair_client : object
        Lasair API client.
    alerce_client : Alerce
        Configured ALeRCE client (fallback).
    dia_object_id : int
        LSST diaObject identifier.
    dia_source_id : int
        LSST diaSource identifier.
    config : ExperimentConfig
        Supplies survey name, stamp options, and plot style.

    Returns
    -------
    Figure
        Figure from :func:`plot_main_figure`.
    """
    log = get_app_logger()
    try:
        log.info(
            "Lasair stamps: oid=%s diaSourceId=%s (object API, lite=False)",
            dia_object_id,
            dia_source_id,
        )
        obj = fetch_lasair_object_full(
            lasair_client,
            dia_object_id,
            dia_source_id=dia_source_id,
            early_exit="plotly",
        )
        cutouts = get_lasair_stamp_cutouts(lasair_client, obj, dia_source_id)
        if cutouts is not None:
            return plot_main_figure(
                cutouts["cutoutScience"],
                cutouts["cutoutTemplate"],
                cutouts["cutoutDifference"],
                style=config.plot_style,
                show=True,
            )
    except Exception:
        log.warning("Lasair stamp download failed; using ALeRCE", exc_info=True)

    log.info(
        "ALeRCE get_stamps: survey=%r oid=%s measurement_id=%s include_variance=%s",
        config.alerce_survey,
        dia_object_id,
        dia_source_id,
        config.alerce_include_variance_and_mask,
    )
    cutouts = get_alerce_stamps(
        alerce_client,
        dia_object_id,
        dia_source_id,
        config.alerce_survey,
        config.alerce_include_variance_and_mask,
    )
    log.debug("ALeRCE cutout keys: %s", list(cutouts.keys()))
    return plot_main_figure(
        cutouts["cutoutScience"],
        cutouts["cutoutTemplate"],
        cutouts["cutoutDifference"],
        style=config.plot_style,
        show=True,
    )


# -----------------------------------------------------------------------------
# Workflow entry points
# -----------------------------------------------------------------------------


def run_simple_query(client: Any, config: ExperimentConfig) -> None:
    """
    Run a minimal Lasair query and log diaObjectId, RA, and Dec.

    Parameters
    ----------
    client : object
        Lasair client.
    config : ExperimentConfig
        SELECT list, tables, classification, and ``simple_query_limit``.
    """
    log = get_app_logger()
    log.info(
        "Simple Lasair query: limit=%s tables=%r classification=%r "
        "simple_query_recent=%s delta_t=%s z_min=%s z_max=%s "
        "bright_max_mag=%s brightness_band=%s",
        config.simple_query_limit,
        config.lasair_objects_sherlock_tables,
        config.sherlock_classification,
        config.apply_delta_t_to_simple_query,
        config.delta_t,
        config.z_min,
        config.z_max,
        config.bright_max_mag,
        config.brightness_band,
    )
    conditions = build_simple_query_conditions(config)
    log.debug("Simple query WHERE: %s", " ".join(conditions.split()))
    sel = config.simple_query_select
    if config.bright_max_mag is not None:
        sel = augment_select_with_brightness_columns(sel, config.brightness_band)
    results = client.query(
        sel,
        config.lasair_objects_sherlock_tables,
        conditions,
        limit=config.simple_query_limit,
    )
    log.info("Simple query returned %d row(s)", len(results))
    bf = config.brightness_band
    fk_flux = f"{bf}_psfFlux"
    fk_mjd = f"{bf}_latestMJD"
    for row in results:
        log.info(
            "object: diaObjectId=%s ra=%s decl=%s distance(Mpc)=%s direct(Mpc)=%s "
            "separation_kpc=%s z=%s %s=%s %s=%s",
            row["diaObjectId"],
            row["ra"],
            row["decl"],
            row.get("distance"),
            row.get("direct_distance"),
            row.get("physical_separation_kpc"),
            row.get("z"),
            fk_flux,
            row.get(fk_flux),
            fk_mjd,
            row.get(fk_mjd),
        )


def run_single_object_demo(
    client: Any,
    alerce_client: Alerce,
    config: ExperimentConfig,
) -> None:
    """
    Plot stamps for the first SN candidate and its first recent diaSource.

    Parameters
    ----------
    client : object
        Lasair client.
    alerce_client : Alerce
        ALeRCE client.
    config : ExperimentConfig
        Query limits, ``delta_t``, and plotting options.
    """
    log = get_app_logger()
    log.info("Single-object demo mode: using first SN candidate from detailed query")
    mjd_now = Time.now().mjd
    log.debug("Current MJD for recency filter: %.6f", mjd_now)
    results = query_sn_candidates(client, config)
    if not results:
        log.warning("No objects returned from Lasair query; nothing to plot")
        return
    dia_object_ids = [results[i]["diaObjectId"] for i in range(len(results))]
    dia_object_id = int(dia_object_ids[0])
    log.info("Using first candidate diaObjectId=%s (of %d)", dia_object_id, len(results))
    r0 = results[0]
    log.info(
        "First candidate Sherlock context: distance(Mpc,lum)=%s direct_distance(Mpc)=%s "
        "physical_separation_kpc=%s z=%s separationArcsec=%s",
        r0.get("distance"),
        r0.get("direct_distance"),
        r0.get("physical_separation_kpc"),
        r0.get("z"),
        r0.get("separationArcsec"),
    )
    obj_result = client.object(
        dia_object_id, lasair_added=config.lasair_object_lasair_added
    )
    sid_pick = pick_latest_dia_source_in_window(obj_result, mjd_now, config.delta_t)
    if sid_pick is None:
        log.warning(
            "No diaSources within delta_t=%s for diaObjectId=%s; try a larger --delta-t",
            config.delta_t,
            dia_object_id,
        )
        return
    log.info("Using latest diaSource in window: diaSourceId=%s", int(sid_pick))
    fetch_and_plot_stamps(client, alerce_client, dia_object_id, int(sid_pick), config)


def run_full_pipeline(
    client: Any,
    alerce_client: Alerce,
    config: ExperimentConfig,
) -> None:
    """
    Query candidates, build diaObject/diaSource pairs, preview a table, plot stamps.

    Parameters
    ----------
    client : object
        Lasair client.
    alerce_client : Alerce
        ALeRCE client.
    config : ExperimentConfig
        Full experiment parameters including ``max_plots`` and ``preview_table_rows``.
    """
    log = get_app_logger()
    log.info(
        "Full pipeline: preview_rows=%s max_plots=%s",
        config.preview_table_rows,
        config.max_plots,
    )
    mjd_now = Time.now().mjd
    log.debug("Current MJD for recency filter: %.6f", mjd_now)
    results = query_sn_candidates(client, config)
    if not results:
        log.warning("No objects returned from Lasair query; pipeline stops")
        return
    dia_object_ids = [int(results[i]["diaObjectId"]) for i in range(len(results))]
    data = collect_dia_object_source_pairs(
        client,
        dia_object_ids,
        mjd_now,
        config,
        candidate_rows=results,
    )
    if data.empty:
        log.warning("No source rows within delta_t=%s", config.delta_t)
        return
    n_preview = min(config.preview_table_rows, len(data))
    log.info(
        "Scan table preview (%d of %d rows):\n%s",
        n_preview,
        len(data),
        data.head(n_preview).to_string(),
    )
    n_plot = min(config.max_plots, len(data))
    log.info("Plotting %d stamp figure(s)", n_plot)
    for i in range(n_plot):
        oid = int(data["diaObjectId"].iloc[i])
        sid = int(data["diaSourceId"].iloc[i])
        extra = ""
        if "distance" in data.columns:
            extra = f" distance(Mpc)={data['distance'].iloc[i]!r}"
        if "physical_separation_kpc" in data.columns:
            extra += f" separation_kpc={data['physical_separation_kpc'].iloc[i]!r}"
        log.info(
            "Figure %d/%d: diaObjectId=%s diaSourceId=%s%s",
            i + 1,
            n_plot,
            oid,
            sid,
            extra,
        )
        fetch_and_plot_stamps(client, alerce_client, oid, sid, config)
    log.info("Full pipeline finished (%d figure(s))", n_plot)


def load_transient_scan_pairs(
    client: Any,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """
    Run the Lasair candidate query and return one row per qualifying source by default.

    Each row is a ``diaObjectId`` with a representative ``diaSourceId`` (latest
    detection in ``delta_t``). The candidate SQL applies redshift, object-level
    alert recency, and brightness filters on ``objects``.

    Used by the Plotly/Dash scanner. Columns include ``ra``, ``decl``, and
    Sherlock metadata when present on candidate rows.

    Parameters
    ----------
    client : object
        Lasair API client.
    config : ExperimentConfig
        Query and filter configuration.

    Returns
    -------
    pandas.DataFrame
        Scan table, possibly empty.
    """
    log = get_app_logger()
    mjd_now = Time.now().mjd
    results = query_sn_candidates(client, config)
    if not results:
        log.warning("Lasair query returned no candidates for scan")
        return pd.DataFrame()
    dia_object_ids = [int(results[i]["diaObjectId"]) for i in range(len(results))]
    data = collect_dia_object_source_pairs(
        client,
        dia_object_ids,
        mjd_now,
        config,
        candidate_rows=results,
    )
    if config.lasair_one_row_per_source:
        log.info("Scan data: %d source row(s) (diaObject + latest in-window diaSource)", len(data))
    else:
        log.info("Scan data: %d (diaObjectId, diaSourceId) pair(s)", len(data))
    return data


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """
    Build :class:`ExperimentConfig` from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace produced by :func:`build_arg_parser`.

    Returns
    -------
    ExperimentConfig
        Populated configuration object.
    """
    plot_style = PlotStyle(
        snr_vmin=args.plot_snr_vmin,
        snr_vmax=args.plot_snr_vmax,
        crosshair_arm_fraction=args.crosshair_arm_fraction,
        crosshair_gap_fraction=args.crosshair_gap_fraction,
    )
    return ExperimentConfig(
        lasair_endpoint=args.lasair_endpoint,
        delta_t=args.delta_t,
        query_limit=args.query_limit,
        max_plots=args.max_plots,
        simple_query_limit=args.simple_query_limit,
        preview_table_rows=args.preview_rows,
        lasair_objects_sherlock_tables=args.lasair_tables,
        sherlock_classification=args.sherlock_classification,
        alerce_survey=args.alerce_survey,
        shuffle_pairs=not args.no_shuffle,
        random_state=args.random_state,
        plot_style=plot_style,
        sherlock_distance_column=ALLOWED_SHERLOCK_DISTANCE_COLUMNS[
            args.distance_metric
        ],
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        apply_distance_filter_to_simple_query=args.simple_query_distance_filter,
        z_min=args.z_min,
        z_max=args.z_max,
        apply_delta_t_to_simple_query=args.simple_query_recent,
        bright_max_mag=args.bright_max_mag,
        brightness_band=args.brightness_band,
        brightness_recency_days=args.brightness_within_days,
        lasair_max_calls_per_hour=args.lasair_max_calls_per_hour,
        lasair_client_cache_dir=(
            str(args.lasair_client_cache).strip() or None
            if args.lasair_client_cache
            else None
        ),
        lasair_use_sql_for_pairs=resolve_lasair_use_sql_for_pairs(
            True
            if args.lasair_sql_pairs
            else (False if args.no_lasair_sql_pairs else None)
        ),
        lasair_diasources_table=args.lasair_diasources_table,
        lasair_pair_query_row_limit=args.lasair_pair_query_limit,
        lasair_one_row_per_source=not args.all_window_dia_sources,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line interface for this module.

    Returns
    -------
    argparse.ArgumentParser
        Parser with experiment and plotting options.
    """
    p = argparse.ArgumentParser(
        description="Lasair + ALeRCE SN cutout experiment (see README.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="Log file path (parent directory is created if missing).",
    )
    p.add_argument(
        "--console-log-level",
        default="INFO",
        help="Minimum level for messages echoed to stderr.",
    )
    p.add_argument(
        "--file-log-level",
        default="DEBUG",
        help="Minimum level for messages written to the log file.",
    )
    p.add_argument(
        "--lasair-token",
        default=None,
        help=f"Lasair API token (default: {DEFAULT_LASAIR_TOKEN_ENV} env var).",
    )
    p.add_argument(
        "--lasair-endpoint",
        default=DEFAULT_LASAIR_ENDPOINT,
        help="Lasair API base URL.",
    )
    p.add_argument(
        "--lasair-max-calls-per-hour",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Client-side cap on Lasair HTTP calls per rolling hour (query, object, POST "
            "object, authenticated cutouts). Default: "
            f"{DEFAULT_LASAIR_MAX_CALLS_PER_HOUR_ENV} env or {DEFAULT_LASAIR_MAX_CALLS_PER_HOUR} "
            "(Lasair standard tokens allow ~100/h). Use 0 to disable."
        ),
    )
    p.add_argument(
        "--lasair-client-cache",
        default=None,
        metavar="DIR",
        help=(
            "Writable directory for the Lasair client's on-disk JSON cache "
            f"(default: {DEFAULT_LASAIR_CLIENT_CACHE_ENV} env if set)."
        ),
    )
    sql_pairs = p.add_mutually_exclusive_group()
    sql_pairs.add_argument(
        "--lasair-sql-pairs",
        action="store_true",
        help=(
            "Build (diaObjectId, diaSourceId) pairs with one SQL join query when supported "
            "(LSST Lasair often omits the detection table; default is per-object object())."
        ),
    )
    sql_pairs.add_argument(
        "--no-lasair-sql-pairs",
        action="store_true",
        help="Disable SQL pair query even if LASAIR_SQL_PAIRS is set in the environment.",
    )
    p.add_argument(
        "--all-window-dia-sources",
        action="store_true",
        help=(
            "Emit one scan/plot row per in-window diaSource (legacy). Default is one row "
            "per diaObject, using the latest detection in the window for stamps."
        ),
    )
    p.add_argument(
        "--lasair-diasources-table",
        default=DEFAULT_LASAIR_DIASOURCES_TABLE,
        metavar="NAME",
        help=(
            "Lasair SQL table name for detections (joined to objects). "
            f"Override with {DEFAULT_LASAIR_DIASOURCES_TABLE_ENV}."
        ),
    )
    p.add_argument(
        "--lasair-pair-query-limit",
        type=int,
        default=None,
        metavar="N",
        help="LIMIT rows for the SQL pair query (default: min(10000, max(200, N_objects*50))).",
    )
    p.add_argument(
        "--delta-t",
        type=float,
        default=ExperimentConfig.delta_t,
        help="Keep diaSources with mjd_now - midpointMjdTai <= this value (days).",
    )
    p.add_argument(
        "--query-limit",
        type=int,
        default=ExperimentConfig.query_limit,
        help="SQL LIMIT for the detailed SN candidate query.",
    )
    p.add_argument(
        "--max-plots",
        type=int,
        default=ExperimentConfig.max_plots,
        help="Maximum stamp figures in the full pipeline.",
    )
    p.add_argument(
        "--simple-query-limit",
        type=int,
        default=ExperimentConfig.simple_query_limit,
        help="SQL LIMIT for --simple-query mode.",
    )
    p.add_argument(
        "--preview-rows",
        type=int,
        default=ExperimentConfig.preview_table_rows,
        help="Rows of the pair table to print before plotting.",
    )
    p.add_argument(
        "--lasair-tables",
        default=DEFAULT_OBJECTS_SHERLOCK_TABLES,
        help="Comma-separated Lasair tables for object + Sherlock joins.",
    )
    p.add_argument(
        "--sherlock-classification",
        default=DEFAULT_SHERLOCK_CLASSIFICATION,
        help='Sherlock classification filter (e.g. "SN").',
    )
    p.add_argument(
        "--distance-metric",
        choices=sorted(ALLOWED_SHERLOCK_DISTANCE_COLUMNS.keys()),
        default="luminosity",
        help=(
            "Sherlock column for distance bounds: luminosity=luminosity distance from z "
            "(Mpc), direct=direct_distance (Mpc), separation_kpc=physical_separation_kpc. "
            "See https://lasair.lsst.ac.uk/schema/ (sherlock_classifications)."
        ),
    )
    p.add_argument(
        "--distance-min",
        type=float,
        default=None,
        help="Minimum value for --distance-metric (inclusive); requires non-NULL column.",
    )
    p.add_argument(
        "--distance-max",
        type=float,
        default=None,
        help="Maximum value for --distance-metric (inclusive); requires non-NULL column.",
    )
    p.add_argument(
        "--simple-query-distance-filter",
        action="store_true",
        help="Apply --distance-min/--distance-max to --simple-query as well.",
    )
    p.add_argument(
        "--simple-query-recent",
        action="store_true",
        help=(
            "With --simple-query, also require mjdnow()-lastDiaSourceMjdTai <= --delta-t "
            "(same recency idea as the detailed query)."
        ),
    )
    p.add_argument(
        "--z-min",
        type=float,
        default=None,
        help="Minimum sherlock_classifications.z (inclusive); NULL z excluded.",
    )
    p.add_argument(
        "--z-max",
        type=float,
        default=None,
        help="Maximum sherlock_classifications.z (strict <); NULL z excluded.",
    )
    p.add_argument(
        "--bright-max-mag",
        type=float,
        default=None,
        help=(
            "Require AB magnitude < this in --brightness-band using objects.{band}_psfFlux "
            "(nJy); brighter = lower mag number (e.g. 22 keeps m < 22)."
        ),
    )
    p.add_argument(
        "--brightness-band",
        choices=sorted(ALLOWED_BRIGHTNESS_BAND_FLUX.keys()),
        default="r",
        help="Filter band for --bright-max-mag (Lasair objects table).",
    )
    p.add_argument(
        "--brightness-within-days",
        type=float,
        default=None,
        help=(
            "Require mjdnow() - {band}_latestMJD <= this many days when using "
            "--bright-max-mag (default: same as --delta-t)."
        ),
    )
    p.add_argument(
        "--alerce-survey",
        default=DEFAULT_ALERCE_SURVEY,
        help="Survey name passed to ALeRCE get_stamps.",
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep deterministic row order for diaObject/diaSource pairs.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed for pair shuffling (pandas sample).",
    )
    p.add_argument(
        "--plot-snr-vmin",
        type=float,
        default=PlotStyle.snr_vmin,
        help="Lower SNR color scale in the fourth panel.",
    )
    p.add_argument(
        "--plot-snr-vmax",
        type=float,
        default=PlotStyle.snr_vmax,
        help="Upper SNR color scale in the fourth panel.",
    )
    p.add_argument(
        "--crosshair-arm-fraction",
        type=float,
        default=PlotStyle.crosshair_arm_fraction,
        help="Crosshair arm length as a fraction of stamp width.",
    )
    p.add_argument(
        "--crosshair-gap-fraction",
        type=float,
        default=PlotStyle.crosshair_gap_fraction,
        help="Crosshair gap from center as a fraction of stamp width.",
    )
    p.add_argument(
        "--plotly-host",
        default="127.0.0.1",
        help="Bind address for --plotly-scan (Dash server).",
    )
    p.add_argument(
        "--plotly-port",
        type=int,
        default=8050,
        help="TCP port for --plotly-scan.",
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Maximum objects in --plotly-scan after filters (default: no cap).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--simple-query",
        action="store_true",
        help="Only run a small Lasair SN query and print objectId, ra, dec.",
    )
    mode.add_argument(
        "--single-demo",
        action="store_true",
        help="First SN candidate from the detailed query; one ALeRCE stamp plot.",
    )
    mode.add_argument(
        "--plotly-scan",
        action="store_true",
        help=(
            "Start a Dash/Plotly viewer: browse stamp cutouts with ← / → (focus the page) "
            "or Prev/Next; uses the same Lasair filters as the full pipeline."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, build clients, and dispatch to a workflow mode.

    Parameters
    ----------
    argv : list of str or None, optional
        If None, use ``sys.argv[1:]``.
    """
    args = build_arg_parser().parse_args(argv)
    try:
        console_level = parse_loglevel_name(args.console_log_level)
        file_level = parse_loglevel_name(args.file_log_level)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    configure_application_logging(
        Path(args.log_file),
        console_level=console_level,
        file_level=file_level,
    )
    log = get_app_logger()
    mode = (
        "simple-query"
        if args.simple_query
        else "single-demo"
        if args.single_demo
        else "plotly-scan"
        if args.plotly_scan
        else "full-pipeline"
    )
    log.info("Run started (mode=%s)", mode)
    log.debug(
        "CLI snapshot: endpoint=%r delta_t=%s query_limit=%s log_file=%r",
        args.lasair_endpoint,
        args.delta_t,
        args.query_limit,
        args.log_file,
    )

    try:
        token = get_lasair_token(args.lasair_token)
        cfg = config_from_args(args)
        client = make_lasair_client(
            token,
            cfg.lasair_endpoint,
            max_calls_per_hour=cfg.lasair_max_calls_per_hour,
            cache_dir=cfg.lasair_client_cache_dir,
        )

        if args.plotly_scan:
            from plotly_scanner import (
                dataframe_to_scan_records,
                json_safe_scan_records_for_dash,
                run_plotly_scan_app,
            )

            pairs = load_transient_scan_pairs(client, cfg)
            if pairs.empty:
                log.warning("No Lasair source rows to scan; exiting.")
                return
            recs = dataframe_to_scan_records(pairs)
            alerce_client = Alerce()
            log.info("ALeRCE client initialized (bulk prefetch for Plotly scan)")
            mjd_now = Time.now().mjd
            recs = assign_best_alerce_stamp_detection_for_scan(
                alerce_client,
                recs,
                cfg.alerce_survey,
                mjd_now,
                cfg.delta_t,
            )
            if not recs:
                log.warning(
                    "No source rows after ALeRCE stamp/detection selection; exiting."
                )
                return
            recs = json_safe_scan_records_for_dash(recs)
            if args.max_scan is not None:
                recs = recs[: max(0, args.max_scan)]
            if not recs:
                log.warning("No records after --max-scan; exiting.")
                return
            prefetch = prefetch_alerce_plotly_scan_data(alerce_client, recs, cfg)
            log.info(
                "Starting Plotly scan: %d source row(s); preview from ALeRCE prefetch — "
                "open http://%s:%s/ in a browser",
                len(recs),
                args.plotly_host,
                args.plotly_port,
            )
            run_plotly_scan_app(
                cfg,
                recs,
                prefetch,
                host=args.plotly_host,
                port=args.plotly_port,
            )
        elif args.simple_query:
            run_simple_query(client, cfg)
        else:
            alerce_client = Alerce()
            log.info("ALeRCE client initialized")
            if args.single_demo:
                run_single_object_demo(client, alerce_client, cfg)
            else:
                run_full_pipeline(client, alerce_client, cfg)
        log.info("Run finished successfully (mode=%s)", mode)
    except Exception:
        log.exception("Run failed (mode=%s)", mode)
        raise


if __name__ == "__main__":
    main()
