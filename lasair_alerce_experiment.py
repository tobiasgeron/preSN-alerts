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
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

from alerce.core import Alerce
from lasair import lasair_client as lasair

# -----------------------------------------------------------------------------
# Defaults (CLI and :class:`ExperimentConfig` use these as factory defaults)
# -----------------------------------------------------------------------------

DEFAULT_LASAIR_ENDPOINT = "https://api.lasair.lsst.ac.uk/api"
DEFAULT_LASAIR_TOKEN_ENV = "LASAIR_API_TOKEN"

DEFAULT_OBJECTS_SHERLOCK_TABLES = "objects,sherlock_classifications"
DEFAULT_SHERLOCK_CLASSIFICATION = "SN"
DEFAULT_ALERCE_SURVEY = "lsst"


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
    simple_query_select: str = "objects.diaObjectId, objects.ra, objects.decl"
    alerce_survey: str = DEFAULT_ALERCE_SURVEY
    alerce_include_variance_and_mask: bool = True
    lasair_object_lasair_added: bool = False
    shuffle_pairs: bool = True
    random_state: int | None = None
    plot_style: PlotStyle = field(default_factory=PlotStyle)


# -----------------------------------------------------------------------------
# Lasair client and authentication
# -----------------------------------------------------------------------------


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
        If no token is available (message printed to stderr).
    """
    token = cli_token or os.environ.get(env_var)
    if not token or not str(token).strip():
        print(
            f"Missing Lasair API token. Set {env_var} or pass --lasair-token.",
            file=sys.stderr,
        )
        sys.exit(1)
    return str(token).strip()


def make_lasair_client(token: str, endpoint: str) -> Any:
    """
    Construct a Lasair API client.

    Parameters
    ----------
    token : str
        Lasair API token.
    endpoint : str
        API base URL.

    Returns
    -------
    object
        Client instance returned by ``lasair_client.lasair`` (typed as ``Any``).
    """
    return lasair(token, endpoint=endpoint)


# -----------------------------------------------------------------------------
# SQL fragments
# -----------------------------------------------------------------------------


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
) -> str:
    """
    Build a WHERE clause for recent Sherlock-classified transients.

    Parameters
    ----------
    delta_t : float
        Upper bound on ``mjdnow() - lastDiaSourceMjdTai`` (Lasair SQL units).
    classification : str
        Sherlock classification to match (quoted in SQL).
    objects_table, sherlock_table : str, optional
        Table names used in the join.

    Returns
    -------
    str
        SQL condition string.
    """
    classification = _assert_safe_sql_token(classification, "classification")
    return f"""
    {objects_table}.diaObjectId={sherlock_table}.diaObjectId
  AND {sherlock_table}.classification IN ("{classification}")
  AND mjdnow() - {objects_table}.lastDiaSourceMjdTai <= {delta_t}
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


# -----------------------------------------------------------------------------
# Object / diaSource helpers
# -----------------------------------------------------------------------------


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
    conditions = build_conditions_sn_recent(
        config.delta_t,
        config.sherlock_classification,
    )
    return client.query(
        config.sn_candidate_select,
        config.lasair_objects_sherlock_tables,
        conditions,
        limit=config.query_limit,
    )


def collect_dia_object_source_pairs(
    client: Any,
    dia_object_ids: Sequence[int],
    mjd_now: float,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """
    For each diaObject, append (diaObjectId, diaSourceId) rows for recent diaSources.

    Parameters
    ----------
    client : object
        Lasair API client.
    dia_object_ids : sequence of int
        Candidate object IDs from :func:`query_sn_candidates`.
    mjd_now : float
        Reference MJD for recency filtering.
    config : ExperimentConfig
        ``delta_t`` and ``lasair_object_lasair_added``; optional shuffle settings.

    Returns
    -------
    pandas.DataFrame
        Columns ``diaObjectId`` and ``diaSourceId``. Empty if no pairs qualify.
    """
    rows: list[dict[str, int]] = []
    for i in tqdm(range(len(dia_object_ids)), desc="Lasair objects"):
        oid = int(dia_object_ids[i])
        obj_result = client.object(oid, lasair_added=config.lasair_object_lasair_added)
        for sid in dia_source_ids_within_delta_t(obj_result, mjd_now, config.delta_t):
            rows.append({"diaObjectId": oid, "diaSourceId": int(sid)})
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    if config.shuffle_pairs:
        return data.sample(frac=1.0, random_state=config.random_state).reset_index(
            drop=True
        )
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
    st = style or PlotStyle()
    calexp_cutout = np.asarray(calexp[0].data)
    template_cutout = np.asarray(template[0].data)
    diff_cutout = np.asarray(diff[0].data)
    variance = np.asarray(diff[1].data)
    snr = diff_cutout / np.sqrt(variance)

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
    if show:
        plt.show()
    return fig


def fetch_and_plot_stamps(
    alerce_client: Alerce,
    dia_object_id: int,
    dia_source_id: int,
    config: ExperimentConfig,
) -> Figure:
    """
    Request stamps from ALeRCE and render :func:`plot_main_figure`.

    Parameters
    ----------
    alerce_client : Alerce
        Configured ALeRCE client.
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
    cutouts = alerce_client.get_stamps(
        oid=dia_object_id,
        measurement_id=dia_source_id,
        survey=config.alerce_survey,
        include_variance_and_mask=config.alerce_include_variance_and_mask,
    )
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
    Run a minimal Lasair query and print diaObjectId, RA, and Dec.

    Parameters
    ----------
    client : object
        Lasair client.
    config : ExperimentConfig
        SELECT list, tables, classification, and ``simple_query_limit``.
    """
    conditions = build_conditions_simple_sn(config.sherlock_classification)
    results = client.query(
        config.simple_query_select,
        config.lasair_objects_sherlock_tables,
        conditions,
        limit=config.simple_query_limit,
    )
    for row in results:
        print(row["diaObjectId"], row["ra"], row["decl"])


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
    mjd_now = Time.now().mjd
    results = query_sn_candidates(client, config)
    if not results:
        print("No objects returned from Lasair query.")
        return
    dia_object_ids = [results[i]["diaObjectId"] for i in range(len(results))]
    dia_object_id = int(dia_object_ids[0])
    obj_result = client.object(
        dia_object_id, lasair_added=config.lasair_object_lasair_added
    )
    ids = dia_source_ids_within_delta_t(obj_result, mjd_now, config.delta_t)
    if ids.size == 0:
        print(
            "No diaSources within delta_t for the first object; try a larger --delta-t."
        )
        return
    fetch_and_plot_stamps(alerce_client, dia_object_id, int(ids[0]), config)


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
    mjd_now = Time.now().mjd
    results = query_sn_candidates(client, config)
    if not results:
        print("No objects returned from Lasair query.")
        return
    dia_object_ids = [int(results[i]["diaObjectId"]) for i in range(len(results))]
    data = collect_dia_object_source_pairs(client, dia_object_ids, mjd_now, config)
    if data.empty:
        print("No diaObject/diaSource pairs within delta_t.")
        return
    n_preview = min(config.preview_table_rows, len(data))
    print(data.head(n_preview).to_string())
    n_plot = min(config.max_plots, len(data))
    for i in range(n_plot):
        fetch_and_plot_stamps(
            alerce_client,
            int(data["diaObjectId"].iloc[i]),
            int(data["diaSourceId"].iloc[i]),
            config,
        )


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
    token = get_lasair_token(args.lasair_token)
    cfg = config_from_args(args)
    client = make_lasair_client(token, cfg.lasair_endpoint)
    alerce_client = Alerce()

    if args.simple_query:
        run_simple_query(client, cfg)
        return
    if args.single_demo:
        run_single_object_demo(client, alerce_client, cfg)
        return
    run_full_pipeline(client, alerce_client, cfg)


if __name__ == "__main__":
    main()
