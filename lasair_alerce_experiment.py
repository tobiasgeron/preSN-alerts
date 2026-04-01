#!/usr/bin/env python3
"""
Lasair LSST API + ALeRCE cutout experiment (refactored from notebook/Lasair_and_alerce_experiment.ipynb).

Query Sherlock SN-classified objects from Lasair, resolve recent diaSources, then fetch
science/template/difference stamps from ALeRCE and plot them.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

from alerce.core import Alerce
from lasair import lasair_client as lasair

DEFAULT_LASAIR_ENDPOINT = "https://api.lasair.lsst.ac.uk/api"

SELECT_SN_FIELDS = """
        objects.diaObjectId,
       objects.ra,
       objects.decl,
       mjdnow()-objects.lastDiaSourceMjdTai AS since,
       objects.lastDiaSourceMjdTai,
       objects.latestR,
       objects.nDiaSources,
       sherlock_classifications.classification,
       sherlock_classifications.association_type,
       sherlock_classifications.distance,
       sherlock_classifications.z,
       sherlock_classifications.classificationReliability,
       sherlock_classifications.major_axis_arcsec,
       sherlock_classifications.separationArcsec
"""

TABLES_OBJECTS_SHERLOCK = "objects,sherlock_classifications"


def get_lasair_token(cli_token: str | None) -> str:
    token = cli_token or os.environ.get("LASAIR_API_TOKEN")
    if not token or not str(token).strip():
        print(
            "Missing Lasair API token. Set LASAIR_API_TOKEN or pass --lasair-token.",
            file=sys.stderr,
        )
        sys.exit(1)
    return str(token).strip()


def make_lasair_client(token: str, endpoint: str) -> Any:
    return lasair(token, endpoint=endpoint)


def conditions_sn_recent(delta_t: float) -> str:
    return f"""
    objects.diaObjectId=sherlock_classifications.diaObjectId
  AND objects.diaObjectId=sherlock_classifications.diaObjectId
  AND sherlock_classifications.classification IN ("SN")
  AND mjdnow() - objects.lastDiaSourceMjdTai <= {delta_t}
  """


def query_sn_candidates(
    L: Any, delta_t: float, limit: int
) -> list[dict[str, Any]]:
    conditions = conditions_sn_recent(delta_t)
    return L.query(SELECT_SN_FIELDS, TABLES_OBJECTS_SHERLOCK, conditions, limit=limit)


def collect_dia_object_source_pairs(
    L: Any,
    dia_object_ids: list[int],
    mjd_now: float,
    delta_t: float,
) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    for i in tqdm(range(len(dia_object_ids)), desc="Lasair objects"):
        dia_object_id = dia_object_ids[i]
        obj_result = L.object(dia_object_id, lasair_added=False)
        src_list = obj_result["diaSourcesList"]
        dia_source_ids = np.array([src_list[ii]["diaSourceId"] for ii in range(len(src_list))])
        mjds = np.array([src_list[ii]["midpointMjdTai"] for ii in range(len(src_list))])
        time_since_mjd = mjd_now - mjds
        idx_keep = np.where(time_since_mjd <= delta_t)[0]
        dia_source_ids = dia_source_ids[idx_keep]
        for j in range(len(dia_source_ids)):
            rows.append(
                {"diaObjectId": int(dia_object_id), "diaSourceId": int(dia_source_ids[j])}
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    return data.sample(frac=1).reset_index(drop=True)


def draw_crosshair(ax, x, y, L=0.2, gap=0.05, **kwargs):
    ax.plot([x - L - gap, x - gap], [y, y], **kwargs)
    ax.plot([x + gap, x + L + gap], [y, y], **kwargs)
    ax.plot([x, x], [y - L - gap, y - gap], **kwargs)
    ax.plot([x, x], [y + gap, y + L + gap], **kwargs)


def plot_main_figure(
    calexp,
    template,
    diff,
    zoom_size=30,
    ncol=4,
    nrow=1,
    col="red",
    titles_fontsize=22,
    figsize=(4, 4.3),
    scale="asinh",
):
    """
    calexp, template, and diff should be HDU lists / fits, with [0] data, [1] variance,
    [2] unused (e.g. PSF mask in API docs).
    """
    calexp_cutout = calexp[0].data
    template_cutout = template[0].data
    diff_cutout = diff[0].data
    snr = diff[0].data / np.sqrt(diff[1].data)

    zoom_size = calexp_cutout.shape[0]
    x = zoom_size / 2
    y = zoom_size / 2

    vmin_p = 30
    vmax_p = 99

    plt.figure(figsize=(figsize[0] * ncol, figsize[1] * nrow))

    plt.subplot(nrow, ncol, 1)
    vmin = np.percentile(template_cutout, vmin_p)
    vmax = np.percentile(template_cutout, vmax_p)
    plt.imshow(
        calexp_cutout,
        origin="lower",
        norm=scale,
        cmap="grey",
        vmin=vmin,
        vmax=vmax,
    )
    draw_crosshair(
        plt.gca(), x, y, L=zoom_size / 10, gap=zoom_size / 20, color=col, linewidth=2
    )
    plt.title("Science Image", fontsize=titles_fontsize)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(nrow, ncol, 2)
    vmin = np.percentile(template_cutout, vmin_p)
    vmax = np.percentile(template_cutout, vmax_p)
    plt.imshow(
        template_cutout,
        origin="lower",
        norm=scale,
        cmap="grey",
        vmin=vmin,
        vmax=vmax,
    )
    draw_crosshair(
        plt.gca(), x, y, L=zoom_size / 10, gap=zoom_size / 20, color=col, linewidth=2
    )
    plt.title("Template Image", fontsize=titles_fontsize)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(nrow, ncol, 3)
    vmin = np.percentile(diff_cutout, 1)
    vmax = np.percentile(diff_cutout, 99)
    plt.imshow(
        diff_cutout,
        origin="lower",
        norm="linear",
        cmap="grey",
        vmin=vmin,
        vmax=vmax,
    )
    draw_crosshair(
        plt.gca(), x, y, L=zoom_size / 10, gap=zoom_size / 20, color=col, linewidth=2
    )
    plt.title("Difference Image", fontsize=titles_fontsize)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(nrow, ncol, 4)
    plt.imshow(
        snr,
        origin="lower",
        norm="linear",
        cmap="viridis",
        vmin=0,
        vmax=5,
    )
    draw_crosshair(
        plt.gca(), x, y, L=zoom_size / 10, gap=zoom_size / 20, color=col, linewidth=2
    )
    plt.title("SNR Map", fontsize=titles_fontsize)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()


def run_simple_query(L: Any, limit: int) -> None:
    selected = "objects.diaObjectId, objects.ra, objects.decl"
    tables = TABLES_OBJECTS_SHERLOCK
    conditions = 'classification="SN"'
    results = L.query(selected, tables, conditions, limit=limit)
    for row in results:
        print(row["diaObjectId"], row["ra"], row["decl"])


def run_single_object_demo(
    L: Any,
    alerce_client: Alerce,
    delta_t: float,
    query_limit: int,
) -> None:
    mjd_now = Time.now().mjd
    results = query_sn_candidates(L, delta_t, limit=query_limit)
    if not results:
        print("No objects returned from Lasair query.")
        return
    dia_object_ids = [results[i]["diaObjectId"] for i in range(len(results))]
    dia_object_id = dia_object_ids[0]
    obj_result = L.object(dia_object_id, lasair_added=False)
    src_list = obj_result["diaSourcesList"]
    dia_source_ids = np.array([src_list[i]["diaSourceId"] for i in range(len(src_list))])
    mjds = np.array([src_list[i]["midpointMjdTai"] for i in range(len(src_list))])
    time_since_mjd = mjd_now - mjds
    idx_keep = np.where(time_since_mjd <= delta_t)[0]
    dia_source_ids = dia_source_ids[idx_keep]
    if len(dia_source_ids) == 0:
        print("No diaSources within delta_t for the first object; try a larger --delta-t.")
        return
    dia_source_id = int(dia_source_ids[0])
    cutouts = alerce_client.get_stamps(
        oid=dia_object_id,
        measurement_id=dia_source_id,
        survey="lsst",
        include_variance_and_mask=True,
    )
    plot_main_figure(
        cutouts["cutoutScience"],
        cutouts["cutoutTemplate"],
        cutouts["cutoutDifference"],
    )


def run_full_pipeline(
    L: Any,
    alerce_client: Alerce,
    delta_t: float,
    query_limit: int,
    max_plots: int,
) -> None:
    mjd_now = Time.now().mjd
    results = query_sn_candidates(L, delta_t, limit=query_limit)
    if not results:
        print("No objects returned from Lasair query.")
        return
    dia_object_ids = [results[i]["diaObjectId"] for i in range(len(results))]
    data = collect_dia_object_source_pairs(L, dia_object_ids, mjd_now, delta_t)
    if data.empty:
        print("No diaObject/diaSource pairs within delta_t.")
        return
    print(data.head(10).to_string())
    n = min(max_plots, len(data))
    for i in range(n):
        dia_object_id = int(data["diaObjectId"].iloc[i])
        dia_source_id = int(data["diaSourceId"].iloc[i])
        cutouts = alerce_client.get_stamps(
            oid=dia_object_id,
            measurement_id=dia_source_id,
            survey="lsst",
            include_variance_and_mask=True,
        )
        plot_main_figure(
            cutouts["cutoutScience"],
            cutouts["cutoutTemplate"],
            cutouts["cutoutDifference"],
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lasair + ALeRCE SN cutout experiment (see README.md).",
    )
    p.add_argument(
        "--lasair-token",
        default=None,
        help="Lasair API token (default: LASAIR_API_TOKEN env var).",
    )
    p.add_argument(
        "--lasair-endpoint",
        default=DEFAULT_LASAIR_ENDPOINT,
        help=f"Lasair API base URL (default: {DEFAULT_LASAIR_ENDPOINT}).",
    )
    p.add_argument(
        "--delta-t",
        type=float,
        default=1.0,
        help="Max days since last diaSource (mjdnow - lastDiaSourceMjdTai), same units as notebook.",
    )
    p.add_argument(
        "--query-limit",
        type=int,
        default=10,
        help="Lasair SQL LIMIT for SN candidate query.",
    )
    p.add_argument(
        "--max-plots",
        type=int,
        default=10,
        help="Maximum cutout figures to display (full pipeline only).",
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
    args = build_arg_parser().parse_args(argv)
    token = get_lasair_token(args.lasair_token)
    L = make_lasair_client(token, args.lasair_endpoint)
    alerce_client = Alerce()

    if args.simple_query:
        run_simple_query(L, limit=8)
        return
    if args.single_demo:
        run_single_object_demo(L, alerce_client, args.delta_t, args.query_limit)
        return
    run_full_pipeline(
        L,
        alerce_client,
        args.delta_t,
        args.query_limit,
        args.max_plots,
    )


if __name__ == "__main__":
    main()
