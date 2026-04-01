"""
Dash + Plotly transient scanner: Lasair selects candidates; preview uses prefetched ALeRCE data.

Candidate ``diaObjectId`` rows (with a representative ``diaSourceId`` per source) come
from Lasair (see ``pre_sn_alerts``).
Stamp FITS and forced photometry are downloaded in bulk from ALeRCE before the Dash server
starts; the UI only reads that in-memory cache (no per-view Lasair object calls).

Run via ``pre_sn_alerts.py --plotly-scan`` (see :func:`run_plotly_scan_app`).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.io.fits import HDUList
from astropy.time import Time
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash_extensions import Keyboard
from plotly.subplots import make_subplots

from pre_sn_alerts import (
    AlercePlotlyPrefetch,
    ExperimentConfig,
    LSST_AB_MAG_ZP_NJY,
    PlotStyle,
    get_app_logger,
)

# JavaScript JSON and ``dcc.Store`` use IEEE-754 doubles; LSST IDs exceed ``MAX_SAFE_INTEGER``.
_JS_MAX_SAFE_INT = 9007199254740991


def _scan_id_to_int(value: Any) -> int:
    """Parse ``diaObjectId`` / ``diaSourceId`` from Dash store (str or int, full precision)."""
    if value is None:
        raise TypeError("scan id is None")
    if isinstance(value, str):
        return int(value.strip(), 10)
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, float):
        # Should not happen if we stringify for Dash; avoid silent corruption
        raise TypeError(f"refuse float scan id (precision loss): {value!r}")
    return int(value)


def dataframe_to_scan_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a scan table (sources + representative diaSource) to dicts for ``dcc.Store``."""
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for rec in df.replace({np.nan: None}).to_dict(orient="records"):
        row: dict[str, Any] = {}
        for k, v in rec.items():
            if v is None:
                row[k] = None
            elif isinstance(v, (np.integer, np.int64)):
                row[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                row[k] = None if np.isnan(v) else float(v)
            else:
                row[k] = v
        out.append(row)
    return out


def json_safe_scan_records_for_dash(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Encode LSST IDs as decimal strings so ``dcc.Store`` JSON survives the browser.

    Values beyond ``Number.MAX_SAFE_INTEGER`` are rounded by JavaScript if sent as numbers,
    which breaks prefetched stamp / forced-photometry lookups in callbacks.
    """
    out: list[dict[str, Any]] = []
    for r in records:
        row = dict(r)
        for key in ("diaObjectId", "diaSourceId"):
            if key not in row or row[key] is None:
                continue
            i = int(row[key])
            if abs(i) > _JS_MAX_SAFE_INT:
                row[key] = str(i)
            else:
                row[key] = i
        out.append(row)
    return out


def _title_for_row(row: Mapping[str, Any]) -> str:
    oid = row.get("diaObjectId")
    sid = row.get("diaSourceId")
    ra = row.get("ra")
    decl = row.get("decl")
    z = row.get("z")
    parts = [
        f"diaObjectId={oid}",
        f"diaSourceId={sid}",
    ]
    if ra is not None and decl is not None:
        try:
            parts.append(f"RA={float(ra):.6f}°, Dec={float(decl):.6f}°")
        except (TypeError, ValueError):
            parts.append(f"RA={ra}, Dec={decl}")
    if z is not None:
        try:
            parts.append(f"z={float(z):.5f}")
        except (TypeError, ValueError):
            parts.append(f"z={z}")
    return " · ".join(str(p) for p in parts)


def _flip_for_display(arr: np.ndarray) -> np.ndarray:
    """Match matplotlib ``origin='lower'`` in Plotly heatmaps."""
    return np.flipud(np.asarray(arr, dtype=float))


def _linear_vmin_vmax(data: np.ndarray, lo_p: float, hi_p: float) -> tuple[float, float]:
    flat = np.nan_to_num(data.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    if flat.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(flat, (lo_p, hi_p))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return float(vmin), float(vmax)


_FORCED_LC_DAY_WINDOW: float = 100.0
_FORCED_SNR_DET_THRESHOLD: float = 3.0

_BAND_COLORS: dict[str, str] = {
    "u": "#6b4dff",
    "g": "#2ca02c",
    "r": "#d62728",
    "i": "#ff7f0e",
    "z": "#9467bd",
    "y": "#7f7f7f",
}


def _pick_time_column(df: pd.DataFrame) -> str | None:
    preferred = ("mjd", "midpointmjdtai", "midpointMjdTai", "midPointTai", "epoch")
    lower_map = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    for c in df.columns:
        if "mjd" in c.lower():
            return c
    return None


def _stamp_primary_2d(hdul: Any) -> np.ndarray:
    """First HDU with non-empty 2D ``data`` (Lasair FITS may use a trivial primary)."""
    for hdu in _stamp_hdu_sequence(hdul):
        d = getattr(hdu, "data", None)
        if d is None:
            continue
        arr = np.asarray(d, dtype=float)
        if arr.ndim == 2 and arr.size > 0:
            return arr
    raise ValueError("No 2D image data in stamp HDUList")


def _stamp_hdu_sequence(hdul: Any) -> list[Any]:
    """``HDUList`` expanded; a single HDU (variance-off stamps) as a one-element list."""
    if isinstance(hdul, HDUList):
        return list(hdul)
    return [hdul]


def _stamp_primary_image_hdu(hdul: Any) -> Any | None:
    """First HDU with non-empty 2D image data (same selection as :func:`_stamp_primary_2d`)."""
    for hdu in _stamp_hdu_sequence(hdul):
        d = getattr(hdu, "data", None)
        if d is None:
            continue
        arr = np.asarray(d, dtype=float)
        if arr.ndim == 2 and arr.size > 0:
            return hdu
    return None


def _cutout_stamp_observation_label(cutouts: Mapping[str, Any]) -> str | None:
    """
    Short caption from science-stamp FITS headers: epoch and filter, when present.

    Tries common LSST / ALeRCE stamp header keys; returns ``None`` if nothing usable is found.
    """
    sci = cutouts.get("cutoutScience")
    if sci is None:
        return None
    hdu = _stamp_primary_image_hdu(sci)
    if hdu is None or not hasattr(hdu, "header"):
        return None
    hdr = hdu.header
    filt: str | None = None
    for key in ("FILTER", "FILTER1", "FILT", "BAND", "LC_FILTER", "FILTNAM"):
        if key not in hdr:
            continue
        raw = hdr[key]
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            filt = s
            break

    mjd_part: str | None = None
    for key in ("MJDOBS", "MJD-OBS", "MJD_OBS", "MJD", "AVRO_MSMT_MJD", "MIDPOINTMJDTAI"):
        if key not in hdr:
            continue
        raw = hdr[key]
        try:
            mjd = float(raw)
            mjd_part = f"MJD {mjd:.5f}"
            break
        except (TypeError, ValueError):
            s = str(raw).strip()
            if s:
                mjd_part = s
                break

    if mjd_part is None and "DATE-OBS" in hdr:
        raw = hdr["DATE-OBS"]
        try:
            t = Time(str(raw).strip())
            mjd_part = f"MJD {float(t.mjd):.5f}"
        except Exception:
            mjd_part = str(raw).strip() or None

    parts: list[str] = []
    if mjd_part:
        parts.append(mjd_part)
    if filt:
        parts.append(f"filter {filt}")
    if not parts:
        return None
    return " · ".join(parts)


def _flux_to_ab_mag_njy(flux: np.ndarray, flux_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.maximum(np.asarray(flux, dtype=float), 1e-12)
    fe = np.asarray(flux_err, dtype=float)
    mag = LSST_AB_MAG_ZP_NJY - 2.5 * np.log10(f)
    mag_err = (2.5 / np.log(10)) * np.abs(fe) / f
    return mag, mag_err


def _ab_mag_three_sigma_flux_upper_limit(flux_err: np.ndarray) -> np.ndarray:
    """AB magnitude at 3× flux uncertainty (nJy), i.e. a 3σ flux upper limit."""
    fe = np.maximum(np.asarray(flux_err, dtype=float), 1e-300)
    f_ul = np.maximum(3.0 * fe, 1e-300)
    return LSST_AB_MAG_ZP_NJY - 2.5 * np.log10(f_ul)


def _forced_photometry_traces(
    df: pd.DataFrame,
    survey: str,
) -> tuple[list[Any], str | None]:
    """
    Build Plotly scatter traces for forced photometry.

    For LSST AB magnitudes, points with ``|flux|/flux_err < 3`` are drawn as 3σ upper
    limits (downward triangles at ``m(ZP - 2.5 log10(3 σ))``) without error bars.

    Returns
    -------
    traces, warning
        ``warning`` is set if the light curve could not be built (empty message otherwise).
    """
    if df is None or df.empty:
        return [], "No forced photometry in Lasair object (diaForcedSourcesList empty)."
    tcol = _pick_time_column(df)
    if tcol is None:
        return [], "Forced photometry table has no MJD/time column."

    flux_col = "psfFlux" if "psfFlux" in df.columns else None
    err_col = "psfFluxErr" if "psfFluxErr" in df.columns else None
    if flux_col is None:
        for alt in ("scienceFlux", "science_flux", "fpFluxMean"):
            if alt in df.columns:
                flux_col = alt
                break
    if flux_col is None:
        return [], "Forced photometry has no psfFlux/scienceFlux column."
    if err_col is None:
        candidates = (
            "psfFluxErr",
            "scienceFluxErr",
            "science_flux_err",
            f"{flux_col}Err",
        )
        for alt in candidates:
            if alt in df.columns:
                err_col = alt
                break

    sub = df[[tcol, flux_col] + ([err_col] if err_col else [])].copy()
    if "band_name" in df.columns:
        sub["band_name"] = df["band_name"].values
    else:
        sub["band_name"] = "?"
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[tcol, flux_col])
    if sub.empty:
        return [], "No valid forced photometry points after cleaning."

    traces: list[Any] = []
    use_ab = survey.lower() == "lsst"
    for band, g in sub.groupby("band_name"):
        t = g[tcol].astype(float).values
        flux = g[flux_col].astype(float).values
        if err_col:
            fe = g[err_col].astype(float).values
        else:
            fe = np.zeros_like(flux)
        color = _BAND_COLORS.get(str(band).lower(), "#1f77b4")

        if use_ab and err_col is not None:
            fe_ok = np.isfinite(fe) & (fe > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.abs(flux) / np.where(fe_ok, fe, np.nan)
            is_det = fe_ok & (snr >= _FORCED_SNR_DET_THRESHOLD)

            if np.any(is_det):
                y_d, yerr_d = _flux_to_ab_mag_njy(flux[is_det], fe[is_det])
                traces.append(
                    go.Scatter(
                        x=t[is_det],
                        y=y_d,
                        mode="markers",
                        name=str(band),
                        marker=dict(size=6, color=color),
                        error_y=dict(type="data", array=yerr_d, thickness=1, width=2),
                        legendgroup=str(band),
                    )
                )
            is_lim = fe_ok & ~is_det
            if np.any(is_lim):
                m_ul = _ab_mag_three_sigma_flux_upper_limit(fe[is_lim])
                traces.append(
                    go.Scatter(
                        x=t[is_lim],
                        y=m_ul,
                        mode="markers",
                        name=f"{band} (3σ UL)",
                        marker=dict(
                            size=9,
                            color=color,
                            symbol="triangle-down",
                            line=dict(width=0.5, color="white"),
                        ),
                        legendgroup=str(band),
                    )
                )
            if not np.any(is_det) and not np.any(is_lim):
                y, _ = _flux_to_ab_mag_njy(flux, fe)
                traces.append(
                    go.Scatter(
                        x=t,
                        y=y,
                        mode="markers",
                        name=str(band),
                        marker=dict(size=6, color=color),
                        legendgroup=str(band),
                    )
                )
        elif use_ab:
            y, _ = _flux_to_ab_mag_njy(flux, fe)
            traces.append(
                go.Scatter(
                    x=t,
                    y=y,
                    mode="markers",
                    name=str(band),
                    marker=dict(size=6, color=color),
                    legendgroup=str(band),
                )
            )
        else:
            scatter_kw: dict[str, Any] = dict(
                x=t,
                y=flux,
                mode="markers",
                name=str(band),
                marker=dict(size=6, color=color),
                legendgroup=str(band),
            )
            if err_col is not None:
                scatter_kw["error_y"] = dict(type="data", array=fe, thickness=1, width=2)
            traces.append(go.Scatter(**scatter_kw))
    return traces, None


def _apply_light_curve_window_and_now(
    fig: go.Figure,
    mjd_now: float,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Set default MJD range to the past ``_FORCED_LC_DAY_WINDOW`` days and mark ``now``."""
    lo = float(mjd_now) - _FORCED_LC_DAY_WINDOW
    hi = float(mjd_now)
    subplot_kw = {"row": row, "col": col} if row is not None else {}
    fig.update_xaxes(range=[lo, hi], title_text="MJD", **subplot_kw)
    fig.add_vline(
        x=mjd_now,
        line_dash="dash",
        line_color="rgba(55,55,55,0.88)",
        line_width=1.5,
        annotation_text="now",
        annotation_position="top",
        annotation_font_size=11,
        **subplot_kw,
    )


def _legend_layout_above_yaxis(fig: go.Figure, yaxis_attr: str) -> dict[str, Any]:
    """Horizontal legend centered just above the given subplot's y-axis domain (paper coords)."""
    ya = getattr(fig.layout, yaxis_attr)
    top = float(ya.domain[1])
    return {
        "orientation": "h",
        "xref": "paper",
        "yref": "paper",
        "x": 0.5,
        "xanchor": "center",
        "y": top + 0.028,
        "yanchor": "bottom",
    }


def build_scan_figure(
    cutouts: Mapping[str, Any],
    style: PlotStyle,
    title: str,
    forced_df: pd.DataFrame | None,
    survey: str,
    *,
    mjd_now: float | None = None,
) -> go.Figure:
    """
    Stamps in the top row and forced-photometry light curve below (AB mag for LSST from nJy flux).

    ``mjd_now`` sets the dashed ``now`` line and default x-axis window (past
    ``_FORCED_LC_DAY_WINDOW`` days). Defaults to :func:`astropy.time.Time.now` MJD.
    """
    mjd_ref = float(mjd_now) if mjd_now is not None else float(Time.now().mjd)
    stamp_caption = _cutout_stamp_observation_label(cutouts)
    calexp = _stamp_primary_2d(cutouts["cutoutScience"])
    template = _stamp_primary_2d(cutouts["cutoutTemplate"])
    diff = _stamp_primary_2d(cutouts["cutoutDifference"])
    diff_stack = cutouts["cutoutDifference"]
    if len(diff_stack) > 1 and getattr(diff_stack[1], "data", None) is not None:
        variance = np.asarray(diff_stack[1].data, dtype=float)
    else:
        sigma = float(np.nanstd(diff)) or 1.0
        variance = np.full_like(diff, sigma**2, dtype=float)
    var_safe = np.maximum(variance, 1e-24)
    snr = diff / np.sqrt(var_safe)
    snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)

    vmin_st, vmax_st = _linear_vmin_vmax(template, style.vmin_percentile, style.vmax_percentile)
    vmin_df, vmax_df = _linear_vmin_vmax(
        diff, style.diff_vmin_percentile, style.diff_vmax_percentile
    )

    stamp_px = float(calexp.shape[0])
    cx = stamp_px / 2.0
    cy = stamp_px / 2.0
    arm = stamp_px * style.crosshair_arm_fraction
    gap = stamp_px * style.crosshair_gap_fraction

    panels: list[tuple[np.ndarray, str, float | None, float | None, str]] = [
        (_flip_for_display(calexp), style.science_cmap, vmin_st, vmax_st, style.panel_titles[0]),
        (_flip_for_display(template), style.template_cmap, vmin_st, vmax_st, style.panel_titles[1]),
        (_flip_for_display(diff), style.diff_cmap, vmin_df, vmax_df, style.panel_titles[2]),
        (
            _flip_for_display(snr),
            style.snr_cmap,
            style.snr_vmin,
            style.snr_vmax,
            style.panel_titles[3],
        ),
    ]

    fig = make_subplots(
        rows=2,
        cols=4,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.09,
        specs=[
            [
                {"type": "heatmap"},
                {"type": "heatmap"},
                {"type": "heatmap"},
                {"type": "heatmap"},
            ],
            [{"colspan": 4, "type": "scatter"}, None, None, None],
        ],
        subplot_titles=(
            style.panel_titles[0],
            style.panel_titles[1],
            style.panel_titles[2],
            style.panel_titles[3],
            (
                "Forced Photometry (AB mag)"
                if survey.lower() == "lsst"
                else "Forced Photometry (psfFlux)"
            ),
        ),
    )

    grey_cs = "Greys_r"
    for j, (zimg, cmap, vmin, vmax, _) in enumerate(panels, start=1):
        cs = grey_cs if cmap == "grey" else ("Viridis" if "viridis" in cmap.lower() else "Greys_r")
        fig.add_trace(
            go.Heatmap(
                z=zimg,
                colorscale=cs,
                zmin=vmin,
                zmax=vmax,
                showscale=False,
                showlegend=False,
            ),
            row=1,
            col=j,
        )
        fig.add_shape(
            type="line",
            x0=cx - gap - arm,
            x1=cx - gap,
            y0=cy,
            y1=cy,
            line=dict(color=style.crosshair_color, width=style.crosshair_linewidth),
            row=1,
            col=j,
        )
        fig.add_shape(
            type="line",
            x0=cx + gap,
            x1=cx + gap + arm,
            y0=cy,
            y1=cy,
            line=dict(color=style.crosshair_color, width=style.crosshair_linewidth),
            row=1,
            col=j,
        )
        fig.add_shape(
            type="line",
            x0=cx,
            x1=cx,
            y0=cy - gap - arm,
            y1=cy - gap,
            line=dict(color=style.crosshair_color, width=style.crosshair_linewidth),
            row=1,
            col=j,
        )
        fig.add_shape(
            type="line",
            x0=cx,
            x1=cx,
            y0=cy + gap,
            y1=cy + gap + arm,
            line=dict(color=style.crosshair_color, width=style.crosshair_linewidth),
            row=1,
            col=j,
        )

    for j in range(1, 5):
        xa = "xaxis" if j == 1 else f"xaxis{j}"
        ya = "yaxis" if j == 1 else f"yaxis{j}"
        y_anchor = "y" if j == 1 else f"y{j}"
        fig.layout[xa].update(scaleanchor=y_anchor, scaleratio=1, visible=False)
        fig.layout[ya].update(visible=False)

    lc_traces, lc_warn = _forced_photometry_traces(
        forced_df if forced_df is not None else pd.DataFrame(),
        survey,
    )
    if lc_traces:
        for tr in lc_traces:
            fig.add_trace(tr, row=2, col=1)
        y_title = "AB magnitude" if survey.lower() == "lsst" else "psfFlux"
        fig.update_yaxes(autorange="reversed", title_text=y_title, row=2, col=1)
        _apply_light_curve_window_and_now(fig, mjd_ref, row=2, col=1)
    else:
        msg = lc_warn or "No forced photometry"
        fig.add_annotation(
            text=msg,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.14,
            showarrow=False,
            font=dict(size=13),
            xanchor="center",
            yanchor="middle",
        )
        fig.update_xaxes(visible=False, row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)

    top_margin = 120 if stamp_caption else 100
    layout_kw: dict[str, Any] = dict(
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=top_margin, b=20),
        height=780,
        paper_bgcolor="white",
    )
    if lc_traces:
        fig.update_layout(**layout_kw, legend=_legend_layout_above_yaxis(fig, "yaxis5"))
    else:
        fig.update_layout(**layout_kw)

    if stamp_caption:
        y1_top = float(fig.layout.yaxis.domain[1])
        fig.add_annotation(
            text=stamp_caption,
            xref="paper",
            yref="paper",
            x=0.5,
            y=y1_top + 0.004,
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=12, color="#222"),
        )
    return fig


def run_plotly_scan_app(
    config: ExperimentConfig,
    records: list[dict[str, Any]],
    alerce_prefetch: AlercePlotlyPrefetch,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """
    Start a Dash app to step through stamp cutouts with arrow keys and Prev/Next buttons.

    ``alerce_prefetch`` must contain stamp FITS dicts and forced photometry for the same
    ``records`` (from :func:`pre_sn_alerts.prefetch_alerce_plotly_scan_data`).

    ``records`` must be JSON-serializable dicts with at least ``diaObjectId`` and
    ``diaSourceId``; optional ``ra``, ``decl``, ``z`` for the title line.
    """
    log = get_app_logger()
    if not records:
        log.warning("run_plotly_scan_app: no records; not starting server")
        return

    nmax = len(records) - 1
    style = config.plot_style
    survey = config.alerce_survey

    app = Dash(__name__)
    app.title = "preSN-alerts transient scan"

    app.layout = html.Div(
        [
            Keyboard(
                id="scan-keyboard",
                captureKeys=["ArrowLeft", "ArrowRight"],
            ),
            html.Div(
                [
                    html.H2("preSN-alerts · transient scan"),
                    html.P(
                        "Lasair candidate list; stamps and forced photometry from prefetched "
                        "ALeRCE data. Use ← / → to step (click the page if keys do nothing) "
                        "or the buttons below."
                    ),
                    html.Div(
                        [
                            html.Button("← Previous", id="btn-prev", n_clicks=0),
                            html.Button("Next →", id="btn-next", n_clicks=0, style={"marginLeft": "12px"}),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(id="scan-counter", style={"marginBottom": "8px"}),
                    dcc.Store(id="idx-store", data=0),
                    dcc.Store(id="nmax-store", data=nmax),
                    dcc.Store(id="meta-store", data=records),
                    dcc.Loading(
                        dcc.Graph(id="main-graph", style={"height": "820px"}),
                        type="circle",
                    ),
                ],
                style={"maxWidth": "1400px", "margin": "0 auto", "padding": "12px"},
            ),
        ]
    )

    @app.callback(
        Output("idx-store", "data"),
        Input("btn-next", "n_clicks"),
        Input("btn-prev", "n_clicks"),
        Input("scan-keyboard", "n_keydowns"),
        State("idx-store", "data"),
        State("nmax-store", "data"),
        State("scan-keyboard", "keydown"),
        prevent_initial_call=True,
    )
    def navigate(
        _n_next: int,
        _n_prev: int,
        _n_kbd: int,
        idx: int,
        nmax_val: int,
        keydown: dict[str, Any] | None,
    ) -> int:
        trig = callback_context.triggered_id
        if trig is None and callback_context.triggered:
            trig = callback_context.triggered[0]["prop_id"].split(".")[0]
        cur = int(idx)
        top = int(nmax_val)
        if trig == "btn-next":
            return min(top, cur + 1)
        if trig == "btn-prev":
            return max(0, cur - 1)
        if trig == "scan-keyboard":
            key = (keydown or {}).get("key")
            if key == "ArrowLeft":
                return max(0, cur - 1)
            if key == "ArrowRight":
                return min(top, cur + 1)
        return cur

    @app.callback(
        Output("main-graph", "figure"),
        Output("scan-counter", "children"),
        Input("idx-store", "data"),
        State("meta-store", "data"),
    )
    def update_graph(idx: int, meta: list[dict[str, Any]] | None):
        meta = meta or records
        i = int(idx) if idx is not None else 0
        i = max(0, min(i, len(meta) - 1))
        row = meta[i]
        oid = _scan_id_to_int(row["diaObjectId"])
        sid = _scan_id_to_int(row["diaSourceId"])
        title = _title_for_row(row)
        counter = f"Object {i + 1} of {len(meta)}"
        log.info(
            "Plotly scan: index=%d/%d diaObjectId=%s diaSourceId=%s",
            i,
            len(meta) - 1,
            oid,
            sid,
        )
        forced_df = alerce_prefetch.forced_by_oid.get(oid)
        if forced_df is None:
            forced_df = pd.DataFrame()
        cutouts: Mapping[str, Any] | None = alerce_prefetch.stamps.get((oid, sid))
        mjd_now = float(Time.now().mjd)

        if cutouts is not None:
            fig = build_scan_figure(
                cutouts, style, title, forced_df, survey, mjd_now=mjd_now
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text=f"{title} — ALeRCE stamps unavailable",
                    x=0.5,
                    xanchor="center",
                ),
                annotations=[
                    dict(
                        text=(
                            "No prefetched stamp bundle for this detection (ALeRCE get_stamps "
                            "failed or returned no cutouts). See startup prefetch logs."
                        ),
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.62,
                        showarrow=False,
                    )
                ],
            )
            lc_traces, _lc_warn = _forced_photometry_traces(
                forced_df if forced_df is not None else pd.DataFrame(),
                survey,
            )
            for tr in lc_traces:
                fig.add_trace(tr)
            if lc_traces:
                yax = "AB magnitude" if survey.lower() == "lsst" else "psfFlux"
                fig.update_layout(
                    yaxis=dict(autorange="reversed", title=yax),
                    height=520,
                    margin=dict(t=80),
                )
                _apply_light_curve_window_and_now(fig, mjd_now)
                fig.update_layout(legend=_legend_layout_above_yaxis(fig, "yaxis"))
        return fig, counter

    runner = getattr(app, "run", None)
    if callable(runner):
        runner(debug=debug, host=host, port=port)
    else:
        app.run_server(debug=debug, host=host, port=port)
