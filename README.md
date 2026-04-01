# preSN-alerts — Lasair and ALeRCE experiment

This repository contains a small experiment that queries the [Lasair LSST](https://lasair.lsst.ac.uk/) API for objects Sherlock classifies as supernovae, then pulls science, template, and difference image stamps from the [ALeRCE](https://alerce.science/) broker for matching `diaObjectId` / `diaSourceId` pairs.

The original exploratory notebook is `notebook/Lasair_and_alerce_experiment.ipynb`. The same workflow is available as a command-line script with configurable parameters.

### Sherlock distances (Lasair)

Lasair exposes host / transient geometry and distances on **`sherlock_classifications`** (see the [Lasair LSST schema browser](https://lasair.lsst.ac.uk/schema/#sherlock_classifications-schema)):

| Column | Meaning (from schema) |
|--------|------------------------|
| `distance` | Luminosity distance from spectral redshift (**Mpc**) |
| `direct_distance` | Distance from non-redshift methods, e.g. standard candles (**Mpc**) |
| `physical_separation_kpc` | Projected separation between transient and host (**kpc**) |

The script selects these in the detailed candidate query and can **filter** in SQL with `--distance-min` / `--distance-max` and `--distance-metric` (`luminosity`, `direct`, or `separation_kpc`). Use `--simple-query-distance-filter` to apply the same bounds in `--simple-query` mode.

**Redshift:** Sherlock’s **`sherlock_classifications.z`** is the host redshift (schema browser). Use **`--z-min`** / **`--z-max`** to filter in SQL. **`--z-max 0.03`** applies a strict **`z < 0.03`** (rows with NULL `z` are excluded when you set either bound).

**Brightness:** Lasair stores latest per-band PSF fluxes as **`objects.{u,g,r,i,z,y}_psfFlux`** in **nanojanskys** ([objects table](https://lasair.lsst.ac.uk/schema/#objects-schema)), with **`{band}_latestMJD`** for the epoch of that flux. The script converts a magnitude limit using **AB mag ≈ 31.4 − 2.5 log10(flux_nJy)** (Rubin-style nJy zeropoint). **`--bright-max-mag 22`** means **AB magnitude < 22** (brighter than 22nd mag). **`--brightness-within-days`** requires **`mjdnow() - {band}_latestMJD`** ≤ that value (default: same as **`--delta-t`**) so the bright measurement is recent.

### Example: past 30 days, z < 0.03, brighter than mag 22

`--delta-t` applies to **`mjdnow() - objects.lastDiaSourceMjdTai`** (the object still has a detection within that many days), which matches how the detailed query defines “recent.” It is **not** the same as “first seen in the last 30 days” (that would use `firstDiaSourceMjdTai` in custom SQL).

**List candidates only** (logs positions, distances, and `z`; increase `--simple-query-limit` as needed):

```bash
python pre_sn_alerts.py --simple-query \
  --simple-query-recent \
  --delta-t 30 \
  --z-max 0.03 \
  --bright-max-mag 22 \
  --brightness-band r \
  --simple-query-limit 2000
```

(`--brightness-band` defaults to `r`; omit it if the default is fine. With `--delta-t 30` and no `--brightness-within-days`, the bright flux must satisfy **`mjdnow() − r_latestMJD ≤ 30`** as well.)

Default Sherlock class is **`SN`**. For other classes (e.g. `VS`, `AGN`), set `--sherlock-classification`. There is no single switch for “every” Sherlock type; use the class you care about or build a custom Lasair filter.

**Full pipeline** (same Lasair cuts, then ALeRCE stamps; cap plots):

```bash
python pre_sn_alerts.py \
  --delta-t 30 \
  --z-max 0.03 \
  --bright-max-mag 22 \
  --query-limit 200 \
  --max-plots 5
```

## Requirements

- [Conda](https://docs.conda.io/) (or [Mamba](https://mamba.readthedocs.io/)) with the `conda-forge` channel.

Python dependencies are declared in `environment.yml`: the scientific stack is installed from **conda-forge**; the Lasair and ALeRCE API clients are installed with **pip** inside that environment (supported by `conda env create` and `conda install pip`).

## Setup

```bash
conda env create -f environment.yml
conda activate presn-alerts
```

Obtain a Lasair API token from your Lasair account and export it:

```bash
export LASAIR_API_TOKEN="your-token-here"
```

## Logging

Progress and results are sent through Python’s `logging` module (not raw `print`). By default the script writes to `logs/pre_sn_alerts.log` (the directory is created automatically) and echoes messages at `INFO` and above to stderr.

| Option | Role |
|--------|------|
| `--log-file` | Path to the log file |
| `--console-log-level` | Level for stderr (`DEBUG`, `INFO`, `WARNING`, …) |
| `--file-log-level` | Level for the file (often `DEBUG` for full detail) |

`tqdm` progress for Lasair object fetches is forwarded to the log at `DEBUG`. Use `--console-log-level DEBUG` if you want the same detail on the terminal.

Shared setup lives in `utils/utilities.py` (`configure_application_logging`, `get_app_logger`, etc.).

## Usage

Full pipeline (default): query recent SN candidates, build `(diaObjectId, diaSourceId)` pairs within `--delta-t` days, then plot up to `--max-plots` ALeRCE stamp sets.

```bash
python pre_sn_alerts.py
```

Other modes:

```bash
# Minimal Lasair query (object id, RA, Dec only)
python pre_sn_alerts.py --simple-query

# One object from the detailed query, single cutout figure
python pre_sn_alerts.py --single-demo
```

Useful options (defaults are listed in `python pre_sn_alerts.py --help`):

| Option | Meaning |
|--------|---------|
| `--lasair-token` | Lasair API token (else env `LASAIR_API_TOKEN`) |
| `--lasair-endpoint` | Lasair API base URL |
| `--lasair-tables` | Comma-separated tables for joins (default `objects,sherlock_classifications`) |
| `--sherlock-classification` | Sherlock class to match (default `SN`) |
| `--delta-t` | Keep diaSources with `mjd_now - midpointMjdTai` ≤ this value (days) |
| `--query-limit` | SQL `LIMIT` on the detailed SN candidate query |
| `--simple-query-limit` | SQL `LIMIT` for `--simple-query` |
| `--max-plots` | Maximum stamp figures in the full pipeline |
| `--preview-rows` | Rows of the pair table printed before plotting |
| `--alerce-survey` | Survey name for ALeRCE `get_stamps` (e.g. `lsst`) |
| `--no-shuffle` | Do not randomize `(diaObjectId, diaSourceId)` order |
| `--random-state` | Seed for shuffling when shuffle is enabled |
| `--plot-snr-vmin`, `--plot-snr-vmax` | SNR panel color scale |
| `--crosshair-arm-fraction`, `--crosshair-gap-fraction` | Crosshair size vs stamp width |
| `--distance-metric` | `luminosity` / `direct` / `separation_kpc` (Sherlock column for bounds) |
| `--distance-min`, `--distance-max` | Inclusive SQL bounds (Mpc or kpc per metric) |
| `--simple-query-distance-filter` | Apply distance bounds in `--simple-query` too |
| `--simple-query-recent` | With `--simple-query`, also apply `--delta-t` on `lastDiaSourceMjdTai` |
| `--z-min`, `--z-max` | Sherlock host redshift bounds (`z_max` is strict `<`; NULL `z` excluded) |
| `--bright-max-mag` | Require AB mag < this in `--brightness-band` (PSF flux in nJy, ZP ≈ 31.4) |
| `--brightness-band` | `u`/`g`/`r`/`i`/`z`/`y` for the flux/MJD columns (default `r`) |
| `--brightness-within-days` | `mjdnow - {band}_latestMJD` ≤ this (default: `--delta-t`) |

You can pass the token on the command line instead of the environment variable if you prefer (`--lasair-token`).

For programmatic use, construct `ExperimentConfig` and optional `PlotStyle` in Python; public functions in `pre_sn_alerts.py` use NumPy-style docstrings.

## License

See [LICENSE](LICENSE).
