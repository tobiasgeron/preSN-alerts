# preSN-alerts — Lasair and ALeRCE experiment

This repository contains a small experiment that queries the [Lasair LSST](https://lasair.lsst.ac.uk/) API for objects Sherlock classifies as supernovae, then pulls science, template, and difference image stamps from the [ALeRCE](https://alerce.science/) broker for matching `diaObjectId` / `diaSourceId` pairs.

The original exploratory notebook is `notebook/Lasair_and_alerce_experiment.ipynb`. The same workflow is available as a command-line script with configurable parameters.

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

You can pass the token on the command line instead of the environment variable if you prefer (`--lasair-token`).

For programmatic use, construct `ExperimentConfig` and optional `PlotStyle` in Python; public functions in `pre_sn_alerts.py` use NumPy-style docstrings.

## License

See [LICENSE](LICENSE).
