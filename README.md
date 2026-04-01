# preSN-alerts — Lasair and ALeRCE experiment

This repository contains a small experiment that queries the [Lasair LSST](https://lasair.lsst.ac.uk/) API for objects Sherlock classifies as supernovae, then pulls science, template, and difference image stamps from the [ALeRCE](https://alerce.science/) broker for matching `diaObjectId` / `diaSourceId` pairs.

The original exploratory notebook is `Lasair_and_alerce_experiment.ipynb`. The same workflow is available as a command-line script with configurable parameters.

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

## Usage

Full pipeline (default): query recent SN candidates, build `(diaObjectId, diaSourceId)` pairs within `--delta-t` days, then plot up to `--max-plots` ALeRCE stamp sets.

```bash
python lasair_alerce_experiment.py
```

Other modes:

```bash
# Minimal Lasair query (object id, RA, Dec only)
python lasair_alerce_experiment.py --simple-query

# One object from the detailed query, single cutout figure
python lasair_alerce_experiment.py --single-demo
```

Useful options:

| Option | Default | Meaning |
|--------|---------|---------|
| `--lasair-token` | `$LASAIR_API_TOKEN` | Lasair API token |
| `--lasair-endpoint` | `https://api.lasair.lsst.ac.uk/api` | API base URL |
| `--delta-t` | `1` | Max days since last diaSource (same idea as the notebook) |
| `--query-limit` | `10` | SQL `LIMIT` on the SN candidate query |
| `--max-plots` | `10` | Cap on interactive figures (full pipeline) |

You can pass the token on the command line instead of the environment variable if you prefer (`--lasair-token`).

## License

See [LICENSE](LICENSE).
