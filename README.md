# Anvil Pipeline

Automated daily update pipeline for **Anvil**, an integrated archive of Earth
events — seismic, space weather, oceanographic, and typhoon data (1990–2026).

- 📦 Dataset: [Anvil on Zenodo](https://doi.org/10.5281/zenodo.21422922)
- 🔁 This repo automates fetching the latest data from USGS, ISC, OMNI2, and
  JMA (typhoon best track), and appends it to the archive on a daily schedule.

## What this does

1. Pulls new earthquake events (USGS FDSN) since the last run
2. Matches nearby space-weather and typhoon conditions within a 7-day window
3. Appends the results as versioned Parquet files
4. Runs automatically via GitHub Actions (see `.github/workflows/`)

## Stack

- Python (pandas, requests)
- GitHub Actions (scheduled runs, no server required)
- Parquet for storage

## License

Code in this repository is licensed under the MIT License.
The Anvil dataset itself is licensed separately under **CC BY 4.0** — see the
[Zenodo record](https://doi.org/10.5281/zenodo.21422922) for details.

## Status

🚧 In progress — core dataset published; automation pipeline under active
development.
