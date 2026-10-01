# A321 Fuel Consumption Prediction and Cruise Fuel-Efficiency Analysis

This repository contains the final R project for the HarvardX/edX Choose Your Own Project assignment.

## Files

- `A321_Fuel_Optimization.R` — complete project pipeline in one R script.
- `A321_Fuel_Optimization.Rmd` — English report source.
- `A321_Fuel_Optimization.pdf` — knit this from the Rmd before submission.

## Dataset

The script automatically downloads the required training data from the official Zenodo record:

https://zenodo.org/records/19184662

Only the files required by this project are downloaded:

- `flightlist_train.parquet`
- `fuel_train.parquet`
- `flights_train.zip`

The trajectory archive is large (about 3.3 GB), so the first run can take time and requires several GB of free disk space.

## Run

Run from the repository root:

    Rscript A321_Fuel_Optimization.R

The script creates the required intermediate data files, model results, diagnostics, and figures under `data/`.

Then open `A321_Fuel_Optimization.Rmd` in RStudio and knit it to PDF.

## Reproducibility

All project file paths are relative. Missing R packages are installed automatically. The download block checks the Zenodo MD5 checksums before the analysis starts.
