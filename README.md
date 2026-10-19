# A321 Fuel Consumption Prediction and Cruise Fuel-Efficiency Analysis

![R](https://img.shields.io/badge/Language-R-blue.svg)
![Course](https://img.shields.io/badge/edX-HarvardX%20Data%20Science-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the final R project submission for the **HarvardX/edX Data Science Professional Certificate** (*Choose Your Own Project* assignment). The project focuses on predicting fuel consumption and analyzing fuel efficiency during the cruise phase for Airbus A321 flights.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [System Requirements](#-system-requirements)
- [How to Run](#-how-to-run)
- [Reproducibility](#-reproducibility)
- [License](#-license)

---

## ✈️ Overview
The primary objective of this project is to model and analyze fuel burn dynamics of Airbus A321 aircraft. By combining flight trajectory data and operational variables, the pipeline assesses fuel consumption behavior, identifies key efficiency drivers during cruise altitude, and evaluates machine learning models for predictive accuracy.

---

## 📁 Project Structure

```text
.
├── A321_Fuel_Optimization.R     # Complete data processing & modeling pipeline
├── A321_Fuel_Optimization.Rmd   # R Markdown source report (English)
├── A321_Fuel_Optimization.pdf   # Compiled final PDF report
├── README.md                    # Project documentation
└── data/                        # Generated data, models, diagnostics & figures
```

### File Details
- **`A321_Fuel_Optimization.R`**: An all-in-one R script that handles data downloading, checksum verification, cleaning, feature engineering, model training, evaluation, and plot generation.
- **`A321_Fuel_Optimization.Rmd`**: The structured project report containing detailed methodology, literature background, exploratory data analysis (EDA), model comparisons, and insights.
- **`A321_Fuel_Optimization.pdf`**: Final compiled report generated from the Rmd file.

---

## 📊 Dataset
The pipeline automatically fetches the training datasets directly from the official **Zenodo** repository:
👉 [Zenodo Record #19184662](https://zenodo.org/records/19184662)

The script downloads only the required files:
- `flightlist_train.parquet` — Flight metadata and schedule records.
- `fuel_train.parquet` — Target fuel burn records.
- `flights_train.zip` — High-resolution flight trajectory archives (~3.3 GB).

---

## 💻 System Requirements
Due to the size of the flight trajectory archive, ensure your system meets the following recommendations:
- **RAM:** Minimum 8 GB (16 GB recommended).
- **Disk Space:** At least 10–15 GB of free disk space for downloading, unzipping, and storing intermediate data files.
- **R Version:** R 4.0 or higher.

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/expertdust/A321-Fuel-Consumption.git](https://github.com/expertdust/A321-Fuel-Consumption.git)
   cd A321-Fuel-Consumption
   ```

2. **Execute the R Pipeline:**
   Run the full pipeline script from the repository root:
   ```bash
   Rscript A321_Fuel_Optimization.R
   ```
   *Note: On the first run, downloading and processing the trajectory data may take some time depending on your internet connection and disk speed.*

3. **Generate the PDF Report:**
   Open `A321_Fuel_Optimization.Rmd` in **RStudio** and click **Knit** (or run `rmarkdown::render("A321_Fuel_Optimization.Rmd")`) to compile `A321_Fuel_Optimization.pdf`.

---

## 🔄 Reproducibility
- **Relative File Paths:** All paths in the codebase are strictly relative to the repository root.
- **Automatic Package Management:** Missing R dependencies/libraries are automatically detected and installed upon running `A321_Fuel_Optimization.R`.
- **Integrity Check:** The download block verifies dataset integrity using MD5 checksums from Zenodo before launching the analysis.

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
