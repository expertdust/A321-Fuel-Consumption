# ========================================================================
# A321 FUEL CONSUMPTION PROJECT
# FINAL SUBMISSION R SCRIPT
#
# This single script contains the complete machine-learning project.
#
# Project workflow:
#   1. Dataset access and package setup
#   2. Dataset inspection
#   3. A321 flight and fuel-data matching
#   4. Data quality control
#   5. Feature engineering and modeling-data preparation
#   6. Exploratory data analysis and figures
#   7. Flight-level train/validation split
#   8. Linear regression
#   9. Random forest
#  10. Gradient boosting
#  11. Model comparison and robustness analysis
#  12. Random-forest hyperparameter tuning
#  13. Prediction diagnostics
#  14. Cruise fuel-efficiency what-if analysis
#  15. Model-based optimization validation
#  16. Robust cruise optimization
#  17. Out-of-sample optimization validation
#  18. Final summary and figures
#
# IMPORTANT:
# - Run this script from the PROJECT ROOT directory.
# - All project paths are relative (for example, data/...).
# - If the raw dataset is already present, it is not downloaded again.
# - The trajectory archive is about 3.3 GB and is downloaded only when
#   data/flights_train/ is missing or empty.
# - The code below is the original project pipeline consolidated into
#   one submission file rather than being split across step files.
# ========================================================================

options(stringsAsFactors = FALSE)
set.seed(2026)

# ------------------------------------------------------------------------
# 0. PROJECT ROOT CHECK
# ------------------------------------------------------------------------
if (!dir.exists("data")) {
  dir.create("data", recursive = TRUE, showWarnings = FALSE)
}

# The script is intended to be run from the project root.
# All data paths below are relative to the current working directory.
if (!file.exists("A321_Fuel_Optimization.Rmd") &&
    !file.exists("A321_Fuel_Optimization_submission_auto_download.R")) {
  message(
    "Run this script from the flight_sub project root so that all paths remain relative."
  )
}

# ------------------------------------------------------------------------
# 0. PACKAGE INSTALLATION
# ------------------------------------------------------------------------
# The course rubric requires automatic installation of missing packages.
# The packages are loaded with if(!require(...)) logic as requested.

required_packages <- c(
  "arrow",
  "readr",
  "dplyr",
  "ggplot2",
  "randomForest",
  "gbm"
)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(
      pkg,
      repos = "https://cloud.r-project.org"
    )
    library(pkg, character.only = TRUE)
  }
}

# ------------------------------------------------------------------------
# 0. DATASET ACCESS
# ------------------------------------------------------------------------
# Official source:
# Sun, J., Spinielli, E., & Strohmeier, M. (2026).
# Aircraft Fuel Burn Estimation Data Challenge.
# Zenodo record: https://zenodo.org/records/19184662
#
# Only the files needed by this project are downloaded.
# The script downloads them only if they are not already available.
#
# The trajectory archive is large, so no unnecessary re-download occurs.

zenodo_base <- "https://zenodo.org/records/19184662/files"

# Zenodo checksums for the three files required by this project.
expected_md5 <- c(
  flightlist_train.parquet = "1c7b9c86aa9e366236828e2439184d37",
  fuel_train.parquet       = "4ff4df3aca1a2d0850f48ef08395f1f9",
  flights_train.zip        = "255cef578697dc79d5d98958b8a880f5"
)

# Increase the download timeout because flights_train.zip is about 3.3 GB.
options(timeout = max(getOption("timeout"), 3600))

check_md5 <- function(path, expected) {
  if (!file.exists(path)) return(FALSE)
  actual <- unname(tools::md5sum(path))
  isTRUE(actual == expected)
}

download_and_check <- function(url, destination, expected) {
  if (check_md5(destination, expected)) {
    message("Already present and checksum verified: ", destination)
    return(invisible(destination))
  }

  if (file.exists(destination)) {
    message("Existing file failed checksum; replacing: ", destination)
    unlink(destination)
  }

  message("Downloading: ", destination)
  download.file(
    url = url,
    destfile = destination,
    mode = "wb",
    method = "libcurl",
    quiet = FALSE
  )

  if (!check_md5(destination, expected)) {
    unlink(destination)
    stop("Checksum verification failed for ", destination, ". Please rerun the script.")
  }

  message("Checksum verified: ", destination)
  invisible(destination)
}

flight_file <- file.path("data", "flightlist_train.parquet")
fuel_file <- file.path("data", "fuel_train.parquet")
trajectory_dir <- file.path("data", "flights_train")
trajectory_zip <- file.path("data", "flights_train.zip")

download_and_check(
  paste0(zenodo_base, "/flightlist_train.parquet?download=1"),
  flight_file,
  expected_md5[["flightlist_train.parquet"]]
)

download_and_check(
  paste0(zenodo_base, "/fuel_train.parquet?download=1"),
  fuel_file,
  expected_md5[["fuel_train.parquet"]]
)

trajectory_files <- character(0)
if (dir.exists(trajectory_dir)) {
  trajectory_files <- list.files(
    trajectory_dir,
    pattern = "\\.parquet$",
    full.names = TRUE,
    recursive = FALSE
  )
}

# The zip archive contains the trajectory files used to construct the A321 dataset.
# It is extracted into data/ so the final structure is data/flights_train/*.parquet.
if (length(trajectory_files) == 0) {

  download_and_check(
    paste0(zenodo_base, "/flights_train.zip?download=1"),
    trajectory_zip,
    expected_md5[["flights_train.zip"]]
  )

  message("Extracting flights_train.zip. This can take several minutes...")
  dir.create(
    "data",
    recursive = TRUE,
    showWarnings = FALSE
  )

  unzip(
    trajectory_zip,
    exdir = "data",
    overwrite = FALSE
  )

  trajectory_files <- list.files(
    trajectory_dir,
    pattern = "\\.parquet$",
    full.names = TRUE,
    recursive = FALSE
  )

  if (length(trajectory_files) == 0) {
    stop(
      "The trajectory archive was extracted, but data/flights_train/*.parquet was not found. ",
      "Please inspect the archive layout before rerunning."
    )
  }
}

message("Trajectory files available: ", length(trajectory_files))
message("Dataset access check complete.\n")

# ========================================================================
# 1. DATASET INSPECTION
# Original file: inspect_dataset.R
# ========================================================================



# ========================================================================
# inspect_dataset.R
# ========================================================================

library(arrow)

file_path <- "data/flightlist_train.parquet"

cat("Checking file...\n")

if (!file.exists(file_path)) {
  stop("File not found: ", file_path)
}

cat("File found successfully.\n")

flight_data <- read_parquet(file_path)

cat("\n============================================\n")
cat("DATASET INFORMATION\n")
cat("============================================\n")

cat("Rows:", nrow(flight_data), "\n")
cat("Columns:", ncol(flight_data), "\n")

cat("\nColumn names:\n")
print(names(flight_data))

cat("\nFirst 5 rows:\n")
print(head(flight_data, 5))

cat("\n============================================\n")
cat("DONE\n")
cat("============================================\n")

# ========================================================================
# check_fuel.R
# ========================================================================

# ============================================================
# CHECK FUEL DATA AND A321 FLIGHTS
# ============================================================

library(arrow)

# ------------------------------------------------------------
# 1. Read flight list
# ------------------------------------------------------------

flight_file <- "data/flightlist_train.parquet"
fuel_file <- "data/fuel_train.parquet"

if (!file.exists(flight_file)) {
  stop("Flight list not found: ", flight_file)
}

if (!file.exists(fuel_file)) {
  stop("Fuel data not found: ", fuel_file)
}

cat("Reading flight list...\n")

flight_data <- read_parquet(
  flight_file
)

cat("Flight list loaded.\n")

# ------------------------------------------------------------
# 2. Read fuel data
# ------------------------------------------------------------

cat("\nReading fuel data...\n")

fuel_data <- read_parquet(
  fuel_file
)

cat("Fuel data loaded.\n")


# ------------------------------------------------------------
# 3. Basic information
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FLIGHT LIST\n")
cat("============================================\n")

cat(
  "Rows:",
  nrow(flight_data),
  "\n"
)

cat(
  "Columns:",
  ncol(flight_data),
  "\n"
)

cat("\nFlight columns:\n")
print(names(flight_data))


cat("\n============================================\n")
cat("FUEL DATA\n")
cat("============================================\n")

cat(
  "Rows:",
  nrow(fuel_data),
  "\n"
)

cat(
  "Columns:",
  ncol(fuel_data),
  "\n"
)

cat("\nFuel columns:\n")
print(names(fuel_data))


# ------------------------------------------------------------
# 4. A321 flights
# ------------------------------------------------------------

a321 <- flight_data[
  flight_data$aircraft_type == "A321",
]

cat("\n============================================\n")
cat("A321 FLIGHTS\n")
cat("============================================\n")

cat(
  "Number of A321 flights:",
  nrow(a321),
  "\n"
)

cat("\nFirst A321 flights:\n")

print(
  head(a321, 10)
)


# ------------------------------------------------------------
# 5. Check common flight_id
# ------------------------------------------------------------

common_ids <- intersect(
  a321$flight_id,
  fuel_data$flight_id
)

cat("\n============================================\n")
cat("A321 + FUEL MATCHING\n")
cat("============================================\n")

cat(
  "A321 flights:",
  nrow(a321),
  "\n"
)

cat(
  "A321 flights with fuel data:",
  length(common_ids),
  "\n"
)

cat(
  "Percentage matched:",
  round(
    length(common_ids) / nrow(a321) * 100,
    2
  ),
  "%\n"
)


# ------------------------------------------------------------
# 6. Check number of fuel records for A321
# ------------------------------------------------------------

a321_fuel <- fuel_data[
  fuel_data$flight_id %in% common_ids,
]

cat("\nNumber of fuel records for A321:\n")

cat(
  nrow(a321_fuel),
  "\n"
)


# ------------------------------------------------------------
# 7. Show sample fuel data
# ------------------------------------------------------------

cat("\n============================================\n")
cat("A321 FUEL DATA SAMPLE\n")
cat("============================================\n")

print(
  head(
    a321_fuel,
    10
  )
)


cat("\n============================================\n")
cat("STEP COMPLETE\n")
cat("============================================\n")

# ========================================================================
# build_a321_dataset.R
# ========================================================================

# ============================================================
# A321 FUEL CONSUMPTION DATASET
# One row = one fuel interval
# ============================================================

library(arrow)


# ------------------------------------------------------------
# 1. Load flight metadata
# ------------------------------------------------------------

cat("Reading flight metadata...\n")

flights <- read_parquet(
  "data/flightlist_train.parquet"
)

flights <- as.data.frame(flights)

# Keep only Airbus A321
a321 <- flights[
  flights$aircraft_type == "A321",
]

cat("A321 flights:", nrow(a321), "\n")


# ------------------------------------------------------------
# 2. Load fuel intervals
# ------------------------------------------------------------

cat("Reading fuel data...\n")

fuel <- read_parquet(
  "data/fuel_train.parquet"
)

fuel <- as.data.frame(fuel)

# Keep only fuel records belonging to A321
a321_fuel <- fuel[
  fuel$flight_id %in% a321$flight_id,
]

cat(
  "A321 fuel intervals:",
  nrow(a321_fuel),
  "\n"
)


# ------------------------------------------------------------
# 3. Helper function: Haversine distance
# ------------------------------------------------------------

haversine_km <- function(
  lon1,
  lat1,
  lon2,
  lat2
) {

  rad <- pi / 180

  lon1 <- lon1 * rad
  lat1 <- lat1 * rad
  lon2 <- lon2 * rad
  lat2 <- lat2 * rad

  dlon <- lon2 - lon1
  dlat <- lat2 - lat1

  a <- sin(dlat / 2)^2 +
       cos(lat1) *
       cos(lat2) *
       sin(dlon / 2)^2

  2 * 6371 * asin(
    pmin(1, sqrt(a))
  )
}


# ------------------------------------------------------------
# 4. Prepare result container
# ------------------------------------------------------------

results <- vector(
  "list",
  length = nrow(a321_fuel)
)

result_index <- 0


# ------------------------------------------------------------
# 5. Process each A321 flight
# ------------------------------------------------------------

cat("\nProcessing A321 trajectory files...\n")

for (i in seq_len(nrow(a321))) {

  flight_id <- a321$flight_id[i]

  trajectory_file <- file.path(
    "data",
    "flights_train",
    paste0(flight_id, ".parquet")
  )

  if (!file.exists(trajectory_file)) {
    cat(
      "Missing trajectory:",
      flight_id,
      "\n"
    )
    next
  }

  # Read trajectory
  traj <- read_parquet(
    trajectory_file
  )

  traj <- as.data.frame(traj)

  # Convert time
  traj$timestamp <- as.POSIXct(
    as.character(traj$timestamp),
    tz = "UTC"
  )

  # Fuel intervals for this flight
  fuel_flight <- a321_fuel[
    a321_fuel$flight_id == flight_id,
  ]

  if (nrow(fuel_flight) == 0) {
    next
  }

  fuel_flight$start <- as.POSIXct(
    as.character(fuel_flight$start),
    tz = "UTC"
  )

  fuel_flight$end <- as.POSIXct(
    as.character(fuel_flight$end),
    tz = "UTC"
  )


  # ----------------------------------------------------------
  # Process each fuel interval
  # ----------------------------------------------------------

  for (j in seq_len(nrow(fuel_flight))) {

    interval_start <- fuel_flight$start[j]
    interval_end <- fuel_flight$end[j]

    inside <- (
      traj$timestamp >= interval_start &
      traj$timestamp <= interval_end
    )

    segment <- traj[inside, ]

    if (nrow(segment) == 0) {
      next
    }


    # --------------------------------------------------------
    # Calculate trajectory statistics
    # --------------------------------------------------------

    interval_seconds <- as.numeric(
      difftime(
        interval_end,
        interval_start,
        units = "secs"
      )
    )

    observed_seconds <- 0

    if (nrow(segment) >= 2) {

      observed_seconds <- as.numeric(
        difftime(
          max(segment$timestamp),
          min(segment$timestamp),
          units = "secs"
        )
      )
    }


    # Distance travelled inside interval
    distance_km <- 0

    if (nrow(segment) >= 2) {

      lon1 <- segment$longitude[-nrow(segment)]
      lat1 <- segment$latitude[-nrow(segment)]

      lon2 <- segment$longitude[-1]
      lat2 <- segment$latitude[-1]

      valid <- (
        !is.na(lon1) &
        !is.na(lat1) &
        !is.na(lon2) &
        !is.na(lat2)
      )

      if (any(valid)) {

        d <- haversine_km(
          lon1[valid],
          lat1[valid],
          lon2[valid],
          lat2[valid]
        )

        distance_km <- sum(
          d,
          na.rm = TRUE
        )
      }
    }


    # --------------------------------------------------------
    # Create observation
    # --------------------------------------------------------

    result_index <- result_index + 1

    results[[result_index]] <- data.frame(

      flight_id = flight_id,

      aircraft_type = "A321",

      interval_start = interval_start,

      interval_end = interval_end,

      interval_minutes =
        interval_seconds / 60,

      fuel_kg =
        fuel_flight$fuel_kg[j],

      trajectory_points =
        nrow(segment),

      observed_minutes =
        observed_seconds / 60,

      coverage_ratio =
        ifelse(
          interval_seconds > 0,
          observed_seconds / interval_seconds,
          NA
        ),

      distance_km =
        distance_km,

      mean_altitude =
        mean(
          segment$altitude,
          na.rm = TRUE
        ),

      min_altitude =
        min(
          segment$altitude,
          na.rm = TRUE
        ),

      max_altitude =
        max(
          segment$altitude,
          na.rm = TRUE
        ),

      altitude_change =
        ifelse(
          nrow(segment) >= 2,
          tail(segment$altitude, 1) -
            head(segment$altitude, 1),
          NA
        ),

      mean_groundspeed =
        mean(
          segment$groundspeed,
          na.rm = TRUE
        ),

      max_groundspeed =
        max(
          segment$groundspeed,
          na.rm = TRUE
        ),

      mean_vertical_rate =
        mean(
          segment$vertical_rate,
          na.rm = TRUE
        ),

      max_vertical_rate =
        max(
          abs(segment$vertical_rate),
          na.rm = TRUE
        ),

      mean_mach =
        ifelse(
          all(is.na(segment$mach)),
          NA,
          mean(
            segment$mach,
            na.rm = TRUE
          )
        ),

      mean_TAS =
        ifelse(
          all(is.na(segment$TAS)),
          NA,
          mean(
            segment$TAS,
            na.rm = TRUE
          )
        ),

      mean_CAS =
        ifelse(
          all(is.na(segment$CAS)),
          NA,
          mean(
            segment$CAS,
            na.rm = TRUE
          )
        )
    )
  }


  if (i %% 25 == 0) {

    cat(
      "Processed",
      i,
      "of",
      nrow(a321),
      "flights\n"
    )
  }
}


# ------------------------------------------------------------
# 6. Combine results
# ------------------------------------------------------------

results <- results[
  seq_len(result_index)
]

a321_dataset <- do.call(
  rbind,
  results
)


# ------------------------------------------------------------
# 7. Add flight metadata
# ------------------------------------------------------------

a321_dataset <- merge(
  a321_dataset,
  a321[
    ,
    c(
      "flight_id",
      "flight_date",
      "origin_icao",
      "origin_name",
      "destination_icao",
      "destination_name"
    )
  ],
  by = "flight_id",
  all.x = TRUE
)


# ------------------------------------------------------------
# 8. Save dataset
# ------------------------------------------------------------

write.csv(
  a321_dataset,
  "data/a321_fuel_dataset.csv",
  row.names = FALSE
)

saveRDS(
  a321_dataset,
  "data/a321_fuel_dataset.rds"
)


# ------------------------------------------------------------
# 9. Summary
# ------------------------------------------------------------

cat("\n============================================\n")
cat("A321 DATASET CREATED\n")
cat("============================================\n")

cat(
  "Rows:",
  nrow(a321_dataset),
  "\n"
)

cat(
  "Columns:",
  ncol(a321_dataset),
  "\n"
)

cat(
  "Unique flights:",
  length(unique(a321_dataset$flight_id)),
  "\n"
)

cat(
  "Total fuel intervals:",
  nrow(a321_fuel),
  "\n"
)

cat(
  "Intervals with trajectory:",
  nrow(a321_dataset),
  "\n"
)

cat(
  "Mean fuel per interval:",
  mean(
    a321_dataset$fuel_kg,
    na.rm = TRUE
  ),
  "kg\n"
)

cat("\nFuel summary:\n")
print(
  summary(a321_dataset$fuel_kg)
)

cat("\nCoverage ratio summary:\n")
print(
  summary(a321_dataset$coverage_ratio)
)

cat("\nMissing values:\n")
print(
  colSums(
    is.na(a321_dataset)
  )
)

cat("\n============================================\n")
cat("STEP COMPLETE\n")
cat("============================================\n")

# ========================================================================
# quality_check.R
# ========================================================================

library(readr)
library(dplyr)

cat("Reading A321 dataset...\n")

data <- read_csv(
  "data/a321_fuel_dataset.csv",
  show_col_types = FALSE
)

cat("Rows:", nrow(data), "\n")
cat("Columns:", ncol(data), "\n")


cat("\n============================================\n")
cat("1. COVERAGE CHECK\n")
cat("============================================\n")

print(
  summary(data$coverage_ratio)
)

cat("\nCoverage groups:\n")

coverage_groups <- data %>%
  mutate(
    coverage_group = case_when(
      coverage_ratio == 0 ~ "0%",
      coverage_ratio < 0.5 ~ "<50%",
      coverage_ratio < 0.9 ~ "50-90%",
      coverage_ratio < 1 ~ "90-99%",
      coverage_ratio == 1 ~ "100%"
    )
  ) %>%
  count(coverage_group)

print(coverage_groups)


cat("\n============================================\n")
cat("2. MISSING VALUES\n")
cat("============================================\n")

missing_table <- data.frame(
  variable = names(data),
  missing = sapply(
    data,
    function(x) sum(is.na(x))
  )
)

missing_table$missing_percent <- (
  missing_table$missing / nrow(data)
) * 100

missing_table <- missing_table[
  order(
    -missing_table$missing_percent
  ),
]

print(
  missing_table
)


cat("\n============================================\n")
cat("3. FUEL CHECK\n")
cat("============================================\n")

cat(
  "Minimum fuel:",
  min(data$fuel_kg, na.rm = TRUE),
  "kg\n"
)

cat(
  "Maximum fuel:",
  max(data$fuel_kg, na.rm = TRUE),
  "kg\n"
)

cat(
  "Mean fuel:",
  mean(data$fuel_kg, na.rm = TRUE),
  "kg\n"
)

cat(
  "Median fuel:",
  median(data$fuel_kg, na.rm = TRUE),
  "kg\n"
)


cat("\n============================================\n")
cat("4. INTERVAL CHECK\n")
cat("============================================\n")

cat(
  "Minimum interval:",
  min(data$interval_minutes, na.rm = TRUE),
  "minutes\n"
)

cat(
  "Maximum interval:",
  max(data$interval_minutes, na.rm = TRUE),
  "minutes\n"
)

cat(
  "Minimum trajectory points:",
  min(data$trajectory_points, na.rm = TRUE),
  "\n"
)


cat("\n============================================\n")
cat("5. EXTREME FUEL VALUES\n")
cat("============================================\n")

extreme_fuel <- data %>%
  arrange(desc(fuel_kg)) %>%
  select(
    flight_id,
    interval_start,
    interval_end,
    interval_minutes,
    fuel_kg,
    coverage_ratio,
    trajectory_points
  ) %>%
  head(10)

print(extreme_fuel)


cat("\n============================================\n")
cat("6. VARIABLES TO KEEP\n")
cat("============================================\n")

keep_variables <- c(
  "flight_id",
  "aircraft_type",
  "interval_start",
  "interval_end",
  "interval_minutes",
  "fuel_kg",
  "trajectory_points",
  "observed_minutes",
  "coverage_ratio",
  "distance_km",
  "mean_altitude",
  "min_altitude",
  "max_altitude",
  "altitude_change",
  "mean_groundspeed",
  "max_groundspeed",
  "mean_vertical_rate",
  "max_vertical_rate",
  "mean_mach",
  "mean_TAS",
  "flight_date",
  "origin_icao",
  "destination_icao"
)

print(keep_variables)


cat("\n============================================\n")
cat("STEP COMPLETE\n")
cat("============================================\n")

# ========================================================================
# prepare_model_data.R
# ========================================================================

# ============================================================
# STEP 5: PREPARE A321 MODELING DATA
# ============================================================

library(readr)
library(dplyr)


# ------------------------------------------------------------
# 1. Read dataset
# ------------------------------------------------------------

cat("Reading A321 dataset...\n")

data <- read_csv(
  "data/a321_fuel_dataset.csv",
  show_col_types = FALSE
)

cat("Rows:", nrow(data), "\n")


# ------------------------------------------------------------
# 2. Remove intervals with insufficient trajectory coverage
# ------------------------------------------------------------

data_clean <- data %>%
  filter(
    coverage_ratio >= 0.90
  )

cat(
  "Rows after coverage filtering:",
  nrow(data_clean),
  "\n"
)


# ------------------------------------------------------------
# 3. Remove variables that are not useful for modeling
# ------------------------------------------------------------

data_clean <- data_clean %>%
  select(
    flight_id,
    flight_date,
    origin_icao,
    destination_icao,

    interval_start,
    interval_end,
    interval_minutes,

    fuel_kg,

    trajectory_points,
    observed_minutes,
    coverage_ratio,

    distance_km,

    mean_altitude,
    min_altitude,
    max_altitude,
    altitude_change,

    mean_groundspeed,
    max_groundspeed,

    mean_vertical_rate,
    max_vertical_rate
  )


# ------------------------------------------------------------
# 4. Create fuel rate
# ------------------------------------------------------------

data_clean <- data_clean %>%
  mutate(
    fuel_rate_kg_min =
      fuel_kg / interval_minutes
  )


# ------------------------------------------------------------
# 5. Create flight phase
# ------------------------------------------------------------

data_clean <- data_clean %>%
  mutate(
    flight_phase = case_when(

      is.na(mean_vertical_rate)
        ~ "Unknown",

      mean_vertical_rate > 500
        ~ "Climb",

      mean_vertical_rate < -500
        ~ "Descent",

      TRUE
        ~ "Cruise"
    )
  )


# ------------------------------------------------------------
# 6. Convert categorical variables
# ------------------------------------------------------------

data_clean$flight_phase <- factor(
  data_clean$flight_phase,
  levels = c(
    "Climb",
    "Cruise",
    "Descent",
    "Unknown"
  )
)


# ------------------------------------------------------------
# 7. Basic sanity checks
# ------------------------------------------------------------

cat("\n============================================\n")
cat("CLEAN DATASET SUMMARY\n")
cat("============================================\n")

cat(
  "Rows:",
  nrow(data_clean),
  "\n"
)

cat(
  "Columns:",
  ncol(data_clean),
  "\n"
)

cat(
  "Unique flights:",
  length(
    unique(data_clean$flight_id)
  ),
  "\n"
)


# ------------------------------------------------------------
# 8. Fuel rate summary
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FUEL RATE SUMMARY\n")
cat("============================================\n")

print(
  summary(
    data_clean$fuel_rate_kg_min
  )
)


# ------------------------------------------------------------
# 9. Flight phase distribution
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FLIGHT PHASE\n")
cat("============================================\n")

print(
  table(
    data_clean$flight_phase,
    useNA = "ifany"
  )
)


# ------------------------------------------------------------
# 10. Missing values
# ------------------------------------------------------------

cat("\n============================================\n")
cat("MISSING VALUES\n")
cat("============================================\n")

missing <- data.frame(
  variable = names(data_clean),
  missing = sapply(
    data_clean,
    function(x) sum(is.na(x))
  )
)

missing$percent <- (
  missing$missing /
  nrow(data_clean)
) * 100

print(
  missing[
    order(
      -missing$percent
    ),
  ]
)


# ------------------------------------------------------------
# 11. Fuel rate by flight phase
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FUEL RATE BY FLIGHT PHASE\n")
cat("============================================\n")

phase_summary <- data_clean %>%
  group_by(flight_phase) %>%
  summarise(
    observations = n(),

    mean_fuel_rate =
      mean(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    median_fuel_rate =
      median(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    mean_fuel =
      mean(
        fuel_kg,
        na.rm = TRUE
      ),

    .groups = "drop"
  )

print(
  phase_summary
)


# ------------------------------------------------------------
# 12. Save modeling dataset
# ------------------------------------------------------------

write_csv(
  data_clean,
  "data/a321_modeling_data.csv"
)

saveRDS(
  data_clean,
  "data/a321_modeling_data.rds"
)


# ------------------------------------------------------------
# 13. Complete
# ------------------------------------------------------------

cat("\n============================================\n")
cat("STEP 5 COMPLETE\n")
cat("============================================\n")

cat(
  "Saved:\n",
  "data/a321_modeling_data.csv\n",
  "data/a321_modeling_data.rds\n"
)

# ========================================================================
# eda_a321.R
# ========================================================================

# ============================================================
# STEP 6: EDA FOR A321 FUEL CONSUMPTION
# ============================================================

# ------------------------------------------------------------
# 0. Install missing packages automatically
# ------------------------------------------------------------

if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr", repos = "https://cloud.r-project.org")
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = "https://cloud.r-project.org")
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "https://cloud.r-project.org")
}

library(readr)
library(dplyr)
library(ggplot2)


# ------------------------------------------------------------
# 1. Read modeling dataset
# ------------------------------------------------------------

cat("Reading modeling dataset...\n")

data <- read_csv(
  "data/a321_modeling_data.csv",
  show_col_types = FALSE
)

cat("Rows:", nrow(data), "\n")
cat(
  "Unique flights:",
  n_distinct(data$flight_id),
  "\n"
)


# ------------------------------------------------------------
# 2. Basic descriptive statistics
# ------------------------------------------------------------

cat("\n============================================\n")
cat("1. BASIC STATISTICS\n")
cat("============================================\n")

cat("\nFuel (kg):\n")
print(summary(data$fuel_kg))

cat("\nFuel rate (kg/min):\n")
print(summary(data$fuel_rate_kg_min))

cat("\nInterval duration (minutes):\n")
print(summary(data$interval_minutes))

cat("\nAltitude:\n")
print(summary(data$mean_altitude))

cat("\nGroundspeed:\n")
print(summary(data$mean_groundspeed))


# ------------------------------------------------------------
# 3. Short interval analysis
# ------------------------------------------------------------

cat("\n============================================\n")
cat("2. SHORT INTERVALS\n")
cat("============================================\n")

short_intervals <- data %>%
  summarise(
    less_than_1_min = sum(
      interval_minutes < 1,
      na.rm = TRUE
    ),

    less_than_5_min = sum(
      interval_minutes < 5,
      na.rm = TRUE
    ),

    less_than_10_min = sum(
      interval_minutes < 10,
      na.rm = TRUE
    ),

    total = n()
  )

print(short_intervals)


cat("\nFuel rate for intervals shorter than 1 minute:\n")

data %>%
  filter(
    interval_minutes < 1
  ) %>%
  select(
    flight_id,
    interval_minutes,
    fuel_kg,
    fuel_rate_kg_min,
    flight_phase,
    coverage_ratio
  ) %>%
  arrange(
    desc(fuel_rate_kg_min)
  ) %>%
  head(20) %>%
  print()


# ------------------------------------------------------------
# 4. High fuel-rate observations
# ------------------------------------------------------------

cat("\n============================================\n")
cat("3. HIGH FUEL-RATE OBSERVATIONS\n")
cat("============================================\n")

high_fuel_rate <- data %>%
  arrange(
    desc(fuel_rate_kg_min)
  ) %>%
  select(
    flight_id,
    interval_start,
    interval_minutes,
    fuel_kg,
    fuel_rate_kg_min,
    mean_altitude,
    mean_groundspeed,
    mean_vertical_rate,
    flight_phase
  ) %>%
  head(20)

print(high_fuel_rate)


# ------------------------------------------------------------
# 5. Fuel rate by flight phase
# ------------------------------------------------------------

cat("\n============================================\n")
cat("4. FUEL RATE BY FLIGHT PHASE\n")
cat("============================================\n")

phase_stats <- data %>%
  group_by(flight_phase) %>%
  summarise(
    observations = n(),

    mean_rate =
      mean(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    median_rate =
      median(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    sd_rate =
      sd(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    min_rate =
      min(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    max_rate =
      max(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    .groups = "drop"
  )

print(phase_stats)


# ------------------------------------------------------------
# 6. Create plots
# ------------------------------------------------------------

cat("\n============================================\n")
cat("5. CREATING EDA PLOTS\n")
cat("============================================\n")


# Fuel rate distribution

p1 <- ggplot(
  data,
  aes(
    x = fuel_rate_kg_min
  )
) +
  geom_histogram(
    bins = 50
  ) +
  labs(
    title = "Distribution of A321 Fuel Rate",
    x = "Fuel rate (kg/min)",
    y = "Number of intervals"
  )

ggsave(
  "data/eda_fuel_rate_distribution.png",
  p1,
  width = 8,
  height = 5,
  dpi = 150
)


# Fuel rate by phase

p2 <- ggplot(
  data,
  aes(
    x = flight_phase,
    y = fuel_rate_kg_min
  )
) +
  geom_boxplot(
    na.rm = TRUE
  ) +
  labs(
    title = "Fuel Rate by Flight Phase",
    x = "Flight phase",
    y = "Fuel rate (kg/min)"
  )

ggsave(
  "data/eda_fuel_rate_by_phase.png",
  p2,
  width = 8,
  height = 5,
  dpi = 150
)


# Fuel rate vs altitude

p3 <- ggplot(
  data,
  aes(
    x = mean_altitude,
    y = fuel_rate_kg_min
  )
) +
  geom_point(
    alpha = 0.35,
    na.rm = TRUE
  ) +
  geom_smooth(
    method = "loess",
    se = FALSE,
    na.rm = TRUE
  ) +
  labs(
    title = "Fuel Rate vs Mean Altitude",
    x = "Mean altitude",
    y = "Fuel rate (kg/min)"
  )

ggsave(
  "data/eda_fuel_rate_vs_altitude.png",
  p3,
  width = 8,
  height = 5,
  dpi = 150
)


# Fuel rate vs groundspeed

p4 <- ggplot(
  data,
  aes(
    x = mean_groundspeed,
    y = fuel_rate_kg_min
  )
) +
  geom_point(
    alpha = 0.35,
    na.rm = TRUE
  ) +
  geom_smooth(
    method = "loess",
    se = FALSE,
    na.rm = TRUE
  ) +
  labs(
    title = "Fuel Rate vs Groundspeed",
    x = "Mean groundspeed",
    y = "Fuel rate (kg/min)"
  )

ggsave(
  "data/eda_fuel_rate_vs_groundspeed.png",
  p4,
  width = 8,
  height = 5,
  dpi = 150
)


# Fuel rate vs interval duration

p5 <- ggplot(
  data,
  aes(
    x = interval_minutes,
    y = fuel_rate_kg_min
  )
) +
  geom_point(
    alpha = 0.35,
    na.rm = TRUE
  ) +
  geom_smooth(
    method = "loess",
    se = FALSE,
    na.rm = TRUE
  ) +
  labs(
    title = "Fuel Rate vs Interval Duration",
    x = "Interval duration (minutes)",
    y = "Fuel rate (kg/min)"
  )

ggsave(
  "data/eda_fuel_rate_vs_duration.png",
  p5,
  width = 8,
  height = 5,
  dpi = 150
)


# ------------------------------------------------------------
# 7. Correlation analysis
# ------------------------------------------------------------

cat("\n============================================\n")
cat("6. CORRELATIONS\n")
cat("============================================\n")

numeric_data <- data %>%
  select(
    fuel_kg,
    fuel_rate_kg_min,
    interval_minutes,
    distance_km,
    mean_altitude,
    min_altitude,
    max_altitude,
    altitude_change,
    mean_groundspeed,
    max_groundspeed,
    mean_vertical_rate,
    max_vertical_rate,
    trajectory_points,
    observed_minutes
  )

correlation_matrix <- cor(
  numeric_data,
  use = "pairwise.complete.obs"
)

print(
  round(
    correlation_matrix,
    3
  )
)


# ------------------------------------------------------------
# 8. Save EDA statistics
# ------------------------------------------------------------

write_csv(
  phase_stats,
  "data/phase_fuel_statistics.csv"
)

saveRDS(
  correlation_matrix,
  "data/a321_correlation_matrix.rds"
)


# ------------------------------------------------------------
# 9. Complete
# ------------------------------------------------------------

cat("\n============================================\n")
cat("STEP 6 COMPLETE\n")
cat("============================================\n")

cat("Plots created:\n")
cat(
  " - data/eda_fuel_rate_distribution.png\n"
)
cat(
  " - data/eda_fuel_rate_by_phase.png\n"
)
cat(
  " - data/eda_fuel_rate_vs_altitude.png\n"
)
cat(
  " - data/eda_fuel_rate_vs_groundspeed.png\n"
)
cat(
  " - data/eda_fuel_rate_vs_duration.png\n"
)

# ========================================================================
# step7_train_validation.R
# ========================================================================

# ============================================================
# STEP 7: TRAIN / VALIDATION SPLIT
# A321 Fuel Rate Prediction
# ============================================================

if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr", repos = "https://cloud.r-project.org")
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = "https://cloud.r-project.org")
}

library(readr)
library(dplyr)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

data <- read_csv(
  "data/a321_modeling_data.csv",
  show_col_types = FALSE
)

cat("Rows:", nrow(data), "\n")
cat(
  "Flights:",
  n_distinct(data$flight_id),
  "\n"
)


# ------------------------------------------------------------
# 2. Create derived operational variables
# ------------------------------------------------------------

data <- data %>%
  mutate(

    # Estimate vertical rate when missing
    derived_vertical_rate = ifelse(
      is.na(mean_vertical_rate) &
        interval_minutes > 0,
      altitude_change / interval_minutes,
      mean_vertical_rate
    ),

    # Estimate groundspeed when missing
    derived_groundspeed = ifelse(
      is.na(mean_groundspeed) &
        interval_minutes > 0,
      distance_km / interval_minutes * 60 / 1.852,
      mean_groundspeed
    )
  )


# ------------------------------------------------------------
# 3. Recalculate flight phase
# ------------------------------------------------------------

data <- data %>%
  mutate(
    flight_phase = case_when(

      is.na(derived_vertical_rate) ~
        "Unknown",

      derived_vertical_rate > 500 ~
        "Climb",

      derived_vertical_rate < -500 ~
        "Descent",

      TRUE ~
        "Cruise"
    )
  )

data$flight_phase <- factor(
  data$flight_phase,
  levels = c(
    "Climb",
    "Cruise",
    "Descent",
    "Unknown"
  )
)


# ------------------------------------------------------------
# 4. Split by FLIGHT, not by individual rows
# ------------------------------------------------------------

set.seed(2026)

all_flights <- unique(
  data$flight_id
)

n_validation_flights <- ceiling(
  length(all_flights) * 0.20
)

validation_flights <- sample(
  all_flights,
  n_validation_flights
)

train_flights <- setdiff(
  all_flights,
  validation_flights
)


train_data <- data %>%
  filter(
    flight_id %in% train_flights
  )

validation_data <- data %>%
  filter(
    flight_id %in% validation_flights
  )


cat("\n============================================\n")
cat("TRAIN / VALIDATION SPLIT\n")
cat("============================================\n")

cat(
  "Training flights:",
  length(train_flights),
  "\n"
)

cat(
  "Validation flights:",
  length(validation_flights),
  "\n"
)

cat(
  "Training rows:",
  nrow(train_data),
  "\n"
)

cat(
  "Validation rows:",
  nrow(validation_data),
  "\n"
)


# ------------------------------------------------------------
# 5. Select modeling variables
# ------------------------------------------------------------

model_variables <- c(
  "flight_id",
  "fuel_rate_kg_min",
  "interval_minutes",
  "distance_km",
  "mean_altitude",
  "min_altitude",
  "max_altitude",
  "altitude_change",
  "derived_groundspeed",
  "derived_vertical_rate",
  "flight_phase"
)

train_model <- train_data %>%
  select(
    all_of(model_variables)
  )

validation_model <- validation_data %>%
  select(
    all_of(model_variables)
  )


# ------------------------------------------------------------
# 6. Training-set median imputation
# ------------------------------------------------------------

numeric_variables <- c(
  "distance_km",
  "mean_altitude",
  "min_altitude",
  "max_altitude",
  "altitude_change",
  "derived_groundspeed",
  "derived_vertical_rate"
)

medians <- list()

for (v in numeric_variables) {

  medians[[v]] <- median(
    train_model[[v]],
    na.rm = TRUE
  )

  train_model[[v]][
    is.na(train_model[[v]])
  ] <- medians[[v]]

  validation_model[[v]][
    is.na(validation_model[[v]])
  ] <- medians[[v]]
}


# ------------------------------------------------------------
# 7. Remove Unknown phase if any remain
# ------------------------------------------------------------

train_model <- train_model %>%
  filter(
    flight_phase != "Unknown"
  )

validation_model <- validation_model %>%
  filter(
    flight_phase != "Unknown"
  )


# ------------------------------------------------------------
# 8. Baseline: overall mean
# ------------------------------------------------------------

baseline_mean <- mean(
  train_model$fuel_rate_kg_min,
  na.rm = TRUE
)

baseline_predictions <- rep(
  baseline_mean,
  nrow(validation_model)
)

baseline_rmse <- sqrt(
  mean(
    (
      validation_model$fuel_rate_kg_min -
      baseline_predictions
    )^2
  )
)

baseline_mae <- mean(
  abs(
    validation_model$fuel_rate_kg_min -
    baseline_predictions
  )
)


# ------------------------------------------------------------
# 9. Baseline: flight-phase mean
# ------------------------------------------------------------

phase_means <- train_model %>%
  group_by(
    flight_phase
  ) %>%
  summarise(
    phase_mean =
      mean(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),
    .groups = "drop"
  )

validation_phase <- validation_model %>%
  left_join(
    phase_means,
    by = "flight_phase"
  )

phase_predictions <-
  validation_phase$phase_mean

phase_rmse <- sqrt(
  mean(
    (
      validation_phase$fuel_rate_kg_min -
      phase_predictions
    )^2
  )
)

phase_mae <- mean(
  abs(
    validation_phase$fuel_rate_kg_min -
    phase_predictions
  )
)


# ------------------------------------------------------------
# 10. Results
# ------------------------------------------------------------

results <- data.frame(

  Model = c(
    "Overall Mean",
    "Flight Phase Mean"
  ),

  RMSE = c(
    baseline_rmse,
    phase_rmse
  ),

  MAE = c(
    baseline_mae,
    phase_mae
  )
)


cat("\n============================================\n")
cat("BASELINE RESULTS\n")
cat("============================================\n")

print(results)


# ------------------------------------------------------------
# 11. Save objects
# ------------------------------------------------------------

saveRDS(
  train_model,
  "data/a321_train_model.rds"
)

saveRDS(
  validation_model,
  "data/a321_validation_model.rds"
)

saveRDS(
  medians,
  "data/a321_training_medians.rds"
)

write.csv(
  results,
  "data/a321_baseline_results.csv",
  row.names = FALSE
)


cat("\n============================================\n")
cat("STEP 7 COMPLETE\n")
cat("============================================\n")

# ========================================================================
# step8_linear_regression.R
# ========================================================================

# ============================================================
# STEP 8: MULTIPLE LINEAR REGRESSION
# ============================================================

if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr", repos = "https://cloud.r-project.org")
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = "https://cloud.r-project.org")
}

library(readr)
library(dplyr)


# ------------------------------------------------------------
# 1. Load train and validation data
# ------------------------------------------------------------

train_data <- readRDS(
  "data/a321_train_model.rds"
)

validation_data <- readRDS(
  "data/a321_validation_model.rds"
)

cat("Training rows:", nrow(train_data), "\n")
cat("Validation rows:", nrow(validation_data), "\n")


# ------------------------------------------------------------
# 2. Fit multiple linear regression
# ------------------------------------------------------------

linear_model <- lm(
  fuel_rate_kg_min ~
    distance_km +
    mean_altitude +
    altitude_change +
    derived_groundspeed +
    derived_vertical_rate +
    flight_phase,
  data = train_data
)


# ------------------------------------------------------------
# 3. Generate validation predictions
# ------------------------------------------------------------

predictions <- predict(
  linear_model,
  newdata = validation_data
)


# ------------------------------------------------------------
# 4. Calculate metrics
# ------------------------------------------------------------

actual <- validation_data$fuel_rate_kg_min

rmse <- sqrt(
  mean(
    (actual - predictions)^2
  )
)

mae <- mean(
  abs(
    actual - predictions
  )
)

sst <- sum(
  (actual - mean(actual))^2
)

sse <- sum(
  (actual - predictions)^2
)

r_squared <- 1 - sse / sst


# ------------------------------------------------------------
# 5. Results
# ------------------------------------------------------------

results <- data.frame(
  Model = "Multiple Linear Regression",
  RMSE = rmse,
  MAE = mae,
  R_squared = r_squared
)

cat("\n============================================\n")
cat("LINEAR REGRESSION RESULTS\n")
cat("============================================\n")

print(results)


# ------------------------------------------------------------
# 6. Model coefficients
# ------------------------------------------------------------

cat("\n============================================\n")
cat("MODEL COEFFICIENTS\n")
cat("============================================\n")

print(
  summary(linear_model)
)


# ------------------------------------------------------------
# 7. Save model and results
# ------------------------------------------------------------

saveRDS(
  linear_model,
  "data/a321_linear_model.rds"
)

write.csv(
  results,
  "data/a321_linear_results.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 8. Save validation predictions
# ------------------------------------------------------------

prediction_output <- validation_data %>%
  select(
    fuel_rate_kg_min,
    flight_phase,
    mean_altitude,
    altitude_change,
    derived_groundspeed,
    derived_vertical_rate
  ) %>%
  mutate(
    predicted_fuel_rate = predictions,
    residual = fuel_rate_kg_min - predicted_fuel_rate
  )

write.csv(
  prediction_output,
  "data/a321_linear_predictions.csv",
  row.names = FALSE
)


cat("\n============================================\n")
cat("STEP 8 COMPLETE\n")
cat("============================================\n")

# ========================================================================
# step9_random_forest.R
# ========================================================================

# ============================================================
# STEP 9: RANDOM FOREST
# ============================================================

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages(
    "randomForest",
    repos = "https://cloud.r-project.org"
  )
}

library(randomForest)


# ------------------------------------------------------------
# 1. Load train and validation data
# ------------------------------------------------------------

train_data <- readRDS(
  "data/a321_train_model.rds"
)

validation_data <- readRDS(
  "data/a321_validation_model.rds"
)

cat(
  "Training rows:",
  nrow(train_data),
  "\n"
)

cat(
  "Validation rows:",
  nrow(validation_data),
  "\n"
)


# ------------------------------------------------------------
# 2. Keep predictors
# ------------------------------------------------------------

predictor_formula <- fuel_rate_kg_min ~
  distance_km +
  mean_altitude +
  altitude_change +
  derived_groundspeed +
  derived_vertical_rate +
  flight_phase


# ------------------------------------------------------------
# 3. Train Random Forest
# ------------------------------------------------------------

set.seed(2026)

rf_model <- randomForest(
  predictor_formula,
  data = train_data,

  ntree = 500,

  importance = TRUE
)


# ------------------------------------------------------------
# 4. Predict validation set
# ------------------------------------------------------------

predictions <- predict(
  rf_model,
  newdata = validation_data
)

actual <- validation_data$fuel_rate_kg_min


# ------------------------------------------------------------
# 5. Calculate metrics
# ------------------------------------------------------------

rmse <- sqrt(
  mean(
    (actual - predictions)^2
  )
)

mae <- mean(
  abs(
    actual - predictions
  )
)

sst <- sum(
  (actual - mean(actual))^2
)

sse <- sum(
  (actual - predictions)^2
)

r_squared <- 1 - sse / sst


# ------------------------------------------------------------
# 6. Results
# ------------------------------------------------------------

results <- data.frame(
  Model = "Random Forest",
  RMSE = rmse,
  MAE = mae,
  R_squared = r_squared
)

cat("\n============================================\n")
cat("RANDOM FOREST RESULTS\n")
cat("============================================\n")

print(results)


# ------------------------------------------------------------
# 7. Feature importance
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FEATURE IMPORTANCE\n")
cat("============================================\n")

importance_table <- importance(
  rf_model
)

print(
  importance_table
)


cat("\nMean Decrease in Accuracy:\n")

print(
  importance(
    rf_model,
    type = 1
  )
)


# ------------------------------------------------------------
# 8. Save model
# ------------------------------------------------------------

saveRDS(
  rf_model,
  "data/a321_random_forest.rds"
)

write.csv(
  results,
  "data/a321_random_forest_results.csv",
  row.names = FALSE
)

write.csv(
  importance_table,
  "data/a321_random_forest_importance.csv"
)


# ------------------------------------------------------------
# 9. Save predictions
# ------------------------------------------------------------

prediction_output <- data.frame(
  actual = actual,
  predicted = predictions,
  residual = actual - predictions
)

write.csv(
  prediction_output,
  "data/a321_random_forest_predictions.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 10. Complete
# ------------------------------------------------------------

cat("\n============================================\n")
cat("STEP 9 COMPLETE\n")
cat("============================================\n")

# ========================================================================
# step10_gradient_boosting.R
# ========================================================================

# ============================================================
# STEP 10: GRADIENT BOOSTING
# ============================================================

if (!requireNamespace("gbm", quietly = TRUE)) {
  install.packages(
    "gbm",
    repos = "https://cloud.r-project.org"
  )
}

library(gbm)


# ------------------------------------------------------------
# 1. Load train and validation data
# ------------------------------------------------------------

train_data <- readRDS(
  "data/a321_train_model.rds"
)

validation_data <- readRDS(
  "data/a321_validation_model.rds"
)

cat(
  "Training rows:",
  nrow(train_data),
  "\n"
)

cat(
  "Validation rows:",
  nrow(validation_data),
  "\n"
)


# ------------------------------------------------------------
# 2. Set seed
# ------------------------------------------------------------

set.seed(2026)


# ------------------------------------------------------------
# 3. Train Gradient Boosting model
# ------------------------------------------------------------

gbm_model <- gbm(

  formula =
    fuel_rate_kg_min ~
      distance_km +
      mean_altitude +
      altitude_change +
      derived_groundspeed +
      derived_vertical_rate +
      flight_phase,

  data = train_data,

  distribution = "gaussian",

  n.trees = 2000,

  interaction.depth = 3,

  shrinkage = 0.03,

  n.minobsinnode = 10,

  bag.fraction = 0.8,

  cv.folds = 5,

  verbose = FALSE
)


# ------------------------------------------------------------
# 4. Find optimal number of trees
# ------------------------------------------------------------

best_iteration <- gbm.perf(
  gbm_model,
  method = "cv",
  plot.it = FALSE
)

cat(
  "\nBest number of trees:",
  best_iteration,
  "\n"
)


# ------------------------------------------------------------
# 5. Predict validation set
# ------------------------------------------------------------

predictions <- predict(
  gbm_model,
  newdata = validation_data,
  n.trees = best_iteration
)

actual <- validation_data$fuel_rate_kg_min


# ------------------------------------------------------------
# 6. Calculate metrics
# ------------------------------------------------------------

rmse <- sqrt(
  mean(
    (actual - predictions)^2
  )
)

mae <- mean(
  abs(
    actual - predictions
  )
)

sst <- sum(
  (actual - mean(actual))^2
)

sse <- sum(
  (actual - predictions)^2
)

r_squared <- 1 - sse / sst


# ------------------------------------------------------------
# 7. Results
# ------------------------------------------------------------

results <- data.frame(

  Model = "Gradient Boosting",

  RMSE = rmse,

  MAE = mae,

  R_squared = r_squared
)

cat("\n============================================\n")
cat("GRADIENT BOOSTING RESULTS\n")
cat("============================================\n")

print(results)


# ------------------------------------------------------------
# 8. Variable importance
# ------------------------------------------------------------

cat("\n============================================\n")
cat("VARIABLE IMPORTANCE\n")
cat("============================================\n")

importance_table <- summary(
  gbm_model,
  n.trees = best_iteration,
  plotit = FALSE
)

print(
  importance_table
)


# ------------------------------------------------------------
# 9. Save model
# ------------------------------------------------------------

saveRDS(
  gbm_model,
  "data/a321_gradient_boosting.rds"
)

write.csv(
  results,
  "data/a321_gradient_boosting_results.csv",
  row.names = FALSE
)

write.csv(
  importance_table,
  "data/a321_gradient_boosting_importance.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 10. Save predictions
# ------------------------------------------------------------

prediction_output <- data.frame(

  actual = actual,

  predicted = predictions,

  residual = actual - predictions

)

write.csv(
  prediction_output,
  "data/a321_gradient_boosting_predictions.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 11. Complete
# ------------------------------------------------------------

cat("\n============================================\n")
cat("STEP 10 COMPLETE\n")
cat("============================================\n")

# ========================================================================
# step11_model_comparison.R
# ========================================================================

# ============================================================
# STEP 11: MODEL COMPARISON + ROBUSTNESS CHECK
# ============================================================

if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr", repos = "https://cloud.r-project.org")
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = "https://cloud.r-project.org")
}

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages(
    "randomForest",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("gbm", quietly = TRUE)) {
  install.packages(
    "gbm",
    repos = "https://cloud.r-project.org"
  )
}

library(readr)
library(dplyr)
library(randomForest)
library(gbm)


# ------------------------------------------------------------
# 1. Load validation data
# ------------------------------------------------------------

validation_data <- readRDS(
  "data/a321_validation_model.rds"
)

cat(
  "Validation rows:",
  nrow(validation_data),
  "\n"
)


# ------------------------------------------------------------
# 2. Load saved models
# ------------------------------------------------------------

linear_model <- readRDS(
  "data/a321_linear_model.rds"
)

rf_model <- readRDS(
  "data/a321_random_forest.rds"
)

gbm_model <- readRDS(
  "data/a321_gradient_boosting.rds"
)


# ------------------------------------------------------------
# 3. Generate predictions
# ------------------------------------------------------------

linear_pred <- predict(
  linear_model,
  newdata = validation_data
)

rf_pred <- predict(
  rf_model,
  newdata = validation_data
)


# Use the best number of trees from the stored GBM model
best_trees <- 783

gbm_pred <- predict(
  gbm_model,
  newdata = validation_data,
  n.trees = best_trees
)

actual <- validation_data$fuel_rate_kg_min


# ------------------------------------------------------------
# 4. Metric function
# ------------------------------------------------------------

calculate_metrics <- function(
  actual,
  predicted,
  model_name
) {

  rmse <- sqrt(
    mean(
      (actual - predicted)^2,
      na.rm = TRUE
    )
  )

  mae <- mean(
    abs(actual - predicted),
    na.rm = TRUE
  )

  sst <- sum(
    (
      actual -
      mean(actual, na.rm = TRUE)
    )^2,
    na.rm = TRUE
  )

  sse <- sum(
    (actual - predicted)^2,
    na.rm = TRUE
  )

  r2 <- 1 - sse / sst

  data.frame(
    Model = model_name,
    RMSE = rmse,
    MAE = mae,
    R_squared = r2
  )
}


# ------------------------------------------------------------
# 5. Compare models
# ------------------------------------------------------------

linear_results <- calculate_metrics(
  actual,
  linear_pred,
  "Multiple Linear Regression"
)

rf_results <- calculate_metrics(
  actual,
  rf_pred,
  "Random Forest"
)

gbm_results <- calculate_metrics(
  actual,
  gbm_pred,
  "Gradient Boosting"
)

comparison <- rbind(
  linear_results,
  rf_results,
  gbm_results
)


# ------------------------------------------------------------
# 6. Add baseline results
# ------------------------------------------------------------

baseline <- read_csv(
  "data/a321_baseline_results.csv",
  show_col_types = FALSE
)

baseline <- baseline %>%
  mutate(
    R_squared = NA_real_
  ) %>%
  select(
    Model,
    RMSE,
    MAE,
    R_squared
  )

all_results <- rbind(
  baseline,
  comparison
)


cat("\n============================================\n")
cat("MODEL COMPARISON\n")
cat("============================================\n")

print(all_results)


# ------------------------------------------------------------
# 7. Identify best model
# ------------------------------------------------------------

best_model <- all_results %>%
  filter(
    !is.na(RMSE)
  ) %>%
  arrange(
    RMSE
  ) %>%
  slice(1)

cat("\n============================================\n")
cat("CURRENT BEST MODEL\n")
cat("============================================\n")

print(best_model)


# ------------------------------------------------------------
# 8. Robustness check: interval >= 1 minute
# ------------------------------------------------------------

cat("\n============================================\n")
cat("ROBUSTNESS CHECK: INTERVAL >= 1 MINUTE\n")
cat("============================================\n")

keep <- validation_data$interval_minutes >= 1

cat(
  "Validation intervals >= 1 minute:",
  sum(keep),
  "\n"
)


robust_linear <- calculate_metrics(
  actual[keep],
  linear_pred[keep],
  "Linear Regression >= 1 min"
)

robust_rf <- calculate_metrics(
  actual[keep],
  rf_pred[keep],
  "Random Forest >= 1 min"
)

robust_gbm <- calculate_metrics(
  actual[keep],
  gbm_pred[keep],
  "Gradient Boosting >= 1 min"
)

robust_results <- rbind(
  robust_linear,
  robust_rf,
  robust_gbm
)

print(robust_results)


# ------------------------------------------------------------
# 9. Save results
# ------------------------------------------------------------

write_csv(
  all_results,
  "data/a321_all_model_results.csv"
)

write_csv(
  robust_results,
  "data/a321_robustness_results.csv"
)


# ------------------------------------------------------------
# 10. Save validation predictions
# ------------------------------------------------------------

prediction_table <- validation_data %>%
  select(
    interval_minutes,
    fuel_rate_kg_min,
    flight_phase
  ) %>%
  mutate(
    linear_prediction = linear_pred,
    random_forest_prediction = rf_pred,
    gradient_boosting_prediction = gbm_pred
  )

write_csv(
  prediction_table,
  "data/a321_validation_predictions_all_models.csv"
)


cat("\n============================================\n")
cat("STEP 11 COMPLETE\n")
cat("============================================\n")

# ========================================================================
# step12_tune_random_forest.R
# ========================================================================

# ============================================================
# STEP 12: RANDOM FOREST HYPERPARAMETER TUNING
# Grouped 5-fold CV by flight_id
# ============================================================

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages(
    "randomForest",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages(
    "dplyr",
    repos = "https://cloud.r-project.org"
  )
}

library(randomForest)
library(dplyr)


# ------------------------------------------------------------
# 1. Load training and validation data
# ------------------------------------------------------------

train_data <- readRDS(
  "data/a321_train_model.rds"
)

validation_data <- readRDS(
  "data/a321_validation_model.rds"
)

cat("Training rows:", nrow(train_data), "\n")
cat("Validation rows:", nrow(validation_data), "\n")


# ------------------------------------------------------------
# 2. Model formula
# ------------------------------------------------------------

rf_formula <- fuel_rate_kg_min ~
  distance_km +
  mean_altitude +
  altitude_change +
  derived_groundspeed +
  derived_vertical_rate +
  flight_phase


# ------------------------------------------------------------
# 3. Create grouped folds by flight
# ------------------------------------------------------------

set.seed(2026)

flight_ids <- unique(
  train_data$flight_id
)

n_folds <- 5

fold_assignment <- sample(
  rep(
    1:n_folds,
    length.out = length(flight_ids)
  )
)

flight_fold_table <- data.frame(
  flight_id = flight_ids,
  fold = fold_assignment
)

train_cv <- train_data %>%
  left_join(
    flight_fold_table,
    by = "flight_id"
  )


cat("\n============================================\n")
cat("CROSS-VALIDATION SETUP\n")
cat("============================================\n")

cat(
  "Training flights:",
  length(flight_ids),
  "\n"
)

cat(
  "Number of folds:",
  n_folds,
  "\n"
)


# ------------------------------------------------------------
# 4. Hyperparameter grid
# ------------------------------------------------------------

parameter_grid <- expand.grid(
  mtry = c(2, 3, 4, 5),
  nodesize = c(5, 10)
)

print(parameter_grid)


# ------------------------------------------------------------
# 5. Cross-validation
# ------------------------------------------------------------

cv_results <- data.frame()

for (p in seq_len(nrow(parameter_grid))) {

  current_mtry <- parameter_grid$mtry[p]
  current_nodesize <- parameter_grid$nodesize[p]

  fold_rmse <- numeric(n_folds)

  cat(
    "\nTesting mtry =",
    current_mtry,
    "nodesize =",
    current_nodesize,
    "\n"
  )

  for (fold in 1:n_folds) {

    validation_fold <- train_cv %>%
      filter(
        fold == !!fold
      )

    training_fold <- train_cv %>%
      filter(
        fold != !!fold
      )

    model <- randomForest(
      rf_formula,
      data = training_fold,
      ntree = 300,
      mtry = current_mtry,
      nodesize = current_nodesize,
      importance = FALSE
    )

    prediction <- predict(
      model,
      newdata = validation_fold
    )

    actual <- validation_fold$fuel_rate_kg_min

    fold_rmse[fold] <- sqrt(
      mean(
        (actual - prediction)^2
      )
    )

    cat(
      "  Fold",
      fold,
      "RMSE:",
      round(
        fold_rmse[fold],
        4
      ),
      "\n"
    )
  }

  cv_results <- rbind(
    cv_results,
    data.frame(
      mtry = current_mtry,
      nodesize = current_nodesize,
      mean_RMSE = mean(fold_rmse),
      sd_RMSE = sd(fold_rmse)
    )
  )
}


# ------------------------------------------------------------
# 6. Sort CV results
# ------------------------------------------------------------

cv_results <- cv_results %>%
  arrange(
    mean_RMSE
  )


cat("\n============================================\n")
cat("CROSS-VALIDATION RESULTS\n")
cat("============================================\n")

print(
  cv_results
)


# ------------------------------------------------------------
# 7. Select best parameters
# ------------------------------------------------------------

best_parameters <- cv_results %>%
  slice(1)

cat("\n============================================\n")
cat("BEST PARAMETERS\n")
cat("============================================\n")

print(
  best_parameters
)


best_mtry <- best_parameters$mtry
best_nodesize <- best_parameters$nodesize


# ------------------------------------------------------------
# 8. Train tuned model on ALL training data
# ------------------------------------------------------------

cat("\nTraining final tuned RF on training data...\n")

set.seed(2026)

tuned_rf <- randomForest(
  rf_formula,
  data = train_data,
  ntree = 500,
  mtry = best_mtry,
  nodesize = best_nodesize,
  importance = TRUE
)


# ------------------------------------------------------------
# 9. Evaluate on untouched validation set
# ------------------------------------------------------------

validation_prediction <- predict(
  tuned_rf,
  newdata = validation_data
)

actual <- validation_data$fuel_rate_kg_min

rmse <- sqrt(
  mean(
    (actual - validation_prediction)^2
  )
)

mae <- mean(
  abs(
    actual - validation_prediction
  )
)

sst <- sum(
  (
    actual -
    mean(actual)
  )^2
)

sse <- sum(
  (
    actual -
    validation_prediction
  )^2
)

r_squared <- 1 - sse / sst


tuned_results <- data.frame(
  Model = "Tuned Random Forest",
  RMSE = rmse,
  MAE = mae,
  R_squared = r_squared,
  mtry = best_mtry,
  nodesize = best_nodesize,
  ntree = 500
)


cat("\n============================================\n")
cat("TUNED RANDOM FOREST RESULTS\n")
cat("============================================\n")

print(
  tuned_results
)


# ------------------------------------------------------------
# 10. Feature importance
# ------------------------------------------------------------

cat("\n============================================\n")
cat("TUNED RF FEATURE IMPORTANCE\n")
cat("============================================\n")

importance_table <- importance(
  tuned_rf
)

print(
  importance_table
)


# ------------------------------------------------------------
# 11. Save everything
# ------------------------------------------------------------

saveRDS(
  tuned_rf,
  "data/a321_tuned_random_forest.rds"
)

write.csv(
  cv_results,
  "data/a321_rf_cv_results.csv",
  row.names = FALSE
)

write.csv(
  tuned_results,
  "data/a321_tuned_rf_results.csv",
  row.names = FALSE
)

write.csv(
  importance_table,
  "data/a321_tuned_rf_importance.csv"
)

prediction_table <- data.frame(
  actual = actual,
  predicted = validation_prediction,
  residual = actual - validation_prediction
)

write.csv(
  prediction_table,
  "data/a321_tuned_rf_predictions.csv",
  row.names = FALSE
)


cat("\n============================================\n")
cat("STEP 12 COMPLETE\n")
cat("============================================\n")

# ========================================================================
# step13_diagnostics.R
# ========================================================================

# ============================================================
# STEP 13: RANDOM FOREST DIAGNOSTIC ANALYSIS
# ============================================================

if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages(
    "readr",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages(
    "dplyr",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages(
    "ggplot2",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages(
    "randomForest",
    repos = "https://cloud.r-project.org"
  )
}

library(readr)
library(dplyr)
library(ggplot2)
library(randomForest)


# ------------------------------------------------------------
# 1. Load validation data and final tuned RF
# ------------------------------------------------------------

validation_data <- readRDS(
  "data/a321_validation_model.rds"
)

rf_model <- readRDS(
  "data/a321_tuned_random_forest.rds"
)

cat(
  "Validation rows:",
  nrow(validation_data),
  "\n"
)


# ------------------------------------------------------------
# 2. Check required variables
# ------------------------------------------------------------

required_variables <- c(
  "flight_id",
  "fuel_rate_kg_min",
  "interval_minutes",
  "mean_altitude",
  "flight_phase"
)

missing_variables <- setdiff(
  required_variables,
  names(validation_data)
)

if (length(missing_variables) > 0) {

  stop(
    "Missing validation variables: ",
    paste(
      missing_variables,
      collapse = ", "
    )
  )
}


# ------------------------------------------------------------
# 3. Generate predictions
# ------------------------------------------------------------

predicted <- predict(
  rf_model,
  newdata = validation_data
)

diagnostics <- validation_data %>%
  mutate(
    predicted_fuel_rate = predicted,
    residual =
      fuel_rate_kg_min -
      predicted_fuel_rate,

    absolute_error =
      abs(residual),

    percentage_error =
      ifelse(
        fuel_rate_kg_min != 0,
        absolute_error /
          fuel_rate_kg_min * 100,
        NA
      )
  )


# ------------------------------------------------------------
# 4. Overall diagnostic metrics
# ------------------------------------------------------------

rmse <- sqrt(
  mean(
    diagnostics$residual^2,
    na.rm = TRUE
  )
)

mae <- mean(
  diagnostics$absolute_error,
  na.rm = TRUE
)

bias <- mean(
  diagnostics$residual,
  na.rm = TRUE
)

correlation <- cor(
  diagnostics$fuel_rate_kg_min,
  diagnostics$predicted_fuel_rate,
  use = "complete.obs"
)

sst <- sum(
  (
    diagnostics$fuel_rate_kg_min -
    mean(
      diagnostics$fuel_rate_kg_min,
      na.rm = TRUE
    )
  )^2,
  na.rm = TRUE
)

sse <- sum(
  diagnostics$residual^2,
  na.rm = TRUE
)

r_squared <- 1 - sse / sst


cat("\n============================================\n")
cat("1. OVERALL DIAGNOSTICS\n")
cat("============================================\n")

cat(
  "RMSE:",
  round(rmse, 4),
  "\n"
)

cat(
  "MAE:",
  round(mae, 4),
  "\n"
)

cat(
  "R-squared:",
  round(r_squared, 4),
  "\n"
)

cat(
  "Prediction correlation:",
  round(correlation, 4),
  "\n"
)

cat(
  "Mean residual / bias:",
  round(bias, 4),
  "\n"
)


# ------------------------------------------------------------
# 5. Diagnostics by flight phase
# ------------------------------------------------------------

phase_diagnostics <- diagnostics %>%
  group_by(
    flight_phase
  ) %>%
  summarise(

    observations = n(),

    RMSE = sqrt(
      mean(
        residual^2,
        na.rm = TRUE
      )
    ),

    MAE = mean(
      absolute_error,
      na.rm = TRUE
    ),

    mean_residual = mean(
      residual,
      na.rm = TRUE
    ),

    median_fuel_rate =
      median(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    mean_fuel_rate =
      mean(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


cat("\n============================================\n")
cat("2. DIAGNOSTICS BY FLIGHT PHASE\n")
cat("============================================\n")

print(
  phase_diagnostics
)


# ------------------------------------------------------------
# 6. Diagnostics by fuel-rate range
# ------------------------------------------------------------

diagnostics <- diagnostics %>%
  mutate(
    fuel_rate_group = case_when(

      fuel_rate_kg_min < 25 ~
        "<25 kg/min",

      fuel_rate_kg_min < 40 ~
        "25-40 kg/min",

      fuel_rate_kg_min < 60 ~
        "40-60 kg/min",

      fuel_rate_kg_min < 100 ~
        "60-100 kg/min",

      TRUE ~
        ">=100 kg/min"
    )
  )

fuel_group_diagnostics <- diagnostics %>%
  group_by(
    fuel_rate_group
  ) %>%
  summarise(

    observations = n(),

    RMSE = sqrt(
      mean(
        residual^2,
        na.rm = TRUE
      )
    ),

    MAE = mean(
      absolute_error,
      na.rm = TRUE
    ),

    mean_residual =
      mean(
        residual,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


cat("\n============================================\n")
cat("3. DIAGNOSTICS BY FUEL-RATE RANGE\n")
cat("============================================\n")

print(
  fuel_group_diagnostics
)


# ------------------------------------------------------------
# 7. Diagnostics by altitude
# ------------------------------------------------------------

diagnostics <- diagnostics %>%
  mutate(
    altitude_group = case_when(

      mean_altitude < 10000 ~
        "<10,000",

      mean_altitude < 20000 ~
        "10,000-20,000",

      mean_altitude < 30000 ~
        "20,000-30,000",

      mean_altitude < 35000 ~
        "30,000-35,000",

      mean_altitude < 38000 ~
        "35,000-38,000",

      TRUE ~
        ">=38,000"
    )
  )

altitude_diagnostics <- diagnostics %>%
  group_by(
    altitude_group
  ) %>%
  summarise(

    observations = n(),

    RMSE = sqrt(
      mean(
        residual^2,
        na.rm = TRUE
      )
    ),

    MAE = mean(
      absolute_error,
      na.rm = TRUE
    ),

    mean_fuel_rate =
      mean(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


cat("\n============================================\n")
cat("4. DIAGNOSTICS BY ALTITUDE\n")
cat("============================================\n")

print(
  altitude_diagnostics
)


# ------------------------------------------------------------
# 8. Worst prediction errors
# ------------------------------------------------------------

worst_predictions <- diagnostics %>%
  arrange(
    desc(absolute_error)
  ) %>%
  select(
    flight_id,
    interval_minutes,
    flight_phase,
    mean_altitude,
    fuel_rate_kg_min,
    predicted_fuel_rate,
    residual,
    absolute_error
  ) %>%
  head(20)


cat("\n============================================\n")
cat("5. LARGEST PREDICTION ERRORS\n")
cat("============================================\n")

print(
  worst_predictions
)


# ------------------------------------------------------------
# 9. Bias check
# ------------------------------------------------------------

cat("\n============================================\n")
cat("6. BIAS CHECK\n")
cat("============================================\n")

cat(
  "Positive residual = model underpredicts\n"
)

cat(
  "Negative residual = model overpredicts\n"
)

cat(
  "Mean residual:",
  round(bias, 4),
  "\n"
)


# ------------------------------------------------------------
# 10. Plot 1: Actual vs predicted
# ------------------------------------------------------------

p1 <- ggplot(
  diagnostics,
  aes(
    x = fuel_rate_kg_min,
    y = predicted_fuel_rate
  )
) +
  geom_point(
    alpha = 0.5
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  labs(
    title =
      "Actual vs Predicted A321 Fuel Rate",
    x =
      "Actual fuel rate (kg/min)",
    y =
      "Predicted fuel rate (kg/min)"
  )

ggsave(
  "data/diagnostic_actual_vs_predicted.png",
  p1,
  width = 8,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 11. Plot 2: Residual distribution
# ------------------------------------------------------------

p2 <- ggplot(
  diagnostics,
  aes(
    x = residual
  )
) +
  geom_histogram(
    bins = 40
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title =
      "Random Forest Residual Distribution",
    x =
      "Residual (actual - predicted)",
    y =
      "Number of observations"
  )

ggsave(
  "data/diagnostic_residual_distribution.png",
  p2,
  width = 8,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 12. Plot 3: Residual vs altitude
# ------------------------------------------------------------

p3 <- ggplot(
  diagnostics,
  aes(
    x = mean_altitude,
    y = residual
  )
) +
  geom_point(
    alpha = 0.5
  ) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  geom_smooth(
    method = "loess",
    se = FALSE
  ) +
  labs(
    title =
      "Residuals vs Mean Altitude",
    x =
      "Mean altitude",
    y =
      "Residual"
  )

ggsave(
  "data/diagnostic_residual_vs_altitude.png",
  p3,
  width = 8,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 13. Plot 4: Residual vs interval duration
# ------------------------------------------------------------

p4 <- ggplot(
  diagnostics,
  aes(
    x = interval_minutes,
    y = residual
  )
) +
  geom_point(
    alpha = 0.5
  ) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  geom_smooth(
    method = "loess",
    se = FALSE
  ) +
  labs(
    title =
      "Residuals vs Interval Duration",
    x =
      "Interval duration (minutes)",
    y =
      "Residual"
  )

ggsave(
  "data/diagnostic_residual_vs_duration.png",
  p4,
  width = 8,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 14. Plot 5: Absolute error by flight phase
# ------------------------------------------------------------

p5 <- ggplot(
  diagnostics,
  aes(
    x = flight_phase,
    y = absolute_error
  )
) +
  geom_boxplot(
    na.rm = TRUE
  ) +
  labs(
    title =
      "Absolute Prediction Error by Flight Phase",
    x =
      "Flight phase",
    y =
      "Absolute error (kg/min)"
  )

ggsave(
  "data/diagnostic_error_by_phase.png",
  p5,
  width = 8,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 15. Save diagnostic tables
# ------------------------------------------------------------

write_csv(
  diagnostics,
  "data/a321_rf_diagnostics.csv"
)

write_csv(
  phase_diagnostics,
  "data/a321_rf_diagnostics_by_phase.csv"
)

write_csv(
  fuel_group_diagnostics,
  "data/a321_rf_diagnostics_by_fuel_range.csv"
)

write_csv(
  altitude_diagnostics,
  "data/a321_rf_diagnostics_by_altitude.csv"
)

write_csv(
  worst_predictions,
  "data/a321_rf_worst_predictions.csv"
)


# ------------------------------------------------------------
# 16. Complete
# ------------------------------------------------------------

cat("\n============================================\n")
cat("STEP 13 COMPLETE\n")
cat("============================================\n")

cat("Diagnostic files created:\n")
cat(
  " - data/diagnostic_actual_vs_predicted.png\n"
)
cat(
  " - data/diagnostic_residual_distribution.png\n"
)
cat(
  " - data/diagnostic_residual_vs_altitude.png\n"
)
cat(
  " - data/diagnostic_residual_vs_duration.png\n"
)
cat(
  " - data/diagnostic_error_by_phase.png\n"
)

# ========================================================================
# step14_fuel_optimization.R
# ========================================================================

# ============================================================
# STEP 14: A321 FUEL-EFFICIENCY / WHAT-IF ANALYSIS
# ============================================================

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages(
    "dplyr",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages(
    "ggplot2",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages(
    "randomForest",
    repos = "https://cloud.r-project.org"
  )
}

library(dplyr)
library(ggplot2)
library(randomForest)


# ------------------------------------------------------------
# 1. Load training data and final selected model
# ------------------------------------------------------------

train_data <- readRDS(
  "data/a321_train_model.rds"
)

rf_model <- readRDS(
  "data/a321_tuned_random_forest.rds"
)


# ------------------------------------------------------------
# 2. Keep observed CRUISE data only
# ------------------------------------------------------------

cruise_data <- train_data %>%
  filter(
    flight_phase == "Cruise"
  )

cat("\n============================================\n")
cat("CRUISE TRAINING DATA\n")
cat("============================================\n")

cat(
  "Rows:",
  nrow(cruise_data),
  "\n"
)

cat(
  "Flights:",
  n_distinct(cruise_data$flight_id),
  "\n"
)


# ------------------------------------------------------------
# 3. Define plausible observed ranges
#    Use 10th-90th percentiles to avoid extreme extrapolation
# ------------------------------------------------------------

altitude_low <- quantile(
  cruise_data$mean_altitude,
  0.10,
  na.rm = TRUE
)

altitude_high <- quantile(
  cruise_data$mean_altitude,
  0.90,
  na.rm = TRUE
)

speed_low <- quantile(
  cruise_data$derived_groundspeed,
  0.10,
  na.rm = TRUE
)

speed_high <- quantile(
  cruise_data$derived_groundspeed,
  0.90,
  na.rm = TRUE
)

distance_reference <- median(
  cruise_data$distance_km,
  na.rm = TRUE
)

vertical_reference <- median(
  cruise_data$derived_vertical_rate,
  na.rm = TRUE
)


cat("\n============================================\n")
cat("OBSERVED CRUISE RANGES\n")
cat("============================================\n")

cat(
  "Altitude range:",
  round(altitude_low, 1),
  "to",
  round(altitude_high, 1),
  "\n"
)

cat(
  "Groundspeed range:",
  round(speed_low, 1),
  "to",
  round(speed_high, 1),
  "\n"
)

cat(
  "Reference distance:",
  round(distance_reference, 2),
  "km\n"
)

cat(
  "Reference vertical rate:",
  round(vertical_reference, 2),
  "\n"
)


# ------------------------------------------------------------
# 4. Create altitude x groundspeed scenario grid
# ------------------------------------------------------------

scenario_grid <- expand.grid(

  mean_altitude = seq(
    floor(altitude_low / 500) * 500,
    ceiling(altitude_high / 500) * 500,
    by = 500
  ),

  derived_groundspeed = seq(
    floor(speed_low / 10) * 10,
    ceiling(speed_high / 10) * 10,
    by = 10
  )
)


# ------------------------------------------------------------
# 5. Keep remaining predictors at representative cruise values
# ------------------------------------------------------------

scenario_grid <- scenario_grid %>%
  mutate(

    distance_km =
      distance_reference,

    altitude_change =
      0,

    derived_vertical_rate =
      vertical_reference,

    flight_phase =
      factor(
        "Cruise",
        levels = levels(
          train_data$flight_phase
        )
      )
  )


# ------------------------------------------------------------
# 6. Predict fuel rate
# ------------------------------------------------------------

scenario_grid$predicted_fuel_rate <- predict(
  rf_model,
  newdata = scenario_grid
)


# ------------------------------------------------------------
# 7. Find lowest predicted fuel-rate scenarios
# ------------------------------------------------------------

best_scenarios <- scenario_grid %>%
  arrange(
    predicted_fuel_rate
  ) %>%
  head(20)


cat("\n============================================\n")
cat("LOWEST PREDICTED FUEL-RATE SCENARIOS\n")
cat("============================================\n")

print(
  best_scenarios
)


# ------------------------------------------------------------
# 8. Find best altitude for each speed
# ------------------------------------------------------------

best_by_speed <- scenario_grid %>%
  group_by(
    derived_groundspeed
  ) %>%
  slice_min(
    predicted_fuel_rate,
    n = 1
  ) %>%
  ungroup()


cat("\n============================================\n")
cat("BEST ALTITUDE FOR EACH SPEED\n")
cat("============================================\n")

print(
  best_by_speed
)


# ------------------------------------------------------------
# 9. Find best speed for each altitude
# ------------------------------------------------------------

best_by_altitude <- scenario_grid %>%
  group_by(
    mean_altitude
  ) %>%
  slice_min(
    predicted_fuel_rate,
    n = 1
  ) %>%
  ungroup()


cat("\n============================================\n")
cat("BEST SPEED FOR EACH ALTITUDE\n")
cat("============================================\n")

print(
  best_by_altitude
)


# ------------------------------------------------------------
# 10. Overall model-based minimum
# ------------------------------------------------------------

best_scenario <- scenario_grid %>%
  arrange(
    predicted_fuel_rate
  ) %>%
  slice(1)


cat("\n============================================\n")
cat("MODEL-BASED LOWEST FUEL-RATE SCENARIO\n")
cat("============================================\n")

print(
  best_scenario
)


# ------------------------------------------------------------
# 11. Heatmap
# ------------------------------------------------------------

p1 <- ggplot(
  scenario_grid,
  aes(
    x = derived_groundspeed,
    y = mean_altitude,
    fill = predicted_fuel_rate
  )
) +
  geom_tile() +
  labs(
    title =
      "Predicted A321 Cruise Fuel Rate",
    subtitle =
      "Model-based what-if analysis within observed data ranges",
    x =
      "Groundspeed",
    y =
      "Mean altitude",
    fill =
      "Predicted fuel rate\n(kg/min)"
  )

ggsave(
  "data/a321_fuel_optimization_heatmap.png",
  p1,
  width = 9,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 12. Save scenario results
# ------------------------------------------------------------

write.csv(
  scenario_grid,
  "data/a321_fuel_optimization_scenarios.csv",
  row.names = FALSE
)

write.csv(
  best_scenarios,
  "data/a321_best_fuel_scenarios.csv",
  row.names = FALSE
)

write.csv(
  best_by_speed,
  "data/a321_best_altitude_by_speed.csv",
  row.names = FALSE
)

write.csv(
  best_by_altitude,
  "data/a321_best_speed_by_altitude.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 13. Complete
# ------------------------------------------------------------

cat("\n============================================\n")
cat("STEP 14 COMPLETE\n")
cat("============================================\n")

cat(
  "Scenario table saved to:\n",
  "data/a321_fuel_optimization_scenarios.csv\n"
)

cat(
  "Heatmap saved to:\n",
  "data/a321_fuel_optimization_heatmap.png\n"
)

cat(
  "\nImportant: these are model-based research scenarios, ",
  "not operational flight recommendations.\n"
)

# ========================================================================
# step15_validate_optimization.R
# ========================================================================

# ============================================================
# STEP 15: VALIDATE MODEL-BASED FUEL OPTIMIZATION
# ============================================================

if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages(
    "readr",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages(
    "dplyr",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages(
    "ggplot2",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages(
    "randomForest",
    repos = "https://cloud.r-project.org"
  )
}

library(readr)
library(dplyr)
library(ggplot2)
library(randomForest)


# ------------------------------------------------------------
# 1. Load training data and Random Forest
# ------------------------------------------------------------

train_data <- readRDS(
  "data/a321_train_model.rds"
)

rf_model <- readRDS(
  "data/a321_tuned_random_forest.rds"
)


# ------------------------------------------------------------
# 2. Keep cruise observations
# ------------------------------------------------------------

cruise <- train_data %>%
  filter(
    flight_phase == "Cruise"
  )


cat("\n============================================\n")
cat("OBSERVED CRUISE DATA\n")
cat("============================================\n")

cat(
  "Rows:",
  nrow(cruise),
  "\n"
)

cat(
  "Flights:",
  n_distinct(cruise$flight_id),
  "\n"
)


# ------------------------------------------------------------
# 3. Create observed altitude and speed bins
# ------------------------------------------------------------

cruise <- cruise %>%
  mutate(

    altitude_bin =
      floor(mean_altitude / 500) * 500,

    speed_bin =
      floor(derived_groundspeed / 10) * 10
  )


# ------------------------------------------------------------
# 4. Summarise observed fuel rate
# ------------------------------------------------------------

observed_cells <- cruise %>%
  group_by(
    altitude_bin,
    speed_bin
  ) %>%
  summarise(

    observations = n(),

    flights =
      n_distinct(flight_id),

    mean_observed_fuel_rate =
      mean(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    median_observed_fuel_rate =
      median(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    sd_observed_fuel_rate =
      sd(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


# ------------------------------------------------------------
# 5. Keep sufficiently supported regions
# ------------------------------------------------------------

supported_cells <- observed_cells %>%
  filter(
    observations >= 5
  )


cat("\n============================================\n")
cat("OBSERVED SUPPORTED REGIONS\n")
cat("============================================\n")

cat(
  "Cells with >= 5 observations:",
  nrow(supported_cells),
  "\n"
)


# ------------------------------------------------------------
# 6. Prepare prediction data for observed cells
# ------------------------------------------------------------

scenario_data <- supported_cells %>%
  mutate(

    mean_altitude =
      altitude_bin + 250,

    derived_groundspeed =
      speed_bin + 5,

    distance_km =
      median(
        cruise$distance_km,
        na.rm = TRUE
      ),

    altitude_change = 0,

    derived_vertical_rate = 0,

    flight_phase =
      factor(
        "Cruise",
        levels = levels(
          train_data$flight_phase
        )
      )
  )


# ------------------------------------------------------------
# 7. Predict fuel rate for supported cells
# ------------------------------------------------------------

scenario_data$predicted_fuel_rate <- predict(
  rf_model,
  newdata = scenario_data
)


# ------------------------------------------------------------
# 8. Compare predicted and observed values
# ------------------------------------------------------------

scenario_data <- scenario_data %>%
  mutate(

    prediction_error =
      predicted_fuel_rate -
      median_observed_fuel_rate,

    absolute_difference =
      abs(prediction_error)
  )


cat("\n============================================\n")
cat("SUPPORTED SCENARIOS: MODEL VS OBSERVED\n")
cat("============================================\n")

print(
  scenario_data %>%
    arrange(predicted_fuel_rate) %>%
    select(
      altitude_bin,
      speed_bin,
      observations,
      flights,
      median_observed_fuel_rate,
      predicted_fuel_rate,
      prediction_error
    ) %>%
    head(20)
)


# ------------------------------------------------------------
# 9. Model-based lowest supported scenario
# ------------------------------------------------------------

best_supported <- scenario_data %>%
  arrange(
    predicted_fuel_rate
  ) %>%
  slice(1)


cat("\n============================================\n")
cat("LOWEST MODEL-PREDICTED SUPPORTED SCENARIO\n")
cat("============================================\n")

print(
  best_supported
)


# ------------------------------------------------------------
# 10. Observed lowest-fuel supported scenario
# ------------------------------------------------------------

best_observed <- scenario_data %>%
  arrange(
    median_observed_fuel_rate
  ) %>%
  slice(1)


cat("\n============================================\n")
cat("LOWEST OBSERVED SUPPORTED SCENARIO\n")
cat("============================================\n")

print(
  best_observed
)


# ------------------------------------------------------------
# 11. Compare model and observed patterns
# ------------------------------------------------------------

comparison_rmse <- sqrt(
  mean(
    (
      scenario_data$predicted_fuel_rate -
      scenario_data$median_observed_fuel_rate
    )^2,
    na.rm = TRUE
  )
)

comparison_mae <- mean(
  abs(
    scenario_data$predicted_fuel_rate -
    scenario_data$median_observed_fuel_rate
  ),
  na.rm = TRUE
)

cat("\n============================================\n")
cat("SUPPORTED-CELL MODEL VS OBSERVED ERROR\n")
cat("============================================\n")

cat(
  "RMSE:",
  round(comparison_rmse, 4),
  "\n"
)

cat(
  "MAE:",
  round(comparison_mae, 4),
  "\n"
)


# ------------------------------------------------------------
# 12. Plot observed fuel rate
# ------------------------------------------------------------

p1 <- ggplot(
  observed_cells %>%
    filter(observations >= 5),
  aes(
    x = speed_bin,
    y = altitude_bin,
    fill = median_observed_fuel_rate
  )
) +
  geom_tile() +
  labs(
    title =
      "Observed A321 Cruise Fuel Rate",
    subtitle =
      "Only altitude-speed cells with at least 5 observations",
    x =
      "Groundspeed",
    y =
      "Altitude",
    fill =
      "Median fuel rate\n(kg/min)"
  )

ggsave(
  "data/a321_observed_fuel_heatmap.png",
  p1,
  width = 9,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 13. Plot model vs observed
# ------------------------------------------------------------

p2 <- ggplot(
  scenario_data,
  aes(
    x = median_observed_fuel_rate,
    y = predicted_fuel_rate,
    size = observations
  )
) +
  geom_point(
    alpha = 0.7
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  labs(
    title =
      "Predicted vs Observed Fuel Rate",
    x =
      "Observed median fuel rate (kg/min)",
    y =
      "Predicted fuel rate (kg/min)",
    size =
      "Observations"
  )

ggsave(
  "data/a321_predicted_vs_observed_optimization.png",
  p2,
  width = 8,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 14. Save results
# ------------------------------------------------------------

write_csv(
  observed_cells,
  "data/a321_observed_altitude_speed_cells.csv"
)

write_csv(
  scenario_data,
  "data/a321_supported_optimization_validation.csv"
)

write_csv(
  best_supported,
  "data/a321_best_supported_scenario.csv"
)

write_csv(
  best_observed,
  "data/a321_best_observed_scenario.csv"
)


# ------------------------------------------------------------
# 15. Complete
# ------------------------------------------------------------

cat("\n============================================\n")
cat("STEP 15 COMPLETE\n")
cat("============================================\n")

cat(
  "Validation tables and plots have been created.\n"
)

# ========================================================================
# step16_robust_optimization.R
# ========================================================================

# ============================================================
# STEP 16: ROBUST A321 CRUISE OPTIMIZATION
# ============================================================

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages(
    "dplyr",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages(
    "randomForest",
    repos = "https://cloud.r-project.org"
  )
}

library(dplyr)
library(randomForest)


# ------------------------------------------------------------
# 1. Load data and model
# ------------------------------------------------------------

train_data <- readRDS(
  "data/a321_train_model.rds"
)

rf_model <- readRDS(
  "data/a321_tuned_random_forest.rds"
)


# ------------------------------------------------------------
# 2. Main cruise region
# ------------------------------------------------------------

cruise <- train_data %>%
  filter(
    flight_phase == "Cruise",
    mean_altitude >= 30000
  )


cat("\n============================================\n")
cat("MAIN CRUISE REGION\n")
cat("============================================\n")

cat(
  "Observations:",
  nrow(cruise),
  "\n"
)

cat(
  "Flights:",
  n_distinct(cruise$flight_id),
  "\n"
)


# ------------------------------------------------------------
# 3. Create altitude and speed cells
# ------------------------------------------------------------

cruise <- cruise %>%
  mutate(
    altitude_bin =
      floor(mean_altitude / 500) * 500,

    speed_bin =
      floor(derived_groundspeed / 10) * 10
  )


# ------------------------------------------------------------
# 4. Observed cell statistics
# ------------------------------------------------------------

cells <- cruise %>%
  group_by(
    altitude_bin,
    speed_bin
  ) %>%
  summarise(

    observations = n(),

    flights =
      n_distinct(flight_id),

    mean_observed_fuel_rate =
      mean(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    median_observed_fuel_rate =
      median(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    sd_observed_fuel_rate =
      sd(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


# ------------------------------------------------------------
# 5. Keep well-supported cells
# ------------------------------------------------------------

supported <- cells %>%
  filter(
    observations >= 10,
    flights >= 5
  )


cat("\n============================================\n")
cat("SUPPORTED CRUISE CELLS\n")
cat("============================================\n")

cat(
  "Cells:",
  nrow(supported),
  "\n"
)


# ------------------------------------------------------------
# 6. Prepare prediction scenarios
# ------------------------------------------------------------

scenario <- supported %>%
  mutate(

    mean_altitude =
      altitude_bin + 250,

    derived_groundspeed =
      speed_bin + 5,

    distance_km =
      median(
        cruise$distance_km,
        na.rm = TRUE
      ),

    altitude_change = 0,

    derived_vertical_rate = 0,

    flight_phase =
      factor(
        "Cruise",
        levels = levels(
          train_data$flight_phase
        )
      )
  )


# ------------------------------------------------------------
# 7. Model predictions
# ------------------------------------------------------------

scenario$predicted_fuel_rate <- predict(
  rf_model,
  newdata = scenario
)


# ------------------------------------------------------------
# 8. Compare observed and predicted
# ------------------------------------------------------------

scenario <- scenario %>%
  mutate(

    prediction_difference =
      predicted_fuel_rate -
      median_observed_fuel_rate,

    absolute_difference =
      abs(prediction_difference)
  )


cat("\n============================================\n")
cat("SUPPORTED CRUISE SCENARIOS\n")
cat("============================================\n")

print(
  scenario %>%
    arrange(
      predicted_fuel_rate
    ) %>%
    select(
      altitude_bin,
      speed_bin,
      observations,
      flights,
      median_observed_fuel_rate,
      predicted_fuel_rate,
      prediction_difference
    ) %>%
    head(20)
)


# ------------------------------------------------------------
# 9. Best model-supported scenario
# ------------------------------------------------------------

best_model_scenario <- scenario %>%
  arrange(
    predicted_fuel_rate
  ) %>%
  slice(1)


cat("\n============================================\n")
cat("BEST MODEL-SUPPORTED SCENARIO\n")
cat("============================================\n")

print(
  best_model_scenario
)


# ------------------------------------------------------------
# 10. Best observed scenario
# ------------------------------------------------------------

best_observed_scenario <- scenario %>%
  arrange(
    median_observed_fuel_rate
  ) %>%
  slice(1)


cat("\n============================================\n")
cat("BEST OBSERVED SUPPORTED SCENARIO\n")
cat("============================================\n")

print(
  best_observed_scenario
)


# ------------------------------------------------------------
# 11. Agreement between model and observations
# ------------------------------------------------------------

comparison_rmse <- sqrt(
  mean(
    (
      scenario$predicted_fuel_rate -
      scenario$median_observed_fuel_rate
    )^2,
    na.rm = TRUE
  )
)

comparison_mae <- mean(
  abs(
    scenario$predicted_fuel_rate -
    scenario$median_observed_fuel_rate
  ),
  na.rm = TRUE
)


cat("\n============================================\n")
cat("MODEL VS OBSERVED\n")
cat("============================================\n")

cat(
  "RMSE:",
  round(comparison_rmse, 4),
  "\n"
)

cat(
  "MAE:",
  round(comparison_mae, 4),
  "\n"
)


# ------------------------------------------------------------
# 12. Save results
# ------------------------------------------------------------

write.csv(
  supported,
  "data/a321_main_cruise_cells.csv",
  row.names = FALSE
)

write.csv(
  scenario,
  "data/a321_robust_optimization_results.csv",
  row.names = FALSE
)

write.csv(
  best_model_scenario,
  "data/a321_best_model_supported.csv",
  row.names = FALSE
)

write.csv(
  best_observed_scenario,
  "data/a321_best_observed_supported.csv",
  row.names = FALSE
)


cat("\n============================================\n")
cat("STEP 16 COMPLETE\n")
cat("============================================\n")

# ========================================================================
# step17_validation_optimization.R
# ========================================================================

# ============================================================
# STEP 17: OUT-OF-SAMPLE VALIDATION OF FUEL OPTIMIZATION
# ============================================================

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages(
    "dplyr",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages(
    "randomForest",
    repos = "https://cloud.r-project.org"
  )
}

library(dplyr)
library(randomForest)


# ------------------------------------------------------------
# 1. Load untouched validation data and final RF
# ------------------------------------------------------------

validation_data <- readRDS(
  "data/a321_validation_model.rds"
)

rf_model <- readRDS(
  "data/a321_tuned_random_forest.rds"
)


cat("\n============================================\n")
cat("VALIDATION DATA\n")
cat("============================================\n")

cat(
  "Rows:",
  nrow(validation_data),
  "\n"
)

cat(
  "Flights:",
  n_distinct(validation_data$flight_id),
  "\n"
)


# ------------------------------------------------------------
# 2. Generate OUT-OF-SAMPLE predictions
# ------------------------------------------------------------

validation_data$predicted_fuel_rate <- predict(
  rf_model,
  newdata = validation_data
)


# ------------------------------------------------------------
# 3. Keep main cruise region
# ------------------------------------------------------------

cruise_validation <- validation_data %>%
  filter(
    flight_phase == "Cruise",
    mean_altitude >= 30000
  )


cat("\n============================================\n")
cat("VALIDATION CRUISE REGION\n")
cat("============================================\n")

cat(
  "Rows:",
  nrow(cruise_validation),
  "\n"
)

cat(
  "Flights:",
  n_distinct(cruise_validation$flight_id),
  "\n"
)


# ------------------------------------------------------------
# 4. Create altitude and speed bins
# ------------------------------------------------------------

cruise_validation <- cruise_validation %>%
  mutate(

    altitude_bin =
      floor(mean_altitude / 500) * 500,

    speed_bin =
      floor(derived_groundspeed / 10) * 10
  )


# ------------------------------------------------------------
# 5. Summarise actual and predicted values
# ------------------------------------------------------------

cells <- cruise_validation %>%
  group_by(
    altitude_bin,
    speed_bin
  ) %>%
  summarise(

    observations = n(),

    flights =
      n_distinct(flight_id),

    observed_median_fuel_rate =
      median(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    observed_mean_fuel_rate =
      mean(
        fuel_rate_kg_min,
        na.rm = TRUE
      ),

    predicted_median_fuel_rate =
      median(
        predicted_fuel_rate,
        na.rm = TRUE
      ),

    predicted_mean_fuel_rate =
      mean(
        predicted_fuel_rate,
        na.rm = TRUE
      ),

    .groups = "drop"
  )


# ------------------------------------------------------------
# 6. Keep sufficiently supported cells
# ------------------------------------------------------------

supported <- cells %>%
  filter(
    observations >= 5,
    flights >= 3
  )


cat("\n============================================\n")
cat("SUPPORTED VALIDATION CELLS\n")
cat("============================================\n")

cat(
  "Cells:",
  nrow(supported),
  "\n"
)


# ------------------------------------------------------------
# 7. Model vs observed error
# ------------------------------------------------------------

cell_rmse <- sqrt(
  mean(
    (
      supported$predicted_median_fuel_rate -
      supported$observed_median_fuel_rate
    )^2,
    na.rm = TRUE
  )
)

cell_mae <- mean(
  abs(
    supported$predicted_median_fuel_rate -
    supported$observed_median_fuel_rate
  ),
  na.rm = TRUE
)


cat("\n============================================\n")
cat("CELL-LEVEL OUT-OF-SAMPLE ERROR\n")
cat("============================================\n")

cat(
  "RMSE:",
  round(cell_rmse, 4),
  "\n"
)

cat(
  "MAE:",
  round(cell_mae, 4),
  "\n"
)


# ------------------------------------------------------------
# 8. Correlation between observed and predicted cells
# ------------------------------------------------------------

spearman_correlation <- cor(
  supported$observed_median_fuel_rate,
  supported$predicted_median_fuel_rate,
  method = "spearman",
  use = "complete.obs"
)


cat(
  "Spearman correlation:",
  round(
    spearman_correlation,
    4
  ),
  "\n"
)


# ------------------------------------------------------------
# 9. Lowest observed supported cell
# ------------------------------------------------------------

best_observed <- supported %>%
  arrange(
    observed_median_fuel_rate
  ) %>%
  slice(1)


cat("\n============================================\n")
cat("LOWEST OBSERVED VALIDATION CELL\n")
cat("============================================\n")

print(
  best_observed
)


# ------------------------------------------------------------
# 10. Lowest predicted supported cell
# ------------------------------------------------------------

best_predicted <- supported %>%
  arrange(
    predicted_median_fuel_rate
  ) %>%
  slice(1)


cat("\n============================================\n")
cat("LOWEST PREDICTED VALIDATION CELL\n")
cat("============================================\n")

print(
  best_predicted
)


# ------------------------------------------------------------
# 11. Top 10 lowest observed cells
# ------------------------------------------------------------

cat("\n============================================\n")
cat("TOP 10 LOWEST OBSERVED VALIDATION CELLS\n")
cat("============================================\n")

print(
  supported %>%
    arrange(
      observed_median_fuel_rate
    ) %>%
    select(
      altitude_bin,
      speed_bin,
      observations,
      flights,
      observed_median_fuel_rate,
      predicted_median_fuel_rate
    ) %>%
    head(10)
)


# ------------------------------------------------------------
# 12. Top 10 lowest predicted cells
# ------------------------------------------------------------

cat("\n============================================\n")
cat("TOP 10 LOWEST PREDICTED VALIDATION CELLS\n")
cat("============================================\n")

print(
  supported %>%
    arrange(
      predicted_median_fuel_rate
    ) %>%
    select(
      altitude_bin,
      speed_bin,
      observations,
      flights,
      observed_median_fuel_rate,
      predicted_median_fuel_rate
    ) %>%
    head(10)
)


# ------------------------------------------------------------
# 13. Compare rankings
# ------------------------------------------------------------

supported <- supported %>%
  mutate(
    observed_rank =
      rank(
        observed_median_fuel_rate,
        ties.method = "average"
      ),

    predicted_rank =
      rank(
        predicted_median_fuel_rate,
        ties.method = "average"
      ),

    rank_difference =
      abs(
        observed_rank -
        predicted_rank
      )
  )


mean_rank_difference <- mean(
  supported$rank_difference,
  na.rm = TRUE
)


cat("\n============================================\n")
cat("RANK AGREEMENT\n")
cat("============================================\n")

cat(
  "Mean absolute rank difference:",
  round(
    mean_rank_difference,
    2
  ),
  "\n"
)


# ------------------------------------------------------------
# 14. Save results
# ------------------------------------------------------------

write.csv(
  supported,
  "data/a321_out_of_sample_optimization.csv",
  row.names = FALSE
)

write.csv(
  best_observed,
  "data/a321_best_observed_validation_cell.csv",
  row.names = FALSE
)

write.csv(
  best_predicted,
  "data/a321_best_predicted_validation_cell.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 15. Complete
# ------------------------------------------------------------

cat("\n============================================\n")
cat("STEP 17 COMPLETE\n")
cat("============================================\n")

# ========================================================================
# step18_final_summary.R
# ========================================================================

# ============================================================
# STEP 18: FINAL PROJECT SUMMARY
# ============================================================

if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages(
    "readr",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages(
    "dplyr",
    repos = "https://cloud.r-project.org"
  )
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages(
    "ggplot2",
    repos = "https://cloud.r-project.org"
  )
}

library(readr)
library(dplyr)
library(ggplot2)


# ------------------------------------------------------------
# 1. Load results
# ------------------------------------------------------------

model_results <- read_csv(
  "data/a321_all_model_results.csv",
  show_col_types = FALSE
)

rf_importance <- read_csv(
  "data/a321_tuned_rf_importance.csv",
  show_col_types = FALSE
)

phase_results <- read_csv(
  "data/phase_fuel_statistics.csv",
  show_col_types = FALSE
)

optimization <- read_csv(
  "data/a321_robust_optimization_results.csv",
  show_col_types = FALSE
)

validation_optimization <- read_csv(
  "data/a321_out_of_sample_optimization.csv",
  show_col_types = FALSE
)


# ------------------------------------------------------------
# 2. Final model performance table
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FINAL MODEL PERFORMANCE\n")
cat("============================================\n")

print(model_results)


# ------------------------------------------------------------
# 3. Best predictive model
# ------------------------------------------------------------

best_model <- model_results %>%
  filter(
    !is.na(RMSE)
  ) %>%
  arrange(
    RMSE
  ) %>%
  slice(1)


cat("\n============================================\n")
cat("BEST PREDICTIVE MODEL\n")
cat("============================================\n")

print(best_model)


# ------------------------------------------------------------
# 4. Feature importance
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FEATURE IMPORTANCE\n")
cat("============================================\n")

print(
  rf_importance
)


# ------------------------------------------------------------
# 5. Phase results
# ------------------------------------------------------------

cat("\n============================================\n")
cat("FUEL RATE BY FLIGHT PHASE\n")
cat("============================================\n")

print(
  phase_results
)


# ------------------------------------------------------------
# 6. Optimization validation
# ------------------------------------------------------------

cat("\n============================================\n")
cat("OUT-OF-SAMPLE OPTIMIZATION VALIDATION\n")
cat("============================================\n")

cat(
  "Supported validation cells:",
  nrow(validation_optimization),
  "\n"
)

print(
  validation_optimization %>%
    arrange(
      observed_median_fuel_rate
    ) %>%
    select(
      altitude_bin,
      speed_bin,
      observations,
      flights,
      observed_median_fuel_rate,
      predicted_median_fuel_rate
    ) %>%
    head(10)
)


# ------------------------------------------------------------
# 7. Create model comparison plot
# ------------------------------------------------------------

model_plot_data <- model_results %>%
  filter(
    !is.na(RMSE)
  )

p1 <- ggplot(
  model_plot_data,
  aes(
    x = reorder(Model, RMSE),
    y = RMSE
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title =
      "A321 Fuel-Rate Prediction Model Comparison",
    x =
      "Model",
    y =
      "Validation RMSE (kg/min)"
  )

ggsave(
  "data/final_model_comparison.png",
  p1,
  width = 8,
  height = 5,
  dpi = 150
)


# ------------------------------------------------------------
# 8. Feature importance plot
# ------------------------------------------------------------

importance_plot_data <- rf_importance %>%
  arrange(
    `%IncMSE`
  )

p2 <- ggplot(
  importance_plot_data,
  aes(
    x = reorder(
      rownames(importance_plot_data),
      `%IncMSE`
    ),
    y = `%IncMSE`
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title =
      "Random Forest Feature Importance",
    x =
      "Feature",
    y =
      "% Increase in MSE"
  )

ggsave(
  "data/final_feature_importance.png",
  p2,
  width = 8,
  height = 5,
  dpi = 150
)


# ------------------------------------------------------------
# 9. Observed optimization heatmap
# ------------------------------------------------------------

p3 <- ggplot(
  validation_optimization,
  aes(
    x = speed_bin,
    y = altitude_bin,
    fill = observed_median_fuel_rate
  )
) +
  geom_tile() +
  labs(
    title =
      "Observed A321 Cruise Fuel Rate",
    subtitle =
      "Validation cells with sufficient observations",
    x =
      "Groundspeed",
    y =
      "Altitude",
    fill =
      "Median fuel rate"
  )

ggsave(
  "data/final_observed_optimization_heatmap.png",
  p3,
  width = 8,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 10. Predicted optimization heatmap
# ------------------------------------------------------------

p4 <- ggplot(
  validation_optimization,
  aes(
    x = speed_bin,
    y = altitude_bin,
    fill = predicted_median_fuel_rate
  )
) +
  geom_tile() +
  labs(
    title =
      "Predicted A321 Cruise Fuel Rate",
    subtitle =
      "Out-of-sample validation cells",
    x =
      "Groundspeed",
    y =
      "Altitude",
    fill =
      "Predicted fuel rate"
  )

ggsave(
  "data/final_predicted_optimization_heatmap.png",
  p4,
  width = 8,
  height = 6,
  dpi = 150
)


# ------------------------------------------------------------
# 11. Create concise project summary
# ------------------------------------------------------------

summary_lines <- c(

  "A321 FUEL CONSUMPTION PROJECT",
  "",
  "Dataset:",
  "Airbus A321 flights from the training portion of the dataset.",
  "",
  "Modeling target:",
  "Fuel rate (kg/min).",
  "",
  paste(
    "Best predictive model:",
    best_model$Model
  ),

  paste(
    "Validation RMSE:",
    round(best_model$RMSE, 4),
    "kg/min"
  ),

  paste(
    "Validation MAE:",
    round(best_model$MAE, 4),
    "kg/min"
  ),

  paste(
    "Validation R-squared:",
    round(best_model$R_squared, 4)
  ),

  "",
  "Optimization analysis:",
  "Model-based and observed cruise-condition analysis",
  "was performed using altitude and groundspeed.",
  "",
  "Important limitation:",
  "The optimization analysis is observational/model-based",
  "and should not be interpreted as an operational flight recommendation."
)

writeLines(
  summary_lines,
  "data/a321_project_summary.txt"
)


# ------------------------------------------------------------
# 12. Save final results
# ------------------------------------------------------------

write_csv(
  model_results,
  "data/final_model_results.csv"
)

write_csv(
  rf_importance,
  "data/final_feature_importance.csv"
)

write_csv(
  phase_results,
  "data/final_phase_statistics.csv"
)


# ------------------------------------------------------------
# 13. Complete
# ------------------------------------------------------------

cat("\n============================================\n")
cat("STEP 18 COMPLETE\n")
cat("============================================\n")

cat(
  "Final report figures and tables have been created.\n"
)

cat(
  "Summary saved to data/a321_project_summary.txt\n"
)

# ========================================================================
# FINAL STATUS
# ========================================================================

cat("\n")
cat("=======================================================================\n")
cat("A321 FUEL CONSUMPTION PROJECT: PIPELINE COMPLETED\n")
cat("=======================================================================\n")
cat("Working directory: ", getwd(), "\n", sep = "")
cat("Final summary: data/a321_project_summary.txt\n")
cat("Final model results: data/final_model_results.csv\n")
cat("Final model comparison figure: data/final_model_comparison.png\n")
cat("=======================================================================\n")
