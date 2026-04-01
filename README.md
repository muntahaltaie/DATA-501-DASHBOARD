# DATA-501-DASHBOARD

## Oil Sands Lake Analysis Dashboard

**Live App:** https://oil-sands-dashboard.streamlit.app/

## Overview

This project is an interactive Streamlit dashboard for analyzing water chemistry across Near, Mid, and Far distance groups in the Alberta oil sands region. The dashboard was designed to support exploratory analysis, multivariate analysis, and model-based interpretation using a cleaned and standardized lake dataset.

## Features

- Data filtering by lake and distance group
- Interactive study area map with AR6 reference site
- Distance calculations from AR6 using geographic coordinates
- Trend analysis over time for selected variables
- PCA for multivariate structure
- ANOVA for group comparisons
- Random Forest for feature importance and classification
- Standardized units across all included parameters

## Methods

- Distances were calculated using the Haversine formula from the AR6 oil sands reference site.
- Lakes were grouped by distance category:
  - **Near:** less than 20 km
  - **Mid:** 20–50 km
  - **Far:** greater than 50 km
- **PCA** was used to examine multivariate structure and dominant gradients in water chemistry.
- **ANOVA** was used to test for statistically significant differences across groups.
- **Random Forest** was used to identify the strongest variables associated with lake grouping.

## Study Area

The study lakes are located in the Athabasca oil sands region of Alberta. Spatial comparisons are made relative to the AR6 reference site, which represents central oil sands mining activity. The dashboard includes both a static study-area figure and an interactive map for geographic interpretation.

## Data Sources

- **Isadore Lake, Kearl Lake, McClelland Lake, Namur Lake:**  
  Oil Sands Monitoring / OSMP Portal  
  https://osmdataportal.alberta.ca/applications/public.html?publicuser=Guest#waterdata/stationoverview

- **Mildred Lake, Gregoire Lake:**  
  Alberta Water Quality Data Portal  
  https://environment.extranet.gov.ab.ca/apps/WaterQuality/dataportal/

## Project Structure

- `dashboard.py` → main Streamlit dashboard
- `FINAL_NORMALIZED_FULL.csv` → cleaned and standardized dataset
- `study_area_map.png` → study area reference figure
- `requirements.txt` → Python dependencies
- `README.md` → project overview and instructions

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
