# DATA-501-DASHBOARD

## Oil Sands Lake Analysis Dashboard

**Live App:** https://oil-sands-dashboard.streamlit.app/

## Overview

This project is an interactive Streamlit dashboard for analyzing water chemistry across Near, Mid, and Far distance groups in the Alberta oil sands region. The dashboard supports exploratory analysis, statistical testing, and machine learning-based insights using a cleaned and standardized dataset.

---

## Dashboard Structure (Tabs)

The dashboard is organized into multiple tabs, each serving a specific analytical purpose:

### Overview
- Summary of the dataset and study design
- Study area map and classification of lakes (Near, Mid, Far)
- Interactive map showing lake locations relative to AR6
- Distance calculations using geographic coordinates
- Sample counts and lake distribution
- Trend analysis over time for selected variables
- Data cleaning notes and unit reference table

### Distributions
- Boxplots comparing chemical parameters across distance groups
- Allows selection of multiple variables
- Helps visualize spread, outliers, and group differences

### PCA (Principal Component Analysis)
- Reduces dimensionality of chemical variables
- Identifies dominant gradients in water chemistry
- Includes:
  - PCA scatter plot
  - Explained variance table
  - Variable loadings (feature contributions)

### ANOVA
- Tests whether chemical variables differ significantly across groups
- Outputs:
  - F-statistics and p-values
  - Significant variables based on selected alpha
  - Boxplots for key variables

### Random Forest
- Machine learning model predicting lake group (or lake)
- Outputs:
  - Model accuracy
  - Classification report
  - Confusion matrix
  - Feature importance rankings

### Data Preview
- Interactive table of the dataset
- Allows filtering by lake and distance group
- Custom column selection
- Missing values displayed as blank
- Designed for inspecting raw and cleaned data

---

## Features

- Data filtering by lake and distance group
- Interactive geographic mapping with AR6 reference site
- Distance calculations using the Haversine formula
- Trend analysis across multiple variables
- PCA, ANOVA, and Random Forest integration
- Standardized units across all variables
- Customizable data preview and filtering

---

## Methods

- Distances were calculated using the Haversine formula from the AR6 oil sands reference site.
- Lakes were grouped into:
  - **Near:** <20 km  
  - **Mid:** 20–50 km  
  - **Far:** >50 km  
- PCA was used to identify multivariate structure in the dataset.
- ANOVA was used to test statistical differences between groups.
- Random Forest was used to determine the most important chemical predictors.

---

## Study Area

The study lakes are located in the Athabasca oil sands region of Alberta. Spatial comparisons are made relative to the AR6 reference site, which represents central oil sands mining activity.

---

## Data Sources

- **Isadore, Kearl, McClelland, Namur:**  
  Oil Sands Monitoring / OSMP Portal  
  https://osmdataportal.alberta.ca/applications/public.html?publicuser=Guest#waterdata/stationoverview  

- **Mildred, Gregoire:**  
  Alberta Water Quality Data Portal  
  https://environment.extranet.gov.ab.ca/apps/WaterQuality/dataportal/  

---

## Project Structure

- `dashboard.py` → main Streamlit dashboard  
- `FINAL_NORMALIZED_FULL.csv` → cleaned dataset  
- `study_area_map.png` → study area reference  
- `requirements.txt` → dependencies  
- `README.md` → documentation  

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
