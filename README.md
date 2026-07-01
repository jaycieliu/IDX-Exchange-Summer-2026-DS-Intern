# IDX-Exchange-DS-Intern: California Property Close Price Prediction

## Project Overview

This repository contains the internship project for predicting California residential property close prices using historical CRMLS sold property data.

The goal of this project is to build a machine learning workflow that can predict a property's final sales price, also known as `ClosePrice`, based on property-level characteristics such as living area, bedrooms, bathrooms, lot size, location-related fields, and other available listing attributes.

For this project, the analysis focuses only on:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`
- California CRMLS sold property records

The final deliverables include data preprocessing scripts, model training and evaluation code, documentation, and a final presentation summarizing the methodology, findings, model performance, and next steps.

---

## Week 1 Summary: Orientation & Setup

**Week 1 deliverables completed:**

- Reviewed the task prompt and project objective.
- Set up Python, Git, Jupyter Notebook, and the local project environment.
- Connected to the FTP server through FileZilla.
- Located and downloaded CRMLS sold property files.
- Reviewed the metadata file to understand key feature definitions.
- Confirmed dataset access and prepared initial notes on important columns.


## Week 2 Summary: Data Exploration

**Week 2 deliverables completed:**

- Loaded at least six months of CRMLS sold data into pandas.
- Combined multiple monthly files into one working dataset.
- Filtered the data to residential single-family properties.
- Explored the distribution of `ClosePrice`, living area, bedrooms, bathrooms, and lot size.
- Checked missing values and basic data quality issues.
- Documentation:
  - [Week2 Summary](/Week2/week2_summary.md)
- Codes:
  - [Week2 Notebook](/Week2/01_exploration.ipynb)
    
