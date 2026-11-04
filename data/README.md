# Data Directory Guide

This directory holds the raw and preprocessed datasets for the **Airline Satisfaction Intelligence** project.

## Expected Competition Files

To train the models from scratch or generate competition test predictions, download the official dataset from Kaggle:

Competition: **Predicting Airline Satisfaction — Playground Series Season 6 Episode 10**

Place the following three files directly into this directory:
1. `train.csv` — Full training set containing features and the target `satisfaction` column.
2. `test.csv` — Test set containing passenger features and the `id` column.
3. `sample_submission.csv` — Official sample submission template.

### Dataset Schema
- **Target**: `satisfaction` (`satisfied`, `neutral or dissatisfied` / `dissatisfied`)
- **Key Identifier**: `id`
- **Route Proxy**: `Flight Distance`
- **Flight & Delay**: `Departure Delay in Minutes`, `Arrival Delay in Minutes`
- **Demographics**: `Gender`, `Customer Type`, `Age`, `Type of Travel`, `Class`
- **13 Survey Indicators** (0 to 5 scale):
  - `Inflight wifi service`
  - `Departure/Arrival time convenient`
  - `Ease of Online booking`
  - `Gate location`
  - `Food and drink`
  - `Online boarding`
  - `Seat comfort`
  - `Inflight entertainment`
  - `On-board service`
  - `Leg room service`
  - `Baggage handling`
  - `Checkin service`
  - `Cleanliness`

## Built-in Demo Data
- `sample_passengers.csv`: A schema-compliant test dataset provided for immediate CLI verification and Streamlit batch prediction testing.

