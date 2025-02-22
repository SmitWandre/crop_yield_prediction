# Crop Yield Prediction Project

This project demonstrates a complete machine learning pipeline for predicting crop yield based on environmental parameters such as average temperature, total rainfall, and soil quality. It includes data preprocessing, model training, evaluation, and an interactive front-end built with Flask and Plotly that displays a professional, interactive interface and an informative graph.

## Project Overview

This project predicts crop yield using a linear regression model built on historical data. Users can input environmental parameters via a professional web interface. The app provides:
- **Data Preprocessing:** Loads and cleans a CSV dataset.
- **Model Training:** Trains a linear regression model and saves it.
- **Evaluation:** Computes key regression metrics such as MAE, RMSE, and R².
- **Interactive Frontend:** A Flask-based web app that uses Bootstrap for styling and Plotly for interactive data visualization.

## Prerequisites

Before setting up and running this project, ensure that you have the following installed on your system:

- **Python 3.12:**  
  The project is developed and tested using Python 3.12. You can download it from [python.org](https://www.python.org/downloads/).

- **pip:**  
  The Python package installer, which is typically bundled with Python 3.12. You can verify its installation by running:
  ```bash
  pip --version
