# Water Quality Monitoring System

## Overview

The Water Quality Monitoring System is a Streamlit-based application that allows users to analyze water quality and predict potability (whether the water is potable or not) using machine learning techniques. The app provides an intuitive interface for data visualization, preprocessing, and prediction.

---

## Features

- **Dataset Overview**: Upload and explore your water quality dataset, including column properties and data distributions.
- **Preprocessing**: Handle missing values, visualize correlations, and resample the dataset using SMOTE-Tomek for balanced training.
- **Prediction**: Train a Random Forest model to predict water potability based on user-provided feature values.
- **Interactive Visualizations**: Utilize Plotly for interactive histograms and heatmaps.
- **Model Persistence**: Save and reuse trained models with `joblib`.

---

## Installation

To run this application, follow the steps below:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd water-quality-monitoring
   ```

2. Create a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## File Structure

```
water-quality-monitoring/
├── app.py               # Main application file
├── requirements.txt     # List of Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Ignored files and directories
```

---

## Requirements

The required Python libraries are listed in `requirements.txt`. Major dependencies include:

- `streamlit`
- `pandas`
- `scikit-learn`
- `imblearn`
- `plotly`
- `joblib`

---

## Usage

1. Upload a CSV file containing water quality data.
2. Navigate through the tabs:
   - View the dataset overview.
   - Preprocess the data to handle missing values and imbalanced classes.
   - Train and evaluate a Random Forest model.
3. Input water quality parameters and predict potability.

---

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for new features or bug fixes.

---
