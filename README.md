# 🌊 Water Scarcity Streamflow Prediction Hackathon

This repository contains the codebase, data preprocessing pipeline, and modeling framework developed for the **Capgemini Invent Water Scarcity Hackathon**. The goal of this project is to forecast streamflow using a hybrid approach that combines causal inference with machine learning to support water resource planning in the face of increasing scarcity.

---

## 🚀 Project Overview

Streamflow forecasting is essential for understanding hydrological dynamics and managing water availability. This project builds a robust and generalizable model that:

- Integrates **static** and **dynamic** features across diverse geographies
- Estimates **causal effects** of key environmental drivers
- Leverages **gradient boosting models** with temporal cross-validation
- Applies **quantile conformal prediction** for calibrated uncertainty estimates

---

## 🧱 Repository Structure
```
├── data/ # Raw and processed data (or pointers to data sources)
├── notebooks/ # Jupyter notebooks (EDA, preprocessing, experiments)
├── src/
│ ├── preprocessing/ # Modular preprocessing scripts
│ ├── features/ # Feature engineering + selection logic
│ ├── models/ # Model training, evaluation, and calibration
│ └── utils/ # Helper functions, logging, config
├── results/ # Plots, tables, and outputs
├── Makefile # Orchestration of preprocessing steps
├── requirements.txt # Python dependencies
└── README.md # You're here
```


---

## 🔧 Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/Iyeleon/water-scarcity-streamflow.git
   cd water-scarcity-streamflow
   ```

2. (Optional) Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install requirements.txt
```

4. Run Preprocessing pipeline:
```
make prep_data
make prep_mini_dataset
make final_data
make feature_selection
```

4. Training and inference
Run train_model.ipynb

🧪 Methodology
The full modeling approach, including data preprocessing, causal analysis, feature selection, and model training, is described in detail in the 📄 project report.

📈 Results
For detailed evaluation metrics, visualizations, and insights on model performance and generalization across regions, please refer to the 📄 report.


