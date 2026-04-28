# AI-Based Intrusion Detection System for Detecting Evolving and Zero-Day Attacks

## Overview

This project implements a hybrid AI-based Network Intrusion Detection System (NIDS) designed to detect both known attacks and previously unseen or zero-day attack behavior.

Traditional intrusion detection systems often rely on static signatures or models trained on historical attack patterns. However, real-world attackers continuously change their techniques, which creates concept drift and makes static detection unreliable. This project addresses that problem by combining supervised machine learning models with unsupervised anomaly detection techniques.

The system uses the CIC-IDS2017 dataset and evaluates multiple detection models using precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices.

---

## Project Goals

The main goals of this project are:

- Detect known network attacks using supervised machine learning
- Identify unknown or zero-day-like behavior using anomaly detection
- Compare supervised and unsupervised model performance
- Build a hybrid IDS architecture that improves detection coverage
- Analyze the trade-off between detection accuracy and false positives
- Provide a dashboard-based view of model results and alerts

---

## System Architecture

```text
                         ┌──────────────────────────┐
                         │   CIC-IDS2017 Dataset     │
                         │  Network Flow Records     │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │    Data Preprocessing     │
                         │  - Cleaning               │
                         │  - Label Encoding         │
                         │  - Feature Scaling        │
                         │  - Class Balancing        │
                         └─────────────┬────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
        ┌───────────────────────┐           ┌────────────────────────┐
        │  Supervised Branch     │           │  Anomaly Detection      │
        │  Known Attack Detection│           │  Zero-Day Detection     │
        └───────────┬───────────┘           └────────────┬───────────┘
                    │                                    │
                    │                                    │
        ┌───────────▼───────────┐           ┌────────────▼───────────┐
        │ Random Forest          │           │ Isolation Forest        │
        │ Gradient Boosting      │           │ Local Outlier Factor    │
        │ Balanced Random Forest │           │ One-Class SVM           │
        └───────────┬───────────┘           │ Autoencoder             │
                    │                       └────────────┬───────────┘
                    │                                    │
                    └──────────────────┬─────────────────┘
                                       ▼
                         ┌──────────────────────────┐
                         │   Hybrid Decision Layer   │
                         │ Score Fusion + Thresholds │
                         └─────────────┬────────────┘
                                       ▼
                         ┌──────────────────────────┐
                         │  Alerts / Reports / UI    │
                         │  Streamlit Dashboard      │
                         └──────────────────────────┘


##  Dataset

This project uses the CIC-IDS2017 dataset from the Canadian Institute for Cybersecurity.

The dataset contains realistic benign and attack network traffic, including:

Brute force attacks
DoS attacks
DDoS attacks
Botnet traffic
Web attacks
Infiltration attempts
Heartbleed
Benign traffic

The raw and processed dataset files are not included in this repository due to size limitations.

## Repository Structure

NIDS/
│
├── models/
│   ├── best_gb.pkl
│   ├── ensemble_minmax_scalers.pkl
│   ├── isolation_forest.pkl
│   ├── oneclass_svm.pkl
│   └── README.md
│
├── outputs/
│   ├── reports/
│   ├── plots/
│   └── metrics/
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── dashboard/
│
├── notebooks/
│
├── requirements.txt
├── .gitignore
└── README.md

## Files Not Included

Some files are intentionally not uploaded to GitHub.

## Large Model Files

The following model files were not uploaded because of GitHub file size limits:

best_random_forest.pkl
best_rf.pkl
best_balanced_rf.pkl
lof_novelty.pkl

# Setup Instructions
1. Clone the repository

git clone https://github.com/your-username/NIDS.git
cd NIDS

2. Create a virtual environment
python -m venv .venv

Activate it:

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Download the dataset

Download CIC-IDS2017 from:

https://www.unb.ca/cic/datasets/ids-2017.html

Place the raw CSV files inside:

data/raw/

5. Run preprocessing

python src/preprocessing/preprocess.py

6. Train supervised models

python src/training/train_supervised.py

7. Train anomaly detection models

python src/training/train_anomaly.py

8. Run evaluation

python src/evaluation/evaluate_models.py

9. Launch dashboard

streamlit run src/dashboard/app.py
