# 🚨 AI-Based Intrusion Detection System  
### Detecting Evolving and Zero-Day Attacks using Hybrid Machine Learning

---

## 📌 Overview

This project implements a **hybrid AI-based Network Intrusion Detection System (NIDS)** designed to detect both:

- **Known attacks** using supervised machine learning  
- **Previously unseen (zero-day) attacks** using anomaly detection  

Traditional intrusion detection systems rely on static signatures and historical attack data. However, in real-world environments:

- Attack patterns evolve over time (**concept drift**)  
- New attacks appear without prior signatures (**zero-day attacks**)  

To address this, this project combines **supervised classification** and **anomaly detection** into a unified architecture that improves detection coverage.

---

## 🎯 Project Objectives

- Detect known network attacks with high accuracy  
- Identify anomalous or previously unseen behavior  
- Compare supervised and anomaly-based detection approaches  
- Design a hybrid IDS architecture for improved coverage  
- Analyze trade-offs between detection accuracy and false positives  
- Provide an interactive dashboard for monitoring and analysis  

---

## 🧠 Key Results

### Supervised Models
- **Random Forest (Best Performer)**
  - Precision: 0.9882  
  - Recall: 0.9882  
  - ROC-AUC: 0.9991  

### Anomaly Detection (Ensemble)
- Recall: ~0.998  
- Precision: ~0.57  

### Insight
Supervised models provide high accuracy for known attacks, while anomaly detection ensures high recall for unknown threats. A hybrid approach improves overall coverage but introduces false positives.

---

## 🏗️ System Architecture

```text
                         ┌──────────────────────────┐
                         │   CIC-IDS2017 Dataset     │
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
                         │ Alerts / Dashboard Output │
                         └──────────────────────────┘

### **Dataset**

CIC-IDS2017 Dataset (UNB)

Contains realistic network traffic with multiple attack scenarios:
DoS / DDoS
Brute Force
Botnet
Web Attacks
Infiltration
Heartbleed
Benign traffic

⚠️ The dataset is not included in this repository due to size constraints.

## ⚙️ **Methodology**

### **Data Preprocessing**

Merged multiple daily CSV files
Removed null, infinite, and duplicate values
Converted labels to binary (Attack vs Benign)
Applied feature scaling (MinMaxScaler)
Used SMOTE for balancing supervised training data
Created separate datasets for supervised and anomaly detection

### **Supervised Models**

Random Forest
Gradient Boosting
Balanced Random Forest

### **Anomaly Detection Models**

Isolation Forest
Local Outlier Factor (LOF)
One-Class SVM
Autoencoder

### **Hybrid Approach**

Combines supervised predictions with anomaly scores
Applies score normalization and thresholding
Produces final alerts for analysis

## **Repository Structure**

NIDS/
│
├── models/
│   ├── best_gb.pkl
│   ├── ensemble_minmax_scalers.pkl
│   ├── isolation_forest.pkl
│   ├── oneclass_svm.pkl
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
├── requirements.txt
└── README.md

## ⚠️ **Files Not Included**

Due to size and security constraints:

Models

best_random_forest.pkl
best_balanced_rf.pkl
lof_novelty.pkl

Data

CIC-IDS2017 dataset

Environment

.env file

## ▶️ **Setup Instructions**

-- Clone Repository

git clone https://github.com/your-username/NIDS.git
cd NIDS

-- Create Virtual Environment

python -m venv .venv
Activate Environment

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

-- Install Dependencies

pip install -r requirements.txt

-- Run Pipeline

python src/preprocessing/preprocess.py
python src/training/train_supervised.py
python src/training/train_anomaly.py
python src/evaluation/evaluate_models.py

-- Run Dashboard

streamlit run src/dashboard/app.py

## 📊 **Dashboard**

### The Streamlit dashboard provides:

Model performance comparison
ROC and PR curve visualization
Anomaly detection insights
Threshold tuning
Interactive exploration of results

### ⚠️ **Limitations**

High false positive rate in anomaly detection
Evaluation limited to offline dataset
No automated concept drift handling
Limited explainability integration
Increased computational complexity

### 🚀 **Future Work**

Real-time streaming IDS (Kafka / Spark)
Concept drift detection and automated retraining
Deep learning-based anomaly detection
Explainable AI (SHAP, LIME)
Integration with SIEM systems

👩‍💻 Author

Tanishka Ganesh Mali
M.S. Cybersecurity — Penn State
