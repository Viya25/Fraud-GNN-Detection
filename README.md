# 🚀 FraudLens: Graph-Based Credit Card Fraud Detection

## 📌 Overview
FraudLens is an end-to-end data science project for detecting fraudulent credit card transactions using both traditional machine learning models and Graph Neural Networks (GNNs).

The project compares baseline models (Logistic Regression, Random Forest, XGBoost) with a Graph Convolutional Network (GCN) to analyze performance differences in highly imbalanced datasets.

---

## 🎯 Problem Statement
Credit card fraud detection is challenging due to:
- Extreme class imbalance (~0.17% fraud cases)
- Evolving fraud patterns
- Lack of relational modeling in traditional ML methods

---

## 🧠 Solution Approach
This project introduces a **graph-based approach** where:
- Each transaction is treated as a node
- Relationships are built using K-Nearest Neighbors (KNN)
- A Graph Neural Network (GCN) is trained to detect fraud

---

## 📂 Project Structure
Fraud-GNN-Detection/
│
├── data/
│ └── creditcard.csv
│
├── notebooks/
│ ├── 01_data_analysis.ipynb
│ ├── 02_baseline_models.ipynb
│ ├── 03_graph_construction.ipynb
│ └── 04_gnn_model.ipynb
│
├── results/
│ └── fraud_detection_results.csv
│
└── README.md


---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- PyTorch
- PyTorch Geometric
- NetworkX
- Matplotlib, Seaborn

---

## 📊 Model Performance

| Model | ROC-AUC | PR-AUC |
|------|--------|--------|
| Logistic Regression | 0.956 | 0.744 |
| Random Forest | 0.958 | 0.864 |
| XGBoost | **0.965** | **0.880** |
| GCN | **0.969** | 0.636 |

---

## 🔍 Key Insights

- GCN achieved the highest ROC-AUC → strong ranking ability
- XGBoost achieved the highest PR-AUC → best fraud detection
- Graph models capture relationships but struggle with extreme class imbalance

---

## 📈 Output

The model generates:
- Fraud probability for each transaction
- Final prediction (Fraud / Not Fraud)

Example:

| Transaction_ID | Fraud_Probability | Prediction |
|----------------|------------------|------------|
| 10234 | 0.91 | Fraud |
| 56789 | 0.02 | Normal |

---

## 🏦 Real-World Applications

- Credit card fraud detection (Banks)
- Online payment security (FinTech)
- E-commerce fraud prevention
- Cybersecurity anomaly detection

---

## 🚀 How to Run

1. Clone the repository:

git clone https://github.com/Viya25/Fraud-GNN-Detection.git


2. Install dependencies:

pip install -r requirements.txt


3. Run notebooks in order:
- 01 → Data Analysis
- 02 → Baseline Models
- 03 → Graph Construction
- 04 → GNN Model

---

## 📌 Future Work

- Improve GNN with attention mechanisms (GAT)
- Handle imbalance using advanced sampling
- Hybrid model (GNN + XGBoost)
- Real-time fraud detection system
