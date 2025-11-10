# 📞 Telecom Customer Churn Prediction Engine

This project demonstrates a **full-stack deployment** of a Machine Learning model designed to predict **customer churn risk in real time**, focusing on optimizing **high-value customer retention (Recall)**.

---

## 🎯 Project Goal

The primary objective was to **minimize False Negatives (missed churners)** to maximize **revenue protection**.  
This was achieved by optimizing for **high Recall on the minority (churn) class**.

---

## 🧠 Final Model Performance

| Metric | Description | Value |
|:--|:--|:--|
| **Model Used** | Decision Tree Classifier | |
| **Key Optimization** | `class_weight='balanced'` | |
| **Churn Recall (Capture Rate)** | % of actual churners correctly identified | **79%** |
| **Churn Precision (Efficiency)** | % of predicted churners who actually churn | **49%** |
| **Missed Customers (FN)** | False Negatives — churners missed by model | **80** |

> This section highlights the **core business trade-off**:  
> maximizing Recall to minimize revenue loss, even if Precision decreases slightly.

---

## ⚙️ Technical Pipeline (MLOps)

The application transfers data between the frontend, backend, and model in **three key stages**:

### 1️⃣ Data Science & Feature Engineering

- **Feature Creation:** Built a robust **18-feature set**, including metrics such as *Charge Deviation* and *Number of Services*.  
- **Imbalance Handling:** Applied `class_weight='balanced'` during training to increase model sensitivity.  
  - Result: Achieved a **79% Recall** on the minority churn class.

---

### 2️⃣ Deployment Stack (API Backend)

- **Model Artifact:** Optimized Decision Tree saved as  
  ```python
  churn_model.pkl
