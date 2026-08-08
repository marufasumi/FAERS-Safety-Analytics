# FDA FAERS Safety Analytics

> End-to-end machine learning project for predicting serious adverse drug event reports using the FDA Adverse Event Reporting System (FAERS).

<p align="left">
  <a href="https://github.com/marufasumi/FAERS-Safety-Analytics">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github">
  </a>
  <img src="https://img.shields.io/badge/Live%20Demo-Coming%20Soon-red?style=for-the-badge&logo=streamlit">
</p>

<p>
<img src="https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python">
<img src="https://img.shields.io/badge/scikit--learn-1.7-orange?style=flat-square&logo=scikitlearn">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit">
<img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

---

# 📌 Project Overview

This project builds an end-to-end machine learning pipeline using over **406,000 FDA FAERS reports** to predict whether an adverse drug event is classified as **serious**.

The project demonstrates the complete data science lifecycle:

- Data Integration
- Feature Engineering
- Exploratory Data Analysis
- Statistical Hypothesis Testing
- Machine Learning
- Model Comparison
- K-Means Clustering
- Interactive Streamlit Dashboard

---

# 🔄 Workflow

<p align="center">
  <img src="assets/workflow.png" width="900" alt="FAERS Workflow">
</p>

---

# 📊 Dataset

| Item | Value |
|------|------:|
| Source | FDA FAERS |
| Reports | 406,184 |
| Predictors | 49 |
| Target | Serious vs Non-serious |
| Models Evaluated | 9 |
| Clusters | 7 |

---

# 🤖 Machine Learning Models

- Logistic Regression
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- Random Forest ⭐
- Support Vector Machine
- Neural Network (MLP)

---

# 🏆 Best Model Performance

| Metric | Score |
|---------|-------:|
| Best Model | Random Forest |
| Accuracy | **92.6%** |
| ROC-AUC | **0.968** |
| F1 Score | **0.932** |
| Cross Validation Accuracy | **92.4%** |

---

# 🚀 Streamlit Application

The interactive application includes:

- Project Overview
- Exploratory Data Analysis
- Statistical Analysis
- Machine Learning Results
- Clustering Analysis
- Serious Event Prediction

> **Public deployment will be available after publication.**

---

# 📂 Repository Structure

```text
FAERS-Safety-Analytics/
│
├── app.py
├── requirements.txt
├── README.md
│
├── assets/
├── data/
├── models/
├── notebooks/
├── pages/
└── utils/
```

---

# 🛠️ Installation

Clone the repository

```bash
git clone https://github.com/marufasumi/FAERS-Safety-Analytics.git
```

Move into the project

```bash
cd FAERS-Safety-Analytics
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 💻 Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Statistics | SciPy |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |

---

# 🔮 Future Enhancements

- Azure Machine Learning
- MLflow Experiment Tracking
- Model Registry
- GitHub Actions CI/CD
- Docker
- REST API
- Model Monitoring
- Automated Retraining

---

# 📄 Data Source

**FDA Adverse Event Reporting System (FAERS)**

https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

---

# ⚠️ Disclaimer

This project is intended for portfolio and educational purposes. Model predictions should not be used for clinical decision-making.

---

# 👩‍💻 Author

**Marufa Sultana Sumi**

<p>
<a href="https://github.com/marufasumi">
<img src="https://img.shields.io/badge/GitHub-marufasumi-181717?style=for-the-badge&logo=github">
</a>
</p>

---

## 📜 License

This project is licensed under the MIT License.
