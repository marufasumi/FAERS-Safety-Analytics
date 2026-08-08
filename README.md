<div align="center">

# FDA FAERS Safety Analytics

### End-to-End Machine Learning Pipeline for Pharmacovigilance

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/marufasumi/FAERS-Safety-Analytics)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Coming%20Soon-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
](#)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)]
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)]

</div>

---

<p>
<img src="https://img.shields.io/badge/📌%20Project%20Overview-009688?style=for-the-badge"/>
</p>

FAERS Safety Analytics is an end-to-end machine learning project that predicts whether an FDA adverse drug event report is classified as **serious**.

The project demonstrates a complete data science workflow, including relational data integration, feature engineering, exploratory data analysis, statistical hypothesis testing, supervised learning, unsupervised clustering, and deployment through an interactive Streamlit application.

---

<p>
<img src="https://img.shields.io/badge/🧠%20Tech%20Stack-7C3AED?style=for-the-badge"/>
</p>

| Area | Technologies |
|------|--------------|
| **Programming** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Statistics** | SciPy |
| **Visualization** | Plotly, Matplotlib |
| **Dashboard** | Streamlit |
| **Version Control** | Git, GitHub |

<br>

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>

<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>

<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy"/>

<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn"/>

<img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly"/>

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit"/>

</p>

---

<p>
<img src="https://img.shields.io/badge/🔄%20Machine%20Learning%20Workflow-2563EB?style=for-the-badge"/>
</p>

<p align="center">
<img src="assets/workflow.png" width="950">
</p>

---

<p>
<img src="https://img.shields.io/badge/📊%20Dataset%20Summary-F97316?style=for-the-badge"/>
</p>

| Item | Value |
|------|------:|
| Source | FDA FAERS |
| Reports | **406,184** |
| Predictor Variables | **49** |
| Target | Serious vs Non-serious |
| Machine Learning Models | **9** |
| Clusters | **7** |

---

<p>
<img src="https://img.shields.io/badge/🤖%20Machine%20Learning-DC2626?style=for-the-badge"/>
</p>

The project evaluates nine supervised learning algorithms:

- Logistic Regression
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- Random Forest ⭐
- Support Vector Machine
- Neural Network (MLP)

Random Forest achieved the highest predictive performance.

---

<p>
<img src="https://img.shields.io/badge/🏆%20Model%20Performance-16A34A?style=for-the-badge"/>
</p>

| Metric | Result |
|---------|-------:|
| Best Model | Random Forest |
| Accuracy | **92.6%** |
| ROC-AUC | **0.968** |
| F1 Score | **0.932** |
| Cross Validation Accuracy | **92.4%** |

---

<p>
<img src="https://img.shields.io/badge/🖥️%20Streamlit%20Application-0EA5E9?style=for-the-badge"/>
</p>

The interactive application includes:

- Project Overview
- Exploratory Data Analysis
- Statistical Analysis
- Machine Learning Results
- K-Means Clustering
- Serious Event Prediction

> **Live demo will be available after project publication.**

---

<p>
<img src="https://img.shields.io/badge/📂%20Repository%20Structure-8B5CF6?style=for-the-badge"/>
</p>

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

<p>
<img src="https://img.shields.io/badge/🚀%20Getting%20Started-2563EB?style=for-the-badge"/>
</p>

Clone the repository

```bash
git clone https://github.com/marufasumi/FAERS-Safety-Analytics.git
```

Navigate to the project

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

<p>
<img src="https://img.shields.io/badge/🔮%20Future%20Roadmap-F59E0B?style=for-the-badge"/>
</p>

- Azure Machine Learning
- MLflow Experiment Tracking
- Model Registry
- Docker
- GitHub Actions CI/CD
- REST API
- Model Monitoring
- Automated Retraining Pipeline

---

<p>
<img src="https://img.shields.io/badge/📄%20Data%20Source-DC2626?style=for-the-badge"/>
</p>

**FDA Adverse Event Reporting System (FAERS)**

https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

---

<p>
<img src="https://img.shields.io/badge/⚠️%20Disclaimer-B91C1C?style=for-the-badge"/>
</p>

This project is intended for portfolio and educational purposes. Predictions generated by the machine learning models should **not** be used for clinical decision-making or patient care.

---

<p>
<img src="https://img.shields.io/badge/👩‍💻%20Author-0A66C2?style=for-the-badge"/>
</p>

**Marufa Sultana Sumi**

[![GitHub](https://img.shields.io/badge/GitHub-marufasumi-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/marufasumi)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Marufa%20Sumi-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marufasumi/)

[![Portfolio](https://img.shields.io/badge/Portfolio-marufasumi.vercel.app-000000?style=flat-square&logo=vercel&logoColor=white)](https://marufasumi.vercel.app)

---

<p>
<img src="https://img.shields.io/badge/📜%20License-16A34A?style=for-the-badge"/>
</p>

This project is licensed under the **MIT License**.
