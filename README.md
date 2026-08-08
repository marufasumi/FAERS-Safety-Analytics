<div align="center">

# FDA FAERS Safety Analytics

### End-to-End Machine Learning Pipeline for Pharmacovigilance

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/marufasumi/FAERS-Safety-Analytics)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Coming%20Soon-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-16A34A?style=flat-square)](#)

</div>

---

<p>
<img src="https://img.shields.io/badge/📌%20Project%20Overview-009688?style=for-the-badge"/>
</p>

FAERS Safety Analytics is an end-to-end machine learning project that predicts whether an FDA adverse drug event report is classified as **serious**.

The workflow combines **relational data integration, feature engineering, EDA, statistical testing, supervised learning, K-Means clustering, and Streamlit-based model inference**.

<p align="center">

<img src="https://img.shields.io/badge/Reports-406,184-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Predictors-49-success?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Models-9-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Best%20Accuracy-92.6%25-red?style=for-the-badge"/>

</p>

---

<p>
<img src="https://img.shields.io/badge/🔄%20Machine%20Learning%20Workflow-2563EB?style=for-the-badge"/>
</p>

<p align="center">
<img src="assets/workflow.png" width="900" alt="FAERS Machine Learning Workflow"/>
</p>

---

<p>
<img src="https://img.shields.io/badge/📊%20Project%20Summary-F97316?style=for-the-badge"/>
</p>

<table>
<tr>

<td width="50%" valign="top">

### Dataset

| Item | Value |
|---|---:|
| Source | FDA FAERS |
| Reports | **406,184** |
| Predictors | **49** |
| Target | Serious vs Non-serious |
| Models | **9** |
| K-Means Clusters | **7** |

</td>

<td width="50%" valign="top">

### Best Model

| Metric | Value |
|---|---:|
| Model | **Random Forest ⭐** |
| Accuracy | **92.6%** |
| ROC-AUC | **0.968** |
| F1 Score | **0.932** |
| CV Accuracy | **92.4%** |

</td>

</tr>
</table>

---

<p>
<img src="https://img.shields.io/badge/🤖%20Models%20&%20Tech%20Stack-7C3AED?style=for-the-badge"/>
</p>

<table>
<tr>

<td width="50%" valign="top">

### Models Evaluated

- Logistic Regression
- LDA / QDA
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- **Random Forest ⭐**
- Support Vector Machine
- Neural Network (MLP)

</td>

<td width="50%" valign="top">

### Technologies

<p>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit"/>
<img src="https://img.shields.io/badge/Git-181717?style=flat-square&logo=git"/>
</p>

</td>

</tr>
</table>

---

<table>
<tr>

<td width="50%" valign="top">

---

<table>
<tr>

<td width="33%" valign="top">

<p>
<img src="https://img.shields.io/badge/🖥️%20Streamlit%20Application-0EA5E9?style=for-the-badge"/>
</p>

The application provides:

- Exploratory Data Analysis
- Statistical Analysis
- Machine Learning Results
- Clustering Analysis
- Serious Event Prediction

> 🚀 **Live deployment coming soon**

</td>

<td width="33%" valign="top">

<p>
<img src="https://img.shields.io/badge/📂%20Repository%20Structure-8B5CF6?style=for-the-badge"/>
</p>

<pre>
FAERS-Safety-Analytics/
├── app.py
├── assets/
├── data/
├── models/
├── notebooks/
├── pages/
├── utils/
└── requirements.txt
</pre>

</td>

<td width="34%" valign="top">

<p>
<img src="https://img.shields.io/badge/🚀%20Run%20Locally-2563EB?style=for-the-badge"/>
</p>

<pre>
git clone
https://github.com/
marufasumi/
FAERS-Safety-Analytics.git

cd FAERS-Safety-Analytics

pip install -r
requirements.txt

streamlit run app.py
</pre>

</td>

</tr>
</table>

---

<p>
<img src="https://img.shields.io/badge/🔮%20Roadmap-F59E0B?style=for-the-badge"/>
</p>

<p align="center">

<img src="https://img.shields.io/badge/Azure%20ML-0078D4?style=flat-square&logo=microsoftazure&logoColor=white"/>

<img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white"/>

<img src="https://img.shields.io/badge/Model%20Registry-6E40C9?style=flat-square"/>

<img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white"/>

<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white"/>

<img src="https://img.shields.io/badge/REST%20API-009688?style=flat-square"/>

<img src="https://img.shields.io/badge/Model%20Monitoring-FF9800?style=flat-square"/>

<img src="https://img.shields.io/badge/Automated%20Retraining-4CAF50?style=flat-square"/>

</p>

---

<p>
<img src="https://img.shields.io/badge/⚠️%20Disclaimer-B91C1C?style=for-the-badge"/>
</p>

This project is intended for portfolio and educational purposes only. Model predictions should **not** be used for clinical decision-making.

---

<p>
<img src="https://img.shields.io/badge/👩‍💻%20Author-0A66C2?style=for-the-badge"/>
</p>

**Marufa Sultana Sumi**

[![GitHub](https://img.shields.io/badge/GitHub-marufasumi-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/marufasumi)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Marufa%20Sumi-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marufasumi/)
[![Portfolio](https://img.shields.io/badge/Portfolio-marufasumi.vercel.app-000000?style=flat-square&logo=vercel&logoColor=white)](https://marufasumi.vercel.app)

---

<p align="center">
<b>Data Source:</b> FDA Adverse Event Reporting System (FAERS) • <b>License:</b> MIT
</p>
