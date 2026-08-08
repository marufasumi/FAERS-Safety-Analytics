# FDA FAERS Safety Analytics
**End-to-End Machine Learning Pipeline for Pharmacovigilance**

Marufa Sultana Sumi | [GitHub](https://github.com/marufasumi/FAERS-Safety-Analytics) | [LinkedIn](https://www.linkedin.com/in/marufasumi/) | [Portfolio](https://marufasumi.vercel.app)

---

## Overview
An end-to-end machine learning project predicting whether an FDA adverse drug event report is classified as **serious**. Demonstrates the full data science lifecycle: relational data integration, feature engineering, EDA, statistical hypothesis testing, supervised learning, unsupervised clustering, and deployment via an interactive Streamlit app.

## Key Numbers
| Reports | Predictors | Models | Clusters | Best Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| 406,184 | 49 | 9 | 7 | **92.6%** |

## Best Model — Random Forest ⭐
| Accuracy | ROC-AUC | F1 Score | CV Accuracy |
|:---:|:---:|:---:|:---:|
| 92.6% | 0.968 | 0.932 | 92.4% |

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · SciPy · Plotly · Matplotlib · Streamlit · Git/GitHub

## Models Evaluated
Logistic Regression, LDA, QDA, Gaussian Naive Bayes, KNN, Decision Tree, **Random Forest**, SVM, Neural Network (MLP)

## Highlights
✅ Multi-table relational data integration &nbsp;·&nbsp; ✅ Feature engineering &nbsp;·&nbsp; ✅ EDA &nbsp;·&nbsp; ✅ Statistical hypothesis testing &nbsp;·&nbsp; ✅ Model comparison &nbsp;·&nbsp; ✅ K-Means clustering &nbsp;·&nbsp; ✅ Interactive prediction dashboard

## Streamlit App
Project Overview · Exploratory Data Analysis · Statistical Analysis · ML Results · Clustering Analysis · Serious Event Prediction

## Quick Start
```bash
git clone https://github.com/marufasumi/FAERS-Safety-Analytics.git
cd FAERS-Safety-Analytics
pip install -r requirements.txt
streamlit run app.py
```

## Data Source
FDA Adverse Event Reporting System (FAERS) — fis.fda.gov/extensions/FPD-QDE-FAERS

## Roadmap
Azure ML · MLflow Tracking · Model Registry · Docker · GitHub Actions CI/CD · REST API · Model Monitoring · Automated Retraining

---
*Portfolio/educational project. Predictions should not be used for clinical decision-making. Licensed under MIT.*
