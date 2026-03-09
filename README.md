# 🏥 Hospital Readmissions: Predicting Quality & Identifying At-Risk Hospitals

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python)
![Spark](https://img.shields.io/badge/Platform-Azure%20Databricks%20%7C%20PySpark-E25A1C?style=flat-square&logo=apachespark)
![ML](https://img.shields.io/badge/Models-Random%20Forest%20%7C%20Decision%20Tree%20%7C%20Logistic%20Regression-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

## Overview

This group project applies machine learning to publicly available U.S. hospital data to predict which hospitals are at highest risk of excess patient readmissions — and to identify the factors driving those outcomes. Using classification and clustering on data from the Centers for Medicare and Medicaid Services (CMS), the analysis supports data-driven decisions for hospital administrators, policymakers, and healthcare investors.

> **Best model:** Random Forest achieved the highest AUC (0.632) for classifying hospital quality. Decision Tree maximised true positive detection (TP = 694), making it the preferred model when minimising missed high-risk hospitals is the priority.

---

## Problem Statement

Given a hospital's excess readmission ratio, payment levels, and medical condition type, can a reliable model determine which factors most strongly increase readmission rates — and which hospitals are in the greatest need of intervention?

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | U.S. Centers for Medicare & Medicaid Services (CMS) |
| **Datasets** | Hospital Readmissions Reduction Program + Hospital Payment and Value of Care |
| **Conditions Studied** | Pneumonia, Heart Attack, Hip/Knee Replacement, Heart Failure |
| **Target Variable** | Hospital Quality — Binary: Bad (ERR ≥ 1) vs. Good (ERR < 1) |
| **Platform** | Azure Databricks (PySpark + Pandas + Matplotlib/Seaborn) |

**Key variables:** Excess Readmission Ratio (ERR), Average Payment, State, Mortality Performance, Expenditure Category, Measure Category.

---

## Methodology

### Exploratory Data Analysis
- ERR is approximately normally distributed (mean ≈ 1.0, SD = 0.1) across U.S. hospitals
- Hip/Knee Replacement has the highest average payment (~$28,000) and widest ERR variance
- Payment levels alone cannot reliably distinguish good vs. bad hospitals — minimal difference between quartiles

### Classification — 7 Models Compared

Feature processing used StringIndexer, One-Hot Encoding, and VectorAssembler. Distance-sensitive models (Logistic Regression, OvR, Linear SVC) used feature scaling; tree-based models did not. Hyperparameter tuning via K-fold cross-validation (k=3) and ParamGrid search, with 80/20 train-test split.

| Model | AUC | Notes |
|-------|-----|-------|
| **Random Forest** | **0.632** | Best overall — balanced precision, avoids false flags |
| Logistic Regression | 0.630 | Close second; also balanced |
| Decision Tree | — | Highest TP (694); best for catching all bad hospitals |
| Gradient Boosted Trees | — | Tested; underperformed top models |
| Naïve Bayes | — | Baseline comparison |
| One-vs-Rest (OvR) | — | Tested with Logistic Regression base |
| Linear SVC | — | Tested; comparable to logistic variants |

### Clustering — K-Means (k=4)
Explored the relationship between payment levels and readmission rates. Optimal cluster count selected via silhouette score analysis.

| Cluster | Profile |
|---------|---------|
| **Cluster 1** | Highest ERR + Highest Payment — potential overpricing or chronic condition burden |
| **Cluster 2** | Low ERR + High Payment — effective but expensive care |
| **Cluster 3** | Lowest ERR + Lowest Payment — likely strongest preventive care model |
| **Cluster 4** | Low ERR + Low Payment — cost-efficient, moderate performance |

**Key finding:** Payment alone does not predict readmission outcomes. Disease type and state health benefit structures appear to play a larger role.

---

## Key Recommendations

- Deploy Decision Tree models as early warning systems to flag hospitals at risk before outcomes worsen
- Use clustering to benchmark hospitals against peers — learn from Cluster 3's low-cost, low-readmission profile
- Collect richer patient-level data: comorbidities, insurance coverage, socioeconomic status, follow-up access
- Improve data infrastructure — 8,000+ "Not Available" payment entries limit model accuracy

---

## Repository Structure

```
├── BDA_Final_Project_Report.pdf       # Full written report with methodology and findings
├── BDA_Final_Presentation.pdf         # Presentation deck (Learning Team 4)
└── README.md
```

> **Note:** Code was developed and executed in Azure Databricks (PySpark). Notebooks are not included in this repository. The report and presentation contain full methodology documentation.

---

## Limitations

- Missing payment data (8,000+ entries replaced with 0) introduces potential distortion
- AUC scores of ~0.63 indicate moderate predictive power — key clinical variables (patient comorbidities, staffing ratios, hospital size) were not available
- K-means clustering is sensitive to feature scaling and may not generalise to future hospital configurations

---

## Collaborators

Developed as a group project (Learning Team 4) at Ivey Business School, MSc Business Analytics program.

---

## Author

**Shafaq Karim** — Graduate in Business Analytics, Ivey Business School
[LinkedIn](https://www.linkedin.com/in/shafaqkarim/) · [Portfolio](https://gamma.app/docs/I-am-Shafaq-Karim-kyo9jrf3keo5ykz?mode=doc)
