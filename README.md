# Healthcare-Patient-Segmentation
# 🏥 Healthcare Analytics: Patient Risk Segmentation & Manifold Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains a two-part data science project focused on healthcare analytics. The primary goal is to demonstrate how unsupervised machine learning techniques can be applied to clinical data to identify at-risk patients and visualize complex, high-dimensional medical datasets.

1. **Diabetic Risk Segmentation:** Using K-Means clustering to group patients based on clinical markers (Glucose and BMI) and assigning clinical action thresholds based on mathematical confidence scores.
2. **High-Dimensional Visualization:** A comparative analysis of **t-SNE** and **UMAP** for discovering hidden disease subtypes and mapping patient progression spectrums.

## 🚀 Key Features & Methodologies
* **Unsupervised Clustering (K-Means):** Segmented patients into Low, Medium, and High-risk categories.
* **Clinical Confidence Scoring:** Calculated the Euclidean distance of patients to cluster centroids to generate a 0-1 "confidence score", bridging the gap between ML outputs and real-world clinical triage.
* **Automated Triage Logic:** Created decision rules (e.g., Auto-care pathway vs. Manual Doctor Review) based on the model's confidence.
* **Manifold Learning (t-SNE & UMAP):** Applied PCA preprocessing followed by non-linear dimensionality reduction to identify hidden clinical subgroups.

## 📊 Visual Insights
*(Note: Upload screenshots of your plots to the `images/` folder on GitHub and replace these placeholder links!)*

**1. Diabetic Risk Clusters & Decision Boundaries**
<img src="images/risk_clusters.png" width="600" alt="K-Means Risk Clusters">
> *Patients closest to the centroid (red 'X') are routed to auto-care, while borderline patients are flagged for manual physician review.*

**2. t-SNE vs. UMAP in Healthcare Data**
<img src="images/tsne_vs_umap.png" width="600" alt="t-SNE and UMAP Comparison">
> *t-SNE excels at separating distinct disease subtypes, while UMAP preserves the global structure, making it ideal for visualizing the continuous spectrum of disease progression.*

## 📂 Repository Structure
* `notebooks/01_Diabetic_Risk_Segmentation_KMeans.ipynb`: Contains the clustering logic, confidence score mathematical formulas, and triage threshold logic.
* `notebooks/02_tSNE_vs_UMAP_Healthcare_Manifolds.ipynb`: Contains the comparative analysis of dimensionality reduction techniques on synthetic 10-feature patient data.

## 🛠️ Installation & Usage
To run these notebooks locally, clone the repository and install the required dependencies:
```bash
git clone [https://github.com/YourUsername/Healthcare-Patient-Segmentation.git](https://github.com/YourUsername/Healthcare-Patient-Segmentation.git)
cd Healthcare-Patient-Segmentation
pip install -r requirements.txt
