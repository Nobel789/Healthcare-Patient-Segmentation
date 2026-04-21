import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import importlib

from healthcare_workflows import (
    RiskDatasetConfig,
    ManifoldDatasetConfig,
    create_diabetes_risk_dataset,
    cluster_patients,
    add_confidence_scores,
    add_actions,
    find_uncertain_patients,
    summarize_clusters,
    create_manifold_dataset,
    preprocess_for_manifold,
    run_tsne,
    run_umap,
)

st.set_page_config(page_title="Healthcare Analytics Web App", layout="wide")

st.title("Healthcare Analytics: Patient Risk Segmentation & Manifold Learning")

tab1, tab2 = st.tabs(["Risk Segmentation", "t-SNE vs UMAP"])

# =========================
# TAB 1: Risk Segmentation
# =========================
with tab1:
    st.header("Diabetic Risk Segmentation")

    config = RiskDatasetConfig(seed=42, group_size=50)
    df = create_diabetes_risk_dataset(config)

    clustered_df, model = cluster_patients(df)
    scored_df = add_confidence_scores(clustered_df, model)
    final_df = add_actions(scored_df)

    st.subheader("Patient Data")
    st.dataframe(final_df)

    st.subheader("Cluster Summary")
    st.dataframe(summarize_clusters(final_df))

    st.subheader("Cluster Plot")
    fig, ax = plt.subplots()

    ax.scatter(
        final_df["Glucose"],
        final_df["BMI"],
        c=final_df["Cluster"]
    )

    centers = model.cluster_centers_
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=200
    )

    ax.set_xlabel("Glucose")
    ax.set_ylabel("BMI")

    st.pyplot(fig)

    # ===== Live Patient =====
    st.subheader("Test Patient")

    glucose = st.slider("Glucose", 50.0, 250.0, 120.0)
    bmi = st.slider("BMI", 10.0, 50.0, 28.0)

    patient_df = pd.DataFrame([{"Glucose": glucose, "BMI": bmi}])

    cluster = model.predict(patient_df)[0]
    distance = model.transform(patient_df).min(axis=1)[0]
    confidence = 1 / (1 + distance)

    if confidence > 0.7:
        action = "Auto-care pathway"
    elif confidence > 0.5:
        action = "Doctor review"
    else:
        action = "Standard care"

    st.write(f"Cluster: {cluster}")
    st.write(f"Confidence: {confidence:.2f}")
    st.write(f"Action: {action}")

# =========================
# TAB 2: t-SNE vs UMAP
# =========================
with tab2:
    st.header("t-SNE vs UMAP")

    config = ManifoldDatasetConfig(seed=42, samples_per_group=100, n_features=10)
    df = create_manifold_dataset(config)

    data, _ = preprocess_for_manifold(df)

    # t-SNE
    tsne_result = run_tsne(data)
    tsne_df = pd.DataFrame(tsne_result, columns=["x", "y"])

    fig1, ax1 = plt.subplots()
    ax1.scatter(tsne_df["x"], tsne_df["y"])
    ax1.set_title("t-SNE")
    st.pyplot(fig1)

    # UMAP
    try:
        umap_module = importlib.import_module("umap")

        umap_result = run_umap(data, umap_module)
        umap_df = pd.DataFrame(umap_result, columns=["x", "y"])

        fig2, ax2 = plt.subplots()
        ax2.scatter(umap_df["x"], umap_df["y"])
        ax2.set_title("UMAP")
        st.pyplot(fig2)

    except ImportError:
        st.error("Install UMAP with: pip install umap-learn")
