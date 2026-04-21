import importlib

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    workflows = importlib.import_module("healthcare_workflows")
except SyntaxError as exc:
    st.set_page_config(page_title="Healthcare Patient Segmentation", layout="wide")
    st.title("🏥 Healthcare Patient Segmentation")
    st.error(
        "The `healthcare_workflows.py` file contains invalid syntax (often unresolved merge markers like "
        "`<<<<<<<`, `=======`, `>>>>>>>`). Please resolve that file and rerun the app."
    )
    st.code(f"{exc.__class__.__name__}: {exc}")
    st.stop()

ManifoldDatasetConfig = workflows.ManifoldDatasetConfig
RiskDatasetConfig = workflows.RiskDatasetConfig
action_rule = workflows.action_rule
add_actions = workflows.add_actions
add_confidence_scores = workflows.add_confidence_scores
cluster_patients = workflows.cluster_patients
create_diabetes_risk_dataset = workflows.create_diabetes_risk_dataset
create_manifold_dataset = workflows.create_manifold_dataset
preprocess_for_manifold = workflows.preprocess_for_manifold
run_tsne = workflows.run_tsne
run_umap = workflows.run_umap
summarize_clusters = workflows.summarize_clusters

st.set_page_config(page_title="Healthcare Patient Segmentation", layout="wide")
st.title("🏥 Healthcare Patient Segmentation")
st.caption("Train a baseline clustering model and score new patient data locally.")


def train_risk_model(seed: int, group_size: int, n_clusters: int):
    config = RiskDatasetConfig(seed=seed, group_size=group_size)
    train_df = create_diabetes_risk_dataset(config)
    clustered_df, model = cluster_patients(train_df, n_clusters=n_clusters, seed=seed)
    scored_df = add_confidence_scores(clustered_df, model)
    final_df = add_actions(scored_df)
    return final_df, model


with st.sidebar:
    st.header("Model setup")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42)
    group_size = st.slider("Synthetic records per risk group", min_value=30, max_value=400, value=120)
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=6, value=3)

baseline_df, risk_model = train_risk_model(seed=seed, group_size=group_size, n_clusters=n_clusters)

risk_tab, manifold_tab = st.tabs(["Risk segmentation", "t-SNE vs UMAP"])

with risk_tab:
    st.subheader("1) Baseline training data")
    st.dataframe(baseline_df.head(20), use_container_width=True)
    st.dataframe(summarize_clusters(baseline_df), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        baseline_df["Glucose"],
        baseline_df["BMI"],
        c=baseline_df["Cluster"],
        cmap="viridis",
        alpha=0.7,
    )
    centers = risk_model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=260, color="red", label="Centroids")
    ax.set_xlabel("Glucose")
    ax.set_ylabel("BMI")
    ax.set_title("Baseline patient clusters")
    ax.legend(loc="best")
    st.pyplot(fig)

    st.subheader("2) Score new patient data")
    st.write("Upload CSV with **Glucose** and **BMI** columns, or test a single patient manually.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### CSV upload")
        uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            required = {"Glucose", "BMI"}
            if not required.issubset(set(new_df.columns)):
                st.error("CSV must contain Glucose and BMI columns.")
            else:
                scored_new = new_df.copy()
                distances = risk_model.transform(scored_new[["Glucose", "BMI"]])
                scored_new["Cluster"] = risk_model.predict(scored_new[["Glucose", "BMI"]])
                scored_new["DistanceToCenter"] = distances.min(axis=1)
                scored_new["Confidence"] = 1 / (1 + scored_new["DistanceToCenter"])
                scored_new["Action"] = scored_new["Confidence"].apply(action_rule)
                st.success(f"Scored {len(scored_new)} patients")
                st.dataframe(scored_new, use_container_width=True)
                st.download_button(
                    "Download scored results",
                    scored_new.to_csv(index=False).encode("utf-8"),
                    file_name="scored_patients.csv",
                    mime="text/csv",
                )

    with col2:
        st.markdown("#### Single patient test")
        glucose = st.slider("Glucose", min_value=50.0, max_value=280.0, value=120.0)
        bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=28.0)

        sample = pd.DataFrame([{"Glucose": glucose, "BMI": bmi}])
        cluster = int(risk_model.predict(sample)[0])
        distance = float(risk_model.transform(sample).min(axis=1)[0])
        confidence = 1 / (1 + distance)

        st.metric("Predicted cluster", cluster)
        st.metric("Confidence", f"{confidence:.3f}")
        st.metric("Recommended action", action_rule(confidence))

with manifold_tab:
    st.subheader("t-SNE vs UMAP on synthetic high-dimensional patient data")

    manifold_config = ManifoldDatasetConfig(seed=seed, samples_per_group=100, n_features=10)
    manifold_df = create_manifold_dataset(manifold_config)
    manifold_data, _ = preprocess_for_manifold(manifold_df)

    tsne_result = run_tsne(manifold_data)
    tsne_df = pd.DataFrame(tsne_result, columns=["x", "y"])

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(tsne_df["x"], tsne_df["y"], alpha=0.7)
    ax1.set_title("t-SNE projection")
    st.pyplot(fig1)

    try:
        umap_module = importlib.import_module("umap")
        umap_result = run_umap(manifold_data, umap_module)
        umap_df = pd.DataFrame(umap_result, columns=["x", "y"])
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(umap_df["x"], umap_df["y"], alpha=0.7)
        ax2.set_title("UMAP projection")
        st.pyplot(fig2)
    except ImportError:
        st.warning("Install optional dependency for UMAP: pip install umap-learn")
