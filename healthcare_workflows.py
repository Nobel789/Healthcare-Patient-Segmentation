"""Reusable workflow helpers for the healthcare segmentation notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RiskDatasetConfig:
    seed: int = 42
    group_size: int = 50


def create_diabetes_risk_dataset(config: RiskDatasetConfig = RiskDatasetConfig()) -> pd.DataFrame:
    """Build toy diabetes risk data with glucose and BMI columns."""
    rng = np.random.default_rng(config.seed)
    low_risk = rng.normal(loc=[90, 22], scale=[8, 2], size=(config.group_size, 2))
    medium_risk = rng.normal(loc=[130, 28], scale=[10, 3], size=(config.group_size, 2))
    high_risk = rng.normal(loc=[180, 35], scale=[12, 4], size=(config.group_size, 2))

    data = np.vstack([low_risk, medium_risk, high_risk])
    return pd.DataFrame(data, columns=["Glucose", "BMI"])


def cluster_patients(df: pd.DataFrame, n_clusters: int = 3, seed: int = 42) -> tuple[pd.DataFrame, KMeans]:
    """Cluster patient records and return a copy with cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clustered = df.copy()
    clustered["Cluster"] = model.fit_predict(clustered[["Glucose", "BMI"]])
    return clustered, model


def _normalize_inverse_distances(assigned_dist: np.ndarray) -> np.ndarray:
    """Convert raw distances to a stable 0..1 confidence score."""
    max_dist = float(np.max(assigned_dist))
    min_dist = float(np.min(assigned_dist))

    if np.isclose(max_dist, min_dist):
        # Degenerate case: model gives nearly identical distance for every record.
        return np.zeros_like(assigned_dist, dtype=float)

    normalized = (assigned_dist - min_dist) / (max_dist - min_dist)
    return 1.0 - normalized


def diagnose_confidence_collapse(confidence: pd.Series) -> str | None:
    """Return a diagnosis string when confidence appears collapsed to uniformly low values."""
    if confidence.empty:
        return "No confidence values available for diagnosis."

    spread = float(confidence.std())
    max_conf = float(confidence.max())
    if spread < 0.02 and max_conf < 0.2:
        return (
            "Confidence collapse detected: scores are tightly grouped at low values. "
            "This typically happens when raw distances are used without normalization, "
            "which compresses confidence on large-scale feature ranges like Glucose."
        )
    return None


def add_confidence_scores(df: pd.DataFrame, model: KMeans) -> pd.DataFrame:
    """Add normalized confidence scores based on distance to assigned cluster center."""
    scored = df.copy()
    distances = model.transform(scored[["Glucose", "BMI"]])
    assigned_dist = np.min(distances, axis=1)
    scored["DistanceToCenter"] = assigned_dist
    scored["Confidence"] = _normalize_inverse_distances(assigned_dist)

    diagnosis = diagnose_confidence_collapse(scored["Confidence"])
    if diagnosis is not None:
        scored["ConfidenceDiagnosis"] = diagnosis

    return scored


def action_rule(confidence: float, uncertain_threshold: float = 0.35) -> str:
    """Map confidence value to recommended triage pathway with low-confidence guardrail."""
    if confidence < uncertain_threshold:
        return "Uncertain — clinician review required"
    if confidence > 0.7:
        return "Auto-care pathway"
    return "Doctor review"


def add_actions(df: pd.DataFrame, uncertain_threshold: float = 0.35) -> pd.DataFrame:
    """Append rule-based action labels from confidence values."""
    updated = df.copy()
    updated["Action"] = updated["Confidence"].apply(lambda c: action_rule(c, uncertain_threshold=uncertain_threshold))
    return updated


def find_uncertain_patients(df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """Return patients with confidence below a threshold."""
    return df[df["Confidence"] < threshold]


@dataclass(frozen=True)
class ManifoldDatasetConfig:
    seed: int = 42
    samples_per_group: int = 100
    n_features: int = 10


def create_manifold_dataset(config: ManifoldDatasetConfig = ManifoldDatasetConfig()) -> pd.DataFrame:
    """Build toy healthcare feature matrix for manifold learning demos."""
    rng = np.random.default_rng(config.seed)
    group1 = rng.normal(0, 1, (config.samples_per_group, config.n_features))
    group2 = rng.normal(3, 1, (config.samples_per_group, config.n_features))
    data = np.vstack([group1, group2])
    return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(config.n_features)])


def preprocess_for_manifold(df: pd.DataFrame, n_components: int = 5) -> tuple[np.ndarray, Dict[str, object]]:
    """Scale then run PCA as a pre-step for manifold methods."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled)
    return pca_data, {"scaler": scaler, "pca": pca}


def run_tsne(data: np.ndarray, seed: int = 42, perplexity: float = 30) -> np.ndarray:
    """Run t-SNE on preprocessed feature data."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    return tsne.fit_transform(data)


def run_umap(data: np.ndarray, umap_module, seed: int = 42, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Run UMAP using an injected umap module (keeps optional dependency out of this file)."""
    model = umap_module.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    return model.fit_transform(data)


def summarize_clusters(df: pd.DataFrame, metrics: Iterable[str] = ("Glucose", "BMI", "Confidence")) -> pd.DataFrame:
    """Convenience summary table by cluster."""
    return df.groupby("Cluster")[list(metrics)].mean().round(2)
