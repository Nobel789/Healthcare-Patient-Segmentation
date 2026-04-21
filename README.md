# Healthcare Patient Segmentation (Local Streamlit App)

This repository provides a local Streamlit application to:
- Train a baseline **K-Means** risk segmentation model on synthetic healthcare data.
- Score **new patient data** (CSV upload or manual input) using `Glucose` and `BMI`.
- View confidence and triage-style action recommendations.
- Compare **t-SNE** and **UMAP** visualizations on synthetic high-dimensional healthcare features.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## CSV format for new data scoring

Upload a `.csv` file with at least these columns:

| Glucose | BMI |
|--------:|----:|
| 118     | 27  |
| 172     | 33  |

The app will return:
- `Cluster`
- `DistanceToCenter`
- `Confidence`
- `Action`

## Repository files

- `app.py` — Streamlit UI for model setup, scoring, and visualization.
- `healthcare_workflows.py` — reusable ML workflow helpers.
- `01_Diabetic_Risk_Segmentation_KMeans.ipynb.ipynb` — notebook artifact.
- `02_tSNE_vs_UMAP_Healthcare_Manifolds.ipynb.ipynb` — notebook artifact.

## Troubleshooting

If deployment fails with a `SyntaxError` showing `<<<<<<< HEAD`, your local copy has unresolved merge markers.

Run:

```bash
rg -n "<<<<<<<|=======|>>>>>>>"
```

If any lines are returned, remove those markers and keep only valid Python/JSON content, then rerun:

```bash
streamlit run app.py
```
