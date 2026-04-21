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

## Troubleshooting

If Streamlit shows a `SyntaxError` in `healthcare_workflows.py` mentioning `<<<<<<< HEAD`, your local file still has unresolved Git merge markers.

1. Open `healthcare_workflows.py`.
2. Remove any conflict marker lines:
   - `<<<<<<< HEAD`
   - `=======`
   - `>>>>>>> ...`
3. Keep the final intended code, save, and run:

```bash
streamlit run app.py
```

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
