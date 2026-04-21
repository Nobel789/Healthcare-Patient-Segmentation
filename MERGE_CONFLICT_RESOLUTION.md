# Merge Conflict Resolution Guide

This repo's current PR conflict can be resolved by keeping the **guardrail-safe** variants in both files.

## 1) `healthcare_workflows.py`

Keep these functions together (do not delete either helper name):

```python
def find_uncertain_patients(df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """Return patients with confidence below a threshold."""
    return df[df["Confidence"] < threshold]


def find_uncertain(df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """Backward-compatible alias for find_uncertain_patients."""
    return find_uncertain_patients(df, threshold=threshold)
```

Why: one side of the PR may reference `find_uncertain_patients`, while another may still reference `find_uncertain`.
Keeping both prevents breakage and avoids recurring conflicts.

## 2) `01_Diabetic_Risk_Segmentation_KMeans.ipynb.ipynb`

Resolve the `add_confidence_scores` cell conflict by keeping:

```python
"df = add_confidence_scores(df, kmeans)\n",
"\n",
"if \"ConfidenceDiagnosis\" in df.columns:\n",
"    print(df[\"ConfidenceDiagnosis\"].iloc[0])\n"
```

Resolve the uncertain-patients cell conflict by keeping:

```python
"uncertain = find_uncertain_patients(df)\n",
"print(uncertain.head())\n"
```

Why: this preserves the low-confidence guardrail default (`0.35`) from the module and keeps the collapse diagnosis visibility.

## 3) Recommended Git workflow (local)

```bash
git checkout <your-pr-branch>
git fetch origin
git merge origin/main
# edit files above, remove conflict markers
git add healthcare_workflows.py 01_Diabetic_Risk_Segmentation_KMeans.ipynb.ipynb
git commit -m "Resolve conflicts in workflow + diabetes notebook"
git push
```

## 4) If using GitHub web conflict editor

- Choose the blocks above exactly.
- Ensure all conflict markers are removed:
  - `<<<<<<<`
  - `=======`
  - `>>>>>>>`
- Click **Mark as resolved** for each file, then **Commit merge**.
