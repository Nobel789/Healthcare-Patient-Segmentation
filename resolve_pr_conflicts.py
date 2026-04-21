#!/usr/bin/env python3
"""Resolve known merge conflicts for this repository from the command line.

Usage:
  python resolve_pr_conflicts.py
"""

from __future__ import annotations

from pathlib import Path


NOTEBOOK = Path("01_Diabetic_Risk_Segmentation_KMeans.ipynb.ipynb")
WORKFLOWS = Path("healthcare_workflows.py")


def resolve_notebook(text: str) -> str:
    conflict_1 = '''<<<<<<<\n    "df = add_confidence_scores(df, kmeans)\\n",\n    "\\n",\n    "if \\\"ConfidenceDiagnosis\\\" in df.columns:\\n",\n    "    print(df[\\\"ConfidenceDiagnosis\\\"].iloc[0])\\n"\n=======\n    "df = add_confidence_scores(df, kmeans)\\n"\n>>>>>>>'''

    resolved_1 = '''    "df = add_confidence_scores(df, kmeans)\\n",\n    "\\n",\n    "if \\\"ConfidenceDiagnosis\\\" in df.columns:\\n",\n    "    print(df[\\\"ConfidenceDiagnosis\\\"].iloc[0])\\n"'''

    conflict_2 = '''<<<<<<<\n    "uncertain = find_uncertain_patients(df, threshold=0.35)\\n",\n    "print(uncertain.head())\\n",\n    "print(df[\\\"Action\\\"].value_counts())\\n"\n=======\n    "uncertain = find_uncertain_patients(df, threshold=0.5)\\n",\n    "print(uncertain.head())\\n"\n>>>>>>>'''

    resolved_2 = '''    "uncertain = find_uncertain_patients(df)\\n",\n    "print(uncertain.head())\\n"'''

    updated = text.replace(conflict_1, resolved_1).replace(conflict_2, resolved_2)
    return updated


def resolve_workflows(text: str) -> str:
    # Keep both helper names to avoid branch API drift.
    if "def find_uncertain_patients" in text and "def find_uncertain(" in text:
        return text

    marker = 'def find_uncertain_patients(df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:\n    """Return patients with confidence below a threshold."""\n    return df[df["Confidence"] < threshold]\n'
    alias = '\n\ndef find_uncertain(df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:\n    """Backward-compatible alias for find_uncertain_patients."""\n    return find_uncertain_patients(df, threshold=threshold)\n'

    if marker in text and "def find_uncertain(" not in text:
        return text.replace(marker, marker + alias)
    return text


def process_file(path: Path, resolver) -> bool:
    if not path.exists():
        return False
    original = path.read_text(encoding="utf-8")
    updated = resolver(original)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    if process_file(NOTEBOOK, resolve_notebook):
        changed.append(str(NOTEBOOK))
    if process_file(WORKFLOWS, resolve_workflows):
        changed.append(str(WORKFLOWS))

    if changed:
        print("Updated:")
        for file in changed:
            print(f"- {file}")
    else:
        print("No changes made. Either conflicts are already resolved or patterns did not match.")


if __name__ == "__main__":
    main()
