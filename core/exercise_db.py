from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from config import EXERCISES_CSV_PATH


class ExerciseDB:
    """
    Loads the exercises_working_gifs.csv once, and lets you
    query rows by arbitrary column→substring filters, plus
    pull gifUrl by id.
    """

    def __init__(self, csv_path: str = None):
        """
        csv_path: optional override; if None, uses EXERCISES_CSV_PATH from config.py
        """

        path = csv_path or EXERCISES_CSV_PATH
        self._df = pd.read_csv(path)

    def get_rows_by_dict(self, params: Dict[str, List[str]]) -> pd.DataFrame:
        """
        params: e.g. {"bodyPart": ["legs"], "target": ["quadriceps"]}

        Returns the subset of rows matching all filters.
        """

        df = self._df
        for col, val in params.items():
            if col not in df.columns or not val:
                continue

            df = df[df[col].isin(val)]

        return df

    def get_url_by_id(self, ex_id: int) -> str:
        """
        Return the 'gifUrl' for a single exercise id.
        """

        row = self._df[self._df["id"] == ex_id]
        if row.empty:
            raise KeyError(f"No exercise with id={ex_id}")
        return row.iloc[0]["gifUrl"]

    def all_values(self) -> Dict[str, List[str]]:
        """
        Return the unique values for bodyPart, equipment, target
        (for prompt reference).
        """

        return {
            "bodyPart": sorted(self._df["bodyPart"].unique().tolist()),
            "equipment": sorted(self._df["equipment"].unique().tolist()),
            "target": sorted(self._df["target"].unique().tolist()),
        }
