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

        return df.copy()

    def get_rows_by_ids(self, ids: List[int]) -> pd.DataFrame:
        """
        Return a DataFrame containing only the rows whose "id" column is in 'ids'.

        If an ID is not found, it is simply omitted from the result (no error).
        """

        id_set = set(ids)
        filtered = self._df[self._df["id"].isin(id_set)].copy()

        return filtered

    def reorder_and_filter_columns(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame 'rows',cproduce a new DataFrame with:
          1. All columns except for "url" column.
          2. Reordered so that "id" is first, "name" (exercise name) is second, then all other
             columns (in the same relative order as they appeared originally).
        """
        # Build a list of columns that do not contain "url"
        all_cols = list(rows.columns)
        filtered_cols = [c for c in all_cols if "url" not in c.lower()]

        # Force "id" first, "name" second, then append anything else
        ordered_cols = []
        if "id" in filtered_cols:
            ordered_cols.append("id")
        if "name" in filtered_cols:
            ordered_cols.append("name")

        for c in filtered_cols:
            if c not in ("id", "name"):
                ordered_cols.append(c)

        # Return a new DataFrame with only those columns, in the specified order
        return rows.loc[:, ordered_cols].copy()

    def get_url_by_id(self, ex_id: int) -> str:
        """
        Return the 'gifUrl' for a single exercise id.
        """

        row = self._df[self._df["id"] == ex_id]
        if row.empty:
            raise KeyError(f"No exercise with id={ex_id}")
        return row.iloc[0]["gifUrl"]

    def get_name_by_id(self, ex_id: int) -> str:
        """
        Return the 'name' for a single exercise id.
        """

        row = self._df[self._df["id"] == ex_id]
        if row.empty:
            raise KeyError(f"No exercise with id={ex_id}")
        return row.iloc[0]["name"]

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
