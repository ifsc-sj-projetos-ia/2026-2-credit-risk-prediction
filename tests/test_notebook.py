"""Testes offline das mesmas funções usadas no notebook, sem duplicar o pipeline."""

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_credit_risk_prediction.ipynb"


@lru_cache(maxsize=1)
def definitions():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and "definitions" in cell.metadata.get("tags", []):
            exec(compile(cell.source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def example_data(size=100):
    """Perfis fictícios apenas para testes; não são a base do experimento."""
    return pd.DataFrame({
        "person_age": np.full(size, 30),
        "person_income": np.arange(size) * 100 + 20_000,
        "person_home_ownership": ["RENT", "OWN"] * (size // 2),
        "person_emp_length": np.full(size, 5.0),
        "loan_intent": ["EDUCATION", "PERSONAL"] * (size // 2),
        "loan_grade": ["A", "B"] * (size // 2),
        "loan_amnt": np.full(size, 5_000),
        "loan_int_rate": np.full(size, 10.0),
        "loan_status": [0, 0, 0, 1, 1] * (size // 5),
        "loan_percent_income": np.full(size, 0.2),
        "cb_person_default_on_file": ["N", "Y"] * (size // 2),
        "cb_person_cred_hist_length": np.full(size, 10),
    })


class NotebookFormatTests(unittest.TestCase):
    def test_every_ipynb_is_a_real_jupyter_notebook(self):
        paths = list((ROOT / "notebooks").glob("*.ipynb"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                nbformat.validate(nbformat.read(path, as_version=4))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = definitions()

    def test_missing_target_is_not_replaced_by_last_column(self):
        data = example_data().drop(columns="loan_status")
        with self.assertRaisesRegex(ValueError, "loan_status"):
            self.code["clean_data"](data)

    def test_local_csv_with_wrong_fingerprint_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "different.csv"
            path.write_text("loan_status\n0\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.code["load_data"](path)

    def test_invalid_or_missing_targets_are_rejected(self):
        for value in (2, -1, np.nan, "default"):
            with self.subTest(value=value):
                data = example_data()
                data["loan_status"] = data["loan_status"].astype(object)
                data.loc[0, "loan_status"] = value
                with self.assertRaisesRegex(ValueError, "loan_status"):
                    self.code["clean_data"](data)

    def test_non_numeric_and_infinite_features_are_rejected(self):
        for value in ("not-an-income", np.inf):
            data = example_data()
            data["person_income"] = data["person_income"].astype(object)
            data.loc[0, "person_income"] = value
            with self.assertRaises(ValueError):
                self.code["clean_data"](data)

    def test_duplicate_and_impossible_profiles_are_removed(self):
        data = example_data()
        data.loc[1, "person_age"] = 144
        data.loc[2, "person_emp_length"] = 123
        data.loc[3, "cb_person_cred_hist_length"] = 40
        data.loc[4, "person_income"] = 0
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        cleaned, audit = self.code["clean_data"](data)
        self.assertEqual(len(cleaned), 96)
        self.assertEqual(audit["duplicates_removed"], 1)
        self.assertEqual(audit["invalid_rows_removed"], 4)
        self.assertEqual(audit["rows_raw"], 101)

    def test_missing_predictors_remain_for_train_only_imputation(self):
        data = example_data()
        data.loc[0, "person_emp_length"] = np.nan
        cleaned, _ = self.code["clean_data"](data)
        self.assertTrue(pd.isna(cleaned.loc[0, "person_emp_length"]))

    def test_splits_are_stratified_disjoint_and_reproducible(self):
        data, _ = self.code["clean_data"](example_data())
        splits = self.code["split_data"](data)
        again = self.code["split_data"](data)
        self.assertEqual([len(X) for X, _ in splits.values()], [60, 20, 20])
        indexes = [set(X.index) for X, _ in splits.values()]
        self.assertFalse(indexes[0] & indexes[1] or indexes[0] & indexes[2])
        self.assertFalse(indexes[1] & indexes[2])
        self.assertEqual(set.union(*indexes), set(data.index))
        for name, (X, y) in splits.items():
            self.assertEqual(y.mean(), 0.4)
            self.assertTrue(X.index.equals(again[name][0].index))
            self.assertNotIn("loan_status", X)
            self.assertNotIn("loan_grade", X)
            self.assertNotIn("loan_int_rate", X)

    def test_preprocessing_learns_only_training_statistics(self):
        data = example_data()
        data.loc[:2, "person_emp_length"] = [1.0, 3.0, np.nan]
        train = data.iloc[:3][self.code["FEATURES"]]
        unseen = data.iloc[[3]][self.code["FEATURES"]].copy()
        unseen["person_emp_length"] = 999.0
        unseen["loan_intent"] = "UNSEEN_PURPOSE"
        preprocessor = self.code["make_preprocessor"](scale=True)
        preprocessor.fit(train)
        transformed = preprocessor.transform(unseen)
        numeric = preprocessor.named_transformers_["numeric"]
        employment_index = self.code["NUMERIC_FEATURES"].index("person_emp_length")
        self.assertEqual(numeric.named_steps["imputer"].statistics_[employment_index], 2)
        self.assertEqual(numeric.named_steps["scaler"].mean_[employment_index], 2)
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        self.assertNotIn("UNSEEN_PURPOSE", encoder.categories_[1])
        self.assertTrue(np.isfinite(transformed).all())

    def test_threshold_maximizes_positive_class_f1(self):
        target = np.array([0, 0, 1, 1])
        scores = np.array([0.05, 0.1, 0.3, 0.4])
        threshold = self.code["choose_threshold"](target, scores)
        self.assertAlmostEqual(threshold, 0.3)
        self.assertEqual(f1_score(target, scores >= threshold), 1.0)

    def test_constant_scores_have_a_valid_threshold(self):
        target = np.array([0, 0, 0, 1])
        threshold = self.code["choose_threshold"](target, np.full(4, 0.25))
        self.assertEqual(threshold, 0.25)

    def test_threshold_tie_favors_recall(self):
        threshold = self.code["choose_threshold"]([1, 0, 0, 1], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(threshold, 0.1)

    def test_metrics_use_one_as_default_and_scores_for_auc(self):
        result = self.code["evaluate"]([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)

    def test_all_models_handle_missing_values_and_unknown_categories(self):
        data, _ = self.code["clean_data"](example_data())
        train = data[self.code["FEATURES"]].copy()
        train.loc[0, "person_emp_length"] = np.nan
        train.loc[1, "loan_intent"] = np.nan
        unseen = train.iloc[[0]].copy()
        unseen["loan_intent"] = "UNSEEN_PURPOSE"
        with self.code["threadpool_limits"](limits=2):
            for name, model in self.code["make_models"]().items():
                with self.subTest(model=name):
                    model.fit(train, data["loan_status"])
                    scores = model.predict_proba(unseen)
                    self.assertEqual(scores.shape, (1, 2))
                    self.assertTrue(np.isfinite(scores).all())
                    self.assertAlmostEqual(scores.sum(), 1)


if __name__ == "__main__":
    unittest.main()
