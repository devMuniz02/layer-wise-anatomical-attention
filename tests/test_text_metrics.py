from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lana_radgen import metrics


class _StubCheXbertScorer:
    def __init__(self, label_map):
        self.label_map = label_map

    def label_reports(self, reports):
        return [self.label_map[report] for report in reports]


def test_chexpert_metrics_use_model_backed_labels(monkeypatch):
    no_finding = [0] * 13 + [1]
    pleural_effusion = [0] * 9 + [1] + [0] * 4
    label_map = {
        "pred normal": no_finding,
        "ref normal": no_finding,
        "pred effusion": pleural_effusion,
        "ref effusion": pleural_effusion,
    }
    monkeypatch.setattr(metrics, "_CHEXBERT_SCORER", None)
    monkeypatch.setattr(metrics, "_get_chexbert_scorer", lambda: _StubCheXbertScorer(label_map))

    result = metrics.chexpert_label_f1(
        predictions=["pred normal", "pred effusion"],
        references=["ref normal", "ref effusion"],
    )

    assert result["chexpert_available"] is True
    assert result["chexpert_error"] is None
    assert result["chexpert_f1_14_micro"] == pytest.approx(1.0)
    assert result["chexpert_f1_14_macro"] == pytest.approx(2.0 / 14.0)
    assert result["chexpert_per_label_f1"]["Pleural Effusion"] == pytest.approx(1.0)
    assert result["chexpert_per_label_f1"]["No Finding"] == pytest.approx(1.0)


def test_chexpert_metrics_from_reference_labels_preserve_label_order(monkeypatch):
    no_finding = [0] * 13 + [1]
    pneumonia = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    label_map = {
        "pred normal": no_finding,
        "pred pneumonia": pneumonia,
    }
    monkeypatch.setattr(metrics, "_CHEXBERT_SCORER", None)
    monkeypatch.setattr(metrics, "_get_chexbert_scorer", lambda: _StubCheXbertScorer(label_map))

    reference_labels = [
        {label: value for label, value in zip(metrics.CHEXPERT_14_LABELS, no_finding)},
        {label: value for label, value in zip(metrics.CHEXPERT_14_LABELS, pneumonia)},
    ]
    result = metrics.chexpert_label_f1_from_reference_labels(
        predictions=["pred normal", "pred pneumonia"],
        reference_labels=reference_labels,
    )

    assert result["chexpert_available"] is True
    assert result["chexpert_f1_14_micro"] == pytest.approx(1.0)
    assert result["chexpert_per_label_f1"]["Pneumonia"] == pytest.approx(1.0)
    assert result["chexpert_per_label_f1"]["No Finding"] == pytest.approx(1.0)


def test_chexpert_reports_unavailable_error_when_scorer_fails(monkeypatch):
    monkeypatch.setattr(metrics, "_CHEXBERT_SCORER", None)

    def _raise():
        raise RuntimeError("weights missing")

    monkeypatch.setattr(metrics, "_get_chexbert_scorer", _raise)

    result = metrics.chexpert_label_f1(["a"], ["b"])

    assert result["chexpert_available"] is False
    assert "weights missing" in result["chexpert_error"]
    assert result["chexpert_f1_14_micro"] is None
    assert result["chexpert_per_label_f1"]["Pleural Effusion"] is None
