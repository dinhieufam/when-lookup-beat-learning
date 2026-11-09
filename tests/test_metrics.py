import numpy as np
import pytest
import scipy.sparse as sp

from mealrec.metrics import health_metrics, mask_scores, ranking_metrics, topk


def csr(rows):
    return sp.csr_matrix(np.array(rows, dtype=np.float32))


def test_topk_orders_best_first_and_breaks_ties_by_index():
    scores = np.array([[0.1, 0.9, 0.5, 0.9]])
    assert topk(scores, 3).tolist() == [[1, 3, 2]]


def test_mask_scores_excludes_training_pairs():
    s = mask_scores(np.ones((1, 3)), csr([[0, 1, 0]]))
    assert np.isneginf(s[0, 1]) and s[0, 0] == 1.0


def test_ranking_metrics_hand_computed():
    # User 0: positives {0, 3}; ranking (meal 1 excluded) = 3, 2, 0.
    # User 1: no positives, so it is not averaged.
    scores = np.array([[0.2, 0.9, 0.5, 0.8], [1.0, 0.0, 0.0, 0.0]])
    truth = csr([[1, 0, 0, 1], [0, 0, 0, 0]])
    exclude = csr([[0, 1, 0, 0], [0, 0, 0, 0]])
    m = ranking_metrics(scores, truth, exclude, ks=(1, 2, 3))
    assert m["recall@1"] == pytest.approx(0.5)
    assert m["recall@2"] == pytest.approx(0.5)
    assert m["recall@3"] == pytest.approx(1.0)
    idcg2 = 1 + 1 / np.log2(3)
    assert m["ndcg@2"] == pytest.approx(1 / idcg2)
    assert m["ndcg@3"] == pytest.approx((1 + 1 / np.log2(4)) / idcg2)


def test_health_metrics_exposure_gap_sign():
    meal_fsa = np.array([5.0, 6.0, 11.0, 12.0])  # meals 0, 1 are the healthier half
    meal_who = np.array([5.0, 4.0, 1.0, 0.0])
    users = np.array([0])
    healthy_first = health_metrics(np.array([[0, 1]]), users, meal_fsa, meal_who, np.array([9.0]), np.array([2.0]))
    unhealthy_first = health_metrics(np.array([[3, 2]]), users, meal_fsa, meal_who, np.array([9.0]), np.array([2.0]))
    assert healthy_first["expo_gap"] == pytest.approx(1.0)
    assert unhealthy_first["expo_gap"] == pytest.approx(-1.0)
    assert healthy_first["fsa_delta"] == pytest.approx(5.5 - 9.0)
    assert healthy_first["who_mean"] == pytest.approx(4.5)
