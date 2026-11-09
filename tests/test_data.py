import numpy as np
import pytest

from mealrec.data import DATA_ROOT, DATASETS, SIZES, load

pytestmark = pytest.mark.skipif(
    not all((DATA_ROOT / d).exists() for d in DATASETS),
    reason="MealRec+ not downloaded; run scripts/download_mealrecplus.sh",
)


@pytest.fixture(scope="module", params=DATASETS)
def data(request):
    return load(request.param)


def test_sizes_match_upstream_loader(data):
    nu, nm, nc = SIZES[data.name]
    assert data.user_course.shape == (nu, nc)
    assert data.meal_course.shape == (nm, nc)
    assert all(m.shape == (nu, nm) for m in data.user_meal.values())


def test_every_meal_is_one_course_per_category(data):
    assert (data.meal_course.getnnz(axis=1) == 3).all()
    for m in range(data.n_meals):
        cats = data.course_category[data.meal_course[m].indices]
        assert sorted(cats.tolist()) == [0, 1, 2]


def test_splits_are_disjoint(data):
    tr, tu, te = (data.user_meal[s] for s in ("train", "tune", "test"))
    assert tr.multiply(te).nnz == 0
    assert tr.multiply(tu).nnz == 0
    assert tu.multiply(te).nnz == 0


def test_meal_scores_in_documented_ranges(data):
    assert data.meal_fsa.min() >= 4 and data.meal_fsa.max() <= 12
    assert data.meal_who.min() >= 0 and data.meal_who.max() <= 7
    assert np.isfinite(data.user_fsa).all()


def test_user_fsa_is_mean_over_train_meals(data):
    train = data.user_meal["train"]
    n = np.asarray(train.sum(axis=1)).ravel()
    has = n > 0
    mean_train = np.asarray(train @ data.meal_fsa).ravel()[has] / n[has]
    assert np.allclose(data.user_fsa[has], mean_train)
