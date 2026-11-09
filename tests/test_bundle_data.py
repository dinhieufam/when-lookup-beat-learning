import pytest

from mealrec.data import BUNDLE_DATASETS, BUNDLE_ROOT, load_bundle
from mealrec.protocols import mask_heldout

pytestmark = pytest.mark.skipif(
    not all((BUNDLE_ROOT / d).exists() for d in BUNDLE_DATASETS),
    reason="CrossCBR datasets not extracted; see data/README.md",
)


@pytest.fixture(scope="module", params=BUNDLE_DATASETS)
def data(request):
    return load_bundle(request.param)


def test_shapes_match_data_size_file(data):
    assert data.user_course.shape == (data.n_users, data.n_courses)
    assert data.meal_course.shape == (data.n_meals, data.n_courses)
    assert all(m.shape == (data.n_users, data.n_meals) for m in data.user_meal.values())


def test_splits_are_disjoint(data):
    tr, tu, te = (data.user_meal[s] for s in ("train", "tune", "test"))
    assert tr.multiply(te).nnz == 0 and tr.multiply(tu).nnz == 0 and tu.multiply(te).nnz == 0


def test_p1_mask_leaves_heldout_items_only_through_train_bundles(data):
    masked = mask_heldout(data).user_course
    heldout = (data.user_meal["tune"] + data.user_meal["test"]) @ data.meal_course
    train = data.user_meal["train"] @ data.meal_course
    left = masked.multiply(heldout > 0)
    assert left.nnz == left.multiply(train > 0).nnz
    assert masked.nnz <= data.user_course.nnz
