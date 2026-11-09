import numpy as np
import pytest

from mealrec.data import DATA_ROOT, DATASETS, SPLITS, load
from mealrec.protocols import cold_split, completion_times, get, mask_heldout, temporal

pytestmark = pytest.mark.skipif(
    not all((DATA_ROOT / d / "meta_data" / "user_course.zip").exists() for d in DATASETS),
    reason="MealRec+ (with --meta) not downloaded; run scripts/download_mealrecplus.sh --meta",
)


@pytest.fixture(scope="module", params=DATASETS)
def data(request):
    return load(request.param)


@pytest.fixture(scope="module")
def uc(data):
    from mealrec.meta import user_course_meta
    return user_course_meta(data.name)


def full_cover(d, um):
    cov = (d.user_course @ d.meal_course.T).multiply(um)
    return np.asarray(cov[um.nonzero()]).ravel() >= 3


def test_p1_only_removes_edges(data):
    p1 = mask_heldout(data)
    assert (p1.user_course - data.user_course).max() <= 0
    assert all((p1.user_meal[s] != data.user_meal[s]).nnz == 0 for s in SPLITS)


def test_p1_heldout_meals_covered_only_through_train_meals(data):
    p1 = mask_heldout(data)
    train_courses = ((data.user_meal["train"] @ data.meal_course) > 0).astype(np.float32)
    for s in ("tune", "test"):
        um = data.user_meal[s]
        users, meals = um.nonzero()
        via_uc = np.asarray((p1.user_course @ data.meal_course.T)[users, meals]).ravel()
        via_train = np.asarray((train_courses @ data.meal_course.T)[users, meals]).ravel()
        assert (via_uc == via_train).all()


def test_p1_keeps_train_meals_fully_covered(data):
    p1 = mask_heldout(data)
    assert full_cover(p1, data.user_meal["train"]).all()


def test_p2_is_temporal(data, uc):
    p2 = temporal(data, uc=uc)
    pairs = completion_times(data, uc)
    train_until = np.datetime64(p2.cutoffs["train_until"])
    # No user-course edge after the cutoff.
    u, c = p2.user_course.nonzero()
    t = uc.drop_duplicates(["u", "c"]).set_index(["u", "c"])["t"]
    assert (t.reindex(list(zip(u, c))).to_numpy() <= train_until).all()
    # Splits are disjoint, cover every original positive, and are ordered in time.
    assert sum(p2.user_meal[s].nnz for s in SPLITS) == len(pairs)
    for a, b in [("train", "tune"), ("train", "test"), ("tune", "test")]:
        assert p2.user_meal[a].multiply(p2.user_meal[b]).nnz == 0
    when = pairs.set_index(["u", "m"])["t"]
    for s, ok in [("train", lambda x: x <= train_until), ("tune", lambda x: x > train_until)]:
        uu, mm = p2.user_meal[s].nonzero()
        assert ok(when.reindex(list(zip(uu, mm))).to_numpy()).all()
    # Train meals stay fully covered; held-out ones never are.
    assert full_cover(p2, p2.user_meal["train"]).all()
    assert not full_cover(p2, p2.user_meal["test"]).any()


def test_cold_split_partitions_meals_and_keeps_every_interaction(data):
    cold = cold_split(data)
    meals = [set(np.flatnonzero(cold.user_meal[s].getnnz(axis=0))) for s in SPLITS]
    assert not (meals[0] & meals[1] or meals[0] & meals[2] or meals[1] & meals[2])
    assert sum(cold.user_meal[s].nnz for s in SPLITS) == sum(data.user_meal[s].nnz for s in SPLITS)
    assert (cold.user_course != data.user_course).nnz == 0  # the leaky variant keeps user_course as released


def test_cold_p1_covers_cold_meals_only_through_train_meals(data):
    cold = get("cold_p1", data.name)
    train_courses = ((cold.user_meal["train"] @ cold.meal_course) > 0).astype(np.float32)
    for s in ("tune", "test"):
        users, meals = cold.user_meal[s].nonzero()
        via_uc = np.asarray((cold.user_course @ cold.meal_course.T)[users, meals]).ravel()
        via_train = np.asarray((train_courses @ cold.meal_course.T)[users, meals]).ravel()
        assert (via_uc == via_train).all()
