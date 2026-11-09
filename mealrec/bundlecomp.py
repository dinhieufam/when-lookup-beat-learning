"""BundleComp: CompMeal's recipe for any bundle dataset (a bundle scored from its items).

CompMeal scores a meal from per-course signals and is trained by simulating the evaluation
protocol on training data (mealrec.compmeal, mealrec.compmeal_train). Nothing in that recipe
needs meal slots: a bundle is scored from aggregates over its items of the same kinds of
signal. Features of (user u, bundle b), each computed from what the protocol shows:
- cover_ui: share of b's items in u's visible user-item edges;
- cover_known: share of b's items in bundles u is known to have taken;
- item_cf: mean over b's items of user-based CF on user-item (cosine neighbours, self excluded);
- bundle_cf: user-based CF on user-bundle (known pairs, self excluded);
- popularity of b (known pairs) and mean popularity of its items (visible user-item);
- log size of b; user activity (known bundles, visible items).
Works on BundleData (Youshu, NetEase, iFashion) and MealRecPlus alike (meal = bundle,
course = item). Features are built per user chunk, so large datasets fit in memory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

FEATURES = ("cover_ui", "cover_known", "item_cf", "bundle_cf", "log_bundle_pop", "item_pop", "log_size",
            "user_log_known", "user_log_items")


def _bin(m) -> sp.csr_matrix:
    m = sp.csr_matrix(m, dtype=np.float32)
    m.eliminate_zeros()
    m.data[:] = 1.0
    return m


def _rownorm(m: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    return (sp.diags(1 / np.maximum(n, 1e-12)) @ m).tocsr()


@dataclass
class BState:
    ui: sp.csr_matrix       # U x I visible user-item edges
    known: sp.csr_matrix    # U x B known user-bundle pairs
    bi: sp.csr_matrix       # B x I
    ui_n: sp.csr_matrix     # row-normalised ui
    known_n: sp.csr_matrix  # row-normalised known
    size: np.ndarray        # B
    log_bundle_pop: np.ndarray  # B
    item_pop: np.ndarray        # B, mean log1p(item degree in ui) over the bundle's items

    @classmethod
    def build(cls, user_item, known, bundle_item) -> "BState":
        ui, kn, bi = _bin(user_item), _bin(known), _bin(bundle_item)
        size = np.asarray(bi.sum(axis=1)).ravel()
        ideg = np.log1p(np.asarray(ui.sum(axis=0)).ravel())
        return cls(ui, kn, bi, _rownorm(ui), _rownorm(kn), size,
                   np.log1p(np.asarray(kn.sum(axis=0)).ravel()), (bi @ ideg) / np.maximum(size, 1))


def chunk_features(st: BState, users: np.ndarray) -> dict[str, np.ndarray]:
    """Feature arrays (len(users) x B, float32) for the given users."""
    n = len(users)
    bit = st.bi.T.tocsr()
    inv_size = 1 / np.maximum(st.size, 1)
    f = {}
    f["cover_ui"] = (st.ui[users] @ bit).toarray() * inv_size
    kitems = _bin(st.known[users] @ st.bi)
    f["cover_known"] = (kitems @ bit).toarray() * inv_size

    sim = (st.ui_n[users] @ st.ui_n.T).toarray()                 # n x U cosine on user-item
    sim[np.arange(n), users] = 0
    items = (st.ui.T @ sim.T).T                                   # n x I neighbour-weighted item scores
    items /= np.maximum(sim.sum(axis=1, keepdims=True), 1e-12)
    f["item_cf"] = (st.bi @ items.T).T * inv_size                 # mean over each bundle's items

    simb = (st.known_n[users] @ st.known_n.T).toarray()          # n x U cosine on user-bundle
    simb[np.arange(n), users] = 0
    f["bundle_cf"] = np.asarray((st.known.T @ simb.T).T) / np.maximum(simb.sum(axis=1, keepdims=True), 1e-12)

    nb = st.bi.shape[0]
    f["log_bundle_pop"] = np.broadcast_to(st.log_bundle_pop, (n, nb))
    f["item_pop"] = np.broadcast_to(st.item_pop, (n, nb))
    f["log_size"] = np.broadcast_to(np.log1p(st.size), (n, nb))
    f["user_log_known"] = np.broadcast_to(np.log1p(np.asarray(st.known[users].sum(axis=1))), (n, nb))
    f["user_log_items"] = np.broadcast_to(np.log1p(np.asarray(st.ui[users].sum(axis=1))), (n, nb))
    return {k: np.asarray(f[k], dtype=np.float32) for k in FEATURES}
