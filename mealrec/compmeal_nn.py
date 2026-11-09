"""CompMeal-N: a neural, additive version of CompMeal (direction B1, gate G3).

score(u, m) = sum_k s_k(u, c_k)  +  sum_{k<l} g_kl(c_k, c_l)  +  prior(m)

- s_k: slot score of course c in category k for user u, an MLP over the course-level state
  (mealrec.compmeal.State: reviewed, recency, CF, in a known meal, popularity), user
  activity, and the course embedding (learned ID + projected content) interacted with the
  user's intent for that category (recency-weighted mean embedding of reviewed courses).
- g_kl: compatibility of a course pair, bilinear in the embeddings plus a weight on how
  many catalog meals share the pair.
- prior: meal popularity (dropped for cold protocols, where every test meal has none).

The score is additive over slots and pairs, so full ranking over the catalog is cheap and
the best triples outside the catalog can be searched slot by slot (beam search, later).
Training rows come from the same protocol simulation as the GBDT CompMeal; the loss is a
listwise softmax over each user's labelled meals and uniform negatives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from mealrec.compmeal import Catalog, State
from mealrec.metrics import ranking_metrics

PAIRS = ((0, 1), (0, 2), (1, 2))
COURSE_FEATURES = ("reviewed", "log_days_since", "cf", "in_known", "log_pop", "log_pop_recent")


def content_embedding(course_meta: pd.DataFrame, used: np.ndarray, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Cu x dim, from static recipe text only (name, ingredients, tags): no review data."""
    text = (course_meta.reindex(used)[["course_name", "ingredients", "tags"]].fillna("")
            .astype(str).agg(" ".join, axis=1).str.replace(r"[\^;]", " ", regex=True))
    x = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(text)
    z = TruncatedSVD(dim, random_state=seed).fit_transform(x)
    return (z / np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)).astype(np.float32)


@dataclass
class Tensors:
    """A State moved to the device, in the shapes the model reads."""
    x: torch.Tensor          # U x Cu x F course-level features
    user: torch.Tensor       # U x 4
    recency: torch.Tensor    # U x Cu, exp(-days/365) where reviewed, else 0
    meal_prior: torch.Tensor  # M x 2
    pair_log: torch.Tensor    # M x 3, log1p(other counted meals sharing each course pair)


def to_tensors(st: State, device, pair_counts: np.ndarray) -> Tensors:
    nu, ncu = st.reviewed.shape
    lds = np.nan_to_num(st.log_days_since, nan=0.0)
    feats = [st.reviewed, lds, st.cf, st.in_known,
             np.broadcast_to(st.log_pop, (nu, ncu)), np.broadcast_to(st.log_pop_recent, (nu, ncu))]
    x = np.stack([np.asarray(f, np.float32) for f in feats], axis=-1)
    days = np.expm1(np.nan_to_num(st.log_days_since, nan=np.inf))
    recency = np.where(st.reviewed, np.exp(-days / 365.0), 0.0).astype(np.float32)
    user = np.stack([st.user[k] for k in ("log_reviews", "log_reviews_recent", "log_days_since_last",
                                           "log_known_meals")], axis=1).astype(np.float32)
    user = np.nan_to_num(user, nan=0.0)
    prior = np.stack([st.meal_log_pop, st.meal_log_pop_recent], axis=1).astype(np.float32)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return Tensors(t(x), t(user), t(recency), t(prior), t(np.log1p(pair_counts).astype(np.float32)))


class CompMealN(nn.Module):
    def __init__(self, cat: Catalog, course_category: np.ndarray, content: np.ndarray, dim: int = 64,
                 hidden: int = 64, use_prior: bool = True, use_pairs: bool = True, use_content: bool = True):
        super().__init__()
        ncu = len(cat.used)
        self.register_buffer("triples", torch.from_numpy(cat.triples))
        self.register_buffer("slot_of", torch.from_numpy(course_category.astype(np.int64)))
        self.register_buffer("content", torch.from_numpy(content))
        self.use_prior, self.use_pairs, self.use_content = use_prior, use_pairs, use_content
        self.id_emb = nn.Embedding(ncu, dim)
        nn.init.normal_(self.id_emb.weight, std=0.01)
        self.proj = nn.Linear(content.shape[1], dim, bias=False)
        n_in = len(COURSE_FEATURES) + 4 + 2 * dim
        self.slot_mlp = nn.ModuleList(nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                                      for _ in range(3))
        self.pair_w = nn.ParameterList(nn.Parameter(torch.zeros(dim, dim)) for _ in PAIRS)
        self.pair_count_w = nn.Parameter(torch.zeros(len(PAIRS)))
        self.prior_w = nn.Parameter(torch.zeros(2))
        # Input standardisation, set from the training states (set_scaling).
        self.register_buffer("x_mu", torch.zeros(len(COURSE_FEATURES)))
        self.register_buffer("x_sd", torch.ones(len(COURSE_FEATURES)))
        self.register_buffer("u_mu", torch.zeros(4))
        self.register_buffer("u_sd", torch.ones(4))

    def set_scaling(self, ts: list[Tensors]) -> None:
        x = torch.cat([t.x.reshape(-1, t.x.shape[-1]) for t in ts])
        u = torch.cat([t.user for t in ts])
        self.x_mu, self.x_sd = x.mean(0), x.std(0).clamp(min=1e-6)
        self.u_mu, self.u_sd = u.mean(0), u.std(0).clamp(min=1e-6)

    def course_emb(self) -> torch.Tensor:
        e = self.id_emb.weight
        if self.use_content:
            e = e + self.proj(self.content)
        return e

    def slot_scores(self, t: Tensors, users: torch.Tensor) -> torch.Tensor:
        """B x Cu x 3: score of every course in every slot (only its own slot is used)."""
        e = self.course_emb()                                   # Cu x d
        rec = t.recency[users]                                  # B x Cu
        x = (t.x[users] - self.x_mu) / self.x_sd                # B x Cu x F
        u = (t.user[users] - self.u_mu) / self.u_sd             # B x 4
        out = []
        for k in range(3):
            w = rec * (self.slot_of == k)                       # B x Cu
            intent = (w @ e) / w.sum(1, keepdim=True).clamp(min=1e-6)   # B x d
            inter = intent[:, None, :] * e[None]                # B x Cu x d
            h = torch.cat([x, u[:, None, :].expand(-1, x.shape[1], -1), inter,
                           e[None].expand(x.shape[0], -1, -1)], dim=-1)
            out.append(self.slot_mlp[k](h).squeeze(-1))
        return torch.stack(out, dim=-1)

    def meal_bias(self, t: Tensors) -> torch.Tensor:
        """M: pair compatibility plus the popularity prior of every catalog meal."""
        e = self.course_emb()
        bias = torch.zeros(self.triples.shape[0], device=e.device)
        if self.use_pairs:
            for j, (a, b) in enumerate(PAIRS):
                ea, eb = e[self.triples[:, a]], e[self.triples[:, b]]
                bias = bias + ((ea @ self.pair_w[j]) * eb).sum(-1) + self.pair_count_w[j] * t.pair_log[:, j]
        if self.use_prior:
            bias = bias + t.meal_prior @ self.prior_w
        return bias

    def scores(self, t: Tensors, users: torch.Tensor, meals: torch.Tensor | None = None) -> torch.Tensor:
        """B x M (all catalog meals) or B x N for given meals (B x N indices)."""
        s = self.slot_scores(t, users)
        bias = self.meal_bias(t)
        tri = self.triples
        if meals is None:
            return s[:, tri[:, 0], 0] + s[:, tri[:, 1], 1] + s[:, tri[:, 2], 2] + bias[None]
        tm = tri[meals]                                          # B x N x 3
        g = lambda k: torch.gather(s[..., k], 1, tm[..., k])
        return g(0) + g(1) + g(2) + bias[meals]

    def score_triples(self, t: Tensors, users: torch.Tensor, tri: torch.Tensor,
                      pair_log_dense: list[torch.Tensor], pos_in_slot: torch.Tensor) -> torch.Tensor:
        """B x N scores of arbitrary triples (B x N x 3, used-space ids); pair counts by dense lookup."""
        s = self.slot_scores(t, users)
        total = sum(torch.gather(s[..., k], 1, tri[..., k]) for k in range(3))
        if self.use_pairs:
            e = self.course_emb()
            for j, (a, b) in enumerate(PAIRS):
                ea, eb = e[tri[..., a]], e[tri[..., b]]
                cnt = pair_log_dense[j][pos_in_slot[tri[..., a]], pos_in_slot[tri[..., b]]]
                total = total + ((ea @ self.pair_w[j]) * eb).sum(-1) + self.pair_count_w[j] * cnt
        return total

    @torch.no_grad()
    def open_topk(self, t: Tensors, users: torch.Tensor, pair_log_dense: list[torch.Tensor],
                  slot_courses: list[torch.Tensor], k: int, beam: int = 40) -> torch.Tensor:
        """Top-k (appetizer, main, dessert) triples over the whole course space, B x k x 3 (used-space ids).

        Slot scores keep the best `beam` courses per slot; the beam**3 triples are then scored
        exactly with the pair terms. No meal prior: a new triple has no popularity.
        pair_log_dense[j]: |slot a| x |slot b| log1p(pair counts) over slot_courses positions.
        """
        s = self.slot_scores(t, users)
        e = self.course_emb()
        top, pos = [], []
        for j in range(3):
            v, i = torch.topk(s[:, slot_courses[j], j], beam, dim=1)      # B x beam (positions in slot j)
            top.append(v)
            pos.append(i)
        total = top[0][:, :, None, None] + top[1][:, None, :, None] + top[2][:, None, None, :]
        if self.use_pairs:
            for j, (a, b) in enumerate(PAIRS):
                ea = e[slot_courses[a][pos[a]]]                              # B x beam x d
                eb = e[slot_courses[b][pos[b]]]
                bil = torch.einsum("bid,de,bje->bij", ea, self.pair_w[j], eb)
                cnt = pair_log_dense[j][pos[a][:, :, None], pos[b][:, None, :]]
                term = bil + self.pair_count_w[j] * cnt                     # B x beam x beam
                shape = [term.shape[0], 1, 1, 1]
                shape[1 + a], shape[1 + b] = beam, beam
                total = total + term.reshape(shape)
        flat = torch.topk(total.reshape(total.shape[0], -1), k, dim=1).indices
        ia, rem = flat // (beam * beam), flat % (beam * beam)
        ib, ic = rem // beam, rem % beam
        g = lambda j, i: slot_courses[j][torch.gather(pos[j], 1, i)]
        return torch.stack([g(0, ia), g(1, ib), g(2, ic)], dim=-1)


def train_compmeal_nn(train_sets: list[tuple], eval_state: State, tune: sp.csr_matrix,
                      test: sp.csr_matrix, train: sp.csr_matrix, cat: Catalog, course_category: np.ndarray,
                      content: np.ndarray, seed: int, device, use_prior: bool = True, use_pairs: bool = True,
                      use_content: bool = True, n_neg: int = 500, epochs: int = 60, patience: int = 10,
                      batch_users: int = 64, lr: float = 1e-3, weight_decay: float = 1e-5,
                      eval_pair_counts: np.ndarray | None = None, log=print) -> dict:
    """Fit on simulated (state, labels[, pair_counts]) sets; select the epoch by full-ranking tune R@20.

    pair_counts (M x 3) default to cat.pair_counts, for the training sets and for evaluation.
    Returns test/tune metrics, the U x M scores and the tune curve of the selected epoch, and
    the model (weights of that epoch) with the evaluation tensors.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CompMealN(cat, course_category, content, use_prior=use_prior, use_pairs=use_pairs,
                      use_content=use_content).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sets = [(to_tensors(ts[0], device, ts[2] if len(ts) > 2 else cat.pair_counts),
             ts[0].known.toarray() > 0, ts[1].toarray() > 0) for ts in train_sets]
    model.set_scaling([t for t, _, _ in sets])
    ev = to_tensors(eval_state, device, cat.pair_counts if eval_pair_counts is None else eval_pair_counts)
    n_users = eval_state.reviewed.shape[0]

    @torch.no_grad()
    def all_scores() -> np.ndarray:
        model.eval()
        out = [model.scores(ev, torch.arange(s, min(s + 256, n_users), device=device)).cpu().numpy()
               for s in range(0, n_users, 256)]
        return np.concatenate(out)

    best, curve, since = (-1.0, -1, None), [], 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for t, known, lab in sets:
            users = rng.permutation(np.flatnonzero(lab.any(axis=1)))
            for i in range(0, len(users), batch_users):
                ub = users[i:i + batch_users]
                cands, masks = [], []
                for u in ub:
                    pos = np.flatnonzero(lab[u])
                    free = np.flatnonzero(~known[u] & ~lab[u])
                    neg = rng.choice(free, size=min(n_neg, len(free)), replace=False)
                    cands.append(np.concatenate([pos, neg]))
                    masks.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
                width = max(len(c) for c in cands)
                idx = np.zeros((len(ub), width), np.int64)
                ispos = np.zeros((len(ub), width), np.float32)
                valid = np.zeros((len(ub), width), bool)
                for r, (c, m) in enumerate(zip(cands, masks)):
                    idx[r, :len(c)], ispos[r, :len(c)], valid[r, :len(c)] = c, m, True
                idx_t = torch.from_numpy(idx).to(device)
                s = model.scores(t, torch.from_numpy(ub).to(device), idx_t)
                s = s.masked_fill(~torch.from_numpy(valid).to(device), -torch.inf)
                logp = F.log_softmax(s, dim=1)
                p = torch.from_numpy(ispos).to(device)
                loss = -((logp.masked_fill(p == 0, 0) * p).sum(1) / p.sum(1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        sc = all_scores()
        r = ranking_metrics(sc, tune, exclude=train)["recall@20"]
        curve.append(r)
        if r > best[0]:
            best, since = (r, epoch, sc), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        log(f"  epoch {epoch:3d} loss {np.mean(losses):.4f} tune R@20 {r:.4f}")
        if since >= patience:
            break
    sc = best[2]
    model.load_state_dict(best_state)
    model.eval()
    return {"scores": sc, "epoch": best[1], "tune_curve": curve, "model": model, "eval_tensors": ev,
            "tune": ranking_metrics(sc, tune, exclude=train), "test": ranking_metrics(sc, test, exclude=train)}


def train_open(train_sets: list[tuple], eval_state: State, cat: Catalog, course_category: np.ndarray,
               content: np.ndarray, slot_courses: list[np.ndarray], select_fn, seed: int, device,
               n_cat: int = 150, n_rand: int = 150, n_recombined: int = 200, epochs: int = 40, patience: int = 6,
               batch_users: int = 32, lr: float = 3e-4, weight_decay: float = 1e-4, log=print) -> dict:
    """Train CompMeal-N for open composition (no meal prior).

    train_sets: (state, labels U x M, pair_log_dense) per simulated set, where pair_log_dense[j]
    is |slot a| x |slot b| log1p(pair counts over the meals known in that state), on device.
    Each user's candidates: labelled meals, and negatives of three kinds: unknown catalog meals,
    random triples from the slot pools, and triples recombined from the user's reviewed courses
    (the hard case in open space: familiar courses that do not form one of their meals).
    select_fn(model, eval_tensors) -> tune score; the best epoch's weights are kept.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CompMealN(cat, course_category, content, use_prior=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_in_slot = np.full(len(cat.used), 0)
    for j in range(3):
        pos_in_slot[slot_courses[j]] = np.arange(len(slot_courses[j]))
    pos_t = torch.from_numpy(pos_in_slot).to(device)
    zeros = np.zeros((len(cat.triples), 3))
    sets = [(to_tensors(st, device, zeros), st.known.toarray() > 0, lab.toarray() > 0, st.reviewed, dense)
            for st, lab, dense in train_sets]
    model.set_scaling([x[0] for x in sets])
    ev = to_tensors(eval_state, device, zeros)

    def candidates(u, known, lab, reviewed):
        pos = cat.triples[np.flatnonzero(lab[u])]
        free = np.flatnonzero(~known[u] & ~lab[u])
        negs = [cat.triples[rng.choice(free, size=min(n_cat, len(free)), replace=False)],
                np.stack([rng.choice(slot_courses[j], size=n_rand) for j in range(3)], axis=1)]
        mine = [np.intersect1d(np.flatnonzero(reviewed[u]), slot_courses[j]) for j in range(3)]
        negs.append(np.stack([rng.choice(m if len(m) else slot_courses[j], size=n_recombined)
                              for j, m in enumerate(mine)], axis=1))
        neg = np.concatenate(negs)
        posset = {tuple(x) for x in pos}
        neg = neg[[tuple(x) not in posset for x in neg]]
        return np.concatenate([pos, neg]), len(pos)

    best, since, curve, best_state = (-1.0, -1), 0, [], None
    for epoch in range(epochs):
        model.train()
        losses = []
        for t, known, lab, reviewed, dense in sets:
            users = rng.permutation(np.flatnonzero(lab.any(axis=1)))
            for i in range(0, len(users), batch_users):
                ub = users[i:i + batch_users]
                cs = [candidates(u, known, lab, reviewed) for u in ub]
                width = max(len(c) for c, _ in cs)
                tri = np.zeros((len(ub), width, 3), np.int64)
                ispos = np.zeros((len(ub), width), np.float32)
                valid = np.zeros((len(ub), width), bool)
                for r, (c, n_pos) in enumerate(cs):
                    tri[r, :len(c)], ispos[r, :n_pos], valid[r, :len(c)] = c, 1, True
                sc = model.score_triples(t, torch.from_numpy(ub).to(device), torch.from_numpy(tri).to(device),
                                         dense, pos_t)
                sc = sc.masked_fill(~torch.from_numpy(valid).to(device), -torch.inf)
                logp = F.log_softmax(sc, dim=1)
                p = torch.from_numpy(ispos).to(device)
                loss = -((logp.masked_fill(p == 0, 0) * p).sum(1) / p.sum(1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        model.eval()
        r = select_fn(model, ev)
        curve.append(r)
        if r > best[0]:
            best, since = (r, epoch), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        log(f"  epoch {epoch:3d} loss {np.mean(losses):.4f} tune {r:.4f}")
        if since >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "eval_tensors": ev, "epoch": best[1], "tune_curve": curve}
