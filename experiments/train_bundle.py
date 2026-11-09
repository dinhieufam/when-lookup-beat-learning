"""Train one bundle-recommendation model on MealRec+ under one protocol and seed.

Model code comes from the official repos in external/ (scripts/setup_baselines.sh);
this script owns data, negative sampling, seeds and evaluation. Tune AND test are
evaluated after every epoch with mealrec.metrics, so epoch selection by tune and by
test (as in the MealRec+ paper) can be compared afterwards on the same run (D2).

    python -m experiments.train_bundle --model CrossCBR --dataset MealRec+H --protocol orig --seed 0

Writes results/runs/<protocol>/<dataset>/<model>/seed<k>/:
    history.json      per-epoch loss and tune/test metrics
    per_user.npz      per-user metrics at the tune-best and test-best epochs
    scores_tune_best.npy  U x M scores (float16) at the tune-best epoch
    model_tune_best.pt    state_dict at the tune-best epoch
    summary.json      written last; its presence marks the run complete
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from mealrec.data import BUNDLE_DATASETS, DATASETS, MealRecPlus
from mealrec.metrics import per_user_metrics
from mealrec.protocols import COLD_PROTOCOLS, PROTOCOLS, VARIANT_PROTOCOLS, get, mask_heldout

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "external"
SURVEY = EXTERNAL / "bundle-recommendation-survey" / "discriminative BR"
KS = (20, 40, 80)
# Above this many user x meal cells, rank in user chunks and do not save the score matrix
# (iFashion is 53,897 x 27,694). MealRec+ stays on the original full-matrix path.
FULL_MATRIX_MAX = 50_000_000

# Upstream best settings for the CrossCBR-release datasets (CrossCBR/config.yaml, MultiCBR/config.yaml).
UPSTREAM = {
    ("CrossCBR", "iFashion"): dict(aug_type="ED", ed_interval=1, num_layers=1, item_level_ratio=0.2,
                                   bundle_level_ratio=0.2, bundle_agg_ratio=0.0, l2_reg=4e-5, c_lambda=0.25,
                                   c_temp=0.2, lr=1e-3),
    ("MultiCBR", "iFashion"): dict(aug_type="Noise", ed_interval=1, num_layers=2, UB_ratio=0.05, UI_ratio=0.0,
                                   BI_ratio=0.15, fusion_weights={"modal_weight": [0.1, 0.2, 0.7],
                                                                  "UB_layer": [0.5, 0.5, 0.0],
                                                                  "UI_layer": [0.3, 0.2, 0.5],
                                                                  "BI_layer": [0.5, 0.5, 0.0]},
                                   l2_reg=1e-5, c_lambda=0.1, c_temp=0.2, lr=1e-3),
}
UPSTREAM.update({
    ("CrossCBR", "Youshu"): dict(aug_type="ED", ed_interval=1, num_layers=1, item_level_ratio=0.2,
                                 bundle_level_ratio=0.2, bundle_agg_ratio=0.2, l2_reg=1e-4, c_lambda=0.04,
                                 c_temp=0.25, lr=1e-3),
    ("CrossCBR", "NetEase"): dict(aug_type="ED", ed_interval=1, num_layers=1, item_level_ratio=0.1,
                                  bundle_level_ratio=0.3, bundle_agg_ratio=0.1, l2_reg=1e-4, c_lambda=0.1,
                                  c_temp=0.25, lr=1e-3),
    ("MultiCBR", "Youshu"): dict(aug_type="Noise", ed_interval=1, num_layers=2, UB_ratio=0.2, UI_ratio=0.1,
                                 BI_ratio=0.1, fusion_weights={"modal_weight": [0.35, 0.15, 0.5],
                                                               "UB_layer": [0.7, 0.2, 0.1],
                                                               "UI_layer": [0.6, 0.2, 0.2],
                                                               "BI_layer": [0.6, 0.1, 0.3]},
                                 l2_reg=1e-5, c_lambda=0.04, c_temp=0.23, lr=1e-3),
    ("MultiCBR", "NetEase"): dict(aug_type="Noise", ed_interval=1, num_layers=2, UB_ratio=0.0, UI_ratio=0.0,
                                  BI_ratio=0.2, fusion_weights={"modal_weight": [0.5, 0.2, 0.3],
                                                                "UB_layer": [0.35, 0.15, 0.5],
                                                                "UI_layer": [0.25, 0.65, 0.1],
                                                                "BI_layer": [0.4, 0.4, 0.2]},
                                  l2_reg=1e-5, c_lambda=0.05, c_temp=0.2, lr=1e-3),
})
UPSTREAM_EPOCHS = {"iFashion": 50, "NetEase": 50, "Youshu": 100}


# ---------------------------------------------------------------- models

class LightGCN(nn.Module):
    """User-meal LightGCN (He et al., 2020). Sees no course information: the non-leaky reference."""

    def __init__(self, ub: sp.csr_matrix, dim: int, layers: int, device):
        super().__init__()
        nu, nb = ub.shape
        self.nu, self.layers = nu, layers
        self.emb = nn.Parameter(nn.init.xavier_normal_(torch.empty(nu + nb, dim)))
        adj = sp.bmat([[None, ub], [ub.T, None]]).tocsr()
        deg = np.asarray(adj.sum(axis=1)).ravel()
        d = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1)))
        a = (d @ adj @ d).tocoo()
        self.graph = torch.sparse_coo_tensor(
            np.vstack([a.row, a.col]), a.data.astype(np.float32), a.shape).coalesce().to(device)

    def propagate(self):
        x, out = self.emb, [self.emb]
        for _ in range(self.layers):
            x = torch.sparse.mm(self.graph, x)
            out.append(x)
        x = torch.stack(out).mean(0)
        return x[: self.nu], x[self.nu:]


def bpr(pred: torch.Tensor) -> torch.Tensor:
    return F.softplus(pred[:, 1:] - pred[:, :1]).mean()


class Adapter:
    """Uniform wrapper: loss(users [B,1], bundles [B,1+neg], epoch, last_batch) and scores()."""

    batch_size = 2048
    n_neg = 1

    def scores(self, users: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def end_epoch(self):
        pass


def _graphs(data: MealRecPlus):
    f = lambda m: sp.csr_matrix(m, dtype=np.float32)
    return [f(data.user_meal["train"]), f(data.user_course), f(data.meal_course)]


_REPO_MODULES = ("config", "model", "models", "utils", "layers", "manifolds", "optimizers")


def _import_from(path: Path):
    """Put one upstream repo first on sys.path, forgetting modules a previously imported repo
    left under the same top-level names (they all use config/models/utils)."""
    for name in list(sys.modules):
        if name.split(".")[0] in _REPO_MODULES:
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(EXTERNAL))]
    sys.path.insert(0, str(path))


def build(name: str, data: MealRecPlus, a, device) -> Adapter:
    ad = Adapter()
    nu, nb, ni = data.n_users, data.n_meals, data.n_courses
    graphs = _graphs(data)

    if name == "LightGCN":
        model = LightGCN(graphs[0], a.dim, a.layers, device).to(device)
        cfg = {"lr": a.lr or 1e-3, "l2": a.l2 if a.l2 is not None else 1e-4}
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        def loss(u, b, epoch, last):
            uf, bf = model.propagate()
            pred = (uf[u] * bf[b]).sum(-1)
            reg = (model.emb[u.squeeze(1)] ** 2).sum() + (model.emb[nu + b] ** 2).sum()
            return bpr(pred) + cfg["l2"] * reg / u.shape[0]

        def scores(users):
            uf, bf = model.propagate()
            return uf[users] @ bf.T

    elif name in ("CrossCBR", "MultiCBR"):
        _import_from(EXTERNAL / name)
        mod = __import__(f"models.{name}", fromlist=[name])
        conf = {"device": device, "embedding_size": a.dim, "num_users": nu, "num_bundles": nb,
                "num_items": ni, "num_layers": a.layers}
        if name == "CrossCBR":  # config.yaml, Youshu (closest in size); 2 layers as in MealRec+
            conf.update(aug_type="ED", ed_interval=1, item_level_ratio=0.2, bundle_level_ratio=0.2,
                        bundle_agg_ratio=0.2, l2_reg=1e-4, c_lambda=0.04, c_temp=0.25, lr=1e-3)
        else:  # MultiCBR config.yaml, Youshu
            conf.update(aug_type="Noise", ed_interval=1, UB_ratio=0.2, UI_ratio=0.1, BI_ratio=0.1,
                        fusion_weights={"modal_weight": [0.35, 0.15, 0.5], "UB_layer": [0.7, 0.2, 0.1],
                                        "UI_layer": [0.6, 0.2, 0.2], "BI_layer": [0.6, 0.1, 0.3]},
                        l2_reg=1e-5, c_lambda=0.04, c_temp=0.23, lr=1e-3)
        conf.update(UPSTREAM.get((name, a.dataset), {}))
        if a.lr:
            conf["lr"] = a.lr
        if a.l2 is not None:
            conf["l2_reg"] = a.l2
        model = getattr(mod, name)(conf, graphs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["l2_reg"])
        cfg = {k: v for k, v in conf.items() if k != "device"}

        def loss(u, b, epoch, last):
            # Upstream redraws edge dropout once per ed_interval epochs, at the last batch.
            bpr_loss, c_loss = model((u, b), ED_drop=conf["aug_type"] == "ED" and last)
            return bpr_loss + conf["c_lambda"] * c_loss

        if name == "CrossCBR":
            def scores(users):
                return model.evaluate(model.propagate(test=True), users)
        else:
            def scores(users):
                return model.evaluate(model.get_multi_modal_representations(test=True), users)

    elif name == "BGCN":
        _import_from(SURVEY / "BGCN")
        from model import BGCN, BGCN_Info  # noqa: E402  (survey repo, BGCN/model)
        # BGCN/config.py optimal settings; --l2 sets its embedding L2 weight ("decay")
        cfg = {"lr": a.lr or 3e-4, "decay": a.l2 if a.l2 is not None else 1e-7, "mess_dropout": 0.3, "node_dropout": 0}
        info = BGCN_Info(a.dim, cfg["decay"], cfg["mess_dropout"], cfg["node_dropout"], a.layers)
        sizes = SimpleNamespace(num_users=nu, num_bundles=nb, num_items=ni)
        model = BGCN(info, sizes, graphs, device).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        def loss(u, b, epoch, last):
            pred, l2 = model(u, b)
            return bpr(pred) + l2 / u.shape[0]

        def scores(users):
            return model.evaluate(model.propagate(), users)

    elif name == "CoHeat":
        _import_from(EXTERNAL / "CoHeat" / "src")
        from model import CoHeat  # noqa: E402  (snudatalab/CoHeat, src/model.py)
        # src/config.yaml, Youshu; 1 layer as upstream unless --layers says otherwise
        conf = {"device": device, "embedding_size": a.dim, "num_users": nu, "num_bundles": nb, "num_items": ni,
                "num_layers": a.layers, "aff_ed_ratio": 0.5, "hist_ed_ratio": 0.5, "agg_ed_ratio": 0.5,
                "lr": a.lr or 1e-3, "lambda2": a.l2 if a.l2 is not None else 1e-4, "lambda1": 0.1, "max_temp": 10000}
        freq = np.asarray(graphs[0].sum(axis=0)).ravel()  # training popularity; 0 for cold meals
        model = CoHeat(conf, graphs, freq).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["lambda2"])
        cfg = {k: v for k, v in conf.items() if k != "device"}
        heat = {"psi": 1.0}

        def loss(u, b, epoch, last):
            # Curriculum heating over the fixed budget: psi = max_temp ** (epoch / epochs), epochs 1-based.
            heat["psi"] = conf["max_temp"] ** ((epoch + 1) / a.epochs)
            bpr_loss, c_loss = model((u, b), ED_drop=True, psi=heat["psi"])
            return bpr_loss + conf["lambda1"] * c_loss

        def scores(users):
            return model.evaluate(model.propagate(test=True), users, heat["psi"])

    elif name == "HyperMBR":
        _import_from(SURVEY / "HyperMBR")
        from config import parser as hparser  # noqa: E402
        import optimizers as hopt  # noqa: E402
        from models.base_models import BRECModel_Distance, BRECModel_Distance_Info  # noqa: E402
        from utils.loss import WMRBLoss  # noqa: E402
        args = hparser.parse_args([])
        for k, v in vars(args).items():  # list-valued defaults are search grids; take the setting used
            if isinstance(v, list) and k not in ("hard_window", "hard_prob"):
                setattr(args, k, v[0])
        args.dim, args.num_layers = a.dim, a.layers
        if a.lr:
            args.lr = a.lr
        args.device, args.cuda = device, 0
        args.num_users, args.num_bundles, args.num_items = nu, nb, ni
        args.itemLevelCoef, args.bundleLevelCoef = 0.7, 0.3  # main.py search values
        args.itemLevel_c, args.bundleLevel_c, args.temperature = 1.0, 2.0, 1.0
        info = BRECModel_Distance_Info(embedding_size=args.dim, dropout=args.dropout, num_layers=args.num_layers,
                                       manifold=args.manifold, negativeNum=args.negative_num, act=args.act,
                                       embed_L2_norm=args.embed_L2_norm, task="tune")
        model = BRECModel_Distance(args, info, graphs).to(device)
        opt = getattr(hopt, args.optimizer)(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        wmrb = WMRBLoss(args, "mean")
        ad.batch_size, ad.n_neg = 4096, args.negative_num
        cfg = {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
               for k, v in vars(args).items()}

        def loss(u, b, epoch, last):
            return wmrb(model(u, b, epoch + 1), batch_size=ad.batch_size)

        def scores(users):
            return model.evaluate(model.propagate(), users)[2]

        def end_epoch():  # upstream resets these running sums every epoch
            model.item_Teach_bundle_KL_loss = model.bundle_Teach_item_KL_loss = model.kdLossSum = 0
        ad.end_epoch = end_epoch

    else:
        raise ValueError(f"unknown model {name!r}")

    ad.model, ad.opt, ad.loss, ad.scores, ad.config = model, opt, loss, scores, cfg
    return ad


MODELS = ("LightGCN", "BGCN", "CrossCBR", "MultiCBR", "HyperMBR", "CoHeat")
# HyperMBR/config.py: epochs 200 (~9 s/epoch on MealRec+H). CoHeat: src/config.yaml epochs 100; its
# heating schedule spans the whole budget, so run it without early stopping (--min-epochs 100).
MAX_EPOCHS = {"HyperMBR": 200, "CoHeat": 100}


# ---------------------------------------------------------------- training

class NegSampler:
    """Uniform negatives among meals the user has no training interaction with."""

    def __init__(self, train: sp.csr_matrix, rng: np.random.Generator):
        self.n = train.shape[1]
        u, m = train.nonzero()
        self.pos = np.sort(u.astype(np.int64) * self.n + m)
        self.rng = rng

    def __call__(self, users: np.ndarray, k: int) -> np.ndarray:
        neg = self.rng.integers(self.n, size=(len(users), k))
        while True:
            key = users[:, None] * self.n + neg
            idx = np.minimum(np.searchsorted(self.pos, key), len(self.pos) - 1)
            bad = self.pos[idx] == key
            if not bad.any():
                return neg
            neg[bad] = self.rng.integers(self.n, size=int(bad.sum()))


@torch.no_grad()
def all_scores(ad: Adapter, n_users: int, device, chunk: int = 512) -> torch.Tensor:
    ad.model.eval()
    out = [ad.scores(torch.arange(s, min(s + chunk, n_users), device=device))
           for s in range(0, n_users, chunk)]
    return torch.cat(out).float()


@torch.no_grad()
def chunked_top(ad: Adapter, data, device, chunk: int = 2048) -> np.ndarray:
    """Top-max(KS) meals per user, scored in user chunks with training meals masked."""
    ad.model.eval()
    train, tops = data.user_meal["train"].tocsr(), []
    for s in range(0, data.n_users, chunk):
        e = min(s + chunk, data.n_users)
        sc = ad.scores(torch.arange(s, e, device=device)).float()
        r, c = train[s:e].nonzero()
        sc[torch.from_numpy(r).to(device), torch.from_numpy(c).to(device)] = -torch.inf
        tops.append(torch.topk(sc, max(KS), dim=1).indices.cpu().numpy())
    return np.concatenate(tops)


def evaluate(scores: torch.Tensor, data: MealRecPlus, train_mask: torch.Tensor) -> dict:
    return evaluate_top(torch.topk(scores.masked_fill(train_mask, -torch.inf), max(KS), dim=1).indices.cpu().numpy(),
                        data)


def evaluate_top(top: np.ndarray, data) -> dict:
    out = {}
    for split in ("tune", "test"):
        truth = data.user_meal[split]
        users = np.flatnonzero(truth.getnnz(axis=1))
        out[split] = {"users": users, **per_user_metrics(top[users], truth[users], KS)}
    return out


def mean_metrics(ev: dict) -> dict:
    return {s: {m: float(v.mean()) for m, v in d.items() if m != "users"} for s, d in ev.items()}


def graph_attrs(model) -> dict:
    """The graph tensors (and raw scipy graphs) an upstream model keeps as plain attributes.
    Every model here names them *graph* (CrossCBR, MultiCBR, BGCN); parameters are not included."""
    return {k: v for k, v in vars(model).items() if "graph" in k.lower()}


def replay_folds(data, a, device) -> list[tuple[np.ndarray, dict]]:
    """Protocol replay for graph models (--replay K), as for BundleComp.

    Training pairs are split into K folds (fixed per seed). For fold k, the fold's pairs play the
    held-out role: they are removed from the user-bundle graph and the P1 mask is applied to them
    (item edges explained only by fold-k bundles are dropped). Training on fold k sees the graphs
    a test pair sees at evaluation. Returns (fold pairs, graph attributes) per fold.
    """
    train = data.user_meal["train"].tocsr()
    u, m = train.nonzero()
    fold = np.random.default_rng(a.seed + 2).integers(a.replay, size=len(u))
    empty = sp.csr_matrix(train.shape, dtype=np.float32)
    out = []
    for k in range(a.replay):
        sel = fold == k
        pick = lambda s: sp.csr_matrix((np.ones(int(s.sum()), np.float32), (u[s], m[s])), shape=train.shape)
        held, rest = pick(sel), pick(~sel)
        masked = mask_heldout(replace(data, user_meal={"train": rest, "tune": held, "test": empty}))
        fold_data = replace(masked, user_meal={**data.user_meal, "train": rest})
        graphs = graph_attrs(build(a.model, fold_data, a, device).model)
        out.append((np.stack([u[sel], m[sel]], 1).astype(np.int64), graphs))
        print(f"replay fold {k}: {int(sel.sum())} pairs, item edges {data.user_course.nnz} -> {fold_data.user_course.nnz}",
              flush=True)
    return out


def out_dir(a) -> Path:
    """results/<root>/<protocol>/<dataset>/<model>[__<tag>]/seed<k>"""
    model = f"{a.model}__{a.tag}" if a.tag else a.model
    return ROOT / "results" / a.root / a.protocol / a.dataset / model / f"seed{a.seed}"


def run(a) -> Path:
    out = out_dir(a)
    if (out / "summary.json").exists() and not a.force:
        print(f"done already: {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed)
    np.random.seed(a.seed)  # upstream edge dropout uses the global numpy RNG
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get(a.protocol, a.dataset)
    folds = replay_folds(data, a, device) if a.replay else None
    if folds is not None:  # building the fold graphs drew random numbers; restart from the seed
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
    ad = build(a.model, data, a, device)
    eval_graphs = graph_attrs(ad.model) if folds is not None else None
    train = data.user_meal["train"].tocsr()
    tu, tm = train.nonzero()
    pairs = np.stack([tu, tm], 1).astype(np.int64)
    full = data.n_users * data.n_meals <= FULL_MATRIX_MAX
    train_mask = torch.from_numpy(train.toarray() > 0).to(device) if full else None
    sampler = NegSampler(train, np.random.default_rng(a.seed))
    rng = np.random.default_rng(a.seed + 1)
    history, best = [], {"tune": (-1.0, -1), "test": (-1.0, -1)}
    per_user, since_best, t0 = {}, 0, time.time()
    for epoch in range(a.epochs):
        ad.model.train()
        te = time.time()
        losses = []
        # Without replay: one pass over all training pairs on the protocol's graphs. With replay:
        # one pass per fold on that fold's graphs; together the folds cover every training pair once.
        passes = [(pairs, None)] if folds is None else folds
        for fold_pairs, graphs in passes:
            if graphs is not None:
                vars(ad.model).update(graphs)
            order = rng.permutation(len(fold_pairs))
            nb = max(len(fold_pairs) // ad.batch_size, 1)  # drop_last, as upstream
            for i in range(nb):
                batch = fold_pairs[order[i * ad.batch_size:(i + 1) * ad.batch_size]]
                negs = sampler(batch[:, 0], ad.n_neg)
                u = torch.from_numpy(batch[:, :1]).to(device)
                b = torch.from_numpy(np.concatenate([batch[:, 1:], negs], 1)).to(device)
                ad.opt.zero_grad()
                # Edge dropout is redrawn at the last batch of an epoch (upstream); under replay at the
                # first batch of each fold, so the dropout graph is drawn from that fold's graphs.
                redraw = (i == 0) if graphs is not None else (i == nb - 1)
                loss = ad.loss(u, b, epoch, redraw)
                loss.backward()
                ad.opt.step()
                losses.append(float(loss.detach()))
        if eval_graphs is not None:
            vars(ad.model).update(eval_graphs)
        ad.end_epoch()
        train_s = time.time() - te

        if full:
            scores = all_scores(ad, data.n_users, device)
            ev = evaluate(scores, data, train_mask)
        else:
            ev = evaluate_top(chunked_top(ad, data, device), data)
        mm = mean_metrics(ev)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "train_s": train_s,
                        "eval_s": time.time() - te - train_s, **mm})
        improved = False
        for split in ("tune", "test"):
            r = mm[split]["recall@20"]
            if r > best[split][0]:
                best[split] = (r, epoch)
                per_user[f"{split}_best"] = {s: {k: v if k == "users" else v.astype(np.float32) for k, v in ev[s].items()}
                                               for s in ev}
                if split == "tune":
                    improved = True
                    if full:
                        np.save(out / "scores_tune_best.npy", scores.cpu().numpy().astype(np.float16))
                    torch.save(ad.model.state_dict(), out / "model_tune_best.pt")
        since_best = 0 if improved else since_best + 1
        print(f"ep {epoch:3d} loss {history[-1]['loss']:.4f} tune R@20 {mm['tune']['recall@20']:.4f} "
              f"test R@20 {mm['test']['recall@20']:.4f} ({train_s:.1f}s+{history[-1]['eval_s']:.1f}s)", flush=True)
        if epoch + 1 >= a.min_epochs and since_best >= a.patience:
            break

    np.savez_compressed(out / "per_user.npz", **{
        f"{sel}/{split}/{k}": v for sel, d in per_user.items() for split, m in d.items() for k, v in m.items()})
    (out / "history.json").write_text(json.dumps(history) + "\n")
    tb, sb = best["tune"][1], best["test"][1]
    summary = {
        "model": a.model, "dataset": a.dataset, "protocol": a.protocol, "seed": a.seed,
        "epochs_run": len(history), "tune_best_epoch": tb, "test_best_epoch": sb,
        "select_by_tune": history[tb], "select_by_test": history[sb],
        "tag": a.tag, "seconds": time.time() - t0, "sec_per_epoch": (time.time() - t0) / len(history),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "args": vars(a), "model_config": ad.config,
        "cutoffs": getattr(data, "cutoffs", None),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"tune-best ep {tb}: test R@20 {history[tb]['test']['recall@20']:.4f} | "
          f"test-best ep {sb}: test R@20 {history[sb]['test']['recall@20']:.4f} | {summary['seconds']:.0f}s")
    return out


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--dataset", required=True, choices=DATASETS + BUNDLE_DATASETS)
    p.add_argument("--protocol", default="orig", choices=PROTOCOLS + COLD_PROTOCOLS + VARIANT_PROTOCOLS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=None,
                   help="maximum epochs (default: 1000; HyperMBR 200 and iFashion 50, their upstream configs)")
    p.add_argument("--patience", type=int, default=50, help="stop after this many epochs without a tune improvement")
    p.add_argument("--min-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=None, help="override the model's learning rate")
    p.add_argument("--l2", type=float, default=None, help="override the model's L2 weight (BGCN: embedding decay)")
    p.add_argument("--root", default="runs", help="output tree under results/ (runs, runs_tune, runs_tuned, ...)")
    p.add_argument("--tag", default="", help="suffix for the model directory, e.g. a hyperparameter setting")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--replay", type=int, default=0,
                   help="train by protocol replay with this many folds (0: off); needs a model that reads graphs")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if a.epochs is None:
        a.epochs = UPSTREAM_EPOCHS.get(a.dataset, MAX_EPOCHS.get(a.model, 1000))
        a.min_epochs = min(a.min_epochs, a.epochs)
    return a


if __name__ == "__main__":
    run(parse())
