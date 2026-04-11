"""
gru_model_v4.py — Siamese GRU, fourth iteration.

New capability vs v3: Cross-Encoder score as a scalar bridge feature.
  ─────────────────────────────────────────────────────────────────────────
  MOTIVATION
  The previous scalar bridge (v3) gave the MLP head six analytic similarity
  features computed purely from the bi-encoder embeddings.  A cross-encoder
  re-ranks the *full text* of both questions jointly — it attends to every
  token in both questions simultaneously — which lets it capture lexical
  overlap, paraphrase patterns, and subtle nuance that a bi-encoder's fixed
  pooled vector inevitably loses.  Adding this as a single extra scalar
  costs almost nothing in parameters but gives the classifier a near-oracle
  signal to condition on.

  IMPLEMENTATION
  The cross-encoder scores are stored in cross_encoder_scores.zarr, keyed
  by (qid1, qid2).  build_features() loads the zarr, builds a dict lookup,
  and appends the score as the 7th scalar (index 6) after the six analytic
  features inherited from v3.  Pairs that are absent from the zarr (e.g.
  after a max-rows subsample) receive a fallback score of 0.5, which is
  the decision boundary and therefore has zero net bias.

  CHANGES vs v3
  ─────────────────────────────────────────────────────────────────────────
  • n_scalar       : 6  →  7  (adds cross_encoder_score)
  • _N_SCALAR      : 6  →  7
  • build_features : loads cross_encoder_scores.zarr and appends the score
  • __init__       : accepts cross_encoder_zarr path (default ../cross_encoder_scores.zarr)
  • clf_in         : 4*h_full + 7  (one extra input neuron)

  Everything else — attention pooling, deep MLP, AdamW, gradient clipping,
  ReduceLROnPlateau, early stopping, capped pos_weight, recall-friendly
  threshold — is carried over unchanged from v3.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import zarr
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class _AttentionPool(nn.Module):
    """Additive attention over GRU time-steps → context vector."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        scores = self.v(torch.tanh(self.W(outputs)))   # (B, T, 1)
        alpha  = torch.softmax(scores, dim=1)           # (B, T, 1)
        return (alpha * outputs).sum(dim=1)             # (B, H)


# ---------------------------------------------------------------------------
# Siamese GRU v4 network
# ---------------------------------------------------------------------------

class _SiameseGRUv4(nn.Module):
    def __init__(
        self,
        embedding_dim: int  = 2560,
        chunk_size: int     = 256,
        hidden_size: int    = 256,
        num_layers: int     = 2,
        dropout: float      = 0.3,
        mlp_hidden: int     = 512,
        n_scalar: int       = 7,      # 6 analytic + 1 cross-encoder score
    ):
        super().__init__()
        self.seq_len    = embedding_dim // chunk_size
        self.chunk_size = chunk_size
        h_full          = 2 * hidden_size   # bidirectional

        # input normalisation
        self.input_norm = nn.LayerNorm(embedding_dim)

        # GRU encoder (shared weights across both questions — Siamese)
        self.gru = nn.GRU(
            input_size    = chunk_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        # attention pooling
        self.attn = _AttentionPool(h_full)

        # deep MLP classifier
        # input = 4*h_full (GRU interaction) + n_scalar (analytic + cross-encoder)
        clf_in = 4 * h_full + n_scalar
        self.classifier = nn.Sequential(
            nn.LayerNorm(clf_in),
            nn.Linear(clf_in, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(mlp_hidden),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = x.view(-1, self.seq_len, self.chunk_size)
        outputs, _ = self.gru(x)
        return self.attn(outputs)   # (B, 2H)

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        scalars: torch.Tensor,      # (B, n_scalar)
    ) -> torch.Tensor:
        h1 = self.encode(emb1)
        h2 = self.encode(emb2)
        interaction = torch.cat(
            [h1, h2, torch.abs(h1 - h2), h1 * h2, scalars], dim=1
        )
        return self.classifier(interaction)


# ---------------------------------------------------------------------------
# Scalar bridge: analytic similarity + cross-encoder score
# ---------------------------------------------------------------------------

def _compute_scalars(
    emb1_np: np.ndarray,
    emb2_np: np.ndarray,
    ce_scores: np.ndarray,
) -> np.ndarray:
    """
    Returns (N, 7) float32 array of scalar bridge features:
      0: cosine similarity      (normalised dot product)
      1: L2 (Euclidean) distance
      2: dot product of raw embeddings
      3: |norm1 − norm2|        (magnitude difference)
      4: element-wise product mean
      5: element-wise absolute-difference mean
      6: cross-encoder score    (NEW — joint full-text re-ranker probability)

    The cross-encoder score (index 6) is the only feature that requires the
    original question text; all others are computed purely from embeddings.
    It gives the classifier a near-oracle signal while the GRU still learns
    what the cross-encoder cannot easily represent (sequence-level structure
    in the embedding space).
    """
    norm1   = np.linalg.norm(emb1_np, axis=1, keepdims=True).clip(min=1e-12)
    norm2   = np.linalg.norm(emb2_np, axis=1, keepdims=True).clip(min=1e-12)
    ne1     = emb1_np / norm1
    ne2     = emb2_np / norm2

    cos_sim   = (ne1 * ne2).sum(axis=1, keepdims=True)                      # (N,1)
    l2_dist   = np.linalg.norm(emb1_np - emb2_np, axis=1, keepdims=True)   # (N,1)
    dot_raw   = (emb1_np * emb2_np).sum(axis=1, keepdims=True)              # (N,1)
    norm_diff = np.abs(norm1 - norm2)                                        # (N,1)
    prod_mean = (emb1_np * emb2_np).mean(axis=1, keepdims=True)             # (N,1)
    diff_mean = np.abs(emb1_np - emb2_np).mean(axis=1, keepdims=True)      # (N,1)
    ce        = ce_scores.reshape(-1, 1)                                     # (N,1)

    return np.concatenate(
        [cos_sim, l2_dist, dot_raw, norm_diff, prod_mean, diff_mean, ce], axis=1
    ).astype(np.float32)   # (N, 7)


# ---------------------------------------------------------------------------
# Cross-encoder score loader
# ---------------------------------------------------------------------------

def _load_cross_encoder_lookup(zarr_path: str) -> dict[tuple[int, int], float]:
    """
    Load cross_encoder_scores.zarr and return a dict mapping
    (qid1, qid2) → cross_encoder_score (float32).

    The zarr store is expected to contain arrays named 'qid1', 'qid2', and
    'cross_encoder_score', all of length N.
    """
    print(f"  [GRU v4] Loading cross-encoder scores from: {zarr_path}", flush=True)
    store  = zarr.open(zarr_path, mode="r")
    qid1s  = store["qid1"][:].astype(np.int64)
    qid2s  = store["qid2"][:].astype(np.int64)
    scores = store["cross_encoder_score"][:].astype(np.float32)

    lookup: dict[tuple[int, int], float] = {
        (int(q1), int(q2)): float(s)
        for q1, q2, s in zip(qid1s, qid2s, scores)
    }
    print(f"  [GRU v4] Cross-encoder lookup size: {len(lookup):,}", flush=True)
    return lookup


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

_DEFAULTS: dict = dict(
    embedding_dim  = 2560,
    chunk_size     = 256,
    hidden_size    = 256,
    num_layers     = 2,
    dropout        = 0.3,
    mlp_hidden     = 512,
    epochs         = 30,
    batch_size     = 512,
    lr             = 1e-3,
    weight_decay   = 1e-4,
    patience       = 6,
    lr_factor      = 0.5,
    lr_patience    = 4,
    grad_clip      = 1.0,
    val_frac       = 0.05,
    # Recall-oriented tuning — same rationale as v3:
    # Question-deduplication UIs penalise missed duplicates (FN) more harshly
    # than false alarms, so we apply a mild pos_weight cap and shift the
    # decision threshold below 0.5.
    max_pos_weight = 2.0,
    threshold      = 0.42,
    seed           = 42,
    # Cross-encoder zarr path (relative to repo root or absolute).
    # Resolved at build_features() time; can be overridden in the constructor.
    cross_encoder_zarr = "../cross_encoder_scores.zarr",
    # Fallback cross-encoder score for pairs absent from the zarr
    # (e.g. when --max-rows subsamples the dataset).
    # 0.5 sits at the decision boundary and introduces zero net bias.
    ce_fallback = 0.5,
)

_N_SCALAR = 7   # must match _compute_scalars output width


class GRUModelV4:
    name = "SiameseGRU_v4"

    def __init__(self, **overrides):
        self.cfg       = {**_DEFAULTS, **overrides}
        self.threshold = self.cfg["threshold"]
        self._model: _SiameseGRUv4 | None = None
        self._device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feature_names: list[str] | None = None

    # -- interface: build_features -------------------------------------------

    def build_features(self, records):
        """
        X = [emb1 | emb2 | scalars(7)]  shape (N, 2*emb_dim + 7)

        The last 7 columns are the scalar bridge features:
          [cos_sim, l2_dist, dot_raw, norm_diff, prod_mean, diff_mean,
           cross_encoder_score]

        Cross-encoder scores are loaded from cross_encoder_scores.zarr and
        aligned to records by (qid1, qid2).  Any pair missing from the zarr
        (atypical; can occur with --max-rows) falls back to cfg['ce_fallback'].
        """
        ce_zarr_path = self.cfg["cross_encoder_zarr"]
        ce_lookup = _load_cross_encoder_lookup(ce_zarr_path)

        fallback   = float(self.cfg["ce_fallback"])
        emb1       = np.array([r.emb1   for r in records], dtype=np.float32)
        emb2       = np.array([r.emb2   for r in records], dtype=np.float32)
        y          = np.array([r.label  for r in records], dtype=np.int64)
        ce_scores  = np.array(
            [ce_lookup.get((r.qid1, r.qid2), fallback) for r in records],
            dtype=np.float32,
        )

        n_missing = sum(
            1 for r in records
            if (r.qid1, r.qid2) not in ce_lookup
        )
        if n_missing:
            print(
                f"  [GRU v4] {n_missing}/{len(records)} pairs not found in "
                f"cross-encoder zarr; using fallback={fallback:.3f}",
                flush=True,
            )

        scalars = _compute_scalars(emb1, emb2, ce_scores)   # (N, 7)
        X = np.concatenate([emb1, emb2, scalars], axis=1)

        self._feature_names = (
            [f"emb1_{i}" for i in range(emb1.shape[1])] +
            [f"emb2_{i}" for i in range(emb2.shape[1])] +
            [
                "scalar_cos_sim",
                "scalar_l2_dist",
                "scalar_dot_raw",
                "scalar_norm_diff",
                "scalar_prod_mean",
                "scalar_diff_mean",
                "scalar_cross_encoder_score",
            ]
        )
        return X, y, self._feature_names

    # -- interface: fit ------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

        dim = (X_train.shape[1] - _N_SCALAR) // 2   # embedding dimension

        # ---- train / val split ------------------------------------------- #
        val_frac = float(self.cfg["val_frac"])
        n_total  = len(X_train)
        n_val    = max(1, int(n_total * val_frac))
        n_tr     = n_total - n_val

        rng    = np.random.default_rng(self.cfg["seed"])
        perm   = rng.permutation(n_total)
        tr_idx = perm[:n_tr]
        va_idx = perm[n_tr:]

        def _make_loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            X_s  = X_train[idx]
            y_s  = y_train[idx]
            e1   = torch.from_numpy(X_s[:, :dim])
            e2   = torch.from_numpy(X_s[:, dim: 2 * dim])
            sc   = torch.from_numpy(X_s[:, 2 * dim:])
            lab  = torch.from_numpy(y_s)
            ds   = TensorDataset(e1, e2, sc, lab)
            return DataLoader(
                ds,
                batch_size  = self.cfg["batch_size"],
                shuffle     = shuffle,
                num_workers = 4,
                pin_memory  = True,
            )

        train_loader = _make_loader(tr_idx, shuffle=True)
        val_loader   = _make_loader(va_idx, shuffle=False)

        # ---- capped pos_weight ------------------------------------------- #
        y_tr  = y_train[tr_idx]
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        raw_pw    = n_neg / max(n_pos, 1)
        capped_pw = min(raw_pw, float(self.cfg["max_pos_weight"]))
        pos_weight = torch.tensor([capped_pw], dtype=torch.float32).to(self._device)
        print(
            f"  [GRU v4] pos_weight: raw={raw_pw:.3f}  capped={capped_pw:.3f}",
            flush=True,
        )

        # ---- build model ------------------------------------------------- #
        self._model = _SiameseGRUv4(
            embedding_dim = dim,
            chunk_size    = self.cfg["chunk_size"],
            hidden_size   = self.cfg["hidden_size"],
            num_layers    = self.cfg["num_layers"],
            dropout       = self.cfg["dropout"],
            mlp_hidden    = self.cfg["mlp_hidden"],
            n_scalar      = _N_SCALAR,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr           = self.cfg["lr"],
            weight_decay = self.cfg["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode      = "min",
            factor    = self.cfg["lr_factor"],
            patience  = self.cfg["lr_patience"],
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss  = float("inf")
        best_state     = None
        patience_count = 0

        for epoch in range(1, self.cfg["epochs"] + 1):

            # training pass
            self._model.train()
            train_loss = 0.0
            n_batches  = 0
            for e1, e2, sc, lab in train_loader:
                e1  = e1.to(self._device)
                e2  = e2.to(self._device)
                sc  = sc.to(self._device)
                lab = lab.to(self._device).float().unsqueeze(1)

                optimizer.zero_grad()
                logits = self._model(e1, e2, sc)
                loss   = criterion(logits, lab)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._model.parameters(), self.cfg["grad_clip"]
                )
                optimizer.step()
                train_loss += loss.item()
                n_batches  += 1

            avg_train = train_loss / max(n_batches, 1)

            # validation pass
            self._model.eval()
            val_loss = 0.0
            n_vb     = 0
            with torch.no_grad():
                for e1, e2, sc, lab in val_loader:
                    e1  = e1.to(self._device)
                    e2  = e2.to(self._device)
                    sc  = sc.to(self._device)
                    lab = lab.to(self._device).float().unsqueeze(1)
                    val_loss += criterion(self._model(e1, e2, sc), lab).item()
                    n_vb     += 1
            avg_val = val_loss / max(n_vb, 1)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [GRU v4] Epoch {epoch:>2}/{self.cfg['epochs']}  "
                f"train={avg_train:.4f}  val={avg_val:.4f}  "
                f"lr={current_lr:.2e}",
                flush=True,
            )

            scheduler.step(avg_val)

            if avg_val < best_val_loss - 1e-6:
                best_val_loss  = avg_val
                best_state     = {
                    k: v.cpu().clone()
                    for k, v in self._model.state_dict().items()
                }
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.cfg["patience"]:
                    print(
                        f"  [GRU v4] Early stopping at epoch {epoch} "
                        f"(best val={best_val_loss:.4f})",
                        flush=True,
                    )
                    break

        if best_state is not None:
            self._model.load_state_dict(
                {k: v.to(self._device) for k, v in best_state.items()}
            )
            print(
                f"  [GRU v4] Restored best checkpoint (val={best_val_loss:.4f})",
                flush=True,
            )

    # -- interface: predict_proba --------------------------------------------

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        dim  = (X_test.shape[1] - _N_SCALAR) // 2
        emb1 = torch.from_numpy(X_test[:, :dim])
        emb2 = torch.from_numpy(X_test[:, dim: 2 * dim])
        sc   = torch.from_numpy(X_test[:, 2 * dim:])

        ds     = TensorDataset(emb1, emb2, sc)
        loader = DataLoader(
            ds,
            batch_size  = self.cfg["batch_size"],
            shuffle     = False,
            num_workers = 4,
            pin_memory  = True,
        )

        self._model.eval()
        all_proba: list[np.ndarray] = []
        with torch.no_grad():
            for (e1, e2, s) in loader:
                e1 = e1.to(self._device)
                e2 = e2.to(self._device)
                s  = s.to(self._device)
                proba = torch.sigmoid(
                    self._model(e1, e2, s)
                ).cpu().numpy().flatten()
                all_proba.append(proba)

        return np.concatenate(all_proba)

    # -- interface: get_config -----------------------------------------------

    def get_config(self) -> dict:
        params = (
            sum(p.numel() for p in self._model.parameters())
            if self._model is not None else 0
        )
        return {
            "model_class": "SiameseGRU_v4",
            "total_params": params,
            **self.cfg,
        }
