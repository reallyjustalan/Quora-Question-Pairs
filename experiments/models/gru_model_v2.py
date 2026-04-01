"""
gru_model_v2.py — Improved Siamese GRU model for experiment framework.

Key improvements over gru_model.py (baseline):
  1. Attention pooling   — learned weighted sum over all GRU time-steps
                           instead of only the last hidden state, so the
                           model can focus on the most discriminative chunks.
  2. Deep MLP head       — LayerNorm → Linear → GELU → Dropout stacked
                           twice before the final logit; gives the classifier
                           far more capacity to separate the interaction
                           features [h1, h2, |h1-h2|, h1*h2].
  3. Input LayerNorm     — normalises each flat embedding before chunking
                           so gradient magnitudes stay consistent regardless
                           of raw embedding scale.
  4. LR scheduler        — ReduceLROnPlateau on a held-out validation loss;
                           the LR is automatically halved whenever the val
                           loss stalls, replacing the fixed 1e-3 step.
  5. Early stopping      — training terminates when val loss fails to
                           improve for `patience` epochs, preventing wasted
                           compute and overfitting.
  6. Gradient clipping   — clips the global norm to 1.0 before every
                           optimiser step, guarding against exploding
                           gradients that can destabilise GRU training.
  7. Class-balanced loss — BCEWithLogitsLoss pos_weight is set to
                           (#negatives / #positives) from the training
                           labels, so the model is not biased toward the
                           majority class (~63 % non-duplicate).

Drop-in replacement: same build_features / fit / predict_proba / get_config
interface as the original GRUModel.  Register under key "gru_v2".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Attention pooling helper
# ---------------------------------------------------------------------------

class _AttentionPool(nn.Module):
    """
    Single-head additive attention over a sequence of GRU outputs.

    Given outputs of shape (B, T, H), produces a context vector of shape
    (B, H) as a learned weighted sum over the T time steps.

    The weight for each step is:
        alpha_t = softmax( v · tanh(W h_t) )_t
    where W ∈ R^{H×H} and v ∈ R^H are learned parameters.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        # outputs: (B, T, H)
        scores = self.v(torch.tanh(self.W(outputs)))   # (B, T, 1)
        alpha  = torch.softmax(scores, dim=1)           # (B, T, 1)
        ctx    = (alpha * outputs).sum(dim=1)           # (B, H)
        return ctx


# ---------------------------------------------------------------------------
# Improved Siamese GRU network
# ---------------------------------------------------------------------------

class _SiameseGRUv2(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2560,
        chunk_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        mlp_hidden: int = 512,
    ):
        super().__init__()
        self.seq_len    = embedding_dim // chunk_size
        self.chunk_size = chunk_size
        h_full          = 2 * hidden_size   # bidirectional

        # --- input normalisation (improvement 3) ---
        self.input_norm = nn.LayerNorm(embedding_dim)

        # --- GRU encoder ---
        self.gru = nn.GRU(
            input_size    = chunk_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        # --- attention pooling (improvement 1) ---
        self.attn = _AttentionPool(h_full)

        # --- deep MLP classifier (improvement 2) ---
        # interaction dim: [h1; h2; |h1-h2|; h1*h2]  →  4 * h_full
        clf_in = 4 * h_full
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
        """x : (B, embedding_dim)  →  context : (B, 2*hidden_size)"""
        x = self.input_norm(x)                              # improvement 3
        x = x.view(-1, self.seq_len, self.chunk_size)       # (B, T, chunk)
        outputs, _ = self.gru(x)                            # (B, T, 2H)
        ctx = self.attn(outputs)                            # (B, 2H) improvement 1
        return ctx

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        h1 = self.encode(emb1)
        h2 = self.encode(emb2)
        interaction = torch.cat(
            [h1, h2, torch.abs(h1 - h2), h1 * h2], dim=1
        )                                                   # (B, 4*2H)
        return self.classifier(interaction)                 # (B, 1)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

_DEFAULTS: dict = dict(
    embedding_dim = 2560,
    chunk_size    = 256,
    hidden_size   = 128,
    num_layers    = 2,
    dropout       = 0.3,
    mlp_hidden    = 512,
    epochs        = 20,      # more headroom; early-stopping guards overfit
    batch_size    = 512,
    lr            = 1e-3,
    patience      = 3,       # early-stopping patience (improvement 5)
    lr_factor     = 0.5,     # ReduceLROnPlateau reduction factor (improvement 4)
    lr_patience   = 2,       # LR scheduler patience (improvement 4)
    grad_clip     = 1.0,     # max global gradient norm (improvement 6)
    val_frac      = 0.10,    # fraction of training data used for val
    threshold     = 0.5,
    seed          = 42,
)


class GRUModelV2:
    name = "SiameseGRU_v2"

    def __init__(self, **overrides):
        self.cfg       = {**_DEFAULTS, **overrides}
        self.threshold = self.cfg["threshold"]
        self._model: _SiameseGRUv2 | None = None
        self._device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feature_names: list[str] | None = None

    # -- interface: build_features -------------------------------------------

    def build_features(self, records):
        """
        Return raw embedding pairs as features — no hand-crafted features.
        X is (N, 2 * embedding_dim): emb1 concatenated with emb2.
        """
        emb1 = np.array([r.emb1 for r in records], dtype=np.float32)
        emb2 = np.array([r.emb2 for r in records], dtype=np.float32)
        y    = np.array([r.label for r in records], dtype=np.int64)

        X = np.concatenate([emb1, emb2], axis=1)   # (N, 5120)
        self._feature_names = (
            [f"emb1_{i}" for i in range(emb1.shape[1])] +
            [f"emb2_{i}" for i in range(emb2.shape[1])]
        )
        return X, y, self._feature_names

    # -- interface: fit ------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

        dim = X_train.shape[1] // 2

        # ------------------------------------------------------------------ #
        # Improvement 5 — carve out a small validation split for early-stop  #
        # + LR scheduling.  We do this inside fit() so the interface stays   #
        # identical to every other model.                                     #
        # ------------------------------------------------------------------ #
        val_frac = float(self.cfg["val_frac"])
        n_total  = len(X_train)
        n_val    = max(1, int(n_total * val_frac))
        n_tr     = n_total - n_val

        rng    = np.random.default_rng(self.cfg["seed"])
        perm   = rng.permutation(n_total)
        tr_idx = perm[:n_tr]
        va_idx = perm[n_tr:]

        def _make_loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            X_s = X_train[idx]
            y_s = y_train[idx]
            e1  = torch.from_numpy(X_s[:, :dim])
            e2  = torch.from_numpy(X_s[:, dim:])
            lab = torch.from_numpy(y_s)
            ds  = TensorDataset(e1, e2, lab)
            return DataLoader(
                ds,
                batch_size  = self.cfg["batch_size"],
                shuffle     = shuffle,
                num_workers = 4,
                pin_memory  = True,
            )

        train_loader = _make_loader(tr_idx, shuffle=True)
        val_loader   = _make_loader(va_idx, shuffle=False)

        # ------------------------------------------------------------------ #
        # Improvement 7 — class-balanced pos_weight                          #
        # ------------------------------------------------------------------ #
        y_tr  = y_train[tr_idx]
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32
        ).to(self._device)

        # ------------------------------------------------------------------ #
        # Build model                                                         #
        # ------------------------------------------------------------------ #
        self._model = _SiameseGRUv2(
            embedding_dim = dim,
            chunk_size    = self.cfg["chunk_size"],
            hidden_size   = self.cfg["hidden_size"],
            num_layers    = self.cfg["num_layers"],
            dropout       = self.cfg["dropout"],
            mlp_hidden    = self.cfg["mlp_hidden"],
        ).to(self._device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.cfg["lr"]
        )

        # Improvement 4 — learning-rate scheduler on val loss
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

            # ---- training pass ------------------------------------------- #
            self._model.train()
            train_loss = 0.0
            n_batches  = 0
            for e1, e2, lab in train_loader:
                e1  = e1.to(self._device)
                e2  = e2.to(self._device)
                lab = lab.to(self._device).float().unsqueeze(1)

                optimizer.zero_grad()
                logits = self._model(e1, e2)
                loss   = criterion(logits, lab)
                loss.backward()

                # Improvement 6 — gradient clipping
                nn.utils.clip_grad_norm_(
                    self._model.parameters(), self.cfg["grad_clip"]
                )

                optimizer.step()
                train_loss += loss.item()
                n_batches  += 1

            avg_train = train_loss / max(n_batches, 1)

            # ---- validation pass ----------------------------------------- #
            self._model.eval()
            val_loss = 0.0
            n_vb     = 0
            with torch.no_grad():
                for e1, e2, lab in val_loader:
                    e1  = e1.to(self._device)
                    e2  = e2.to(self._device)
                    lab = lab.to(self._device).float().unsqueeze(1)
                    logits    = self._model(e1, e2)
                    val_loss += criterion(logits, lab).item()
                    n_vb     += 1
            avg_val = val_loss / max(n_vb, 1)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [GRU v2] Epoch {epoch:>2}/{self.cfg['epochs']}  "
                f"train={avg_train:.4f}  val={avg_val:.4f}  "
                f"lr={current_lr:.2e}",
                flush=True,
            )

            # Improvement 4 — step scheduler based on val loss
            scheduler.step(avg_val)

            # Improvement 5 — early stopping
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
                        f"  [GRU v2] Early stopping at epoch {epoch} "
                        f"(best val={best_val_loss:.4f})",
                        flush=True,
                    )
                    break

        # Restore best checkpoint
        if best_state is not None:
            self._model.load_state_dict(
                {k: v.to(self._device) for k, v in best_state.items()}
            )
            print(
                f"  [GRU v2] Restored best checkpoint (val={best_val_loss:.4f})",
                flush=True,
            )

    # -- interface: predict_proba --------------------------------------------

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        dim  = X_test.shape[1] // 2
        emb1 = torch.from_numpy(X_test[:, :dim])
        emb2 = torch.from_numpy(X_test[:, dim:])

        ds     = TensorDataset(emb1, emb2)
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
            for (e1, e2) in loader:
                e1    = e1.to(self._device)
                e2    = e2.to(self._device)
                logits = self._model(e1, e2)
                proba  = torch.sigmoid(logits).cpu().numpy().flatten()
                all_proba.append(proba)

        return np.concatenate(all_proba)

    # -- interface: get_config -----------------------------------------------

    def get_config(self) -> dict:
        params = (
            sum(p.numel() for p in self._model.parameters())
            if self._model is not None else 0
        )
        return {
            "model_class": "SiameseGRU_v2",
            "total_params": params,
            **self.cfg,
        }
