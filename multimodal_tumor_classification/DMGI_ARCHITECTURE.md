# DMGI Architecture (Deep Multiplex Graph Infomax)

Reference: *DMGI: Deep Multiplex Graph Infomax* (AAAI 2023, Park et al.)

## 1. Overview

DMGI learns node embeddings on a **multiplex graph** — a set of graphs sharing
the same nodes but with different edge structures (relations). It maximises
mutual information between local (node-level) and global (graph-level) features
across all relation types, regularises towards a shared **consensus embedding**,
and optionally adds a supervised classification loss.

In this codebase every patient is a node. Four clinical-feature subgroups define
four relation types. Node features are the concatenation of Swin-Tiny image
embeddings (768-d) and one-hot/numerical clinical features (31-d), row-normalised.

## 2. Graph Construction

### Per-relation similarity graphs

For each of the 4 feature groups (receptor/staging, demographics/progression,
treatment, anatomical), pairwise cosine similarity is computed over the
corresponding clinical columns. An edge exists when similarity >= θ_r. Each
relation has its own threshold from the paper's Duke-Breast setting:

| Relation | θ |
|----------|------|
| receptor_staging | 0.75 |
| demographics_progression | 0.90 |
| treatment | 0.75 |
| anatomical | 0.75 |

### Self-loops

Weighted self-connections are added: `A_hat = A + sc * I` where `sc = 3.0`.
This amplifies each node's own features relative to neighbours.

### Symmetric normalisation

`A_norm = D^{-1/2} A_hat D^{-1/2}` where D is the degree matrix of A_hat.
The result is stored as a sparse COO tensor for efficient `spmm`.

## 3. Single-Layer GCN with Dropout

Each relation r has its own GCN:

```
h_r = ReLU( A_norm_r @ (Dropout(X) @ W_r) )
```

- `W_r`: (in_dim, hid_dim) with Xavier uniform init, no bias
- Dropout probability: 0.5 (applied to input features)
- One layer only (no stacking)

## 4. Mutual Information Maximisation

### Summary vector

For each relation, the graph-level summary is:

```
s_r = sigmoid( mean(h_r) )
```

### Discriminator (shared)

A single bilinear discriminator is shared across all relations:

```
score(s, h) = h^T M s     (M is a learnable [hid x hid] matrix)
```

The discriminator takes `(summary, h_pos, h_neg)` and returns logits of shape
`(2N,)` — concatenation of positive scores (real nodes) and negative scores
(corrupted nodes).

### Negative samples

Node features are corrupted by row-permutation (`x_shuf = x[randperm]`).
Corrupted embeddings: `h_neg_r = GCN_r(x_shuf, A_norm_r)`.

### MI loss

Binary cross-entropy with logits, summed (not averaged) across relations:

```
L_MI = sum_r BCE(logits_r, [1...1, 0...0])
```

The MI loss has an implicit coefficient of 1.0 (not tunable).

## 5. Per-Node Attention Fusion

For each relation, a learnable linear projection `a_r: R^{hid} -> R^1`
produces a per-node attention score. Softmax is taken across relations
per node:

```
alpha_{r,i} = softmax_r( a_r(h_{r,i}) )
```

Positive and negative embeddings are fused separately:

```
h_fused_pos = sum_r alpha_r * h_pos_r
h_fused_neg = sum_r alpha_r * h_neg_r
```

## 6. Learnable Consensus H

`H` is an `(N, hid)` parameter matrix initialised with Xavier uniform.
It is **not** derived from any encoder — it is directly optimised.

The classifier operates on H:

```
logits = Linear(H)    # (N, num_classes)
```

### Consensus regularisation loss

```
L_reg_pos = sum( (H - h_fused_pos)^2 )
L_reg_neg = sum( (H - h_fused_neg)^2 )
L_consensus = L_reg_pos - L_reg_neg
```

This pulls H toward the fused positive embeddings and pushes it away from
the fused negative embeddings.

## 7. Three-Part Loss

```
L = L_MI + α * L_consensus + β * L_supervised
```

| Symbol | Config name | Paper value | Description |
|--------|-------------|-------------|-------------|
| (implicit 1.0) | — | 1.0 | MI loss coefficient |
| α | `DMGI_REG_COEF` | 0.001 | Consensus regularisation weight |
| β | `DMGI_SUP_COEF` | 0.01 | Supervised CE weight |
| γ | weight_decay | 0.0001 | L2 regularisation (Adam) |

Supervised loss is cross-entropy on `H[train_mask]` only, with class-balanced
weights computed from the training set.

## 8. Node Features

1. **Image**: Swin-Tiny 768-d embeddings (`swin_features.npy`), cached from
   the Swin pipeline
2. **Clinical**: 31-d one-hot + numerical encoding (`encode_clinical_features`)
3. **Concatenation**: `[image; clinical]` → 799-d
4. **Row normalisation**: each node's feature vector is L2-normalised

This replaces the previous approach of using Swin fused embeddings (which
included a learned projection). Using raw features lets DMGI's GCN learn its
own projection.

## 9. Differences: Current vs Reference Implementation

| Aspect | Previous (this repo) | Current (aligned with reference) |
|--------|---------------------|----------------------------------|
| GCN depth | 2 layers per relation | 1 layer per relation |
| Discriminator | Per-relation (R discriminators) | Shared (1 discriminator) |
| Consensus | Attention-weighted average of relation embeddings | Learnable parameter H (N, hid) |
| Attention | Global scalar weights per relation | Per-node attention fusion |
| MI loss | Averaged across relations | Summed across relations |
| Consensus loss | MSE(h_r, consensus), detached | (H - h_pos)^2 - (H - h_neg)^2 |
| Loss weights | alpha/beta/gamma = 1.0/0.5/1.0 | reg_coef=0.001, sup_coef=0.01 |
| Node features | Swin fused embeddings (proj_dim-d) | Swin raw (768-d) + clinical (31-d), row-normalised |
| Thresholds | Single threshold, grid-searched | Per-relation [0.75, 0.9, 0.75, 0.75] |
| Self-loops | A + I | A + 3.0*I |
| Early stopping | Val macro F1 | Total loss |
| Hyperparameters | Random search (20 combos) | Fixed from paper |
| Input dropout | None | 0.5 |
| Weight decay | Grid-searched | Fixed 0.0001 |
