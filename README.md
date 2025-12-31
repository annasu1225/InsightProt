# InsightProt: Latent Identity Inference from Protein Conformational Ensembles

This repository contains code for a probabilistic study of **protein identity inference from conformational ensembles** using **multiscale Gauss Linking Integral (mGLI) embeddings** and a **variational autoencoder (VAE)**.

The central question is:

> Given multiple structural observations, can we infer whether they arise from a single underlying protein identity or from a mixture of identities, without relying on sequence alignment or residue-level correspondence?

We frame this as a **set-level latent-variable inference problem** and evaluate identity consistency via **latent-space coherence**.

---

## Overview

### Key ideas
- Proteins are better modeled as **distributions over conformations**, not single static structures.
- Individual conformers may differ substantially in geometry while sharing the same molecular identity.
- Identity should therefore be inferred from **sets of observations**, not from a single structure.
- mGLI provides a **global, deformation-stable embedding** of protein geometry.
- A VAE is used as a **representation learner** to induce a latent space in which conformers of the same protein exhibit higher coherence.

### What this project does
- Computes mGLI embeddings for protein conformers
- Trains a VAE on embeddings without identity supervision
- Evaluates whether **same-identity sets** of conformers are more coherent in latent space than **mixed-identity sets**
- Uses a strictly **protein-disjoint test split** and **test-only negatives**

---

## Dataset

We use a **small subset** of the [ProteinConformers](https://github.com/auroua/ProteinConformers) benchmark, which provides large ensembles of simulated protein conformations.

For feasibility:
- A limited number of proteins are selected
- A small number of conformers per protein are sampled using a diversity heuristic
- The goal is **proof-of-concept**, not exhaustive density modeling

---

## mGLI Embeddings

Each protein structure is mapped to a fixed-length embedding using the **multiscale Gauss Linking Integral** (mGLI):

- Backbone-only (Cα) geometry
- Global, nonlocal descriptor
- Stable under continuous deformations
- No residue indexing or alignment required

Implementation is adapted from:
> Shen et al., *PNAS* (2024)

---

## Training

The VAE is trained on mGLI embeddings with:
- Protein-disjoint train/validation split
- Train-only normalization
- Removal of near-constant embedding dimensions
- Early stopping based on validation loss

The VAE is used **only as a representation learner**; likelihood values are not interpreted as physical densities.

Run training:
```
python train.py
```

Run evaluation:
```
python eval.py
```

---

## Acknowledgement
I appreciate the support and helpful discussions with Alex Lew, Mark Gerstein, and Li Shen. 