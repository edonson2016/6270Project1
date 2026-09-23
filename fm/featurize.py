"""Turning 3-D molecular geometries into fixed-length vectors a plain MLP can model.

This module exists because of a symmetry problem, not a dimensionality problem.
A conformer of a fixed-composition molecule with N atoms is only 3N numbers --
27 for ethanol -- which is nothing. The difficulty is that the *same* physical
conformer has infinitely many coordinate representations (any rotation and
translation), and a plain MLP has no way to know that. Train on raw Cartesians
and the model burns its capacity memorizing orientations.

Two ways out are implemented, and comparing them is the point:

  AlignedCartesian  -- break the symmetry by fixing a frame. Kabsch-align every
      structure to one reference. Exactly invertible, keeps 3N dims, dead simple.
      Fails when the molecule has near-symmetric atoms that the aligner can swap.

  PairwiseDistance  -- remove the symmetry by only representing invariants. The
      N(N-1)/2 interatomic distances are rotation- and translation-invariant by
      construction. Reconstruction back to 3-D needs classical MDS and is only
      approximate, because a generated distance matrix need not correspond to
      any real 3-D arrangement. Also it cannot distinguish mirror images.

Neither is equivariant. Both let you train a working conformer model today and
see exactly what the equivariant machinery would be buying you.
"""

from __future__ import annotations

import numpy as np


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Optimal rotation matrix aligning P onto Q. Both (N, 3), both centered."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])  # guard against a reflection
    return Vt.T @ D @ U.T


class AlignedCartesian:
    """Centre each structure and Kabsch-align it to a reference. Dim = 3N."""

    def __init__(self) -> None:
        self.reference: np.ndarray | None = None
        self.n_atoms: int | None = None

    @property
    def dim(self) -> int:
        return 3 * self.n_atoms

    def fit(self, coords: np.ndarray) -> "AlignedCartesian":
        """coords: (M, N, 3). Uses the first structure as the reference frame."""
        coords = np.asarray(coords, dtype=np.float64)
        self.n_atoms = coords.shape[1]
        ref = coords[0]
        self.reference = ref - ref.mean(axis=0, keepdims=True)
        return self

    def transform(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        centered = coords - coords.mean(axis=1, keepdims=True)
        out = np.empty_like(centered)
        for i, c in enumerate(centered):
            out[i] = c @ kabsch_rotation(c, self.reference).T
        return out.reshape(len(coords), -1).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Back to (M, N, 3). Exact -- this featurization loses nothing but the frame."""
        return np.asarray(x, dtype=np.float64).reshape(-1, self.n_atoms, 3)


class PairwiseDistance:
    """Upper-triangular interatomic distances. Dim = N(N-1)/2."""

    def __init__(self) -> None:
        self.n_atoms: int | None = None
        self._iu: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def dim(self) -> int:
        return self.n_atoms * (self.n_atoms - 1) // 2

    def fit(self, coords: np.ndarray) -> "PairwiseDistance":
        self.n_atoms = np.asarray(coords).shape[1]
        self._iu = np.triu_indices(self.n_atoms, k=1)
        return self

    def transform(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        return d[:, self._iu[0], self._iu[1]].astype(np.float32)

    def to_matrix(self, x: np.ndarray) -> np.ndarray:
        n = self.n_atoms
        D = np.zeros((len(x), n, n), dtype=np.float64)
        D[:, self._iu[0], self._iu[1]] = x
        return D + np.transpose(D, (0, 2, 1))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Classical MDS back to 3-D coordinates.

        Approximate by nature: a vector the model generated may not be a valid
        Euclidean distance matrix at all. Large negative eigenvalues in the
        Gram matrix are the diagnostic that the model produced something
        geometrically impossible -- which is useful signal, not a bug.
        """
        D = self.to_matrix(np.asarray(x, dtype=np.float64))
        n = self.n_atoms
        J = np.eye(n) - np.ones((n, n)) / n
        G = -0.5 * J @ (D ** 2) @ J  # double-centred Gram matrix
        w, V = np.linalg.eigh(G)
        w = w[..., ::-1]
        V = V[..., ::-1]
        top_w = np.clip(w[..., :3], 0.0, None)
        return (V[..., :3] * np.sqrt(top_w)[..., None, :])


def embedding_quality(feat: "PairwiseDistance", x: np.ndarray) -> dict:
    """How Euclidean are these distance vectors, really?

    A set of N(N-1)/2 numbers is only a valid 3-D distance matrix if the
    double-centred Gram matrix is positive semi-definite with rank <= 3. A model
    generating distance vectors has no idea about that constraint, so it will
    produce vectors that describe no arrangement of atoms in 3-D space.

    The diagnostic: eigenvalues 4 and beyond should be ~0 for a genuinely 3-D
    structure, and negative eigenvalues should not exist at all. The mass sitting
    outside the top 3 eigenvalues is exactly the geometric information that gets
    silently discarded when MDS projects the sample back to 3-D.
    """
    D = feat.to_matrix(np.asarray(x, dtype=np.float64))
    n = feat.n_atoms
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ (D ** 2) @ J
    w = np.linalg.eigvalsh(G)[:, ::-1]  # descending

    total = np.abs(w).sum(axis=1)
    top3 = np.abs(w[:, :3]).sum(axis=1)
    neg = np.clip(-w, 0.0, None).sum(axis=1)
    return {
        "frac_variance_in_top3": float((top3 / total).mean()),
        "frac_variance_discarded": float(1.0 - (top3 / total).mean()),
        "neg_eig_mass": float((neg / total).mean()),
        "frac_with_neg_eig": float((w[:, 3:] < -1e-6).any(axis=1).mean()),
        "worst_neg_eig": float(w.min()),
    }


# ------------------------------------------------------------------ diagnostics
def bond_lengths(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Lengths of specified atom pairs. coords (M, N, 3), pairs (P, 2) -> (M, P)."""
    a = coords[:, pairs[:, 0], :]
    b = coords[:, pairs[:, 1], :]
    return np.linalg.norm(a - b, axis=-1)


def infer_bonds(coords: np.ndarray, z: np.ndarray, tol: float = 1.3) -> np.ndarray:
    """Bond list inferred from a reference structure by covalent-radius cutoff."""
    radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02}
    ref = coords[0]
    n = len(z)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(ref[i] - ref[j])
            cutoff = tol * (radii.get(int(z[i]), 0.8) + radii.get(int(z[j]), 0.8))
            if d < cutoff:
                pairs.append((i, j))
    return np.array(pairs, dtype=int)


def geometry_report(gen: np.ndarray, ref: np.ndarray, z: np.ndarray, bonds: np.ndarray) -> dict:
    """Cheap physical sanity metrics for generated conformers, no QM needed.

    These are the metrics that tell you whether the model learned chemistry or
    just learned to put atoms roughly in the right region of space.
    """
    gb, rb = bond_lengths(gen, bonds), bond_lengths(ref, bonds)
    # A bond is "broken" if it falls outside the range ever seen in the reference data.
    lo, hi = rb.min(axis=0) * 0.85, rb.max(axis=0) * 1.15
    intact = ((gb >= lo) & (gb <= hi)).all(axis=1)

    # Any non-bonded pair closer than 0.9 A is a steric clash.
    dg = np.linalg.norm(gen[:, :, None, :] - gen[:, None, :, :], axis=-1)
    iu = np.triu_indices(gen.shape[1], k=1)
    bonded = set(map(tuple, bonds))
    nonbonded = np.array([(i, j) for i, j in zip(*iu) if (i, j) not in bonded])
    clash = (dg[:, nonbonded[:, 0], nonbonded[:, 1]] < 0.9).any(axis=1)

    return {
        "n_samples": int(len(gen)),
        "frac_bonds_intact": float(intact.mean()),
        "frac_no_clash": float((~clash).mean()),
        "frac_valid": float((intact & ~clash).mean()),
        "bond_mean_abs_err": float(np.abs(gb.mean(0) - rb.mean(0)).mean()),
        "bond_std_ratio": float((gb.std(0) / rb.std(0)).mean()),
        "rg_gen": float(np.linalg.norm(gen - gen.mean(1, keepdims=True), axis=-1).mean()),
        "rg_ref": float(np.linalg.norm(ref - ref.mean(1, keepdims=True), axis=-1).mean()),
    }
