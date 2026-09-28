"""Thermal-motion animation frames by Metropolis Monte Carlo on the MMFF94 surface.

NOT molecular dynamics. There is no integrator, no momentum and no time axis: this
draws configurations from the canonical ensemble at temperature T by proposing random
Cartesian displacements and accepting them with probability min(1, exp(-dE/kT)).
Amplitude therefore grows with T for the right reason, but nothing here is a rate, a
diffusion coefficient or a trajectory.

Further caveats, all material: MMFF94 is a general organic force field never
parameterised for ionic liquids and it handles charged species poorly; this is a
single ion pair in vacuum, not a condensed phase; and a real melting point is a
collective property of many ions that a two-body calculation cannot express. Purely
illustrative.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

KCAL_PER_K = 0.0019872041  # kB in kcal/mol/K


def prep(smi: str):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    if AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE, useRandomCoords=True) != 0:
        return None
    if AllChem.MMFFOptimizeMolecule(m, maxIters=2000) == -1:
        return None
    return m


def _torsions(m):
    """Rotatable dihedral quadruples -- the soft degrees of freedom at 300-600 K."""
    patt = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    out = []
    for b in m.GetSubstructMatches(patt):
        j, k = b
        aj = [x.GetIdx() for x in m.GetAtomWithIdx(j).GetNeighbors() if x.GetIdx() != k]
        ak = [x.GetIdx() for x in m.GetAtomWithIdx(k).GetNeighbors() if x.GetIdx() != j]
        if aj and ak:
            out.append((aj[0], j, k, ak[0]))
    return out


def mc(m, T, steps=6000, keep=60, seed=0):
    """Metropolis MC over TORSIONS and rigid-body ion-pair motion.

    Cartesian jitter of every atom was the first attempt and it fails: bond stretches
    are stiff (~300 kcal/mol/A^2), so a 0.015 A displacement per atom costs several
    kcal/mol against kT = 0.6 at 300 K and nothing is ever accepted. Bond lengths
    barely move at these temperatures anyway. The motions that matter are torsional
    flexing and the two ions moving relative to each other, so those are the moves.

    Step sizes are adapted to hold acceptance near 0.4, which is what makes the
    amplitude comparable between temperatures rather than an artefact of tuning.
    """
    from rdkit.Chem import rdMolTransforms as rmt
    props = AllChem.MMFFGetMoleculeProperties(m)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(m, props)
    tors = _torsions(m)
    frags = Chem.GetMolFrags(m)
    if len(frags) != 2 or not tors:
        return None
    anion = list(frags[1])
    kT = KCAL_PER_K * T
    rng = np.random.default_rng(seed)
    conf = m.GetConformer()
    e = ff.CalcEnergy(list(np.array(conf.GetPositions()).ravel()))
    s_tor, s_tr = 12.0, 0.25          # degrees, angstrom -- adapted below
    # A free ion pair in vacuum simply dissociates once it has thermal energy: the
    # anion drifts off and RMSF runs to tens of angstroms. In a real ionic liquid each
    # ion is caged by its neighbours. A hard wall on the inter-ion centroid distance
    # stands in for that cage. It is an approximation with no condensed-phase physics
    # behind it, and it is what keeps the picture bounded.
    cation = [i for i in range(m.GetNumAtoms()) if i not in set(frags[1])]
    x_init = np.array(conf.GetPositions())
    d0 = float(np.linalg.norm(x_init[anion].mean(0) - x_init[cation].mean(0)))
    d_max = d0 + 2.0
    frames, acc, win = [np.array(conf.GetPositions())], 0, 0

    def cur():
        return np.array(conf.GetPositions())

    def setpos(x):
        for i in range(m.GetNumAtoms()):
            conf.SetAtomPosition(i, x[i].tolist())

    for s in range(steps):
        x0 = cur()
        kind = rng.random()
        if kind < 0.6:
            q = tors[rng.integers(len(tors))]
            a0 = rmt.GetDihedralDeg(conf, *q)
            rmt.SetDihedralDeg(conf, *q, a0 + float(rng.normal(0, s_tor)))
        elif kind < 0.85:
            x = x0.copy(); x[anion] += rng.normal(0, s_tr, 3); setpos(x)
        else:
            x = x0.copy()
            c = x[anion].mean(0)
            th = rng.normal(0, np.deg2rad(s_tor)); ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
            R = np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)
            x[anion] = (x[anion]-c) @ R.T + c; setpos(x)
        xn = cur()
        if np.linalg.norm(xn[anion].mean(0) - xn[cation].mean(0)) > d_max:
            setpos(x0); continue                     # outside the cage
        en = ff.CalcEnergy(list(xn.ravel()))
        if en < e or rng.random() < math.exp(max(-(en - e) / kT, -50.0)):
            e = en; acc += 1; win += 1
        else:
            setpos(x0)
        if (s + 1) % 200 == 0:                      # adapt toward 40% acceptance
            r = win / 200.0
            f = 1.15 if r > 0.45 else (0.87 if r < 0.35 else 1.0)
            s_tor, s_tr, win = s_tor * f, s_tr * f, 0
        if (s + 1) % max(1, steps // keep) == 0:
            frames.append(cur())
    # remove overall translation before reporting or rendering: drift of the whole
    # complex is not thermal motion of interest and would dominate RMSF
    F = np.array(frames)
    F = F - F[:, cation, :].mean(1, keepdims=True)
    return {"frames": [f.round(3).tolist() for f in F],
            "symbols": [at.GetSymbol() for at in m.GetAtoms()],
            "accept": acc / steps, "T": T,
            "rmsf": float(np.sqrt(((F - F.mean(0)) ** 2).sum(-1).mean())),
            "anion_com_sd": float(np.linalg.norm(F[:, anion].mean(1)
                                                 - F[:, anion].mean(1).mean(0), axis=1).std()),
            "cage_A": round(d_max, 2)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", required=True)
    p.add_argument("--temps", default="300,450,600")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", default="runs/il/results/mc_frames.json")
    a = p.parse_args()
    res = []
    for smi in a.pairs:
        m = prep(smi)
        if m is None:
            print(f"  embed/optimise failed: {smi[:50]}"); continue
        rec = {"smiles": smi, "runs": []}
        for T in [float(t) for t in a.temps.split(",")]:
            r = mc(m, T, steps=a.steps, seed=0)
            if r:
                rec["runs"].append(r)
                print(f"  {smi[:42]:42} T={T:.0f}K  accept {r['accept']:.2f}  "
                      f"RMSF {r['rmsf']:.3f} A")
        if rec["runs"]:
            res.append(rec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    print(f"-> {a.out}  ({len(res)} molecules)")


if __name__ == "__main__":
    main()
