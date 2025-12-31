'''
Script to download and subset ProteinConformers dataset from Hugging Face.
Selects a subset of proteins and, for each, samples a diverse set of conformers
using Farthest Point Sampling (FPS) based on CA-RMSD.
Author: Anna Su
Date: Dec 2025
'''

import os, re, zipfile, random, shutil
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download
from Bio.PDB import PDBParser, MMCIFParser


# -------------------- CONFIG --------------------
REPO_ID = "Jim990908/ProteinConformers"
REPO_TYPE = "dataset"

OUT_ROOT = "./pc_subset_fps"
N_PROTEINS = 50
POOL_SIZE = 300          # extract this many conformers per protein 
K_SELECT = 10             # num conformers to select per protein; must be <= POOL_SIZE
SEED = 0

# Optional: constrain which zips to use (leave None for auto)
ZIP_KEYWORD: Optional[str] = None
# -----------------------------------------------

PID_RE = re.compile(r"(T\d{3,4})")
STRUCT_RE = re.compile(r".*\.(pdb|cif|mmcif|ent)$", re.IGNORECASE)


def list_repo_files() -> List[str]:
    api = HfApi()
    return api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)


def group_zips_by_pid(files: List[str], keyword: Optional[str] = None) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    for f in files:
        if not f.lower().endswith(".zip"):
            continue
        if keyword and keyword not in f:
            continue
        m = PID_RE.search(f)
        if not m:
            continue
        pid = m.group(1)
        d.setdefault(pid, []).append(f)
    for pid in d:
        d[pid] = sorted(d[pid])
    return d


def choose_pids(groups: Dict[str, List[str]], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    pids = sorted(groups.keys())
    rng.shuffle(pids)
    return pids[: min(n, len(pids))]


def download_zip(zip_path: str, cache_dir: str) -> str:
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=zip_path,
        cache_dir=cache_dir,
    )


def extract_pool(zip_fp: str, out_dir: str, pool_size: int, seed: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    with zipfile.ZipFile(zip_fp, "r") as z:
        members = [m for m in z.namelist() if STRUCT_RE.match(m) and not m.endswith("/")]
        if not members:
            return []
        # Sample up to pool_size members; extract only those
        chosen = rng.sample(members, k=min(pool_size, len(members)))
        paths = []
        for m in chosen:
            base = os.path.basename(m)
            out_path = os.path.join(out_dir, base)
            with z.open(m) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            paths.append(out_path)
    return paths


def load_ca_coords(struct_path: str) -> Optional[np.ndarray]:
    """Return (L,3) CA coords for first model. None if parse fails or empty."""
    try:
        if struct_path.lower().endswith((".cif", ".mmcif")):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        st = parser.get_structure("x", struct_path)
        model = next(st.get_models())
        coords = []
        for chain in model:
            for res in chain:
                if "CA" in res:
                    coords.append(res["CA"].get_coord())
        if len(coords) < 20:
            return None
        return np.asarray(coords, dtype=float)
    except Exception:
        return None


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD after optimal superposition using NumPy (P and Q share shape (N,3))."""
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P and Q must both be (N,3) arrays with the same N.")
    P_centered = P - P.mean(axis=0, keepdims=True)
    Q_centered = Q - Q.mean(axis=0, keepdims=True)

    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    Q_aligned = Q_centered @ R
    diff = P_centered - Q_aligned
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

def farthest_point_sample(paths: List[str], k: int, seed: int) -> List[str]:
    """
    Greedy FPS using CA-RMSD:
      pick one seed, then iteratively add the structure maximizing min distance to selected.
    """
    rng = random.Random(seed)

    # Preload CA coords; keep only those with valid coords
    items: List[Tuple[str, np.ndarray]] = []
    for p in paths:
        c = load_ca_coords(p)
        if c is not None:
            items.append((p, c))
    if len(items) <= k:
        return [p for p, _ in items]

    # Ensure consistent length by restricting to the modal length
    lengths = [c.shape[0] for _, c in items]
    L_mode = max(set(lengths), key=lengths.count)
    items = [(p, c) for p, c in items if c.shape[0] == L_mode]
    if len(items) <= k:
        return [p for p, _ in items]

    # Initialize
    start_idx = rng.randrange(len(items))
    selected = [items[start_idx]]
    selected_paths = [items[start_idx][0]]

    # Maintain min distance to selected for each candidate
    min_d = np.full(len(items), np.inf, dtype=float)
    # Update with first selected
    P0 = selected[0][1]
    for i, (_, ci) in enumerate(items):
        min_d[i] = kabsch_rmsd(P0, ci)

    # Greedy additions
    for _ in range(k - 1):
        # Avoid re-selecting already selected indices by setting distance -inf
        for sp, _ in selected:
            idx = next(i for i, (p, _) in enumerate(items) if p == sp)
            min_d[idx] = -np.inf

        next_idx = int(np.argmax(min_d))
        selected.append(items[next_idx])
        selected_paths.append(items[next_idx][0])

        # Update min distances using newly selected
        Pnew = items[next_idx][1]
        for i, (_, ci) in enumerate(items):
            if min_d[i] == -np.inf:
                continue
            d = kabsch_rmsd(Pnew, ci)
            if d < min_d[i]:
                min_d[i] = d

    return selected_paths


def main():
    random.seed(SEED)
    os.makedirs(OUT_ROOT, exist_ok=True)
    cache_dir = os.path.join(OUT_ROOT, ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("Listing HF repo files...")
    files = list_repo_files()
    groups = group_zips_by_pid(files, keyword=ZIP_KEYWORD)
    if not groups:
        raise RuntimeError("No PID-keyed zip files found. Adjust PID_RE / ZIP_KEYWORD.")

    pids = choose_pids(groups, N_PROTEINS, SEED)
    print(f"Selected {len(pids)} proteins: {pids}")

    rows = []
    for pid in tqdm(pids, desc="Proteins"):
        # Choose first zip for this protein 
        zip_path = groups[pid][0]
        zip_local = download_zip(zip_path, cache_dir)

        pool_dir = os.path.join(OUT_ROOT, "pool", pid)
        sel_dir = os.path.join(OUT_ROOT, "selected", pid)
        os.makedirs(sel_dir, exist_ok=True)

        pool_paths = extract_pool(zip_local, pool_dir, POOL_SIZE, SEED)
        if len(pool_paths) < K_SELECT:
            continue

        chosen = farthest_point_sample(pool_paths, K_SELECT, SEED)

        # Copy selected into a clean folder
        for p in chosen:
            dst = os.path.join(sel_dir, os.path.basename(p))
            shutil.copy2(p, dst)
            rows.append((pid, zip_path, dst))

    manifest = pd.DataFrame(rows, columns=["protein_id", "source_zip", "structure_path"])
    manifest_path = os.path.join(OUT_ROOT, "selected_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print("\nWrote:", manifest_path)
    print(manifest.groupby("protein_id")["structure_path"].count().describe())


if __name__ == "__main__":
    main()