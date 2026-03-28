"""Check if pos_backup and pos_excluded can serve as held-out eval sets.

Steps:
1. Check unique source files per tag
2. Check contamination via cosine similarity >0.99 with training positives
3. Score clean embeddings through faph_hardened_s43 and D_combined_bce_s42
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

CACHE = Path("J:/CLAUDE/PROJECTS/Wakeword/experiments/embedding_cache.npz")
MODELS_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/experiments/models")

def load_cache():
    data = np.load(CACHE, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"],
        "tags": data["tags"],
        "files": data["files"],
    }

def cosine_sim_matrix(A, B):
    """Cosine similarity between each row of A and each row of B. Returns max per row of A."""
    # Normalize
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    # Compute in chunks to avoid OOM (A can be large, B is ~13k)
    max_sims = np.zeros(A.shape[0], dtype=np.float32)
    chunk = 500
    for i in range(0, A.shape[0], chunk):
        sims = A_norm[i:i+chunk] @ B_norm.T  # (chunk, B)
        max_sims[i:i+chunk] = sims.max(axis=1)
    return max_sims

def score_with_model(model_path, embeddings):
    """Run embeddings through an ONNX model, return scores."""
    sess = ort.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    # Run in batches
    scores = []
    batch = 1024
    for i in range(0, len(embeddings), batch):
        out = sess.run([output_name], {input_name: embeddings[i:i+batch]})[0]
        scores.append(out.flatten())
    return np.concatenate(scores)

def main():
    print("Loading embedding cache...")
    cache = load_cache()
    tags = cache["tags"]
    embs = cache["embeddings"]
    files = cache["files"]

    # Training positives
    train_mask = np.isin(tags, ["pos_main", "pos_diverse"])
    train_embs = embs[train_mask]
    print(f"Training positives (pos_main + pos_diverse): {train_embs.shape[0]}")

    # Candidate eval sets
    for tag in ["pos_backup", "pos_excluded"]:
        mask = tags == tag
        tag_embs = embs[mask]
        tag_files = files[mask]
        n = mask.sum()
        unique_files = len(set(tag_files))

        print(f"\n{'='*60}")
        print(f"Tag: {tag} — {n} embeddings, {unique_files} unique source files")
        print(f"{'='*60}")

        # Check contamination
        print("  Checking cosine similarity with training positives...")
        max_sims = cosine_sim_matrix(tag_embs, train_embs)
        contaminated = max_sims > 0.99
        n_contaminated = contaminated.sum()
        n_clean = n - n_contaminated

        print(f"  Contaminated (cosine >0.99 with training): {n_contaminated}")
        print(f"  Clean: {n_clean}")
        if n > 0:
            print(f"  Contamination rate: {n_contaminated/n*100:.1f}%")

        # Similarity distribution
        for thresh in [0.999, 0.99, 0.98, 0.95, 0.90]:
            above = (max_sims > thresh).sum()
            print(f"    cosine > {thresh}: {above}")

        if n_clean == 0:
            print("  No clean embeddings — cannot use as eval set.")
            continue

        clean_embs = tag_embs[~contaminated]
        clean_files = tag_files[~contaminated]
        clean_unique_files = len(set(clean_files))
        print(f"  Clean embeddings: {n_clean} from {clean_unique_files} unique files")

        # Score through models
        thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]

        for model_name in ["faph_hardened_s43.onnx", "D_combined_bce_s42.onnx"]:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                print(f"  Model {model_name} not found, skipping.")
                continue

            print(f"\n  Model: {model_name}")
            scores = score_with_model(model_path, clean_embs)
            print(f"    Score stats: min={scores.min():.4f}, max={scores.max():.4f}, "
                  f"mean={scores.mean():.4f}, median={np.median(scores):.4f}")

            for t in thresholds:
                detected = (scores >= t).sum()
                rate = detected / n_clean * 100
                print(f"    Threshold {t:.2f}: {detected}/{n_clean} detected ({rate:.1f}%)")

    # Also show pos_eval_real contamination for reference
    print(f"\n{'='*60}")
    print("Reference: pos_eval_real contamination check")
    print(f"{'='*60}")
    eval_mask = tags == "pos_eval_real"
    eval_embs = embs[eval_mask]
    max_sims = cosine_sim_matrix(eval_embs, train_embs)
    contaminated = (max_sims > 0.99).sum()
    print(f"  pos_eval_real: {eval_mask.sum()} total, {contaminated} contaminated (cosine >0.99)")
    print(f"  Contamination rate: {contaminated/eval_mask.sum()*100:.1f}%")

if __name__ == "__main__":
    main()
