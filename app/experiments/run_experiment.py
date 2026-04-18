import json
import time
import subprocess
from pathlib import Path

RESULTS_FILE = Path("experiments/results.jsonl")
RUNS_DIR = Path("experiments/runs")


def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p.stdout


def parse_metrics(eval_output: str):
    metrics = {}
    for line in eval_output.splitlines():
        line = line.strip()
        if line.startswith("Precision@"):
            k, v = line.split(":")
            metrics[k] = float(v.strip())
        if line.startswith("MRR:"):
            _, v = line.split(":")
            metrics["MRR"] = float(v.strip())
        if line.startswith("Questions:"):
            _, v = line.split(":")
            metrics["Questions"] = int(v.strip())
    return metrics


def run_experiment(chunk_size: int, overlap: int):
    run_dir = RUNS_DIR / f"cs{chunk_size}_ov{overlap}"
    run_dir.mkdir(parents=True, exist_ok=True)

    exp = {
        "run_dir": str(run_dir),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "timestamp": time.time(),
    }

    chunk_file = run_dir / "chunks.jsonl"
    index_file = run_dir / "faiss.index"
    meta_file = run_dir / "chunks_meta.json"

    # 1) Ingest -> run_dir/chunks.jsonl
    t0 = time.time()
    run_cmd([
        "python3", "-u", "app/ingest/ingest.py",
        "--input_dir", "data",
        "--output_file", str(chunk_file),
        "--chunk_size", str(chunk_size),
        "--overlap", str(overlap)
    ])
    exp["ingest_s"] = round(time.time() - t0, 4)

    # 2) Build index -> run_dir/faiss.index + chunks_meta.json
    t0 = time.time()
    run_cmd([
        "python", "app/embed/build_index.py",
        "--chunk_file", str(chunk_file),
        "--index_file", str(index_file),
        "--meta_file", str(meta_file)
    ])
    exp["index_s"] = round(time.time() - t0, 4)

    # 3) Evaluate baseline
    t0 = time.time()
    out_base = run_cmd([
        "python", "-m", "app.eval.eval_retrieval",
        "--run_dir", str(run_dir),
        "--rerank_top_k", "0"
    ])

    # 4) Evaluate reranked
    out_rerank = run_cmd([
        "python", "-m", "app.eval.eval_retrieval",
        "--run_dir", str(run_dir),
        "--rerank_top_k", "5"
    ])
    exp["eval_s"] = round(time.time() - t0, 4)

    base_metrics = parse_metrics(out_base)
    rerank_metrics = parse_metrics(out_rerank)

    exp.update({
        "Questions": base_metrics.get("Questions"),

        "Precision@1": base_metrics.get("Precision@1"),
        "Precision@3": base_metrics.get("Precision@3"),
        "Precision@5": base_metrics.get("Precision@5"),
        "MRR": base_metrics.get("MRR"),

        "Precision@1_rerank": rerank_metrics.get("Precision@1"),
        "Precision@3_rerank": rerank_metrics.get("Precision@3"),
        "Precision@5_rerank": rerank_metrics.get("Precision@5"),
        "MRR_rerank": rerank_metrics.get("MRR"),
    })

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(exp) + "\n")

    print("\n✅ Experiment saved:", exp)


if __name__ == "__main__":
    configs = [
        (800, 0),
        (800, 200),
        (1000, 0),
        (1000, 200),
        (1200, 200),
    ]
    for cs, ov in configs:
        print(f"\n=== Running experiment chunk_size={cs}, overlap={ov} ===")
        run_experiment(cs, ov)