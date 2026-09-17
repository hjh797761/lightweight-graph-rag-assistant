"""Small CUDA allocation probe; optionally verify cached BGE embedding/reranking."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", action="store_true", help="also load cached BGE models; never downloads")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new file before allocating resources, retaining diagnostics on failure.
    with args.out.open("x", encoding="utf-8") as output:
        report = {"status": "running", "job_id": os.environ.get("SLURM_JOB_ID"),
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "python": platform.python_version(), "models_requested": args.models}
        started = time.perf_counter()
        try:
            import torch
            report.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                          visible_gpu_count=torch.cuda.device_count())
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; this is not a successful GPU probe")
            if torch.cuda.device_count() != 1:
                raise RuntimeError("Expected exactly one visible GPU from the single-GPU allocation")
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.reset_peak_memory_stats()
            x = torch.ones((256, 256), device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            if not torch.allclose(y, torch.full_like(y, 256)):
                raise RuntimeError("CUDA matrix multiplication produced an unexpected result")
            report["cuda_matmul"] = "passed"
            if args.models:
                from sentence_transformers import CrossEncoder, SentenceTransformer
                embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda", local_files_only=True)
                texts = ["温度升高后打开冷却阀。", "图书馆周末下午五点关闭。"]
                vectors = embedder.encode(texts, batch_size=2, normalize_embeddings=True, show_progress_bar=False)
                if vectors.shape != (2, 512):
                    raise RuntimeError(f"Unexpected BGE embedding shape: {vectors.shape}")
                reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda", local_files_only=True)
                scores = reranker.predict([("冷却阀何时打开？", text) for text in texts], show_progress_bar=False)
                import numpy as np
                if len(scores) != 2 or not np.isfinite(scores).all() or not np.isfinite(vectors).all():
                    raise RuntimeError("Model probe produced invalid outputs")
                report.update(embedding_shape=list(vectors.shape), reranker_scores=[float(s) for s in scores])
            torch.cuda.synchronize()
            report["peak_torch_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
            report["status"] = "complete"
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report["wall_seconds"] = time.perf_counter() - started
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
