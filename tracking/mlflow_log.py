"""
Logs one experiment run (a specific config: chunk size, embedding model,
top_k, prompt version, etc.) to MLflow, including the eval scores from
eval/run_eval.py. This is what lets you say "I tried 5 configurations and
here's why I picked this one" with a real artifact to show for it.

Usage:
    python -m tracking.mlflow_log --run-name "chunk_size_300"

Change a config value (env var or app/config.py) between runs, e.g.:
    CHUNK_SIZE=300 python -m tracking.mlflow_log --run-name "chunk_300"
    CHUNK_SIZE=500 python -m tracking.mlflow_log --run-name "chunk_500"

Then:
    mlflow ui
to compare runs side by side.
"""
import argparse

import mlflow

from app import config
from app.ingest import build_index
from eval.run_eval import load_testset, run_pipeline, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-reindex", action="store_true",
                         help="Skip rebuilding the index (use existing chroma_db)")
    parser.add_argument("--experiment", default="rag-eval")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "embedding_model": config.EMBEDDING_MODEL,
            "top_k": config.TOP_K,
            "llm_provider": config.LLM_PROVIDER,
        })

        if not args.skip_reindex:
            n_chunks = build_index(reset=True)
            mlflow.log_metric("n_chunks_indexed", n_chunks)

        testset = load_testset(config.TESTSET_PATH)
        if not testset:
            print("No real testset entries found -- skipping scoring, only logging config/index size.")
            return

        dataset_dict = run_pipeline(testset)
        result = score(dataset_dict)
        scores_df = result.to_pandas()

        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric in scores_df.columns:
                mlflow.log_metric(metric, float(scores_df[metric].mean()))

        mlflow.log_metric("n_questions", len(testset))
        print("Logged run to MLflow. Run `mlflow ui` to view.")


if __name__ == "__main__":
    main()
