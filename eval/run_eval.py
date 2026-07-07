"""
Runs the full RAG pipeline against eval/testset.json and scores it with
Ragas (faithfulness, answer relevancy, context precision, context recall).

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --fail-under 0.7   # override threshold

Exit code is non-zero if faithfulness (or the metric given via --gate-metric)
falls below the threshold -- this is what .github/workflows/eval.yml checks
to block a bad change from merging.
"""
import argparse
import json
import sys
import time

from datasets import Dataset

from app import config
from app.retrieval import retrieve
from app.generate import generate_answer


def load_testset(path):
    with open(path) as f:
        raw = json.load(f)
    items = raw.get("examples", raw if isinstance(raw, list) else [])
    items = [
        item for item in items
        if "REPLACE ME" not in item.get("question", "") and item.get("question")
    ]
    return items


def run_pipeline(testset):
    """Runs retrieve + generate for every question, returns a Ragas-shaped dataset dict."""
    questions, answers, contexts_list, ground_truths = [], [], [], []
    for item in testset:
        question = item["question"]
        contexts = retrieve(question)
        answer = generate_answer(question, contexts)

        questions.append(question)
        answers.append(answer)
        contexts_list.append([c["text"] for c in contexts])
        ground_truths.append(item.get("ground_truth", ""))
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def score(dataset_dict):
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    ds = Dataset.from_dict(dataset_dict)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-under", type=float, default=config.FAITHFULNESS_THRESHOLD)
    parser.add_argument("--gate-metric", default="faithfulness")
    args = parser.parse_args()

    testset = load_testset(config.TESTSET_PATH)
    if not testset:
        print(
            "eval/testset.json has no real question/ground_truth pairs yet "
            "(only the placeholder example). Add real pairs based on your "
            "documents before running eval.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Running pipeline over {len(testset)} test questions...")
    t0 = time.time()
    dataset_dict = run_pipeline(testset)
    print(f"Pipeline run took {time.time() - t0:.1f}s")

    print("Scoring with Ragas...")
    result = score(dataset_dict)
    scores = result.to_pandas()

    report = {
        "n_questions": len(testset),
        "config": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "embedding_model": config.EMBEDDING_MODEL,
            "top_k": config.TOP_K,
            "llm_provider": config.LLM_PROVIDER,
        },
        "mean_scores": {
            metric: float(scores[metric].mean())
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            if metric in scores.columns
        },
    }

    config.EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.EVAL_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    gate_value = report["mean_scores"].get(args.gate_metric)
    if gate_value is None:
        print(f"Gate metric '{args.gate_metric}' not found in results.", file=sys.stderr)
        sys.exit(1)

    if gate_value < args.fail_under:
        print(
            f"FAIL: {args.gate_metric} = {gate_value:.3f} is below threshold {args.fail_under}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"PASS: {args.gate_metric} = {gate_value:.3f} >= threshold {args.fail_under}")
    sys.exit(0)


if __name__ == "__main__":
    main()
