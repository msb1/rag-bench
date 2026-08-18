import os
import time
from datetime import timedelta
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from evaluate import (
    evaluate_answer_correctness,
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_faithfulness,
    evaluate_relevancy,
    format_context_list,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
results_columns = [
    "question",
    "answer",
    "ground_truth",
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "answer_correctness"
]
"""
Essential Checklist for a 1000-Record Run:

Packages questions, generated text answers, ground_truth, and chunks into long "LLM-as-a-judge" instructions.
Go into LM Studio's settings panel and change your model's Context Length (max_tokens) to at least 8192 (or higher
if your chunks are massive) to prevent errors.

Start with a Mini-Test: Before committing to a multi-hour 1000-row execution run, test your pipeline integrity
by slicing the dataframe down to just 3 records first:

    evaluation_dataset = Dataset.from_pandas(df.head(3))
"""

def evaluate_row(row: tuple[Any, ...]) -> pd.DataFrame:
    """Processes a single row of the dataframe through all 5 metrics."""
    # Extract data from rows
    question = row.question
    answer = row.answer
    contexts = row.contexts
    ground_truth = row.ground_truth

    print(f"Evaluating Question: {question} with Answer: {answer}")

    # Run your 5 native text-only metrics
    faithfulness = evaluate_faithfulness(contexts, answer)
    relevancy = evaluate_relevancy(question, answer)
    recall = evaluate_context_recall(ground_truth, contexts)
    precision = evaluate_context_precision(question, ground_truth, contexts)
    correctness = evaluate_answer_correctness(answer, ground_truth)

    print(f"faithfulness={faithfulness:.2f}, relevancy={relevancy:.2f}, recall={recall:.2f}, precision={precision:.2f}, correctness={correctness:.2f}")

    # Return as a series to automatically expand into dataframe columns
    return pd.DataFrame({
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_recall": recall,
        "context_precision": precision,
        "answer_correctness": correctness
    }, index=[0])


def main():
    # LOAD AND FORMAT YOUR JSONL FILE
    jsonl_file_path = os.getenv("DATASET_FILE")

    print(f"Loading data from {jsonl_file_path}...")
    # read_json with lines=True reads line-by-line JSON objects flawlessly
    df = pd.read_json(jsonl_file_path, lines=True)
    # df = df.head(3)

    # Quick Sanity Check: Ensure contexts are lists of strings, not a single string
    if not isinstance(df["contexts"].iloc[0], list):
        print("Warning: 'contexts' should be an array of strings in your JSONL file. Converting string to single-item list.")
        df["contexts"] = df["contexts"].apply(lambda x: [x] if isinstance(x, str) else x)

    # convert list of context strings to appropriately formatted string to be inserted in LLM prompt
    df["contexts"] = df["contexts"].apply(lambda x: format_context_list(x))
    # print(df.head())

    results_df = pd.DataFrame(columns=results_columns)
    for index, row in enumerate(df.itertuples()):
        print(f"------------{index}----------------")
        start = time.perf_counter()
        results_df = pd.concat([results_df, evaluate_row(row)], ignore_index=True)
        elapsed_seconds = time.perf_counter() - start
        elapsed_time = str(timedelta(seconds=int(elapsed_seconds)))
        print(f"elapsed_time = {elapsed_time}")
        print("------------------------------")

    # SAVE AND EXPORT THE RESULTS
    print("\n--- Summary Performance Scores ---")
    # Select columns and aggregate
    stats = results_df[[
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
        "answer_correctness"]
    ].agg(['mean', 'min', 'max', 'std'])
    print(stats)

    results_df.to_json(os.getenv("OUTPUT_FILE"), orient="records", lines=True)
    print(f"\nRow-by-row metrics successfully saved to: {os.getenv("OUTPUT_FILE")}")


if __name__ == "__main__":
    load_dotenv(dotenv_path='/Users/msb/Code/rag-bench/.env')
    main()
