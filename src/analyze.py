import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv


def read_data() -> pd.DataFrame:
    # Load the generated evaluation data
    try:
        return pd.read_json(os.getenv("OUTPUT_FILE"), lines=True)
    except FileNotFoundError:
        print(f"Error: Could not find '{os.getenv("OUTPUT_FILE")}'. Please run your RAG eval script first.")
        sys.exit()


def plot_results(df: pd.DataFrame) -> None:

    # List of standard RAG metrics we calculated (equivalent to RAGAS)
    target_metrics = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_recall",
        "context_precision"
    ]

    # Ensure only existing metrics from the CSV are used and drop any NaN values for plotting
    available_metrics = [m for m in target_metrics if m in df.columns]
    df_clean = df[available_metrics].dropna()

    if df_clean.empty:
        print("Error: No evaluation metric data found or all rows contain NaN values.")
        sys.exit()

    # Calculate global mean scores
    mean_scores = df_clean.mean()

    # Setup Plot Canvas (Two side-by-side subplots)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # noqa: RUF059
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: Overall Average Performance Bar Chart ---
    # Map metric names to clean display labels
    clean_labels = [m.replace('_', ' ').title() for m in available_metrics]

    sns.barplot(
        x=mean_scores.values,
        y=clean_labels,
        ax=axes[0],
        palette="viridis",
        hue=clean_labels,
        legend=False
    )
    axes[0].set_title("Overall Average Performance (Higher is Better)", fontsize=14, pad=15)
    axes[0].set_xlabel("Score (0.0 to 1.0)", fontsize=12)
    axes[0].set_xlim(0, 1.0)

    # Add exact score labels to the ends of the bars
    for i, v in enumerate(mean_scores.values):
        axes[0].text(v + 0.02, i, f"{v:.2f}", va='center', fontweight='bold', fontsize=11)


    # --- PLOT 2: Metric Distribution Analysis (Boxplot) ---
    # Reshape the dataframe for seaborn distribution plotting
    df_melted = df_clean.melt(var_name="Metric", value_name="Score")
    df_melted["Metric"] = df_melted["Metric"].str.replace('_', ' ').str.title()

    sns.boxplot(
        x="Score",
        y="Metric",
        data=df_melted,
        ax=axes[1],
        palette="plasma",
        hue="Metric",
        legend=False
    )
    axes[1].set_title("Score Distribution & Pipeline Consistency", fontsize=14, pad=15)
    axes[1].set_xlabel("Score Spread", fontsize=12)
    axes[1].set_xlim(0, 1.0)


    # Clean up formatting and display
    plt.tight_layout()

    # Save the dashboard chart locally as an image file
    output_image = "rag_metrics_dashboard.png"
    plt.savefig(output_image, dpi=300)
    print(f"Success! Dashboard visualization saved as: {output_image}")

    # Show the plot window
    plt.show()


def analyze_defects(df: pd.DataFrame):

    # Set your failure thresholds
    # Standard industry baseline is usually 0.70 to 0.80. Anything under 0.40 is a major system failure.
    FAITHFULNESS_THRESHOLD = 0.30
    CONTEXT_RECALL_THRESHOLD = 0.30

    # Filter for Hallucinations (Low Faithfulness)
    # The LLM hallucinated facts completely missing from your retrieved chunks
    hallucination_failures = df[df["faithfulness"] < FAITHFULNESS_THRESHOLD]

    # Filter for Retrieval Misses (Low Context Recall)
    # Your retriever failed to fetch the factual answers present in the ground truth
    retrieval_failures = df[df["context_recall"] < CONTEXT_RECALL_THRESHOLD]

    # EXPORT AND EXAMINE FAILURES
    print("--- Pipeline Failure Analysis ---")
    print(f"Total Rows Evaluated: {len(df)}")
    print(f"Severe Hallucination Rows (< {FAITHFULNESS_THRESHOLD}): {len(hallucination_failures)}")
    print(f"Severe Retrieval Miss Rows (< {CONTEXT_RECALL_THRESHOLD}): {len(retrieval_failures)}")

    # Export the raw isolated rows to separate files for targeted re-testing
    if not hallucination_failures.empty:
        hallucination_failures.to_json(os.getenv("HALLUCINATIONS_OUTPUT_FILE"), orient="records", lines=True)
        print(f"\nHallucination rows exported to: {os.getenv("HALLUCINATIONS_OUTPUT_FILE")}")

        # Print the top 3 worst offenders cleanly to the console for quick review
        print("\n>>> SAMPLE OFFENDER (Low Faithfulness):")
        worst_row = hallucination_failures.sort_values(by="faithfulness").iloc[0]
        print(f"Prompt / Query: {worst_row['question']}")
        print(f"Generated LLM Response: {worst_row['answer']}")
        print(f"Expected Ground Truth:  {worst_row['ground_truth']}")
        print(f"Faithfulness Score:     {worst_row['faithfulness']}")

    if not retrieval_failures.empty:
        retrieval_failures.to_json(os.getenv("RETRIEVAL_MISSES_OUTPUT_FILE"), orient="records", lines=True)
        print(f"Retrieval failure rows exported to: {os.getenv("RETRIEVAL_MISSES_OUTPUT_FILE")}")

def main():
    df = read_data()
    plot_results(df)
    analyze_defects(df)


if __name__ == "__main__":
    load_dotenv(dotenv_path='/Users/msb/Code/rag-bench/.env')
    main()
