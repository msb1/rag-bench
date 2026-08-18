import json
import os

import requests

"""
Your Final, Embedding-Free Metric Architecture:

    To summarize how your custom framework handles the math without vectors:
        (1) Faithfulness: Pure text extraction and logic validation.
        (2) Answer Relevancy: Pure text intent classification.
        (3) Context Recall: Pure text truth-matching.
        (4) Context Precision: Pure text noise-filtering.
        (5) Answer Correctness: Structured factual overlap via JSON / text grading.
"""

def format_context_list(contexts: list) -> str:
    """Combines a list of context strings into a structured, numbered block."""
    formatted_chunks = []

    for idx, chunk in enumerate(contexts, start=1):
        # Clean up any accidental whitespace or trailing newlines in the raw data
        clean_chunk = chunk.strip()

        # Structure each chunk with clear semantic boundaries
        formatted_chunks.append(f"[Context Chunk {idx}]\n{clean_chunk}")

    # Join everything together using a clear visual break
    return "\n\n".join(formatted_chunks)


def ask_local_llm(prompt: str, temperature: float = 0.0) -> str:
    """Sends a prompt directly to LM Studio and returns the clean text output."""
    payload = {
        "model": os.getenv('EVAL_MODEL'),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lm-studio",
    }

    try:
        response = requests.post(
            f"{os.getenv('OPENAI_LOCAL_ENDPOINT')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        print(f"API Error: {e}")
        return ""


def evaluate_faithfulness(context: str, answer: str) -> float:
    """
    Measures if the generated answer sticks strictly to the retrieved context.
    Returns a score between 0.0 (total hallucination) and 1.0 (fully faithful).
    """
    # Step A: Extract individual facts/claims from the answer
    extraction_prompt = f"""
    Analyze the following Answer and break it down into a bulleted list of independent, single factual claims.
    Do not add or infer anything outside the text.

    Answer: {answer}

    List of claims:
    """
    claims_text = ask_local_llm(extraction_prompt)
    claims = [c.strip("- ").strip() for c in claims_text.split("\n") if c.strip()]

    if not claims:
        return 0.0

    # Step B: Verify each claim against the context
    verified_count = 0
    for claim in claims:
        verification_prompt = f"""
        Determine if the following Claim can be logically inferred using ONLY the provided Context.
        Respond with exactly one word: 'YES' if it is supported, or 'NO' if it is not supported or not mentioned.

        Context: {context}
        Claim: {claim}

        Verdict (YES/NO):
        """
        verdict = ask_local_llm(verification_prompt).upper()
        if "YES" in verdict:
            verified_count += 1

    # Step C: Calculate the final Ragas-style ratio
    return verified_count / len(claims)


def evaluate_relevancy(question: str, answer: str) -> float:
    """
    Measures if the answer directly addresses the user's question.
    Returns a score between 0.0 (irrelevant) and 1.0 (highly relevant).
    """
    # Step A: Have the LLM reverse-engineer the question based on the answer
    reverse_prompt = f"""
    Generate the specific, concise question that the following answer is trying to resolve.
    Only return the question text, nothing else.

    Answer: {answer}
    Generated Question:
    """
    generated_question = ask_local_llm(reverse_prompt)

    # Step B: Compare the real question with the generated question
    # Note: For pure Ragas math, you would generate text embeddings for both strings
    # and calculate Cosine Similarity. If you don't want to use embeddings,
    # you can fallback to a direct semantic classification:

    scoring_prompt = f"""
    Compare the following two questions. Do they have the exact same semantic intent?
    Rate their similarity on a scale from 0.0 (completely different intent) to 1.0 (identical intent).
    Respond with ONLY a numeric float value.

    Original Question: {question}
    Generated Question: {generated_question}

    Similarity Score:
    """
    try:
        score_text = ask_local_llm(scoring_prompt)
        return float(score_text)
    except ValueError:
        return 0.0


def evaluate_answer_correctness(answer: str, ground_truth: str) -> float:
    """
    Measures the factual correctness of the generated answer compared to the ground truth.
    Returns a score between 0.0 (completely wrong) and 1.0 (perfectly correct match).
    """
    classification_prompt = f"""
    Compare the Generated Answer against the Ground Truth answer. Break down the comparison into three clean categories:
    1. TP (True Positives): Facts present in BOTH the Generated Answer and Ground Truth.
    2. FP (False Positives): Facts present in the Generated Answer but contradict or are NOT in the Ground Truth.
    3. FN (False Negatives): Facts present in the Ground Truth but completely missed by the Generated Answer.

    Provide the output strictly in this JSON format:
    {{
        "TP": ["fact 1", "fact 2"],
        "FP": ["fact 3"],
        "FN": ["fact 4"]
    }}

    Generated Answer: {answer}
    Ground Truth: {ground_truth}

    JSON Output:
    """
    raw_json = ask_local_llm(classification_prompt)

    try:
        # Clean potential markdown wrappers if the local model adds them
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0]
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0]

        data = json.loads(raw_json.strip())
        tp = len(data.get("TP", []))
        fp = len(data.get("FP", []))
        fn = len(data.get("FN", []))

        if (tp + fp + fn) == 0:
            return 0.0

        # Standard F1-Score calculation matching Ragas math architecture
        f1_correctness = tp / (tp + 0.5 * (fp + fn))
        return f1_correctness

    except Exception:  # noqa: BLE001
        # Fallback if local model breaks JSON formatting rules: semantic intent evaluation
        fallback_prompt = f"""
        Rate the factual correctness of the Generated Answer compared to the Ground Truth on a scale from 0.0 to 1.0.
        0.0 means completely wrong or factually conflicting. 1.0 means completely accurate.
        Respond with ONLY the float number.

        Ground Truth: {ground_truth}
        Generated Answer: {answer}

        Score:
        """
        try:
            return float(ask_local_llm(fallback_prompt))
        except ValueError:
            return 0.0


def evaluate_context_recall(ground_truth: str, context: str) -> float:
    """
    Measures if the retrieved context contains all facts from the ground truth.
    Returns a score between 0.0 (missed everything) and 1.0 (found all facts).
    """
    # Step A: Extract core factual statements from the Ground Truth
    extraction_prompt = f"""
    Break down the following Ground Truth answer into a bulleted list of independent, single factual statements.
    Only output the bulleted list. Do not write an introduction or conclusion.

    Ground Truth: {ground_truth}

    List of facts:
    """
    facts_text = ask_local_llm(extraction_prompt)
    gt_facts = [f.strip("- ").strip() for f in facts_text.split("\n") if f.strip()]

    if not gt_facts:
        return 0.0

    # Step B: Check if each fact exists within the retrieved Context
    found_count = 0
    for fact in gt_facts:
        verification_prompt = f"""
        Can the following Fact be directly found or clearly deduced using ONLY the provided Context?
        Respond with exactly one word: 'YES' if it is present, or 'NO' if it is missing.

        Context: {context}
        Fact: {fact}

        Verdict (YES/NO):
        """
        verdict = ask_local_llm(verification_prompt).upper()
        if "YES" in verdict:
            found_count += 1

    # Ratio of ground truth facts successfully recalled
    return found_count / len(gt_facts)


def evaluate_context_precision(question: str, ground_truth: str, context: str) -> float:
    """
    Measures how relevant the retrieved context chunks are to answering the question.
    Filters out noise. Returns a score between 0.0 (all noise) and 1.0 (purely relevant).
    """
    # Step A: Break the retrieved context down into individual sentences
    # For advanced setups, pass a list of retrieved chunks directly instead of splitting strings
    context_sentences = [s.strip() for s in context.replace("?", ".").split(".") if s.strip()]

    if not context_sentences:
        return 0.0

    relevant_sentences_count = 0

    # Step B: Determine how many sentences in the context are actually useful
    for sentence in context_sentences:
        precision_prompt = f"""
        Task: Analyze if the given sentence from the context is highly relevant and useful
        for answering the user's question, given the known correct ground truth.

        Question: {question}
        Ground Truth Answer: {ground_truth}
        Sentence to analyze: {sentence}

        Respond with exactly one word: 'YES' if it is relevant/useful, or 'NO' if it is irrelevant noise.

        Verdict (YES/NO):
        """
        verdict = ask_local_llm(precision_prompt).upper()
        if "YES" in verdict:
            relevant_sentences_count += 1

    # Ratio of useful information to total text retrieved
    return relevant_sentences_count / len(context_sentences)
