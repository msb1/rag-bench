import json
import os
import re
import time
from collections.abc import Sequence
from datetime import timedelta

import requests
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever  # noqa: F401
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from utils.clients import create_embeddings, create_openai_client
from utils.qdrt import create_qdrt_client

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, precise AI assistant. Answer user queries using ONLY the facts explicitly provided in the 'Retrieved Context' section.\n"  # noqa: ISC004
            "Do not include source references like (Doc x). Only provide a strictly factual, text answer from the Retrieved Context.\n\n"
            "Rules:\n"
            "1. Grounding: Do not assume or extrapolate. If the answer cannot be found entirely in the context, reply: 'I cannot find the answer in the provided documents.'\n"
            "2. Safeguard: Never use your training knowledge to supplement Retrieved Context"
            "3. Tone: Professional, concise, and direct."
        ),
        (
            "human",
            "[Retrieved Context]\n"  # noqa: ISC004
            "{context}\n\n"
            "[User Query]\n"
            "{user_query}"
        ),
    ]
)


def append_to_jsonl(question: str, answer: str, retrieved_chunks: list[str], ground_truth: str, filepath: str) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        contexts = [f"Chunk {index+1}: {chunk}" for index, chunk in enumerate(retrieved_chunks)]
        data = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def generate_queries_chain(llm, n=3):
    """
    Creates a chain that takes a single user prompt and generates n alternative variations.
    """
    query_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert AI assistant. Your task is to generate {n} alternative variations of the user's query to look up documents in a vector database from multiple perspectives. Provide these variations separated by newlines. Do not add numbering or introductory text."),
        ("human", "Generate variations for: {question}")
    ])

    # Chain components: Prompt -> LLM -> Parse output -> Split by newline into a list of strings
    query_chain = (
        query_generation_prompt
        | llm
        | StrOutputParser()
        | (lambda text: [q.strip() for q in text.strip().split("\n") if q.strip()])
    )
    return query_chain


class RAGState:
    def __init__(self):
        self.final_contexts = []

    def capture_contexts(self, docs):
        # Captures the documents AFTER they pass through the reranker
        self.final_contexts = [doc.page_content for doc in docs]
        return docs  # Pass the docs downstream unchanged


class LocalFastAPIReranker(BaseDocumentCompressor):
    """Stateless client mapping queries to your local FastAPI reranker port."""

    model_name: str = "jina-reranker-v3.5"
    top_n: int = 5
    # Target your newly created FastAPI server route
    endpoint_url: str = "http://192.168.1.50:8000/v1/rerank"

    def compress_documents(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: CallbackManagerForRetrieverRun | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        try:
            # Post straight to a native rerank structure
            response = requests.post(
                self.endpoint_url,
                headers={"Content-Type": "application/json"},
                json={
                    "query": query,
                    "documents": [doc.page_content for doc in documents],
                    "top_n": self.top_n
                },
                timeout=15.0 # True encoder models respond in milliseconds!
            )

            # print(f"reranker response: {response.status_code}")
            response.raise_for_status()
            result_json = response.json()
            # print(result_json)

            ranked_docs = []
            # Map clean index numbers cleanly back to LangChain instances
            for item in result_json.get("results", []):
                idx = item["index"]
                if idx < len(documents):
                    doc = documents[idx]
                    new_doc = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "rerank_score": item.get("relevance_score", 0.0)}
                    )
                    ranked_docs.append(new_doc)

            return list(ranked_docs)[:self.top_n]

        except Exception as e:  # noqa: BLE001
            print(f"FastAPI Reranking backend failed: {e}. Falling back to default order.")
            return list(documents)[:self.top_n]

def clean_and_deduplicate_table(raw_table_input: dict[int, str]) -> str:
    if not raw_table_input:
        return ""

    combined_table = ""
    if isinstance(raw_table_input, dict):
        sorted_keys = sorted(raw_table_input.keys(), key=lambda x: int(x))
        raw_strings = [str(raw_table_input[k]) for k in sorted_keys]
        combined_table = "\n".join(raw_strings)
    elif isinstance(raw_table_input, list):
        combined_table = "\n".join(raw_table_input)
    elif isinstance(raw_table_input, str):
        combined_table = f"\n{raw_table_input}"
    else:
        return ""

    if not combined_table:
        return ""

    # Basic cleanup: standardize cell linebreaks and clean HTML entities
    cleaned = re.sub(re.compile(r"<br\s*/?>\s*<br\s*/?>", re.IGNORECASE), " | ", combined_table)
    cleaned = re.sub(re.compile(r"<br\s*/?>", re.IGNORECASE), " ", cleaned)
    cleaned = re.sub(r'\|\s*\|', '|', cleaned)
    cleaned = cleaned.replace("&amp;lt;", "<").replace("&lt;", "<").replace("&gt;", ">")

    # Split raw payload into individual lines for structural scanning
    raw_lines = [line.strip() for line in cleaned.split('\n') if line.strip()]

    final_lines: list[str] = []
    seen_headers: set[str] = set()
    is_table_started = False

    for line in raw_lines:
        # Strip explicit continuation banner row artifacts dynamically
        if "continued from previous page" in line.lower():
            continue

        # Standardize line layout to easily identify duplicate text signatures
        normalized_line = line.replace(" ", "").lower()

        # Handle separator rows (e.g., |---|---|). Allow only the first one found.
        if is_table_started and re.match(r'^\|[-|\s]+]$', normalized_line):
            continue

        # Detect standard data headers or bold keys
        if "|" in line:
            # If we haven't locked down a primary table header yet, make this line the master header
            if not is_table_started:
                final_lines.append(line)
                seen_headers.add(normalized_line)
                is_table_started = True
                continue

            # If table is active and we see a line identical to a known header structure, skip it
            if normalized_line in seen_headers:
                continue

        final_lines.append(line)

    return "\n".join(final_lines)


# ========================================================
# 2. SEAMLESS RAG FUSION INTEGRATION
# ========================================================
def build_rag_fusion_pipeline(state: RAGState):
    """Assembles the complete RAG Fusion pipeline with proper RunnableLambdas."""
    fusion_llm = create_openai_client(os.getenv('OPENAI_REMOTE_ENDPOINT'), os.getenv('CODING_MODEL'), 0.0)
    rag_llm = create_openai_client(os.getenv('OPENAI_LOCAL_ENDPOINT'), os.getenv('RAG_MODEL'), 0.7)

    # Base Qdrant configuration
    embeddings = create_embeddings()
    sparse_embeddings = FastEmbedSparse(
        model_name="Qdrant/bm25",
        extra_options={"local_files_only": True} # <--- Forces fastembed to skip the remote check
    )
    qdrt_client = create_qdrt_client()

    vector_store = QdrantVectorStore(
        client=qdrt_client,
        collection_name=os.getenv('QDRANT_COLLECTION'),
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    # Retrieve a wide pool for high recall
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 12})

    # Initialize the fast-api compressor
    reranker: LocalFastAPIReranker = LocalFastAPIReranker(top_n=5)
    queries_generator = generate_queries_chain(fusion_llm)

    def execute_fusion_retrieval(input_data: dict) -> list[Document]:
        original_query = input_data["user_query"]
        query_variations = queries_generator.invoke({"question": original_query})

        all_retrieved_docs = []
        for query in query_variations:
            all_retrieved_docs.extend(base_retriever.invoke(query))

        # Unified Deduplication
        seen_contents: set[str] = set()
        unique_documents: list[str] = []
        for doc in all_retrieved_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_documents.append(doc)

        # Fire a single API call to your FastAPI server containing everything
        ranked_docs = reranker.compress_documents(
            query=original_query,
            documents=unique_documents
        )
        for doc in ranked_docs:
            raw_table_data: dict[int, str] = doc.metadata.get("raw_table_content")
            if raw_table_data:
                cleaned_table = clean_and_deduplicate_table(raw_table_data)
                doc.page_content = (
                    f"{doc.page_content}\n\n"
                    f"### Associated Table Data\n"
                    f"{cleaned_table}"
                )

        return list(ranked_docs)

    def format_docs(docs: list[Document]) -> str:
        # Format documents safely into context window exactly as before
        context_string = "\n".join(
            f"---\nSource ID: [Doc {i+1}]\nContent: {doc.page_content}\n---"
            for i, doc in enumerate(docs)
        )
        return context_string

    # Convert standard Python helpers to LCEL components
    runnable_fusion = RunnableLambda(execute_fusion_retrieval)
    runnable_format = RunnableLambda(format_docs)

    # Build the final chain
    rag_fusion_chain = (
        {"user_query": RunnablePassthrough()}

        | RunnablePassthrough.assign(context=runnable_fusion | state.capture_contexts | runnable_format)
        | rag_prompt
        | rag_llm
        | StrOutputParser()
    )

    return rag_fusion_chain


def main():
    """Call RAG pipeline and generate answers"""
    # find last record processed
    last_processed_index = -1
    if os.path.exists(os.getenv("PROGRESS_FILE")) and os.path.getsize(os.getenv("PROGRESS_FILE")) > 0:
        with open(os.getenv("PROGRESS_FILE"), "r") as f:
            last_processed_index = json.load(f).get("last")
    print(f"Last processed OpenRAG index = {last_processed_index}")

    # Read queries and answers json files
    with open(os.getenv("QUERIES_FILE"), 'r', encoding='utf-8') as file:
        query_dict = json.load(file)

    with open(os.getenv("ANSWERS_FILE"), 'r', encoding='utf-8') as file:
        ground_truth_dict = json.load(file)

    print("Initializing complete RAG Fusion Pipeline...")
    state = RAGState()
    pipeline = build_rag_fusion_pipeline(state)
    print("\nExecuting pipeline (Generating queries, executing parallel searches, applying RRF, and generating final answer)...")

    for index, (key, value) in enumerate(query_dict.items()):
        if index < last_processed_index + 1:
            continue
        query = value['query']
        ground_truth = ground_truth_dict[key]

        start = time.perf_counter()
        print(f"\nOpenRAG Query {index+1} -- key={key}: {query}")
        print(f"Ground truth answer: {ground_truth}")

        generated_answer = pipeline.with_config(verbose=True).invoke(query)

        append_to_jsonl(query, generated_answer, state.final_contexts, ground_truth, os.getenv("DATASET_FILE"))

        print("--- Final Generated Answer ---")
        print(generated_answer)
        elapsed_seconds = time.perf_counter() - start
        elapsed_time = str(timedelta(seconds=int(elapsed_seconds)))
        print("------------------------------")
        print(f"elapsed_time = {elapsed_time}")

        # update progress file (in case of interruptions)
        with open(os.getenv("PROGRESS_FILE"), "w") as f:
            json.dump({"last": index}, f)


if __name__ == "__main__":
    load_dotenv(dotenv_path='/Users/msb/Code/rag-bench/.env')
    main()
