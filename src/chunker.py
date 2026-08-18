import os
import re
import time
import uuid
from datetime import timedelta

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from utils.clients import create_embeddings, create_s3_client
from utils.helper import list_files_in_s3_folder
from utils.qdrt import create_qdrt_client

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=51,
)

# Specify which headers to split on and how to label them in metadata
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("####", "Header 3"),
]

def strip_references(markdown_text: str) -> str:
    # Looks for variations of "References", "Bibliography", or "Works Cited"
    # as a markdown header near the end of the document
    pattern = re.compile(r'^(#+\s+(References|Bibliography|Works\s+Cited))\b', re.IGNORECASE | re.MULTILINE)
    match = pattern.search(markdown_text)

    if match:
        # Keep everything before the References header
        return markdown_text[:match.start()]
    return markdown_text


def is_useless_math_scrap(text: str, min_english_words: int = 10) -> bool:
    # Strip out standard block and inline LaTeX symbols ($...$ or $$...$$)
    # This also targets common markdown/latex formatting like \alpha, \sum, \infty
    clean_text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.DOTALL)
    clean_text = re.sub(r'\$.*?\$', ' ', clean_text)
    clean_text = re.sub(r'\\[a-zA-Z]+', ' ', clean_text)

    # Extract only valid, standalone English words
    # (Matches alphabetical strings longer than 1 character)
    english_words = re.findall(r'\b[a-zA-Z]{2,}\b', clean_text)

    # 3. If the total count of valid words is lower than your threshold, discard it
    return len(english_words) < min_english_words


def filter_chunk(chunk_text: str, min_length: int = 60) -> bool:

    # check if useless math chunk (dense math strings but fewer than 10 actual English words)
    if is_useless_math_scrap(chunk_text):
        return True
    # Drop tiny noise snippets (page number remnants)
    if len(chunk_text) < min_length:
        return True
    # Drop isolated image reference strings
    return bool(chunk_text.startswith("![") and chunk_text.endswith(")"))


def create_document_chunks(markdown_text: str, key: str):
    """Chunk documents into smaller segments suitable for embedding and RAG.
       Associates raw markdown tables directly with their neighboring text chunk
       metadata instead of utilizing LLM-generated summaries."""

    # First strip references
    markdown_text = strip_references(markdown_text)
    print(f"references stripped from markdown for key={key}")

    # Extract markdown tables cleanly via Regex before splitting the text
    table_regex = r"(?:^|\n)(\|[^\n]+\|\n\|[ \t]*:?---:?[ \t]*\|[^\n]*\n(?:\|[^\n]+\|\n*)+)"

    # Use an indexed tracking replacement strategy to know exactly where each table sits
    tables = []
    def table_replacer(match):
        table_content = match.group(1).strip()
        tables.append(table_content)
        # Store a sequential tracking token to identify the table placement during chunking
        return f"\n\n[TABLE_REMOVED_{len(tables) - 1}]\n\n"

    text_only_markdown = re.sub(table_regex, table_replacer, markdown_text)

    # Stage 1: Split by structure (extracts context into metadata)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keep the header text inside the chunk content
    )
    md_header_splits = markdown_splitter.split_text(text_only_markdown)

    # Stage 2: Split by length (forces strict sizes within each section)
    # Apply character splitting onto our structural chunks
    doc_chunks = text_splitter.split_documents(md_header_splits)

    print(f"markdown document with key={key} split into {len(doc_chunks)} raw structural segments...")
    documents_to_index = []

    # Map the unique table tokens back to their respective chunk containers
    for doc_chunk in doc_chunks:
        chunk_content = doc_chunk.page_content

        # Filter unnecessary chunks based on user-defined constraints
        if filter_chunk(chunk_content):
            continue

        # Find all structural table keys enclosed within this chunk boundary
        placeholder_regex = r"\[TABLE_REMOVED_(\d+)\]"
        found_table_ids = re.findall(placeholder_regex, chunk_content)

        associated_tables = []
        for id_str in found_table_ids:
            table_idx = int(id_str)
            if table_idx < len(tables):
                associated_tables.append(tables[table_idx])

        # Clean up text content by removing tracking placeholders so they don't corrupt token embedding space
        clean_content = re.sub(placeholder_regex, "", chunk_content).strip()

        # Skip completely empty sections that carry no content or table context
        if not clean_content and not associated_tables:
            continue

        # Build dictionary out of existing metadata (preserving markdown structure properties)
        metadata = dict(doc_chunk.metadata)
        metadata["type"] = "text"

        if associated_tables:
            # Store the raw markdown table data inside an array layout
            metadata["raw_table_content"] = associated_tables
            metadata["has_tables"] = True

        # Re-wrap directly back into standard LangChain Document classes
        documents_to_index.append(Document(page_content=clean_content, metadata=metadata))

    print(f"Text chunks for key={key} compiled seamlessly with structural table metadata...")

    # Establish globally standardized index references conforming to Qdrant validation requirements
    for index, doc in enumerate(documents_to_index):
        # Generate the standard readable chunk string format
        logical_id = f"{key}__chunk__{index+1}__{len(documents_to_index)}"

        # Store the readable string path safely inside metadata so you don't lose audit visibility
        doc.metadata["logical_chunk_id"] = logical_id

        # Convert the string deterministically into a valid UUID string using UUID v5
        # This keeps the IDs consistent across runs while satisfying Qdrant schemas
        doc.id = str(uuid.uuid5(uuid.NAMESPACE_DNS, logical_id))

    return documents_to_index



def main():
    """Main function to read documents, chunk them, embed, and store in Qdrant."""
    s3_client = create_s3_client()
    embeddings = create_embeddings()
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    qdrt_client = create_qdrt_client()
    collections = qdrt_client.get_collections()
    print(f"Qdrant connected. Existing collections: {collections}")

    # This automatically handles the local BM25/sparse pipeline under the hood via fastembed
    vector_store = QdrantVectorStore(
        client=qdrt_client,
        collection_name=os.getenv('QDRANT_COLLECTION'),
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    file_keys = list_files_in_s3_folder(os.getenv('S3_BUCKET'), f"{os.getenv('S3_PATH_MARKDOWN_PREFIX')}/", s3_client)
    print(f"\n📥 Found {len(file_keys)} PDF files in {os.getenv('S3_PATH_MARKDOWN_PREFIX')}/")

    if not file_keys:
        print(f"No files found in {os.getenv('S3_BUCKET')}/{os.getenv('S3_PATH_MARKDOWN_PREFIX')}")
        return

    print(f"Found {len(file_keys)} markdown files. Start Chunking...")

    for index, key in enumerate(file_keys):
        start = time.perf_counter()
        # Retreive document from RustFS
        response = s3_client.get_object(Bucket=os.getenv('S3_BUCKET'), Key=key)
        # Read the file content and decode it from bytes to string
        markdown_content = response['Body'].read().decode('utf-8')
        print(f"\nmarkdown document for key={key} retrieved...")
        # chunk document (with Langchain recursive splitter)
        documents = create_document_chunks(markdown_content, key)
        print(f"document chunks created for markdown doc with key={key}...")
        vector_store.add_documents(documents)

        text_chunks = [document.page_content for document in documents]
        print(f"markdown document {key} stored in Qdrant with {len(documents)} chunks")
        print(f"chunk_max={len(max(text_chunks, key=len))}, chunk_min={len(min(text_chunks, key=len))}")
        elapsed_seconds = time.perf_counter() - start
        elapsed_time = str(timedelta(seconds=int(elapsed_seconds)))
        print(f"Iteration {index+1}: elapsed_time = {elapsed_time}")


if __name__ == "__main__":
    load_dotenv(dotenv_path='/Users/msb/Code/rag-bench/.env')
    main()
