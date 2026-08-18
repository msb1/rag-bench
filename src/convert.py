import os

import pymupdf
import pymupdf4llm
from dotenv import load_dotenv

from utils.clients import create_s3_client
from utils.helper import list_files_in_s3_folder


def pdf_to_markdown(bucket: str, key: str, s3_client):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    pdf_bytes = response["Body"].read()

    print("Converting PDF to Markdown format...")
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        print(f"{key} has Total Pages: {doc.page_count}")
        markdown_text = pymupdf4llm.to_markdown(doc)

    markdown_key = key.replace("pdf/", "markdown/").replace(".pdf", ".md")
    print(f"markdown_key={markdown_key}\n")
    s3_client.put_object(Bucket=bucket, Key=markdown_key, Body=markdown_text, ContentType="text/markdown; charset=utf-8")


def main():

    s3_client = create_s3_client()

    file_keys = list_files_in_s3_folder(os.getenv('S3_BUCKET'), "pdf/", s3_client)
    print(f"\n📥 Found {len(file_keys)} PDF files in {os.getenv('S3_PATH_PDF_PREFIX')}/")

    for key in file_keys:
        pdf_to_markdown(os.getenv('S3_BUCKET'), key, s3_client)

    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    load_dotenv(dotenv_path='/Users/msb/Code/rag-bench/.env')
    print("🚀 Starting PDF to Markdown conversion via pymupdf4llm...")
    print(f"📦 Source: {os.getenv('S3_BUCKET')}/{os.getenv('S3_PATH_PDF_PREFIX')}")
    print(f"📤 Destination: {os.getenv('S3_BUCKET')}/{os.getenv('S3_PATH_MARKDOWN_PREFIX')}")
    print(f"📋 Metadata: {os.getenv('S3_BUCKET')}\n")
    main()
