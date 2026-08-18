
import os

import boto3
from botocore.config import Config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def create_s3_client():
    # Configure client with endpoint for S3-compatible storage
    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"}
    )
    return boto3.client(
        's3',
        endpoint_url=os.getenv('S3_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
        config=config
    )


def create_openai_client(base_url: str, model: str, temperature: float):
    """Initialize client pointing to your Qwen 3.5 provider."""
    return ChatOpenAI(
        model=model,
        base_url=base_url,  # pointing to your LM Studio instance
        api_key="lm-studio",
        temperature=temperature,
    )


# Define a patch class for local OpenAI-compatible APIs like LM Studio
class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # Forces LangChain to call the standard client directly without pre-tokenizing
        response = self.client.create(
            input=texts,
            model=self.model,
            **self.model_kwargs
        )
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> list[float]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.embed_documents([text])[0]


def create_embeddings():
    """Create embeddings using the local Qwen embedding model."""
    embeddings = CustomOpenAIEmbeddings(
        # The endpoint provided by LM Studio (include /v1)
        base_url="http://192.168.1.50:1234/v1",
        model=os.getenv('EMBEDDING_MODEL'),
        api_key="lm-studio"
    )
    return embeddings
