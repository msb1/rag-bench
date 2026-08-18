from qdrant_client import QdrantClient

"""Qdrant client factory for OpenRAG."""


def create_qdrt_client():
    """Create a QDRTClient instance pointing to your Qdrant database.

    Args:
        url: The URL of your Qdrant server (default: http://192.168.1.50:6333)

    Returns:
        A QDRTClient instance connected to the vector database.
    """

    # Default endpoint — configure for your local setup
    url = 'http://192.168.1.50:6333'  # Your Qdrant endpoint

    return QdrantClient(url=url)


def test_qdrt_connection():
    """Test if the Qdrant database is reachable and respond to queries."""
    try:
        client = create_qdrt_client()
        collections = client.get_collections()
        return True, f"Connected. Collections: {collections}"
    except ImportError as e:
        return False, "qdrant-client not installed: pip install qdrant-client"
    except Exception as e:
        return False, str(e)
