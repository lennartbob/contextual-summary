from __future__ import annotations

import os

from attrs import define
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

load_dotenv()

@define
class QdrantConfig:
  """Configuration class for Qdrant connection."""

  host: str = os.getenv("QDRANT_HOSTNAME", "127.0.0.1")
  port: int = int(os.getenv("QDRANT_PORT", "6333"))  # Default HTTP port
  grpc_port: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))  # Default gRPC port
  prefer_grpc: bool = os.getenv("QDRANT_PREFER_GRPC", "False").lower() == "true"
  api_key: str | None = os.getenv("QDRANT_API_KEY")  # Optional API key
  # Add other QdrantClient options if needed (e.g., https, prefix)
  # url: str | None = os.getenv("QDRANT_URL") # Alternatively use full URL

  def get_client_args(self) -> dict:
    """Returns arguments suitable for QdrantClient constructor."""
    args = {
      "host": self.host,
      "port": self.port,
      "grpc_port": self.grpc_port,
      "prefer_grpc": self.prefer_grpc,
      "api_key": self.api_key,
      # Add timeout, etc. if needed
    }
    # If a full URL is provided, it might override host/port
    # url = os.getenv("QDRANT_URL")
    # if url:
    #   args = {"url": url, "api_key": self.api_key} # Adjust based on QdrantClient behavior
    return args

def get_qdrant_client(config: QdrantConfig = QdrantConfig()) -> QdrantClient:
  """Initializes and returns a QdrantClient instance."""
  try:
    client_args = config.get_client_args()
    client = QdrantClient(**client_args)
    # Optionally ping the server to ensure connection (raises exception on failure)
    # client.openapi_client.health_api.healthz() # Simple health check might vary
    # Or try a lightweight operation like listing collections
    client.get_collections()
    print(
      f"Successfully connected to Qdrant at host='{config.host}', "
      f"port={config.port}, grpc_port={config.grpc_port}, prefer_grpc={config.prefer_grpc}"
    )
    return client
  except UnexpectedResponse as ue:
    error_message = f"Failed to connect to Qdrant (API Error): {ue}"
    print(error_message)
    raise ConnectionError(error_message) from ue
  except Exception as e:
    error_message = f"Failed to connect to Qdrant (General Error): {e}"
    print(error_message)
    raise ConnectionError(
      error_message
    ) from e  # Use ConnectionError or a custom exception
