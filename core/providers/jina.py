from __future__ import annotations

import os
from typing import Any, Optional

import requests
from attr import define
from dotenv import load_dotenv
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

load_dotenv()


@define
class RerankerResult:
  """Represents a reranked item with its index, relevance score, and associated text.

  Attributes:
      index (int): The position of the item in the original list.
      relevance_score (float): The relevance score assigned by the reranker.
      text (str): The content of the item.
  """

  index: int
  relevance_score: float
  text: str


def rerank_with_jina(
    query: str, text_list: list[str], top_n: int, model: str = "jina-reranker-v2-base-multilingual"
) -> list[RerankerResult]:
  """
  Reranks a list of text documents based on their relevance to the query using Jina's API.

  Args:
      query (str): The query string for which documents are being reranked.
      text_list (list[str]): The list of documents (texts) to be reranked.
      top_n (int): The number of top relevant documents to return.
      model (str): The name of the Jina reranker model to use.
                   Defaults to "jina-reranker-v2-base-multilingual".

  Returns:
      list[RerankerResult]: A list of reranked items with their relevance scores and text.

  Raises:
      ValueError: If the JINA_API_KEY environment variable is not set,
                  or if the API request fails, or if there's an issue
                  parsing the API response.
  """
  if not text_list:
    return []

  api_key: Optional[str] = os.getenv("JINA_API_KEY")

  if not api_key:
    raise ValueError("No API key for the Jina Reranker has been set. "
                     "Please set the JINA_API_KEY environment variable.")

  url = "https://api.jina.ai/v1/rerank"
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}",
  }
  data = {
      "model": model,
      "query": query,
      "documents": text_list,
      "top_n": top_n,
  }

  try:
    response: requests.Response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    response_json: Any = response.json()

    # Parse the results from the API response
    reranked_results: list[RerankerResult] = [
        RerankerResult(
            index=r["index"],
            relevance_score=r["relevance_score"],
            text=r["document"]["text"],
        )
        for r in response_json.get("results", [])
    ]

    return reranked_results

  except requests.RequestException as e:
    # Catch any request-related errors (connection issues, timeouts, bad status codes after raise_for_status)
    raise ValueError(f"Jina API request failed: {e}")
  except KeyError as e:
      # Catch errors if expected keys are missing in the JSON response
      raise ValueError(f"Something went wrong parsing the Jina API response: Missing key {e}")
  except Exception as e:
      # Catch any other unexpected errors during processing
      raise ValueError(f"An unexpected error occurred during reranking: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_jina_embedding(
  input: list[str], late_chunking: bool = False
) -> list[list[float]]:
  """Get Jina embeddings.

  Args:
    input (list[str]): The list of strings to embed.
    late_chunking (bool): Whether you should use late chunking.
      Set to false if only embedding a single chunk.

  Returns:
    list[list[float]]: A list containing a list of floats as a vector.
  """
  if len(input) == 0:
    return []
  url = "https://api.jina.ai/v1/embeddings"
  api_key: str = os.getenv("JINA_API_KEY")

  headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
  data = {
    "model": "jina-embeddings-v3",
    "task": "retrieval.query",
    "dimensions": 1024,
    "late_chunking": late_chunking,
    "embedding_type": "float",
    "input": input,
  }
  response = requests.post(url, headers=headers, json=data)

  repsonse_json = response.json()
  try:
    z = []
    for e in repsonse_json["data"]:
      z.append(e["embedding"])
    return z
  except Exception as e:
    print(repsonse_json)
    raise e
