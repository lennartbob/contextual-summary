from typing import Optional, Union, List, Dict # Added List, Dict, Union
from uuid import UUID
from qdrant_client import models
from qdrant_client import QdrantClient
# PointStruct is used for add_MANY, SearchRequest for search_many
from qdrant_client.http.models import PointStruct, SearchRequest, ScoredPoint
from qdrant_client.http.exceptions import UnexpectedResponse

from attrs import define, field

from benchmarks.frames.config import DEFAULT_QDRANT_COLLECTION, QDRANT_EMBEDDING_DIMENSION
from core.config.qdrant_config import QdrantConfig, get_qdrant_client
from core.providers.jina import get_jina_embedding

METRIC_MAP = {
  "COSINE": models.Distance.COSINE,
  "L2": models.Distance.EUCLID,
  "IP": models.Distance.DOT,
}
DEFAULT_QDRANT_METRIC = models.Distance.COSINE


@define
class WikiChunk:
  id: UUID
  wiki: str
  content: str

@define
class QdrantRepository:
  """
  A repository class for interacting with a specific Qdrant collection
  using WikiChunk objects.

  Handles common CRUD and search operations. Assumes the collection
  exists in Qdrant or is created via ensure_collection_exists(). Consistently
  uses UUID objects for point IDs in its public interface.

  The Qdrant client is lazily loaded upon first use.
  """

  config: QdrantConfig
  collection_name: str = field(default=DEFAULT_QDRANT_COLLECTION)
  vector_dimension: int = field(default=QDRANT_EMBEDDING_DIMENSION)
  distance_metric: models.Distance = field(default=DEFAULT_QDRANT_METRIC)

  _client: Optional[QdrantClient] = field(init=False, default=None, repr=False)
  _payload_fields: list[str] = field(
    init=False,
    default=["id", "wiki", "content"],
    repr=False,
  )

  @property
  def client(self) -> QdrantClient:
    if self._client is None:
      try:
        qdrant_client_instance = get_qdrant_client(self.config)
        self.ensure_collection_exists(qdrant_client_instance) # Pass the instance here
        self._client = qdrant_client_instance
      except Exception as e:
        self._client = None
        raise RuntimeError(f"Failed to initialize Qdrant client: {e}") from e
    return self._client

  def ensure_collection_exists(self, qdrant_client: QdrantClient) -> None:
    try:
      existing_collections = qdrant_client.get_collections().collections
      collection_names = {col.name for col in existing_collections}

      if self.collection_name not in collection_names:
        print(f"Collection '{self.collection_name}' does not exist. Creating it...")
        qdrant_client.create_collection(
          collection_name=self.collection_name,
          vectors_config=models.VectorParams(
            size=self.vector_dimension,
            distance=self.distance_metric,
          ),
        )
        print(
          f"Collection '{self.collection_name}' created with dim={self.vector_dimension} and distance={self.distance_metric}."
        )
        print(
          f"Attempting to create payload indexes for fields: {self._payload_fields}..."
        )
        for field_name in self._payload_fields:
          try:
            schema_type = models.PayloadSchemaType.TEXT if field_name == "content" else models.PayloadSchemaType.KEYWORD
            if field_name == "id":
                schema_type = models.PayloadSchemaType.KEYWORD

            qdrant_client.create_payload_index(
              collection_name=self.collection_name,
              field_name=field_name,
              field_schema=schema_type,
            )
            print(f"  - Created payload index for '{field_name}' with schema {schema_type}.")
          except Exception as idx_e: # More specific exceptions could be caught if qdrant_client raises them
            # Qdrant might return an error if index already exists or type is incompatible
            print(
              f"  - Warning: Failed to create/update payload index for '{field_name}': {idx_e}. It might already exist with a compatible type."
            )
        print("Payload index creation process finished.")

      else:
        print(
          f"Collection '{self.collection_name}' already exists. Validating config..."
        )
        collection_info = qdrant_client.get_collection(
          collection_name=self.collection_name
        )

        # Robust check for vector_params_config
        vectors_config = collection_info.config.params.vectors
        actual_dim: Optional[int] = None
        actual_dist: Optional[models.Distance] = None

        if isinstance(vectors_config, models.VectorParams): # Single unnamed vector
            actual_dim = vectors_config.size
            actual_dist = vectors_config.distance
        elif isinstance(vectors_config, dict): # Named vectors
            if "" in vectors_config and isinstance(vectors_config[""], models.VectorParams): # Default unnamed vector in a dict
                actual_dim = vectors_config[""].size
                actual_dist = vectors_config[""].distance
            elif self.collection_name in vectors_config and isinstance(vectors_config[self.collection_name], models.VectorParams): # A common pattern is to name the vector config same as collection or a primary name
                 # This case is less likely for the simple setup but good to be aware of.
                 # For this repository, we assume a single, default vector configuration.
                 # If you intend to use named vectors, the repository logic would need to be more complex.
                pass # Fall through to error if no default unnamed vector
            # If you use specific named vectors in your application, you'd check that specific name.
            # For a generic repository assuming one default vector type, we primarily look for the unnamed default.
            if not actual_dim and len(vectors_config) == 1: # If only one named vector, maybe that's what we want to compare against?
                # This is heuristic. Better to rely on the "" default for generic cases.
                # For now, we will require an unnamed default if it's a dict.
                first_vector_name = next(iter(vectors_config))
                print(f"Warning: Collection '{self.collection_name}' uses named vectors. Assuming validation against the first one found: '{first_vector_name}'. "
                      f"For unambiguous validation, ensure a default (empty string key) vector config exists or adapt repository for named vectors.")
                if isinstance(vectors_config[first_vector_name], models.VectorParams):
                    actual_dim = vectors_config[first_vector_name].size
                    actual_dist = vectors_config[first_vector_name].distance


        if actual_dim is None or actual_dist is None:
            raise RuntimeError(
                f"Collection '{self.collection_name}' exists, but its vector configuration format is not a simple VectorParams or a dict with a default ('') VectorParams entry. "
                f"Actual config: {vectors_config}. Cannot automatically validate."
            )

        if actual_dim != self.vector_dimension:
          raise RuntimeError(
            f"Dimension mismatch for collection '{self.collection_name}': Exists with dimension {actual_dim}, but repository configured for {self.vector_dimension}."
          )
        if actual_dist != self.distance_metric:
          print(
            f"Warning: Distance metric mismatch for collection '{self.collection_name}': Exists with metric '{actual_dist}', but repository configured for '{self.distance_metric}'. Operations will use the existing collection metric ('{actual_dist}')."
          )
        print(
          f"Collection '{self.collection_name}' config validated (Actual Dim: {actual_dim}, Actual Distance: {actual_dist})."
        )

    except UnexpectedResponse as ue:
      content = ue.content.decode() if ue.content else "N/A"
      raise RuntimeError(
        f"Qdrant API error during collection check/creation for '{self.collection_name}': Status={ue.status_code}, Content='{content}'"
      ) from ue
    except Exception as e:
      raise RuntimeError(
        f"Failed to ensure collection '{self.collection_name}' exists: {type(e).__name__} - {e}"
      ) from e

  def add_MANY(self, chunks: list[WikiChunk]) -> None:
    if not chunks:
      return

    print(f"Preparing to add {len(chunks)} chunks to collection '{self.collection_name}'.")

    contents_to_embed = [chunk.content for chunk in chunks]
    try:
      embeddings = get_jina_embedding(contents_to_embed)
    except Exception as e:
      raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    if len(embeddings) != len(chunks):
      raise RuntimeError(
          f"Mismatch between number of chunks ({len(chunks)}) and generated embeddings ({len(embeddings)})."
      )

    points_to_upsert: list[PointStruct] = []
    for chunk, vector in zip(chunks, embeddings):
      if len(vector) != self.vector_dimension:
        raise ValueError(
            f"Embedding for chunk ID {chunk.id} has dimension {len(vector)}, "
            f"but collection dimension is {self.vector_dimension}."
        )

      payload = {
          "id": str(chunk.id),
          "wiki": chunk.wiki,
          "content": chunk.content,
      }
      for field_key in self._payload_fields:
          if field_key not in payload:
              raise KeyError(f"Field '{field_key}' missing in payload for chunk ID {chunk.id}. "
                             f"Ensure WikiChunk attributes match _payload_fields.")

      points_to_upsert.append(
          models.PointStruct(
              id=str(chunk.id),
              vector=vector,
              payload=payload,
          )
      )

    try:
      print(f"Upserting {len(points_to_upsert)} points...")
      self.client.upsert(
          collection_name=self.collection_name,
          points=points_to_upsert,
          wait=True
      )
      print(f"Successfully added {len(points_to_upsert)} chunks.")
    except UnexpectedResponse as ue:
      content = ue.content.decode() if ue.content else "N/A"
      raise RuntimeError(
        f"Qdrant API error during upsert to '{self.collection_name}': Status={ue.status_code}, Content='{content}'"
      ) from ue
    except Exception as e:
      raise RuntimeError(f"Failed to add chunks to Qdrant: {e}") from e

  def _scored_point_to_wikichunk(self, hit: ScoredPoint) -> Optional[WikiChunk]:
    """Helper to convert a ScoredPoint to a WikiChunk, handling missing payload."""
    payload = hit.payload
    if payload is None:
        print(f"Warning: Search hit (ID: {hit.id}, Score: {hit.score}) has no payload. Skipping.")
        return None
    try:
      chunk_id_str = payload.get("id")
      wiki_val = payload.get("wiki")
      content_val = payload.get("content")

      if chunk_id_str is None or wiki_val is None or content_val is None:
          missing_fields = [
              f for f in self._payload_fields
              if payload.get(f) is None
          ]
          print(f"Warning: Search hit (ID: {hit.id}) payload is missing required fields: {missing_fields}. Payload: {payload}. Skipping.")
          return None

      return WikiChunk(
          id=UUID(chunk_id_str),
          wiki=wiki_val,
          content=content_val,
      )
    except Exception as e:
      print(f"Warning: Error converting search hit payload to WikiChunk (ID: {hit.id}): {e}. Payload: {payload}. Skipping.")
      return None

  def search(self, query: str, n_result: int) -> list[WikiChunk]:
    if not query:
      return []
    if n_result <= 0:
      return []

    try:
      query_embedding = get_jina_embedding([query])[0]
    except Exception as e:
      raise RuntimeError(f"Failed to generate embedding for query: {e}") from e

    if len(query_embedding) != self.vector_dimension:
        raise ValueError(
            f"Query embedding dimension {len(query_embedding)} does not match "
            f"collection dimension {self.vector_dimension}."
        )

    try:
      search_results: list[ScoredPoint] = self.client.search(
          collection_name=self.collection_name,
          query_vector=query_embedding,
          limit=n_result,
          with_payload=True
      )
    except UnexpectedResponse as ue:
      content = ue.content.decode() if ue.content else "N/A"
      raise RuntimeError(
        f"Qdrant API error during search in '{self.collection_name}': Status={ue.status_code}, Content='{content}'"
      ) from ue
    except Exception as e:
      raise RuntimeError(f"Failed to search in Qdrant: {e}") from e

    wiki_chunks_found: list[WikiChunk] = []
    for hit in search_results:
      chunk = self._scored_point_to_wikichunk(hit)
      if chunk:
        wiki_chunks_found.append(chunk)
    return wiki_chunks_found

  def search_many(self, queries: List[str], n_result_per_query: int) -> List[List[WikiChunk]]:
    """
    Searches the Qdrant collection for multiple queries simultaneously.
    Returns a list of lists, where each inner list contains WikiChunk results
    for the corresponding input query.
    """
    if not queries:
      return []
    if n_result_per_query <= 0:
      return [[] for _ in queries] # Return list of empty lists, matching query count

    print(f"Batch searching for {len(queries)} queries in '{self.collection_name}', requesting {n_result_per_query} results per query.")

    try:
      query_embeddings = get_jina_embedding(queries)
    except Exception as e:
      raise RuntimeError(f"Failed to generate embeddings for batch queries: {e}") from e

    if len(query_embeddings) != len(queries):
        raise RuntimeError(
            f"Mismatch between number of queries ({len(queries)}) and generated embeddings ({len(query_embeddings)})."
        )

    search_requests: List[SearchRequest] = []
    for i, embedding in enumerate(query_embeddings):
      if len(embedding) != self.vector_dimension:
        raise ValueError(
            f"Query embedding {i+1} dimension {len(embedding)} does not match "
            f"collection dimension {self.vector_dimension}."
        )
      search_requests.append(
          SearchRequest(
              vector=embedding,
              limit=n_result_per_query,
              with_payload=True
          )
      )

    try:
      # search_batch returns a list of lists of ScoredPoint
      batched_results: List[List[ScoredPoint]] = self.client.search_batch(
          collection_name=self.collection_name,
          requests=search_requests
      )
    except UnexpectedResponse as ue:
      content = ue.content.decode() if ue.content else "N/A"
      raise RuntimeError(
        f"Qdrant API error during batch search in '{self.collection_name}': Status={ue.status_code}, Content='{content}'"
      ) from ue
    except Exception as e:
      raise RuntimeError(f"Failed to batch search in Qdrant: {e}") from e

    all_query_results: List[List[WikiChunk]] = []
    for single_query_hits in batched_results:
      current_query_chunks: List[WikiChunk] = []
      for hit in single_query_hits:
        chunk = self._scored_point_to_wikichunk(hit)
        if chunk:
          current_query_chunks.append(chunk)
      all_query_results.append(current_query_chunks)

    print(f"Batch search completed. Result counts per query: {[len(res) for res in all_query_results]}")
    return all_query_results


  def _record_to_wikichunk(self, point_record: models.Record) -> Optional[WikiChunk]:
    """Helper to convert a Record to a WikiChunk, handling missing payload."""
    payload = point_record.payload
    if payload is None:
        print(f"Warning: Retrieved point (ID: {point_record.id}) has no payload. Cannot reconstruct WikiChunk.")
        return None
    try:
      chunk_id_str = payload.get("id")
      wiki_val = payload.get("wiki")
      content_val = payload.get("content")

      if chunk_id_str is None or wiki_val is None or content_val is None:
          missing_fields = [
              f for f in self._payload_fields
              if payload.get(f) is None
          ]
          print(f"Warning: Retrieved point (ID: {point_record.id}) payload is missing required fields: {missing_fields}. Payload: {payload}. Cannot reconstruct WikiChunk.")
          return None

      return WikiChunk(
          id=UUID(chunk_id_str),
          wiki=wiki_val,
          content=content_val,
      )
    except Exception as e:
        print(f"Warning: Error converting retrieved point payload to WikiChunk (ID: {point_record.id}): {e}. Payload: {payload}.")
        return None

  def get_ONE(self, wiki_chunk_id: UUID) -> Optional[WikiChunk]:
    print(f"Attempting to retrieve chunk with ID {wiki_chunk_id} from '{self.collection_name}'.")
    try:
      retrieved_points: list[models.Record] = self.client.retrieve(
          collection_name=self.collection_name,
          ids=[str(wiki_chunk_id)],
          with_payload=True
      )
    except UnexpectedResponse as ue:
      content = ue.content.decode() if ue.content else "N/A"
      # Specific check for 404 if Qdrant client throws it this way for retrieve (unlikely for retrieve, more for direct GET point)
      # if ue.status_code == 404:
      #   print(f"Chunk with ID {wiki_chunk_id} not found (API 404).")
      #   return None
      raise RuntimeError(
        f"Qdrant API error retrieving ID '{wiki_chunk_id}': Status={ue.status_code}, Content='{content}'"
      ) from ue
    except Exception as e:
      raise RuntimeError(f"Failed to retrieve chunk ID {wiki_chunk_id} from Qdrant: {e}") from e

    if not retrieved_points:
      print(f"Chunk with ID {wiki_chunk_id} not found.")
      return None

    return self._record_to_wikichunk(retrieved_points[0])


  def get_MANY(self, wiki_chunk_ids: list[UUID]) -> list[WikiChunk]:
    if not wiki_chunk_ids:
      return []

    print(f"Attempting to retrieve {len(wiki_chunk_ids)} chunks from '{self.collection_name}'.")
    ids_str = [str(uid) for uid in wiki_chunk_ids]

    try:
      retrieved_points: list[models.Record] = self.client.retrieve(
          collection_name=self.collection_name,
          ids=ids_str,
          with_payload=True
      )
    except UnexpectedResponse as ue:
      content = ue.content.decode() if ue.content else "N/A"
      raise RuntimeError(
        f"Qdrant API error during batch retrieve from '{self.collection_name}': Status={ue.status_code}, Content='{content}'"
      ) from ue
    except Exception as e:
      raise RuntimeError(f"Failed to retrieve multiple chunks from Qdrant: {e}") from e

    found_chunks: list[WikiChunk] = []
    for point_record in retrieved_points:
      chunk = self._record_to_wikichunk(point_record)
      if chunk:
        found_chunks.append(chunk)

    print(f"Retrieved {len(found_chunks)} out of {len(wiki_chunk_ids)} requested chunks.")
    if len(found_chunks) != len(wiki_chunk_ids):
        print(f"Note: {len(wiki_chunk_ids) - len(found_chunks)} chunk(s) were not found or had payload issues.")
    return found_chunks