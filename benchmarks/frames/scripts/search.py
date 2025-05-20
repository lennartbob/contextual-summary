

from benchmarks.frames.repo.vector_db import QdrantRepository
from core.config.qdrant_config import QdrantConfig


query = "Where did DJ khaled go to highschool?"


r = QdrantRepository(
    QdrantConfig()
)

result = r.search(query, n_result=20)

for re in result:
    print(re)
    print("--")