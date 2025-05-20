import json
from uuid import uuid4

from benchmarks.dataset import Dataset, Row




def get_multi_hop_dataset():


    path = "data/multihoprag.json"

    with open(path, encoding="utf-8") as f:
        data = json.loads(f.read())
    rows = []
    for row in data:
        sources = []
        for f in row["evidence_list"]:
            sources.append(f["url"])

        rows.append(
          Row(
            id = uuid4(),
            prompt=row["query"],
            answer= row["answer"],
            tags = [],
            sources=sources
          )
        )

    return Dataset(rows=rows)
