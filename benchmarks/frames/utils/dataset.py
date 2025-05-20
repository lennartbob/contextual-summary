from benchmarks.dataset import Dataset, Row, ReasoningType
from benchmarks.frames.config import DATA_URL, ROWS_WITH_FAILED_LINKS
import pandas as pd


def get_frames_dataset() -> Dataset:

    print(f"Loading data from {DATA_URL}...")
    df = pd.read_csv(DATA_URL, sep='\t')
    print("Data loaded successfully.")

    rows = []

    for index, row in df.iterrows():
        if index in ROWS_WITH_FAILED_LINKS:
            print("skipping row, as it the links do not exists", row["wiki_links"])
            continue
        types = row["reasoning_types"]
        tags = []
        for t in types.split(" | "):
            tags.append(ReasoningType(t))

        sources = row["wiki_links"]
        r = Row(
            id = index, prompt = row["Prompt"], answer=row["Answer"], tags=tags, sources=sources
        )
        rows.append(r)

    
    return Dataset(rows = rows)

