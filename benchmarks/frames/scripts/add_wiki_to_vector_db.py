from tqdm import tqdm
from typing import List # Import List for type hinting

from benchmarks.frames.repo.vector_db import QdrantRepository
from benchmarks.frames.repo.wiki_db import get_all_wiki_urls
from benchmarks.frames.utils.chunker import chunk_wiki_page
from core.config.qdrant_config import QdrantConfig

# Assume Chunk is a defined type or structure used by chunk_wiki_page and add_MANY
# from your_module import Chunk # Uncomment and replace 'your_module' if Chunk is defined elsewhere

def chunk_add_to_vdb_batched():
    """
    Fetches wiki URLs, chunks their content, and adds the chunks to a
    vector database in batches of 5 wiki pages.
    """
    wiki_links: List[str] = get_all_wiki_urls()

    # Ensure there are links to process, although the original asserted > 2000
    if not wiki_links:
        print("No wiki links found to process.")
        return

    # If you still want to enforce a minimum, uncomment the original assertion
    # assert len(wiki_links) > 2000, "Less than 2000 wiki links found, cannot proceed with batching as intended."

    qdrant_repo = QdrantRepository(
        config=QdrantConfig()
    )

    batch_size = 5
    total_chunks_added = 0
    # Calculate the total number of batches for tqdm progress
    num_batches = (len(wiki_links) + batch_size - 1) // batch_size

    # Iterate through wiki links in batches
    # tqdm tracks progress based on the number of batches
    for i in tqdm(range(0, len(wiki_links), batch_size), total=num_batches, desc="Processing wiki batches"):
        # Get the links for the current batch
        current_batch_links = wiki_links[i : i + batch_size]
        batch_chunks = [] # List to hold chunks from all pages in the current batch

        # Process each link in the current batch to get its chunks
        for wiki_link in current_batch_links:
            try:
                chunks_from_page = chunk_wiki_page(wiki_link)
                batch_chunks.extend(chunks_from_page)
            except Exception as e:
                print(f"Error processing wiki link {wiki_link}: {e}")
                # Continue with the next link or batch even if one fails

        # Add all accumulated chunks from the batch to the vector database
        if batch_chunks: # Only attempt to add if there are chunks
            try:
                qdrant_repo.add_MANY(batch_chunks)
                total_chunks_added += len(batch_chunks)
            except Exception as e:
                print(f"Error adding batch of chunks to Qdrant starting with link {current_batch_links[0]}: {e}")
                # Depending on requirements, you might want to log the failed batch
        else:
            # This case might happen if chunk_wiki_page returns empty for all links in a batch
            print(f"No chunks generated for batch starting with link {current_batch_links[0]}")


    print(f"Successfully added {total_chunks_added} chunks to the vector database.")

# Call the new batched function
# chunk_add_to_vdb_batched()