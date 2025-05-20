
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter

from benchmarks.frames.repo.vector_db import WikiChunk
from benchmarks.frames.repo.wiki_db import get_wiki_content

def chunk_wiki_page(
  wiki: str, chunk_size: int = 800, overlap: int = 200
) -> list[WikiChunk]:
  """
  Chunks a document from a filepath after extracting text from all pages and then chunking.

  Args:
      filepath: Path to the PDF document.
      chapter_token_size: The maximum number of tokens per chunk.

  Returns:
      A list of Chunk objects, where each chunk contains content and the page numbers it originated from.
  """
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    is_separator_regex=False,
    chunk_overlap=overlap,
  )
  full_text = get_wiki_content(wiki)

  texts = text_splitter.create_documents([full_text])
  chunks_content: list[str] = [t.page_content for t in texts]
  all_chunks: list[WikiChunk] = []
  for chunk_content in chunks_content:
    all_chunks.append(
      WikiChunk(content=chunk_content, id = uuid4(), wiki=wiki)
    ) 

  return all_chunks