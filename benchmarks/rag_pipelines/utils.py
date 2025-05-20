
import itertools
import operator
import os
from benchmarks.frames.repo.vector_db import WikiChunk

def save_prompt(type: str, filename: str, prompt: str, answer: str):
  """
  Saves a prompt and its answer to a file within the 'data/prompts' directory using a relative path.

  Args:
    type: The name of the directory (folder) within 'data/prompts' to save the file in.
    filename: The name of the file to create or overwrite.
    prompt: The prompt text.
    answer: The answer text.
  """
  if ".txt" not in filename:
    filename = filename + ".txt"
  if not isinstance(answer, str):
    answer = str(answer)
  content = prompt + "\n=========\n" + answer

  # Define the base directory 'data/prompts'
  base_dir = "data/prompts"
  filename = filename.strip()

  # Create the base directory if it doesn't exist
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)

  # Create the subdirectory (type) if it doesn't exist
  full_dir = os.path.join(base_dir, type)
  if not os.path.exists(full_dir):
    os.makedirs(full_dir)

  # Construct the full file path
  filepath = os.path.join(full_dir, filename)

  # Write the content to the file
  try:
    with open(filepath, "w", encoding="utf-8") as f:
      f.write(content)
    # print(f"Successfully saved to {filepath}")
  except Exception as e:
    print(f"Error saving to {filepath}: {e}")


def chunks_to_string(chunks: list[WikiChunk]) -> str:
    """
    Groups WikiChunk objects by their 'wiki' attribute and formats them into a string.

    Args:
        chunks: A list of WikiChunk objects.

    Returns:
        A formatted string with chunks grouped by source.
    """
    s = ""
    if not chunks:
        return s

    # Sort chunks by wiki source for groupby
    sorted_chunks = sorted(chunks, key=operator.attrgetter('wiki'))

    source_number = 1
    for wiki_source, wiki_chunks_iterator in itertools.groupby(sorted_chunks, key=operator.attrgetter('wiki')):
        s += f"Source {source_number}: {wiki_source}\n"
        for chunk in wiki_chunks_iterator:
            s += f"{chunk.content}\n"
        s += "\n" # Add a blank line between sources
        source_number += 1

    return s.strip() # Remove trailing newline if any
