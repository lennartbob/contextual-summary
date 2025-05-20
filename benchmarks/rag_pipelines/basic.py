import asyncio
from collections import defaultdict
import itertools
import operator
from enum import StrEnum
from pprint import pprint
from attrs import define
import os # Import the os module
from tqdm import tqdm # Corrected import
from benchmarks.dataset import Row, Dataset
from benchmarks.evaluation_check import self_check
from benchmarks.frames.repo.vector_db import QdrantRepository, WikiChunk
from benchmarks.frames.utils.dataset import get_frames_dataset
from core.config.qdrant_config import QdrantConfig
from core.providers.async_gpt import AsyncAzureOpenAIProvider
from core.providers.jina import RerankerResult, rerank_with_jina
from benchmarks.utils import process_template
from benchmarks.rag_pipelines.utils import chunks_to_string

async def basic_rag_pipeline(row: Row, llm: AsyncAzureOpenAIProvider, vector_repo: QdrantRepository) -> str:
    """
    Executes the RAG pipeline steps for a single row.
    """
    # Search vector database
    chunks: list[WikiChunk] = vector_repo.search(query=row.prompt, n_result=50)

    # Rerank retrieved chunks
    # Assuming rerank_with_jina takes the list of WikiChunk objects and returns the top N WikiChunk objects
    reranker_result:list[RerankerResult] = rerank_with_jina(query=row.prompt, text_list=[c.content for c in chunks], top_n=10)
    # Ensure indices are valid before accessing chunks
    relevant_chunks_indices = [re.index for re in reranker_result if re.index < len(chunks)]
    relevant_chunks: list[WikiChunk] = [chunks[index] for index in relevant_chunks_indices]


    # Format chunks for the prompt
    formatted_chunks = chunks_to_string(relevant_chunks)

    # Process the LLM prompt template
    prompt = process_template(
        "rag_template.jinja",
        {"chunks": formatted_chunks, "n_chunks": len(relevant_chunks), "question": row.prompt}
    )

    # Get the LLM response
    answer = await llm.get_response(prompt)

    return answer

async def process_row_and_check(row: Row, llm: AsyncAzureOpenAIProvider, vector_repo: QdrantRepository) -> tuple[Row, bool, str | None]:
    """
    Processes a single row through the RAG pipeline and performs a self-check.

    Args:
        row: The input Row object.
        llm: The language model provider.
        vector_repo: The vector database repository.

    Returns:
        A tuple containing the original Row, a boolean indicating if the check passed,
        and the LLM's answer (or None if an exception occurred).
    """
    answer = None # Initialize answer to None
    try:
        answer = await basic_rag_pipeline(row, llm, vector_repo)
        is_correct = await self_check(row, answer, llm)
        return (row, is_correct, answer) # Return answer
    except Exception as e:
        print(f"Error processing row {row.id} ('{row.prompt[:50]}...'): {e}")
        # Log the error and return False for this row, with answer as None
        return (row, False, None)


async def basic_rag(dataset: Dataset, llm: AsyncAzureOpenAIProvider, vector_repo: QdrantRepository, n:int) -> dict:
    """
    Runs a basic RAG pipeline asynchronously on a dataset and provides an overview
    of correctness, including a breakdown by reasoning type, and saves details
    for incorrect answers.

    Args:
        dataset: The dataset to process.
        llm: The language model provider.
        vector_repo: The vector database repository.
        n: Number of rows to process. Use -1 for all rows.

    Returns:
        A dictionary containing total correct/false counts and counts per reasoning type.
    """
    print(f"Starting RAG processing for {len(dataset.rows)} rows using async...")

    rows_to_process = dataset.rows[:n] if n != -1 and n < len(dataset.rows) else dataset.rows

    results = []
    for row in tqdm(rows_to_process):
        f = await process_row_and_check(row, llm, vector_repo)
        results.append(f)
    # Initialize counters and type breakdowns
    total_correct = 0
    total_false = 0
    correct_by_type = defaultdict(int)
    false_by_type = defaultdict(int)
    error_count = 0 # This will now count tasks that returned (row, False, None) due to exception

    # Define the directory for wrong answers
    wrong_answers_folder = "wrong_answers"
    # Create the directory if it doesn't exist
    os.makedirs(wrong_answers_folder, exist_ok=True)
    print(f"Saving incorrect answers to '{wrong_answers_folder}' folder.")

    # Process the results from the concurrent tasks
    for result in results:
        # Result is a tuple (row, is_correct, answer)
        row, is_correct, answer = result # This line is safe because process_row_and_check always returns a tuple

        if is_correct:
            total_correct += 1
            for tag in row.tags:
                correct_by_type[tag] += 1
        else:
            total_false += 1
            for tag in row.tags:
                false_by_type[tag] += 1

            # Check if the failure was due to an internal exception in process_row_and_check
            # (indicated by answer being None when is_correct is False)
            if answer is None:
                 error_count += 1 # Count tasks that failed internally
                 # The error was already printed by process_row_and_check
            else:
                 # Save details for incorrect answers where an answer was generated
                 filename = os.path.join(wrong_answers_folder, f"false_{row.id}.txt")
                 try:
                     with open(filename, "w", encoding='utf-8') as f:
                         f.write(f"Question:\n{row.prompt}\n\n")
                         f.write(f"Ground Truth:\n{row.answer}\n\n")
                         f.write(f"Answer:\n{answer}\n")
                     # print(f"Saved wrong answer details for row {row.id} to {filename}") # Optional: print for every save
                 except IOError as e:
                     print(f"Error saving wrong answer details for row {row.id} to {filename}: {e}")

    # Reset total_false count and recalculate
    total_false_with_answer = 0
    false_by_type_with_answer = defaultdict(int)

    # Re-process results to distinguish between self-check failure and task error
    for result in results:
        row, is_correct, answer = result
        if not is_correct:
            if answer is not None: # Task completed and produced an answer, but self-check failed
                total_false_with_answer += 1
                for tag in row.tags:
                    false_by_type_with_answer[tag] += 1
            # else: # Task failed internally (answer is None), already counted in error_count


    total_completed_tasks = len(results) # Number of results returned by tqdm.gather
    # Check if this matches len(rows_to_process) - it should, as tasks don't raise unhandled exceptions
    if total_completed_tasks != len(rows_to_process):
         print(f"Warning: Number of results ({total_completed_tasks}) does not match planned rows ({len(rows_to_process)}).")
         # This scenario should ideally not happen with the current error handling
    # Let's structure the overview to clearly show these categories
    overview = {
        "total_rows_in_dataset": len(dataset.rows),
        "total_rows_selected_for_processing": len(rows_to_process),
        "tasks_with_internal_exceptions": error_count, # Tasks where process_row_and_check failed internally
        "tasks_completed_with_answer": len(rows_to_process) - error_count, # total_correct + total_false_with_answer
        "total_correct": total_correct, # Tasks completed & self_check=True
        "total_false_self_check": total_false_with_answer, # Tasks completed & self_check=False
        "correct_percentage": (total_correct / (len(rows_to_process) - error_count) * 100) if (len(rows_to_process) - error_count) > 0 else 0.0,
        "false_self_check_percentage": (total_false_with_answer / (len(rows_to_process) - error_count) * 100) if (len(rows_to_process) - error_count) > 0 else 0.0,
        "internal_exception_percentage": (error_count / len(rows_to_process) * 100) if len(rows_to_process) > 0 else 0.0,
        "correct_by_type": dict(correct_by_type), # Convert defaultdict to dict for final output
        "false_by_type_self_check": dict(false_by_type_with_answer), # Convert defaultdict to dict for final output
        # Note: Tasks with internal exceptions are not broken down by type here
    }


    print("RAG processing finished.")
    return overview


# Example usage (keep as is):
r = asyncio.run(basic_rag(
    dataset=get_frames_dataset(),
    vector_repo=QdrantRepository(QdrantConfig()),
    llm=AsyncAzureOpenAIProvider(resource_model_name="gpt-4.1"),
    n = -1 # Process all rows
))

pprint(r)