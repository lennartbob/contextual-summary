import asyncio
from collections import defaultdict
import itertools
import operator
from enum import StrEnum
from pprint import pprint
from typing import Optional
from uuid import uuid4
from attrs import define
import os # Import the os module
from tqdm import tqdm # Corrected import
from benchmarks.dataset import Row, Dataset
from benchmarks.evaluation_check import self_check
from benchmarks.frames.repo.vector_db import QdrantRepository, WikiChunk
from benchmarks.frames.repo.wiki_db import get_wiki_content
from benchmarks.frames.utils.dataset import get_frames_dataset
from benchmarks.rag_pipelines.query_exp import query_expansion
from benchmarks.rag_pipelines.utils import save_prompt
from core.config.qdrant_config import QdrantConfig
from core.providers.async_gemini import AsyncGeminiProvider
from core.providers.async_gpt import AsyncAzureOpenAIProvider
from core.providers.jina import RerankerResult, rerank_with_jina
from benchmarks.utils import process_template

async def contexual_summ_rag(dataset: Dataset, llm: AsyncAzureOpenAIProvider, vector_repo: QdrantRepository, n:int) -> dict:

    rows_to_process = dataset.rows[:n] if n != -1 and n < len(dataset.rows) else dataset.rows

    results = []
    for row in tqdm(rows_to_process):
        f = await process_row_and_check_summ(row, llm, vector_repo)
        results.append(f)
    # Initialize counters and type breakdowns
    total_correct = 0
    total_false = 0
    correct_by_type = defaultdict(int)
    false_by_type = defaultdict(int)
    error_count = 0 # This will now count tasks that returned (row, False, None) due to exception

    # Define the directory for wrong answers
    wrong_answers_folder = "wrong_answers_context_summ"
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
                         f.write(f"Prompt:\n{row.prompt}\n----\n")
                         f.write(f"Ground Truth:\n{row.answer}\n----\n")
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


    print("RAG CONTEXTUAL SUMMARY processing finished.")

    return overview

async def llm_contextual_summ(query:str, context:str, gemini:AsyncGeminiProvider) -> str:
    prompt = process_template(
        "contextual_summary.jinja",
        {"query":query, "context": context.strip().replace("\n\n",  " ")}
    )
    r = await gemini.get_response(prompt, temperature=0.5)
    save_prompt("contextual_sum",f"{query[:10]}_{uuid4().hex[:5]}", prompt, r)
    return r


async def contextual_summ_pipeline(
    row:Row,
    llm: AsyncAzureOpenAIProvider,
    vector_repo: QdrantRepository,
    max_summaries_per_query:int = 3, # Max unique documents to summarize per query context
    reranker_top_n: int = 10 # How many top results from reranker to consider for summarization
):
    gemini = AsyncGeminiProvider("gemini-2.0-flash") 
    
    expanded_queries:list[str]|None = await query_expansion(row.prompt)

    queries_to_process = [row.prompt]
    if expanded_queries:
        queries_to_process.extend(expanded_queries)

    # Fetch chunks for all queries
    all_query_chunks_nested:list[list[WikiChunk]] = vector_repo.search_many(
        queries=queries_to_process, 
        n_result_per_query=30 # Retrieve more chunks initially for the reranker to work with
    )

    aggregated_summaries_for_rag = []
    total_summarized_sources = 0
    
    for i, current_query in enumerate(queries_to_process):
        query_specific_chunks = all_query_chunks_nested[i]
        
        # print(f"\nProcessing query ({i+1}/{len(queries_to_process)}): \"{current_query}\"")
        # print(f"Retrieved {len(query_specific_chunks)} chunks for this query.")

        relevant_chunks_for_this_query: list[WikiChunk]
        if query_specific_chunks:
            # --- Reranking for the current_query (run synchronously in a thread) ---
            # print(f"Reranking {len(query_specific_chunks)} chunks for query: \"{current_query}\"...")
            # The rerank_with_jina function is synchronous.
            # asyncio.to_thread runs it in a separate thread to avoid blocking.
            reranker_results:list[RerankerResult] = await asyncio.to_thread(
                rerank_with_jina, # The synchronous function
                current_query,    # Arguments to the function
                [c.content for c in query_specific_chunks],
                reranker_top_n    # Ask reranker for its top N results
            )
            # print(f"Reranker returned {len(reranker_results)} results.")

            # Map reranked results (which have indices) back to actual WikiChunk objects
            temp_relevant_chunks = []
            for re_result in reranker_results:
                if 0 <= re_result.index < len(query_specific_chunks):
                    temp_relevant_chunks.append(query_specific_chunks[re_result.index])
                else:
                    print(f"Warning: Reranker returned invalid index {re_result.index} for query_specific_chunks of length {len(query_specific_chunks)}")
            relevant_chunks_for_this_query = temp_relevant_chunks
            
            if not relevant_chunks_for_this_query:
                print(f"Warning: Reranking yielded no valid chunks for query '{current_query}'. Falling back to top initial chunks.")
                # Fallback: use top N from original retrieval if reranking fails or returns nothing
                relevant_chunks_for_this_query = query_specific_chunks[:reranker_top_n]
        else:
            # print(f"No chunks retrieved for query: \"{current_query}\". Skipping reranking.")
            relevant_chunks_for_this_query = []

        # --- Identify unique wiki documents to summarize for THIS query's context ---
        wiki_links_to_summarize_for_this_query = []
        unique_wikis_for_this_query_context = set() 

        for wiki_chunk in relevant_chunks_for_this_query: # Iterate over reranked (or fallback) chunks
            if len(wiki_links_to_summarize_for_this_query) >= max_summaries_per_query:
                break
            if wiki_chunk.wiki not in unique_wikis_for_this_query_context:
                wiki_links_to_summarize_for_this_query.append(wiki_chunk.wiki)
                unique_wikis_for_this_query_context.add(wiki_chunk.wiki)
        
        if not wiki_links_to_summarize_for_this_query:
            # print(f"No unique wiki links selected for summarization for query: \"{current_query}\"")
            continue

        # print(f"Identified {len(wiki_links_to_summarize_for_this_query)} unique wiki links for summarization for this query from reranked/selected chunks.")

        summarization_tasks = []
        valid_links_for_tasks = [] 

        for wiki_link in wiki_links_to_summarize_for_this_query:
            content:str|None = get_wiki_content(wiki_link) # Synchronous call
            if not content:
               print(f"Content for wiki link '{wiki_link}' not found (for query '{current_query}'). Skipping.") 
               continue
            
            summarization_tasks.append(llm_contextual_summ(current_query, content, gemini))
            valid_links_for_tasks.append(wiki_link)

        if not summarization_tasks:
            # print(f"No valid content found for summarization tasks for query: \"{current_query}\"")
            continue
            
        # print(f"Gathering {len(summarization_tasks)} contextual summaries for query: \"{current_query}\"...")
        contextual_summaries:list[str] = await asyncio.gather(*summarization_tasks)
        total_summarized_sources += len(contextual_summaries)

        if contextual_summaries:
            query_context_header = f"--- Summaries relevant to query: \"{current_query}\" ---\n"
            summary_block_for_this_query = [query_context_header]
            
            for idx, summary_text in enumerate(contextual_summaries):
                link = valid_links_for_tasks[idx] 
                summary_block_for_this_query.append(f"Source Document: {str(link)}\n {summary_text}\n\n") # Added extra \n
            
            aggregated_summaries_for_rag.append("".join(summary_block_for_this_query))
        # else:
            # print(f"No summaries generated for query: \"{current_query}\"")

    final_rag_context_string = "\n".join(aggregated_summaries_for_rag).strip()
    
    if not final_rag_context_string:
        # print("No contextual summaries generated from any query. Answering with no context.")
        final_rag_context_string = "No specific context documents were found or summarized for your query."

    prompt_for_final_answer = process_template(
        "rag_template.jinja",
        {
            "question": row.prompt, 
            "chunks": final_rag_context_string,
        }
    )

    # print("\n--- FINAL PROMPT FOR AZURE LLM ---")
    # print(prompt_for_final_answer)
    # print("--- END OF FINAL PROMPT ---\n")

    answer:str = await llm.get_response(prompt_for_final_answer)

    save_prompt("rag_rerank_ctx_multi_query", row.prompt[:20].replace(" ", "_"), prompt_for_final_answer, answer)
    return answer



async def process_row_and_check_summ(row: Row, llm: AsyncAzureOpenAIProvider, vector_repo: QdrantRepository) -> tuple[Row, bool, str | None]:
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
        answer = await contextual_summ_pipeline(row, llm, vector_repo)
        is_correct = await self_check(row, answer, llm)
        return (row, is_correct, answer) # Return answer
    except Exception as e:
        print(f"Error processing row {row.id} ('{row.prompt[:50]}...'): {e}")
        # Log the error and return False for this row, with answer as None
        return (row, False, None)



r = asyncio.run(contexual_summ_rag(
    dataset=get_frames_dataset(),
    vector_repo=QdrantRepository(QdrantConfig()),
    llm=AsyncAzureOpenAIProvider(resource_model_name="gpt-4.1-mini"),
    n = -1
))

pprint(r)



