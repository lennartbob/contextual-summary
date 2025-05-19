import asyncio
import pandas as pd
import ast
import sqlite3
import requests # Still imported, though not used in the main async flow now
import time
import os
import sys
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup

from benchmarks.frames.config import DATA_URL # Import BeautifulSoup

# --- Configuration ---
# URL of the dataset

# User-Agent for requests (good practice)
USER_AGENT = "WikiContentFetcher/2.0 (learning; contact@example.com; +https://github.com/my-project-link-or-contact)"
# Timeout for web requests
REQUEST_TIMEOUT = 15 # seconds (Increased slightly for async)
# Batch size for concurrent requests
BATCH_SIZE = 20
# Delay between batches (Optional, to avoid overwhelming the network/server)
BATCH_DELAY = 1 # seconds
# File to log failed URLs
FAILED_URLS_FILE = "failed.txt" # New constant for the failed URLs log file

# --- Part 1: Get a unique set of wiki links (Same as before) ---

def get_unique_wiki_links(df: pd.DataFrame) -> set:
    """
    Extracts a unique set of Wikipedia links from the 'wiki_links' column.
    """
    all_links = []
    print("Extracting unique links...")
    for index, row in df.iterrows():
        wiki_links_str = row.get('wiki_links') # Use .get for safety

        if pd.isna(wiki_links_str):
            continue # Skip if wiki_links column is missing or NaN

        try:
            # Safely evaluate the string representation of the list
            links_list_raw = ast.literal_eval(wiki_links_str)

            # Ensure it's a list and extract/strip links
            if isinstance(links_list_raw, list):
                # Filter out non-string items and strip whitespace
                cleaned_links = [str(link).strip() for link in links_list_raw if isinstance(link, str)]
                all_links.extend(cleaned_links)
            # else: print(f"Warning: Row {index} 'wiki_links' was not a list after eval: {type(links_list_raw)}") # Optional warning

        except (ValueError, SyntaxError) as e:
            # print(f"Error parsing wiki_links at row {index}: {e} - '{wiki_links_str}'") # Optional error logging
            pass # Skip rows with parsing errors

    unique_links = set(all_links)
    print(f"Found {len(unique_links)} unique links.")
    return unique_links

# --- Part 2 & 3 Helper: Async Fetch and Text Extraction Function ---

async def fetch_and_extract_text(session: aiohttp.ClientSession, original_url: str) -> tuple:
    """
    Async helper to fetch HTML content, extract text, and handle errors.
    Adds https:// if the scheme is missing.
    Returns (success: bool, used_url: str|None, text_content: str|None, original_url: str, error_msg: str|None).
    """
    current_url = original_url
    html_content = None
    text_content = None # This is what we want to store
    used_url = None # This will be the URL that was actually used (original or https)
    error_msg = None
    success = False

    # Check if the URL has a scheme. If not, assume https://
    parsed_url = urlparse(current_url)
    if not parsed_url.scheme:
        current_url = f"https://{current_url}"
        # print(f"  Assuming https:// for {original_url} -> {current_url}") # Optional log

    used_url = current_url # The URL we are attempting to fetch


    try:
        # Use async session.get with timeout
        async with session.get(current_url, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status() # Raise for bad status codes (4xx or 5xx)
            html_content = await response.text() # Get the raw HTML text

            # --- Extract text using BeautifulSoup ---
            if html_content:
                try:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text_content = soup.get_text()
                    # Optional: basic cleaning of text (e.g., remove excessive newlines/spaces)
                    # text_content = ' '.join(text_content.split()) # Example: reduces multiple spaces/newlines to single space
                except Exception as parse_error:
                    error_msg = f"Parsing error: {parse_error}"
                    # print(f"  Parsing failed for {current_url}: {parse_error}", file=sys.stderr) # Optional log
                    success = False # Parsing failure is considered a failure to get usable content
            else:
                 error_msg = "Fetched content was empty"
                 success = False


            # If parsing was successful and we got text
            if text_content is not None:
                success = True
                # print(f"  Success fetching and parsing {used_url}") # Optional success log


    except aiohttp.ClientError as e:
        # Catch aiohttp specific errors (timeout, connection, bad status, etc.)
        error_msg = f"ClientError: {e}"
        # print(f"  Failed fetching {current_url}: {error_msg}", file=sys.stderr) # Optional failure log
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"Unexpected error during fetch: {e}"
        # print(f"  Failed fetching {current_url}: {error_msg}", file=sys.stderr) # Optional failure log

    # Return all relevant info, including the extracted text
    return (success, used_url, text_content, original_url, error_msg)

# --- Part 2 & 3: Fetch text and store in SQLite ---
async def fetch_and_store_content_async(links_set: set, db_path: str):
    """
    Fetches content for links in batches asynchronously, extracts text, and stores in SQLite.
    Logs failed URLs to a file.
    """
    setup_database(db_path) # Ensure database and table exist

    conn = None
    failed_file = None # File handle for failed URLs
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get URLs already in the database to skip fetching
        print("Checking existing entries in database...")
        existing_urls = set()
        cursor.execute(f"SELECT url FROM {TABLE_NAME}")
        for row in cursor.fetchall():
            existing_urls.add(row[0])
        print(f"Found {len(existing_urls)} existing entries.")

        # Filter out links that are already in the database
        links_to_process = []
        for url in sorted(list(links_set)): # Sort for predictable order
            url_to_check = url
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                url_to_check = f"https://{url}" # Check potential https version if scheme was missing

            if url_to_check not in existing_urls and url not in existing_urls:
                 links_to_process.append(url)
            # else: print(f"  Skipping {url} (already in DB)") # Optional skipped log

        print(f"\nStarting fetch process for {len(links_to_process)} links...")

        fetched_count = 0 # Count successful fetches and stores
        skipped_count = len(links_set) - len(links_to_process) # Count initial skips
        error_count = 0 # Count all processing/db errors

        headers = {'User-Agent': USER_AGENT}

        # Use a single ClientSession for all requests
        async with aiohttp.ClientSession(headers=headers) as session:
            # Process links in batches
            total_links_to_process = len(links_to_process)
            for i in range(0, total_links_to_process, BATCH_SIZE):
                batch_urls = links_to_process[i:i + BATCH_SIZE]
                print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(total_links_to_process + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch_urls)} URLs)...")

                # Create tasks for each URL in the batch
                tasks = [asyncio.create_task(fetch_and_extract_text(session, url)) for url in batch_urls]

                # Run tasks concurrently and wait for results
                results = await asyncio.gather(*tasks)

                # Process results and store in DB (synchronously after batch fetch)
                for success, used_url, text_content, original_url, error_msg in results:
                    if success and text_content is not None: # Ensure both fetch/parse was successful AND we got text
                        try:
                            # Store in DB using the URL that actually worked (used_url)
                            cursor.execute(f'''
                                INSERT OR REPLACE INTO {TABLE_NAME} (url, content)
                                VALUES (?, ?)
                            ''', (used_url, text_content)) # Store text_content
                            fetched_count += 1
                        except sqlite3.Error as e:
                            # Log database errors to stderr and count them separately or add to error_count
                            error_count += 1
                            print(f"  Database error storing {used_url}: {e}", file=sys.stderr)
                            # Optionally log DB errors to failed.txt as well? For now, just fetch/parse errors.
                        except Exception as e:
                            error_count += 1
                            print(f"  Unexpected error during database storage for {used_url}: {e}", file=sys.stderr)
                            # Optionally log DB errors to failed.txt as well? For now, just fetch/parse errors.
                    else:
                        # If fetch failed or parsing failed or text_content was None
                        error_count += 1
                        print(f"  Failed to fetch/process {original_url} (used: {used_url}): {error_msg}", file=sys.stderr)

                        # --- Log failed URL to file ---
                        try:
                            # Use 'a' mode to append to the file. Open and close each time for simplicity.
                            with open(FAILED_URLS_FILE, 'a', encoding='utf-8') as f:
                                log_entry = f"Original URL: {original_url}, Used URL: {used_url if used_url else 'N/A'}, Error: {error_msg}\n"
                                f.write(log_entry)
                        except Exception as log_e:
                            print(f"  Error writing to failed URLs file {FAILED_URLS_FILE}: {log_e}", file=sys.stderr)
                        # --- End log failed URL ---


                # Commit after each batch
                conn.commit()
                # print(f"Batch {i//BATCH_SIZE + 1} committed.") # Optional verbose commit log

                # Optional delay between batches
                if BATCH_DELAY > 0 and i + BATCH_SIZE < total_links_to_process:
                    print(f"Waiting for {BATCH_DELAY} seconds before next batch...")
                    await asyncio.sleep(BATCH_DELAY)


        print("\nFetch and store process complete.")
        print(f"Summary: Fetched {fetched_count}, Skipped (already in DB) {skipped_count}, Errors {error_count}")
        if error_count > 0:
             print(f"Details of failed URLs are logged in '{FAILED_URLS_FILE}'.")


    except sqlite3.Error as e:
        print(f"Database error during fetch/store: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
        # File handle is managed by 'with open', so no need to close 'failed_file' here

# --- Main execution ---

if __name__ == "__main__":
    # Optional: Clear the failed.txt file at the start if you want a fresh log each run
    # try:
    #     if os.path.exists(FAILED_URLS_FILE):
    #         os.remove(FAILED_URLS_FILE)
    #         print(f"Cleared previous failed URLs log: {FAILED_URLS_FILE}")
    # except Exception as e:
    #     print(f"Error clearing failed URLs file: {e}", file=sys.stderr)

    print(f"Loading data from {DATA_URL}...")
    try:
        df = pd.read_csv(DATA_URL, sep='\t')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1) # Exit if data loading fails

    # 1. Get unique links
    unique_links_set = get_unique_wiki_links(df)

    # 2 & 3. Fetch text content and store in the database (ASYNC version)
    # Ensure the directory for the database exists if you put it elsewhere
    # os.makedirs(os.path.dirname(DATABASE_NAME), exist_ok=True) # Uncomment if using a subdir

    # Run the asynchronous function using asyncio.run()
    asyncio.run(fetch_and_store_content_async(unique_links_set, DATABASE_NAME))

    # 4. Example of retrieving content (text) from the database
    print("\n--- Example Retrieval ---")
    # Pick a random link from the set for demonstration, if the set is not empty
    if unique_links_set:
        # Note: Retrieval needs the exact URL stored.
        # The async function stores the URL that succeeded (could be original or https://)
        # We'll try retrieving both the original and the potential https:// version.
        # Take one of the first few links for easier testing consistency
        example_original_url = sorted(list(unique_links_set))[0]
        print(f"Attempting to retrieve content (text) for (original from dataset): {example_original_url}")

        retrieved_content = get_content_from_db(example_original_url, DATABASE_NAME) # This now retrieves text

        if retrieved_content:
            print("Found content (text) using original URL.")
        else:
            # If not found with original, try with https:// prepended
            parsed_url = urlparse(example_original_url)
            https_version_url = f"https://{example_original_url}" if not parsed_url.scheme else example_original_url

            # Only try if it was modified AND the modified version is different from original
            if https_version_url != example_original_url:
                 print(f"Not found with original URL, trying with https://: {https_version_url}")
                 retrieved_content = get_content_from_db(https_version_url, DATABASE_NAME)
                 if retrieved_content:
                     print("Found content (text) using https:// URL.")


        if retrieved_content:
            print(f"Successfully retrieved content (text - first 500 chars):")
            # Use repr() to show newlines/tabs if they are present
            print(repr(retrieved_content[:500]) + "..." if len(retrieved_content) > 500 else repr(retrieved_content))
        else:
            print("Content (text) not found in database (might have failed to fetch/parse or the final stored URL was different).")


        # Example of a URL that might not be in the set (or db)
        print("\nAttempting to retrieve content (text) for a non-existent URL:")
        non_existent_url = "https://en.wikipedia.org/wiki/This_Page_Definitely_Does_Not_Exist_XYZ123"
        retrieved_content_nonexistent = get_content_from_db(non_existent_url, DATABASE_NAME)
        if retrieved_content_nonexistent:
             print(f"Unexpectedly retrieved content (text) for {non_existent_url}")
        else:
            print(f"Content (text) for {non_existent_url} not found as expected.")

    else:
        print("No unique links found to demonstrate retrieval.")

    print("\nScript finished.")