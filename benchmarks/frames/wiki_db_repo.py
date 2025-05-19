import pandas as pd
import sqlite3
import sys
from bs4 import BeautifulSoup # Import the BeautifulSoup class

# --- Configuration ---
# Name of the SQLite database file
DATABASE_NAME = "wiki_content.db"
# Name of the table to store wiki content
TABLE_NAME = "wiki_pages"

def get_content_from_db(url: str, db_path: str = DATABASE_NAME) -> str | None:
    """
    Retrieves the stored content for a given URL from the database.
    Returns the content string or None if the URL is not found.
    """
    conn = None
    content = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Note: Retrieval uses the exact URL stored, which might be the https:// version
        cursor.execute(f"SELECT content FROM {TABLE_NAME} WHERE url = ?", (url,))
        result = cursor.fetchone()
        if result:
            content = result[0]
    except sqlite3.Error as e:
        print(f"Database error during retrieval for {url}: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
    return content


# --- Main Execution ---
url_to_get = "https://en.wikipedia.org/wiki/Grazia_Deledda"
html_str = get_content_from_db(url_to_get)

if html_str:
    # 1. Convert the string to HTML (parse it)
    # Use BeautifulSoup to parse the HTML string
    soup = BeautifulSoup(html_str, 'html.parser') # 'html.parser' is a standard parser

    # 2. Extract all text from the HTML
    # Use the get_text() method to get all visible text
    all_text = soup.get_text().strip().replace("\n\n", " ")

    # Print the extracted text
    print(all_text)
else:
    print(f"Could not retrieve content for {url_to_get} from the database.", file=sys.stderr)