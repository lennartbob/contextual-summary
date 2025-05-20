import pandas as pd
import sqlite3
import sys

from benchmarks.frames.config import DATABASE_NAME, TABLE_NAME

def get_wiki_content(url: str, db_path: str = DATABASE_NAME) -> str | None:
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


def get_all_wiki_urls(db_path: str = DATABASE_NAME) -> list[str]:
    """
    Retrieves a list of all stored URLs from the database.
    Returns a list of URL strings. Returns an empty list if an error occurs or no URLs are found.
    """
    conn = None
    urls = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Select all URLs from the table
        cursor.execute(f"SELECT url FROM {TABLE_NAME}")
        results = cursor.fetchall()
        # Extract URLs from the fetched rows
        urls = [row[0] for row in results]
    except sqlite3.Error as e:
        print(f"Database error during retrieval of all URLs: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
    return urls
