import ast
import pandas as pd

from benchmarks.frames.preprocess.web_to_db import DATA_URL
failed_links = ['https://en.wikipedia.org/wiki/2021_French_Open_%E2%80%93_Men%2527s_singles', 'https://en.wikipedia.org/wiki/Jack_Vance_(tennis)', 'https://en.wikipedia.org/wiki/Nemanja_Markovi%C4%87', 'https://en.wikipedia.org/wiki/Pok%C3%A9mon_(NOT_REQUIRED']

df = pd.read_csv(DATA_URL, sep='\t')

def check_faild_links(df: pd.DataFrame) -> set:
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
                for c_link in cleaned_links:
                    if c_link in failed_links:
                        print(f"Row index {index}, contains a filaed link", c_link)
            # else: print(f"Warning: Row {index} 'wiki_links' was not a list after eval: {type(links_list_raw)}") # Optional warning

        except (ValueError, SyntaxError) as e:
            # print(f"Error parsing wiki_links at row {index}: {e} - '{wiki_links_str}'") # Optional error logging
            pass # Skip rows with parsing errors

    unique_links = set(all_links)
    print(f"Found {len(unique_links)} unique links.")
    return unique_links




check_faild_links(df)