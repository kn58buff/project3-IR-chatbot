import wikipedia
import regex as re
import pandas as pd
import numpy as np
import os
import requests
import concurrent.futures
import tqdm
import preprocessor
from dotenv import load_dotenv

load_dotenv()

# Constant defs for scraping
API_URL = "https://en.wikipedia.org/w/api.php?"
PARAMS = {
    "action": "query",
    "titles": "",
    "prop": "extracts",
    "format": "json"
    }

API_KEY = os.getenv("API_KEY")

HEADERS = {
  'Authorization': API_KEY,
  'User-Agent': "CSE535Project3/1.0 (contact: kn58@buffalo.edu)"
}

# Constant defs for indexing
CORE_NAME = "IRF25P3"
VM_IP = "34.63.237.17"

class Scraper:
    """Class to handle scraping Wikipedia articles and exporting to CSV."""
    def __init__(self):
        self.base_url = API_URL
        self.params = PARAMS.copy()
        self.headers = HEADERS.copy()
        self.preprocessor = preprocessor.Preprocessor()

    def _process_text(self, text):
        # Remove special characters and extra spaces
        if text is None or text == "":
            return ""
        else:
            text_clean = re.sub(r"[^a-zA-Z0-9 ]", " ", text) # remove special chars
            text_clean = re.sub(r"/^\s+|\s+$|\s+(?=\s)", "", text_clean).strip() # remove multiple spaces
            return text_clean

    
    def _create_chunks(self, seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]
    
    def _find_topics(self, queries, tol = 6000):
        results = []
        for q in queries:
            if len(results) >= tol:
                break

            res = wikipedia.search(q, results=1000)
            results.extend(res)
        unique_results = list(set(results))
        return unique_results

    def _retrieve_page(self, page_title):
        params = self.params.copy()
        # implement batching logic
        params ={
            "action": "query",
            "prop": "extracts",
            "format": "json",
            "exintro": True,
            "titles": "|".join(page_title),
            "explaintext": True
        }

        response = requests.get(f"{self.base_url}", params=params, headers=self.headers).json()

        pages = response["query"]["pages"]

        batched_data = []

        for k in pages.keys():
            page_id = k

            page_content = self._process_text(pages[page_id].get("extract", ""))
            title = pages[page_id]["title"]
            url = f"https://en.wikipedia.org/w/index.php?curid={page_id}"

            batched_data.append({
                "page_id": page_id,
                "title": title,
                "url": url,
                "summary": page_content,
            })

        return batched_data
    
    def _scrape_search_results(self, page_titles, max_workers=10):
        
        data = []
        batches = list(self._create_chunks(page_titles, 10))
        print(len(batches))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._retrieve_page, batch): batch for batch in batches}
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    batched_data = future.result()
                    data.extend(batched_data)
                except Exception as e:
                    print(f"Error retrieving batch: {e}")
                    continue

        return data

    def run_scraper(self, queries, save = False):
        to_scrape = [self._find_topics(qs, tol=10000) for qs in tqdm.tqdm(queries)]
        topics = ["health", "environment", "technology", "economy", "entertainment",
          "sports", "politics", "education", "travel", "food"]
        
        df = pd.DataFrame()
        for qs, topic in zip(to_scrape, topics):
            print(f"Scraping topic: {topic} with {len(qs)} queries")

            to_add = self._scrape_search_results(qs, max_workers=10)
            temp_df = pd.DataFrame(to_add).drop_duplicates(subset=["page_id"], keep="first").reset_index(drop=True)
            temp_df["topic"] = topic
            df = pd.concat([df, temp_df], ignore_index=True)

        df_cleaned = df.replace("", np.nan).dropna()
        true_df = df_cleaned.loc[df_cleaned["summary"].str.len() >= 200]

        tqdm.tqdm.pandas()
        tokens = true_df["summary"].progress_apply(self.preprocessor.tokenizer)

        true_df["lemmatized_summary"] = tokens
        if save:
            true_df.to_csv("wikipedia_scraped_data.csv", index=False)
        return true_df

