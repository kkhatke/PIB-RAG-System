#%%
import json
import os
import requests
import warnings
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock

warnings.filterwarnings("ignore")

def initialize_session(url):
    """Initializes the session and retrieves the VIEWSTATE and VIEWSTATEGENERATOR."""
    session = requests.Session()
    response = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve initial page. Status code: {response.status_code}")
    soup = BeautifulSoup(response.text, "html.parser")
    viewstate = soup.find("input", {"name": "__VIEWSTATE"})["value"]
    viewstategenerator = soup.find("input", {"name": "__VIEWSTATEGENERATOR"})["value"]
    return session, viewstate, viewstategenerator

def fetch_article_ids(session, url, viewstate, viewstategenerator, year, month, day):
    """Fetches article IDs and ministry-article pairs from the specified URL using the provided VIEWSTATE."""
    payload = {
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "__VIEWSTATE": viewstate,
        "__VIEWSTATEGENERATOR": viewstategenerator,
        "__VIEWSTATEENCRYPTED": "",
        "minname": "0",
        "rdate": day,
        "rmonth": month,
        "ryear": year,
        "__CALLBACKID": "__Page",
        "__CALLBACKPARAM": f"1|{day}|{month}|{year}|0"
    }
    response = session.post(url, data=payload, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve article IDs. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    articles = []
    ministry_elements = soup.find_all("li", class_="rel min-rel")
    
    for ministry_element in ministry_elements:
        ministry_name = ministry_element.text.strip()
        articles_list = ministry_element.find_next("ul", class_="rel-display11")
        
        if articles_list:
            article_items = articles_list.find_all("li", class_="link1")
            for article in article_items:
                article_id = article['id']
                article_title = article.text.strip()
                articles.append({
                    "date": f"{year}-{month:02d}-{day:02d}",
                    "id": article_id,
                    "ministry": ministry_name,
                    "title": article_title
                })

    return articles

def get_article_content(session, article):
    """Retrieves the content of the article based on the provided ID."""
    rid = article['id']
    article_url = f"https://archive.pib.gov.in/newsite/PrintRelease.aspx?relid={rid}"
    response = session.get(article_url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", style="text-align: justify;line-height:1.6;font-size:110%")
        article["content"] = content_div.get_text(separator="\n", strip=True) if content_div else None
        return article
    else:
        print(f"Failed to retrieve content for article ID {rid}")
        return None

def process_day(session, url, viewstate, viewstategenerator, year, month, day, progress_file, data_file):
    """Processes and saves articles for a specific day, with progress tracking and JSON validation."""
    date_key = f"{year}-{month:02d}-{day:02d}"
    if is_date_processed(date_key, progress_file):
        print(f"Skipping {date_key}, already processed")
        return

    articles = fetch_article_ids(session, url, viewstate, viewstategenerator, year, month, day)
    completed_articles = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_article_content, session, article): article for article in articles}
        for future in as_completed(futures):
            article = future.result()
            if article and article.get("content"):
                completed_articles.append(article)

    if completed_articles:
        save_to_json(completed_articles, data_file)
        update_progress(date_key, progress_file)
        print(f"Saved {len(completed_articles)} articles for {date_key}")

def save_to_json(data, filename):
    """Safely appends data to a JSON file using file locking to avoid corruption."""
    filepath = os.path.join("scraped_data", filename)
    os.makedirs("scraped_data", exist_ok=True)
    lock = FileLock(f"{filepath}.lock")

    with lock:
        if os.path.exists(filepath):
            with open(filepath, 'r+', encoding='utf-8') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    print("Corrupted JSON file, resetting.")
                    existing_data = []
                existing_data.extend(data)
                file.seek(0)
                json.dump(existing_data, file, ensure_ascii=False, indent=4)
        else:
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

def update_progress(date_key, progress_file):
    """Updates progress tracking file safely with file locking."""
    lock = FileLock(f"{progress_file}.lock")

    with lock:
        if os.path.exists(progress_file):
            with open(progress_file, 'r+') as file:
                try:
                    progress = json.load(file)
                except json.JSONDecodeError:
                    progress = []
                progress.append(date_key)
                file.seek(0)
                json.dump(progress, file, indent=4)
        else:
            with open(progress_file, 'w') as file:
                json.dump([date_key], file, indent=4)

def is_date_processed(date_key, progress_file):
    """Checks if the date is already processed, with error handling for corrupted progress files."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as file:
                progress = json.load(file)
                return date_key in progress
        except json.JSONDecodeError:
            print("Corrupted progress file, resetting.")
            return False
    return False

def main():
    """Main function to run the scraper."""
    url = "https://archive.pib.gov.in/archive2/erelease.aspx"
    data_file = "all_articles.json"
    progress_file = "progress.json"
    session, viewstate, viewstategenerator = initialize_session(url)

    for year in range(2025, 2026):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    process_day(session, url, viewstate, viewstategenerator, year, month, day, progress_file, data_file)
                except Exception as e:
                    print(f"Error processing {year}-{month}-{day}: {e}")

if __name__ == "__main__":
    main()

# %%
