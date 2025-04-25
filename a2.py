import requests
from requests_kerberos import HTTPKerberosAuth
import json
import hashlib
import os
import logging
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename(url):
    return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + ".json")

def compute_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def fetch_url_with_session(url, cert_path):
    session = requests.Session()
    session.auth = HTTPKerberosAuth()
    session.verify = cert_path
    try:
        logger.info(f"Fetching URL: {url}")
        response = session.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None

def scrape_and_cache(url, cert_path):
    html_content = fetch_url_with_session(url, cert_path)
    if html_content is None:
        return None

    soup = BeautifulSoup(html_content, "html.parser")
    extracted_text = "\n".join([p.get_text() for p in soup.find_all('p')])
    content_hash = compute_hash(extracted_text)

    cache_filename = get_cache_filename(url)

    # Check if cache exists
    if os.path.exists(cache_filename):
        with open(cache_filename, 'r', encoding='utf-8') as file:
            cached_data = json.load(file)
            if cached_data.get('content_hash') == content_hash:
                logger.info(f"No change detected for: {url}")
                return cached_data  # Return cached version

    # If new or changed, update cache
    content = {
        "title": soup.title.string if soup.title else "No title",
        "text": extracted_text,
        "content_hash": content_hash
    }

    with open(cache_filename, 'w', encoding='utf-8') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)

    logger.info(f"Updated cache for: {url}")
    return content

def scrape_app_urls_from_json(json_file, cert_path):
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        logger.error(f"JSON file {json_file} not found.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_file}: {e}")
        return

    for app_name, app_data in data.items():
        logger.info(f"Processing app: {app_name}")
        for url in app_data.get('urls', []):
            content = scrape_and_cache(url, cert_path)
            if content:
                app_data[f"{url}_scraped_data"] = content

    with open('subprime_updated_apps_data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    logger.info("Scraping complete. Data updated.")

# Example usage
json_file = 'subprime.json'
cert_path = '/path/to/your/certificate.crt'
scrape_app_urls_from_json(json_file, cert_path)