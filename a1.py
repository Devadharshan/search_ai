import requests
from requests_kerberos import HTTPKerberosAuth
import json
import hashlib
import os
from bs4 import BeautifulSoup

# Cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename(url):
    """Generate a cache filename based on the URL."""
    return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + ".json")

def fetch_url_with_session(url, cert_path):
    """Use requests session with Kerberos authentication and custom SSL certificate."""
    
    # Create a session
    session = requests.Session()
    
    # Configure Kerberos authentication
    session.auth = HTTPKerberosAuth()
    
    # Set the SSL certificate path (certificate verification)
    session.verify = cert_path
    
    try:
        # Send the GET request
        response = session.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        return response.text  # Return HTML content of the page
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_and_cache(url, cert_path):
    """Scrapes the website content using a session and caches it."""
    cache_filename = get_cache_filename(url)
    
    # Check if the cached file exists
    if os.path.exists(cache_filename):
        print(f"Loading from cache: {url}")
        with open(cache_filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    # If not cached, scrape the page using session
    print(f"Scraping: {url}")
    html_content = fetch_url_with_session(url, cert_path)
    
    if html_content is None:
        return None
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract content, such as title, paragraphs, etc.
    content = {
        "title": soup.title.string if soup.title else "No title",
        "text": [p.get_text() for p in soup.find_all('p')]
    }

    # Save to cache
    with open(cache_filename, 'w', encoding='utf-8') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)

    return content

def scrape_app_urls_from_json(json_file, cert_path):
    """Reads a JSON file, scrapes URLs for each app, and updates the JSON data."""
    # Load the existing JSON data
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Loop through each app
    for app_name, app_data in data.items():
        print(f"Scraping for {app_name}:")
        
        # Loop through URLs for each app
        for url in app_data['urls']:
            scraped_data = scrape_and_cache(url, cert_path)
            
            # Add the scraped data to the app's data (optional)
            if scraped_data:
                app_data[f"{url}_scraped_data"] = scraped_data
    
    # Save the updated data with scraped results
    with open('subprime_updated_apps_data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print("Scraping complete and data saved.")

# Example usage
json_file = 'subprime.json'  # Use the 'subprime' JSON file
cert_path = '/path/to/your/certificate.crt'  # Path to the SSL certificate
scrape_app_urls_from_json(json_file, cert_path)