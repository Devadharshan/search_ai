import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

def fetch_data(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract clean text
        clean_text = soup.get_text(separator="\n", strip=True)

        # Regex extraction
        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", clean_text)
        phones = re.findall(r"\+?\d[\d\s().-]{7,}\d", clean_text)
        urls_found = re.findall(r"https?://[^\s\"']+", response.text)

        # Image URLs
        image_tags = soup.find_all("img")
        image_urls = [urljoin(url, img.get("src")) for img in image_tags if img.get("src")]

        # Tables
        tables = []
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                tables.append(rows)

        # Return all extracted data
        return {
            "text": clean_text,
            "images": image_urls,
            "emails": emails,
            "phones": phones,
            "urls": urls_found,
            "tables": tables
        }

    except Exception as e:
        return {"error": str(e)}