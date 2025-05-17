import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def extract_urls():
    base_url = "https://www.mardep.gov.hk/en/public-services/arrivals-and-departures/vladhist/index.html"

    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "html.parser")

    data = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        text = link.get_text(strip=True)

        if href.lower().endswith(".xml") and any(key in text for key in [
            "Due To Enter Hong Kong Report",
            "Intend To Depart Hong Kong Report",
            "Enter Hong Kong Water Report",
            "Depart Hong Kong Water Report"
        ]):
       

            for report_type in [
                "Due To Enter Hong Kong Report",
                "Intend To Depart Hong Kong Report",
                "Enter Hong Kong Water Report",
                "Depart Hong Kong Water Report"
            ]:
                if report_type in text:
                    full_link = urljoin(base_url, href)
                    data.append({
                        "report_type": report_type,
                        "xml_link": full_link
                    })

    df = pd.DataFrame(data)


    df.to_csv("xml_links.csv", index=False)


