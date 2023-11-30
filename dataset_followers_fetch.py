import requests
from bs4 import BeautifulSoup
import pandas as pd


def search_entity_id(query):
    wikidata_url = "https://www.wikidata.org/w/api.php"

    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": query,
    }

    try:
        response = requests.get(wikidata_url, params=params)
        response.raise_for_status()

        data = response.json()

        if "search" in data and data["search"]:
            # Get the first result (most relevant)
            return data["search"][0]["id"]

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    return None


headers = {
    "authority": "www.hackerearth.com",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.5",
    "cache-control": "max-age=0",
    "sec-ch-ua": '"Brave";v="107", "Chromium";v="107", "Not=A?Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "sec-gpc": "1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
}


def get_twitter_followers_on_date(key):
    response = requests.get(f"https://www.wikidata.org/wiki/{key}", headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    try:
        soup = soup.find("div", {"class": "wikibase-statementgroupview", "id": "P8687"})
        twitter_followers = soup.find(
            "div",
            {"class": "wikibase-snakview-value wikibase-snakview-variation-valuesnak"},
        ).text
        soup_date = soup.find("div", {"class": "wikibase-statementview-qualifiers"})
        soup_date = soup_date.find_all("div", {"class", "wikibase-snakview-body"})[1]
        date_posted = soup_date.find(
            "div",
            {"class": "wikibase-snakview-value wikibase-snakview-variation-valuesnak"},
        ).text
    except AttributeError:
        return 0, None
    return twitter_followers, date_posted


# Read Excel file
excel_file_path = "/content/content_simulation_train.xlsx"
df = pd.read_excel(excel_file_path)


# Unique brands info
max_unique_brands = df["inferred company"].unique()

print(max_unique_brands)
print(len(max_unique_brands))
print("=" * 50)


print("Running API code and finding the Twitter followers for each brand...")

brands_followers_dict = {}
brands_date_posted = {}

for i, brand in enumerate(max_unique_brands):
    brand_id = search_entity_id(brand)
    followers, date = get_twitter_followers_on_date(brand_id)

    brands_followers_dict[brand] = followers
    brands_date_posted[brand] = date


df["followers"] = df["inferred company"].map(brands_followers_dict)
df["date_retrieved"] = df["inferred company"].map(brands_date_posted)

df.to_csv("content_simulation_train_with_followers.csv")
