import csv
import json
import os
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


BASE_AZ_URL = "https://www.allrecipes.com/recipes-a-z-6735880"
DELAY_MS = 2000
PROGRESS_SAVE_EVERY = 5


class AllRecipesScraper:
    def __init__(self, headless=True):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page()

    def close(self):
        self.browser.close()
        self.playwright.stop()

    def fetch_html(self, url: str) -> str:
        print(f"GET {url}")
        self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
        self.page.wait_for_timeout(DELAY_MS)
        return self.page.content()

    @staticmethod
    def clean_text(text):
        if not text:
            return None
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def find_recipe_node(node):
        if isinstance(node, dict):
            node_type = node.get("@type")
            if node_type == "Recipe" or (isinstance(node_type, list) and "Recipe" in node_type):
                return node
            for value in node.values():
                found = AllRecipesScraper.find_recipe_node(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = AllRecipesScraper.find_recipe_node(item)
                if found:
                    return found
        return None

    @staticmethod
    def get_recipe_json_ld(soup: BeautifulSoup):
        scripts = soup.find_all("script", type="application/ld+json")
        for script in scripts:
            raw = script.string or script.get_text()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            found = AllRecipesScraper.find_recipe_node(data)
            if found:
                return found
        return None

    def extract_letter_category_links(self):
        html = self.fetch_html(BASE_AZ_URL)
        soup = BeautifulSoup(html, "html.parser")

        categories = []
        current_letter = None

        for tag in soup.find_all(["h2", "h3", "a"]):
            if tag.name in ["h2", "h3"]:
                text = self.clean_text(tag.get_text())
                if text and len(text) == 1 and text.isalpha():
                    current_letter = text.upper()

            elif tag.name == "a" and current_letter:
                href = tag.get("href")
                text = self.clean_text(tag.get_text())
                if not href or not text:
                    continue

                full_url = urljoin(BASE_AZ_URL, href)

                if "allrecipes.com" in full_url:
                    categories.append({
                        "letter": current_letter,
                        "category_name": text,
                        "category_url": full_url,
                    })

        seen = set()
        deduped = []
        for item in categories:
            key = (item["letter"], item["category_url"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        return deduped

    def extract_recipe_links_from_category(self, category_url: str):
        print(f"GET {category_url}")
        self.page.goto(category_url, wait_until="domcontentloaded", timeout=60000)
        self.page.wait_for_timeout(3000)

        try:
            self.page.wait_for_load_state("networkidle", timeout=10000)
        except:
            pass

        # start from top
        self.page.evaluate("window.scrollTo(0, 0)")
        self.page.wait_for_timeout(1000)

        def collect_links_from_current_html():
            html = self.page.content()
            soup = BeautifulSoup(html, "html.parser")

            links = set()
            for a in soup.find_all("a", href=True):
                href = urljoin(category_url, a["href"])

                # keep recipe pages, skip category/index pages
                if (
                    re.search(r"-recipe-\d+", href)
                    or re.search(r"/recipe/\d+/", href)
                ):
                    links.add(href)

            return links

        # first collect without assuming scrolling is needed
        recipe_links = collect_links_from_current_html()
        print(f"Initial HTML count: {len(recipe_links)} recipes")

        # if page already has lots of recipes, return them
        if len(recipe_links) >= 8:
            print(f"FINAL COUNT: {len(recipe_links)} recipes")
            return sorted(recipe_links)

        # fallback: scroll and try to load more if initial HTML was sparse
        stable_rounds = 0
        previous_count = len(recipe_links)

        for i in range(20):
            # try clicking load more if present
            try:
                load_more = self.page.query_selector("button:has-text('Load More')")
                if load_more and load_more.is_enabled():
                    print("Clicking Load More...")
                    load_more.scroll_into_view_if_needed()
                    self.page.wait_for_timeout(1000)
                    load_more.click()
                    self.page.wait_for_timeout(2500)
            except:
                pass

            # scroll down
            self.page.mouse.wheel(0, 4000)
            self.page.wait_for_timeout(2000)

            try:
                self.page.wait_for_load_state("networkidle", timeout=5000)
            except:
                pass

            new_links = collect_links_from_current_html()
            new_count = len(new_links)

            print(f"Scroll {i}: {new_count} recipes found")

            if new_count == previous_count:
                stable_rounds += 1
            else:
                stable_rounds = 0
                previous_count = new_count
                recipe_links = new_links

            if stable_rounds >= 4:
                break

        print(f"FINAL COUNT: {len(recipe_links)} recipes")
        return sorted(recipe_links)

    def parse_recipe_page(self, recipe_url: str):
        html = self.fetch_html(recipe_url)
        soup = BeautifulSoup(html, "html.parser")
        recipe_ld = self.get_recipe_json_ld(soup)

        result = {
            "url": recipe_url,
            "title": None,
            "description": None,
            "author": None,
            "date_published": None,
            "yield": None,
            "prep_time": None,
            "cook_time": None,
            "total_time": None,
            "category": None,
            "cuisine": None,
            "ingredients": [],
            "directions": [],
            "rating_value": None,
            "rating_count": None,
            "review_count": None,
            "rating_text": None,
            "nutrition_calories": None,
            "nutrition_fat": None,
            "nutrition_carbs": None,
            "nutrition_protein": None,
            "nutrition_raw": None,
        }

        if recipe_ld:
            result["title"] = recipe_ld.get("name")
            result["description"] = recipe_ld.get("description")
            result["date_published"] = recipe_ld.get("datePublished")
            result["yield"] = recipe_ld.get("recipeYield")
            result["prep_time"] = recipe_ld.get("prepTime")
            result["cook_time"] = recipe_ld.get("cookTime")
            result["total_time"] = recipe_ld.get("totalTime")
            result["category"] = recipe_ld.get("recipeCategory")
            result["cuisine"] = recipe_ld.get("recipeCuisine")

            author = recipe_ld.get("author")
            if isinstance(author, list) and author:
                first = author[0]
                result["author"] = first.get("name") if isinstance(first, dict) else str(first)
            elif isinstance(author, dict):
                result["author"] = author.get("name")
            elif isinstance(author, str):
                result["author"] = author

            ingredients = recipe_ld.get("recipeIngredient") or []
            result["ingredients"] = [self.clean_text(x) for x in ingredients if self.clean_text(x)]

            instructions = recipe_ld.get("recipeInstructions") or []
            steps = []
            for step in instructions:
                if isinstance(step, dict):
                    txt = step.get("text")
                    if txt:
                        steps.append(self.clean_text(txt))
                elif isinstance(step, str):
                    steps.append(self.clean_text(step))
            result["directions"] = steps

            agg = recipe_ld.get("aggregateRating")
            if isinstance(agg, dict):
                result["rating_value"] = agg.get("ratingValue")
                result["rating_count"] = agg.get("ratingCount")
                result["review_count"] = agg.get("reviewCount")

            nutrition = recipe_ld.get("nutrition")
            if isinstance(nutrition, dict):
                result["nutrition_raw"] = nutrition
                result["nutrition_calories"] = nutrition.get("calories")
                result["nutrition_fat"] = nutrition.get("fatContent")
                result["nutrition_carbs"] = nutrition.get("carbohydrateContent")
                result["nutrition_protein"] = nutrition.get("proteinContent")

        page_text = soup.get_text("\n", strip=True)
        if "Be the first to rate & review!" in page_text and not result["rating_value"]:
            result["rating_text"] = "Be the first to rate & review!"
            result["rating_count"] = 0
            result["review_count"] = 0

        if not result["title"]:
            h1 = soup.find("h1")
            if h1:
                result["title"] = self.clean_text(h1.get_text())

        return result


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(data, path):
    rows = []
    for item in data:
        row = item.copy()
        row["ingredients"] = json.dumps(row["ingredients"], ensure_ascii=False)
        row["directions"] = json.dumps(row["directions"], ensure_ascii=False)
        row["nutrition_raw"] = json.dumps(row["nutrition_raw"], ensure_ascii=False)
        rows.append(row)

    keys = sorted({k for row in rows for k in row.keys()})

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_progress(path="allrecipes_progress.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_seen_recipe_urls(data):
    return {item["url"] for item in data if "url" in item}


def scrape_all(
    headless=True,
    limit_letters=None,
    limit_categories=None,
    limit_recipes=None,
    progress_json="allrecipes_progress.json",
    progress_csv="allrecipes_progress.csv",
):
    scraper = AllRecipesScraper(headless=headless)

    existing_data = load_progress(progress_json)
    all_data = existing_data[:]
    seen_urls = get_seen_recipe_urls(existing_data)

    print(f"Loaded {len(all_data)} recipes from progress file")

    try:
        categories = scraper.extract_letter_category_links()

        if limit_letters:
            categories = [c for c in categories if c["letter"] in set(limit_letters)]

        if limit_categories is not None:
            categories = categories[:limit_categories]

        print(f"Found {len(categories)} category pages")

        for c_idx, category in enumerate(categories, start=1):
            print(f"\n[{c_idx}/{len(categories)}] {category['letter']} - {category['category_name']}")
            print(category["category_url"])

            try:
                recipe_links = scraper.extract_recipe_links_from_category(category["category_url"])
            except Exception as e:
                print(f"Failed category: {e}")
                continue

            if limit_recipes is not None:
                recipe_links = recipe_links[:limit_recipes]

            recipe_links = [url for url in recipe_links if url not in seen_urls]

            print(f"Found {len(recipe_links)} new recipes")

            for r_idx, recipe_url in enumerate(recipe_links, start=1):
                print(f"  ({r_idx}/{len(recipe_links)}) {recipe_url}")
                try:
                    recipe = scraper.parse_recipe_page(recipe_url)
                    recipe["letter"] = category["letter"]
                    recipe["source_category"] = category["category_name"]
                    recipe["source_category_url"] = category["category_url"]

                    all_data.append(recipe)
                    seen_urls.add(recipe_url)

                    if len(all_data) % PROGRESS_SAVE_EVERY == 0:
                        save_json(all_data, progress_json)
                        save_csv(all_data, progress_csv)
                        print(f"Progress saved: {len(all_data)} recipes")

                except Exception as e:
                    print(f"Failed recipe: {e}")

    finally:
        scraper.close()

    save_json(all_data, progress_json)
    save_csv(all_data, progress_csv)
    print(f"Final progress saved: {len(all_data)} recipes")

    return all_data


if __name__ == "__main__":
    data = scrape_all(
        headless=False,
        limit_letters=None,
        limit_categories=None,
        limit_recipes=None,
        progress_json="allrecipes_progress.json",
        progress_csv="allrecipes_progress.csv",
    )

    save_json(data, "allrecipes_all.json")
    save_csv(data, "allrecipes_all.csv")

    print(f"\nSaved {len(data)} recipes.")