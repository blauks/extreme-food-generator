from time import sleep
from attr import attr, attrs
from numpy import number
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests

import json

from bs4 import BeautifulSoup

options = Options()
options.headless = True
options.add_argument('--disable-browser-side-navigation')
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
driver.set_page_load_timeout(15)

currentPage = 1
url = "https://www.matprat.no/oppskrifter?page="

driver.get(url+str(currentPage))
WebDriverWait(driver, timeout=3).until(lambda d: d.find_element(By.CLASS_NAME, "slim-pagination__range"))
soup = BeautifulSoup(driver.page_source, 'html.parser')

maxPage = int(soup.find(attrs={"class": "slim-pagination__range"}).get_text().split(" ")[1])

recipes = []

def createRecipeObject(a, t, d, s, n, i, time, di):
    return {"attributes": a, "title": t, "description": d, "steps": s, "numberOfPeople": n, "ingredients": i, "time": time, "difficulty": di}

def formatSteps(rawList):
    output = []
    for i, step in enumerate(rawList, start=1):
        stepText = step.get_text().replace("\n", "")
        output.append(f"{i}. {stepText}")

    return output


def formatIngredient(raw):
    amount = raw.find(attrs={"class": "amount"}).get_text()
    unit = raw.find(attrs={"class": "unit"}).get_text()
    # :-)
    ingredient = " ".join(raw.find_all("span")[-1].get_text(strip=True).replace("\n","").split())
    return f"{amount}{unit} {ingredient}"

def extractRecipe(recipeURL):
    source = requests.get("https://www.matprat.no"+recipeURL).content
    soup = BeautifulSoup(source, 'html.parser')
    attributes = [att.get_text() for att in soup.find_all(attrs={"class": "recipe-search-tags__item"})]
    title = soup.find(attrs={"class": "article-title lp_is_start"}).get_text()
    description = soup.find(attrs={"itemprop": "description"}).find("p").get_text()
    steps = formatSteps(soup.find(attrs={"class": "recipe-steps"}).find_all(attrs={"class": "recipe-steps__item"}))
    numberOfPeople = ""
    try:
        numberOfPeople = soup.find(attrs={"id": "portionsInput"})["value"]
    except:
        numberOfPeople = soup.find(attrs={"class": "portions-label"}).get_text().strip()
    # Matprat skal selvfølgelig drive å ha både mobil og pc HTML i samme fil, bare i tilfelle noen plutselig bytter over til mobilen deres mens de ser på oppskrifter på pcen :|
    ingredients = [formatIngredient(item) for item in soup.find(attrs={"class": "grid__item cm-module--white one-whole recipe-ingredients"}).find_all(attrs={"itemprop":"ingredients"})]
    time = soup.find(attrs={"data-epi-property-name": "RecipeTime"}).get_text()
    difficulty = soup.find(attrs={"data-epi-property-name": "RecipeDifficulty"}).get_text()
    return createRecipeObject(attributes, title, description, steps, numberOfPeople, ingredients, time, difficulty)

def findRecipes(pageChildren):
    for i, recipe in enumerate(pageChildren, start=1):
        print(f"Recipe {i}")
        recipeURL = recipe.find("a")['href']
        recipeObject = extractRecipe(recipeURL)
        if not recipeObject == None: recipes.append(recipeObject)

def findRecipeURLs(hits):
    urls = [child["linkUrl"] for child in hits]
    return urls

recipeUrls = []

for p in range(1, maxPage+1):
    print(f"Page {p} of {maxPage} starting...")
    print(url+str(p))
    #source = requests.get(url+str(p), cookies={"consent-set": "true", "marketing-cookies-consent": "granted", "statistics-cookies-consent": "granted", "ARRAffinity": "95e5f71f2cafafb58c0c651bc100e82035e010c66b371ba830f2463be02741ec", ".ASPXANONYMOUS": "Xy_mAmQJR9XyKdLKyD2qCkTyQlWzLw-6OjguJDZFTiww8fmLUkM0GfhQxRVDqwLfKxRJ8JpCgN4o57THNTEV1EFEEjKbm1Edb9CJP1khakEmGM8KUQhZ6mvut6zVs_G_gGLYbA2", "_ab_test": "Q+3o5ipwUEdYF2edMl0Uk3jZwd3ux3rUAFaUuWChVHvfqq7cYRcx2AE218zHUqO0xPk1wg=="}).content
    data = json.loads(requests.get("https://www.matprat.no/api/RecipeSearch/GetRecipes?text=&page=1&sort=new", headers={'referer': url+str(p)}).content)
    recipeUrls += findRecipeURLs(data["searchHits"])
    print(len(recipeUrls))
    print(f"...Page {p} of {maxPage} done, continuing")


for i, url in enumerate(recipeUrls):
    try:
        recipeObject = extractRecipe(url)
        print(f"{i} of {len(recipeUrls)}: ",url)
        if not recipeObject == None: recipes.append(recipeObject) 
    except:
        print("DIDN'T WORK, PAUSE")
        sleep(3)
        pass

open("recipesTEST.json", "w").write(json.dumps(recipes, indent=4, ensure_ascii=False))