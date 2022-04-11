from time import sleep
from attr import attr, attrs
from numpy import number
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

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

    return " | ".join(output)


def formatIngredient(raw):
    amount = raw.find(attrs={"class": "amount"}).get_text()
    unit = raw.find(attrs={"class": "unit"}).get_text()
    # :-)
    ingredient = " ".join(raw.find_all("span")[-1].get_text(strip=True).replace("\n","").split())
    return f"{amount}{unit} {ingredient}"

def extractRecipe(recipeURL):
    driver.get("https://www.matprat.no"+recipeURL)
    WebDriverWait(driver, timeout=3).until(lambda d: d.find_element(By.CLASS_NAME, "recipe-search-tags__item"))
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        attributes = " ".join([att.get_text() for att in soup.find_all(attrs={"class": "recipe-search-tags__item"})])
        title = soup.find(attrs={"class": "article-title lp_is_start"}).get_text()
        description = soup.find(attrs={"itemprop": "description"}).find("p").get_text()
        steps = formatSteps(soup.find(attrs={"class": "recipe-steps"}).find_all(attrs={"class": "recipe-steps__item"}))
        numberOfPeople = soup.find(attrs={"id": "portionsInput"})["value"]

        # Matprat skal selvfølgelig drive å ha både mobil og pc HTML i samme fil, bare i tilfelle noen plutselig bytter over til mobilen deres mens de ser på oppskrifter på pcen :|
        ingredients = " | ".join([formatIngredient(item) for item in soup.find(attrs={"class": "grid__item cm-module--white one-whole recipe-ingredients"}).find_all(attrs={"itemprop":"ingredients"})])
        time = soup.find(attrs={"data-epi-property-name": "RecipeTime"}).get_text()
        difficulty = soup.find(attrs={"data-epi-property-name": "RecipeDifficulty"}).get_text()

        return createRecipeObject(attributes, title, description, steps, numberOfPeople, ingredients, time, difficulty)
    except:
        pass

    return None

def findRecipes(pageChildren):
    for i, recipe in enumerate(pageChildren, start=1):
        print(f"Recipe {i}")
        recipeURL = recipe.find("a")['href']
        recipeObject = extractRecipe(recipeURL)
        if not recipeObject == None: recipes.append(recipeObject)

def findRecipeURLs(s):
    recipeElements = soup.find(attrs={"id": "recipeListResults"}).children
    urls = [child.find("a")['href'] for child in recipeElements]
    return urls

recipeUrls = []

for p in range(1, maxPage+1):
    print(f"Page {p} of {maxPage} starting...")
    print(url+str(p))
    driver.get(url+str(p))
    driver.refresh()
    WebDriverWait(driver, timeout=3).until(lambda d: d.find_element(By.ID, "recipeListResults"))
    print(len(recipeUrls))
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        recipeUrls += findRecipeURLs(soup)
    except: pass
    print(f"...Page {p} of {maxPage} done, continuing")
    # Max for now
    if len(recipeUrls) > 500:
        break


for i, url in enumerate(recipeUrls):
    print(url)
    try:
        recipeObject = extractRecipe(url)
        if not recipeObject == None: recipes.append(recipeObject)
    except: pass
    if i%10 == 0:
        open("recipes.json", "w").write(json.dumps(recipes, indent=4, ensure_ascii=False))
    

open("recipes.json", "w").write(json.dumps(recipes, indent=4, ensure_ascii=False))