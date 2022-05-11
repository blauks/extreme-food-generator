from time import sleep
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
        stepText = step.get_text().replace("\n", "").replace(u"\xa0", "")
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
    
    steps = ""
    try:
        steps = formatSteps(soup.find(attrs={"class": "recipe-steps"}).find_all(attrs={"class": "recipe-steps__item"}))
    except:
        steps = formatSteps(soup.find(attrs={"class": "new-recipe-details__header new-recipe-details__header--left"}).find_next("ol").find_all("li"))
    
    numberOfPeople = ""
    try:
        numberOfPeople = soup.find(attrs={"id": "portionsInput"})["value"]
    except:
        numberOfPeople = soup.find(attrs={"class": "portions-label"})
        if numberOfPeople == None:
            numberOfPeople = ""
        else:
            numberOfPeople = numberOfPeople.get_text().strip()
        
    # Matprat skal selvfølgelig drive å ha både mobil og pc HTML i samme fil, bare i tilfelle noen plutselig bytter over til mobilen deres mens de ser på oppskrifter på pcen :|
    ingredients = [formatIngredient(item) for item in soup.find(attrs={"class": "grid__item cm-module--white one-whole recipe-ingredients"}).find_all(attrs={"itemprop":"ingredients"})]
    
    time = soup.find(attrs={"data-epi-property-name": "RecipeTime"})
    if time == None:
        time = ""
    else: 
        time = time.get_text()

    difficulty = soup.find(attrs={"data-epi-property-name": "RecipeDifficulty"})
    if difficulty == None:
        difficulty = ""
    else: 
        difficulty = difficulty.get_text()

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
    data = json.loads(requests.get(f"https://www.matprat.no/api/RecipeSearch/GetRecipes?text=&page={str(p)}&sort=new", headers={'referer': url+str(p)}).content)
    recipeUrls += findRecipeURLs(data["searchHits"])
    print(len(recipeUrls))
    print(f"...Page {p} of {maxPage} done, continuing")

for i, url in enumerate(recipeUrls):
    print(f"{i} of {len(recipeUrls)}: ",url)
    try:
        recipeObject = extractRecipe(url)
        if not recipeObject == None: recipes.append(recipeObject) 
    except:
        print("DIDN'T WORK, NEXT ONE")

open("recipesTEST.json", "w").write(json.dumps(recipes, indent=4, ensure_ascii=False))