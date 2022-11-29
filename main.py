from bs4 import BeautifulSoup
import requests 
import json

books = []

url = "https://www.goodreads.com/book/show/5100"

result = requests.get(url)
doc = BeautifulSoup(result.text, "html.parser")

with open('html.txt', 'w', encoding="utf-8") as f:
    f.write(str(doc.prettify()))

# Type of Html Received
html = doc.find(class_= "desktop withSiteHeaderTopFullImage")
title = "N/A"

if html:
    h1Title = doc.find(id="bookTitle")
    title = str(h1Title.text.strip())
else: 
    h1Title = doc.find(class_="Text Text__title1")
    title = str(h1Title.text.strip())

book = {
    "title": title
}

books.append(book)

jsonString = json.dumps(books)
jsonFile = open("data.json", "w")
jsonFile.write(jsonString)
jsonFile.close()

# SAMPLE JSON:

# {
# title: title_name
#   
# }

