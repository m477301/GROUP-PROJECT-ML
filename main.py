from bs4 import BeautifulSoup
import requests 
import json
import re
import logging
import sys

# Combining JSON
# ---------------
# data = open('data.json', encoding="utf8")
# books = json.load(data)
# data.close()
# data = open('data9.json', encoding="utf8")
# books_two = json.load(data)
# for book in books_two:
#     books.append(book)

# jsonString = json.dumps(books)
# jsonFile = open("data.json", "w")
# jsonFile.write(jsonString)
# jsonFile.close()
# print(len(books))

# Scraping data with (arg0 = start, arg1 = finish, arg2 = data file number)
books = []
jsonString = json.dumps(books)
jsonFile = open("data" + sys.argv[3] + ".json", "w")
jsonFile.write(jsonString)
jsonFile.close()

start = int(sys.argv[1])
finish = int(sys.argv[2])
for i in range(start, finish):
    if books:
        if any(d['id'] == i for d in books):
            continue
                
    url = f"https://www.goodreads.com/book/show/{i}"

    result = requests.get(url)
    doc = BeautifulSoup(result.text, "html.parser")

    with open('html.txt', 'w', encoding="utf-8") as f:
        f.write(str(doc.prettify()))

    # Type of Html Received
    html = doc.find(class_= "desktop withSiteHeaderTopFullImage")
    title = "N/A"

    if html:
        # Get the title
        h1Title = doc.find(id="bookTitle")
        if h1Title:
            title = str(h1Title.text.strip())
        else:
            continue

        # Get the blurb
        blurbClass = doc.find(class_="readable stacked")
        if blurbClass:
            blurb = str(blurbClass.text.strip())
            blurb = blurb.translate({ord(i): None for i in '"'})
        else:
            continue

        # Get the genres
        actionLinkGenres = doc.find_all(class_="actionLinkLite bookPageGenreLink") 
        if actionLinkGenres:
            genres = []
            for genre in actionLinkGenres:
                if str(genre.text.strip()) != "...more":
                    genres.append(str(genre.text.strip()))
        else:
            continue
        
        # Get author
        authorName = doc.find(class_="authorName")
        if authorName:
            author = str(authorName.text.strip())
        else:
            author = None

        # Get book rating
        bookRating = doc.find("span", itemprop="ratingValue")
        if bookRating:
            rating = str(bookRating.text.strip())
        else:
            rating = None

        # Get pages & format
        PageCount = doc.find("span", itemprop='numberOfPages')
        if PageCount:
            pages = str(PageCount.contents[0].text.strip()).split(" ")[0]
        else:
            pages = None    
    else: 
        # Get the title
        h1Title = doc.find(class_="Text Text__title1")
        if h1Title:
            title = str(h1Title.text.strip())
        else:
            continue

        # Get the blurb
        blurbClass = doc.find(class_="Formatted")
        if blurbClass:
            blurb = str(blurbClass.text.strip())
            blurb = blurb.translate({ord(i): None for i in '"'})
        else:
            continue
        
        # Get the genres of the book
        actionLinkGenres = doc.find_all(class_="Button Button--tag-inline Button--small") 
        if actionLinkGenres:
            genres = []
            for genre in actionLinkGenres:
                if str(genre.text.strip()) != "...more":
                    genres.append(str(genre.text.strip()))
        else:
            continue
        
        # Get author
        authorName = doc.find(class_="ContributorLink__name")
        if authorName:
            author = str(authorName.text.strip())
        else:
            author = None

        # Get book rating
        bookRating = doc.find(class_="RatingStatistics__rating")
        if bookRating:
            rating = str(bookRating.text.strip())
        else:
            rating = None

        # Get pages & format
        extraInfo = doc.find(class_='FeaturedDetails')
        if extraInfo:
            pages = str(extraInfo.contents[0].text.strip()).split(" ")[0]
        else:
            pages = None

    book = {
        "id": i,
        "title": title,
        "blurb": blurb,
        "genre": genres,
        "author": author,
        "rating": rating,
        "pages": pages
    }

    logging.basicConfig(filename='app.log', filemode='w+', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.info("Added Book(" + str(i) + ") to Data.")

    books.append(book)
    jsonString = json.dumps(books)
    jsonFile = open("data" + sys.argv[3] + ".json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print(f"Completed: {((i+1-start)/finish)*100}%")
print(f"Current Number of Datapoints: {len(books)}")