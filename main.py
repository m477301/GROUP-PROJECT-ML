from bs4 import BeautifulSoup
import requests 
import json
import re

start = 1
finish = 50
for i in range(start, finish):
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
        blurb = str(blurbClass.text.strip())
        blurb = blurb.translate({ord(i): None for i in '"'})

        # Get the genres
        actionLinkGenres = doc.find_all(class_="actionLinkLite bookPageGenreLink") 
        genres = []
        for genre in actionLinkGenres:
            if str(genre.text.strip()) != "...more":
                genres.append(str(genre.text.strip()))
        
        # Get author
        authorName = doc.find(class_="authorName")
        author = str(authorName.text.strip())

        # Get book rating
        bookRating = doc.find("span", itemprop="ratingValue")
        rating = str(bookRating.text.strip())

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
        blurb = str(blurbClass.text.strip())
        blurb = blurb.translate({ord(i): None for i in '"'})
        
        # Get the genres of the book
        actionLinkGenres = doc.find_all(class_="Button Button--tag-inline Button--small") 
        genres = []
        for genre in actionLinkGenres:
            if str(genre.text.strip()) != "...more":
                genres.append(str(genre.text.strip()))
        
        # Get author
        authorName = doc.find(class_="ContributorLink__name")
        author = str(authorName.text.strip())

        # Get book rating
        bookRating = doc.find(class_="RatingStatistics__rating")
        rating = str(bookRating.text.strip())

        # Get pages & format
        extraInfo = doc.find(class_='FeaturedDetails')
        pages = str(extraInfo.contents[0].text.strip()).split(" ")[0]

    book = {
        "id": i,
        "title": title,
        "blurb": blurb,
        "genre": genres,
        "author": author,
        "rating": rating,
        "pages": pages
    }

    data = open('data.json', encoding="utf8")
    books = json.load(data)
    data.close()
    books.append(book)
    jsonString = json.dumps(books)
    jsonFile = open("data.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print(f"Completed: {(i/finish)*100}%")