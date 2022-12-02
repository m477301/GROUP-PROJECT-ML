import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# load data
df = pd.read_json("data.json")
# isolate genres array
temp = df.get("genre")
genres = []
for i in range(0, len(temp), 1):  # for(i = 0; i < len(temp); i++)
    # get first genre and add to ne genre array
    if (len(temp[i]) != 0):
        genres.append(temp[i][0])
    # set points with no genres data to noGenre
    else:
        genres.append("NoGenre")

# remove genre arrays from all data points
df.drop("genre", axis=1, inplace=True)
# create new column were every data point only has one genre
df["genre"] = genres
# get arraay of all genres
allGenres = df.get("genre")

# get all unique genres
uniqueGenres = df.genre.unique()
print(len(uniqueGenres))

# assign unique genres number
genreDict = {}
index = 0
for i in range(0, len(uniqueGenres), 1):
    genreDict[uniqueGenres[i]] = i

genreNumbers = [None] * len(allGenres)
print(allGenres)
for i in range(0, len(allGenres), 1):
    result_vector = np.zeros(len(uniqueGenres))
    result_vector[genreDict[allGenres[i]]] = 1
    genreNumbers[i] = result_vector


df['GenreNumber'] = genreNumbers
print(df.head(5))
X_train, X_test, y_train, y_test = train_test_split(
    df["blurb"], df["GenreNumber"], test_size=0.2)

# print(y_train[0])
