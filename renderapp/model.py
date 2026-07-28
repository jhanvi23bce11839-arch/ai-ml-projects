import pandas as pd
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies dataset
movies = pd.read_csv("tmdb_5000_movies.csv")

# Load credits dataset from zip file
import zipfile

with zipfile.ZipFile("tmdb_5000_credits.zip", "r") as zip_ref:
    zip_ref.extractall("credits_data")

credits = pd.read_csv("credits_data/tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Keep required columns
movies = movies[['movie_id',
                 'title',
                 'overview',
                 'genres',
                 'keywords',
                 'cast',
                 'crew']]

# Remove missing values
movies.dropna(inplace=True)
def convert(text):

    L = []

    for i in ast.literal_eval(text):
        L.append(i['name'])

    return L


def convert_cast(text):

    L = []

    counter = 0

    for i in ast.literal_eval(text):

        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break

    return L


def fetch_director(text):

    L = []

    for i in ast.literal_eval(text):

        if i['job'] == 'Director':
            L.append(i['name'])

    return L

    movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

movies['cast'] = movies['cast'].apply(convert_cast)

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):

    movie = movie.lower()

    index = new_df[
        new_df['title'].str.lower() == movie
    ].index

    if len(index) == 0:
        return []

    index = index[0]

    distances = list(enumerate(similarity[index]))

    movies_list = sorted(
        distances,
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movies_list:
        recommendations.append(
            new_df.iloc[i[0]].title
        )

    return recommendations
    print(recommend("Avatar"))