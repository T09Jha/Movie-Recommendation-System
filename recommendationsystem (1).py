import numpy as np
import matplotlib.pyplot as mplot
import pandas as pd
import sklearn as sk
import seaborn as sns

df=pd.read_csv('dataset.csv')

df.columns

missing=df.isnull().sum()
print(missing)

df.columns

df['tags']=df['genre']+df['overview']

columns_to_drop = ['genre', 'overview']

df = df.drop(columns=columns_to_drop)
df.info()

def fill_genre(row):
    if pd.isnull(row['tags']):
        common_genre = df[df['release_date'] == row['release_date']]['tags'].mode()
        return common_genre[0] if not common_genre.empty else 'Unknown'
    return row['tags']
df['tags'] = df.apply(fill_genre, axis=1)
print(df[['tags', 'release_date']].head())

missing=df.isnull().sum()
print(missing)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['tags'])

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
sim=cosine_similarity(tfidf_matrix)

np.testing.assert_allclose(cosine_sim, sim, rtol=1e-5, atol=1e-8)

cosine_sim

def recommend(movie_title):
    try:
        index = df[df['title'] == movie_title].index[0]
    except IndexError:
        print(f"Movie '{movie_title}' not found in the dataset! Give Exact keyword!")
        return
    index = df[df['title'] == movie_title].index[0]
    scoring = list(enumerate(cosine_sim[index]))
    compare = sorted(scoring, reverse=True, key=lambda tfidf: tfidf[1])
    recommendations = compare[1:11]
    for i in recommendations:
        movie_details = df.iloc[i[0]]
        print(f"Title: {movie_details['title']}")
        print(f"Original Language: {movie_details['original_language']}")
        print(f"Popularity: {movie_details['popularity']}")
        print(f"Overview: {movie_details['tags']}")
        print(f"Release Date: {movie_details['release_date']}")
        print("\n")

movie=input("Enter exact movie you want to be recommended: ")
print(recommend(movie))
