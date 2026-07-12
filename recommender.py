import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import ast
import os

class Recommender:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.movies_df = pd.DataFrame()
        self.cosine_sim = None
        self.indices = None
        self.load_data()

    def parse_genres(self, x):
        try:
            genres_list = ast.literal_eval(x)
            return "|".join([i['name'] for i in genres_list])
        except (ValueError, TypeError):
            return ""

    def load_data(self):
        if os.path.exists(self.data_path):
            self.movies_df = pd.read_csv(self.data_path)
            
            # Extract genres from JSON-like string
            self.movies_df['genre_str'] = self.movies_df['genres'].apply(self.parse_genres)
            self.movies_df['genre'] = self.movies_df['genre_str'] # For the frontend to use

            # Combine genre and overview for content-based filtering
            self.movies_df['content'] = self.movies_df['genre_str'] + " " + self.movies_df['overview'].fillna('')
            
            # TF-IDF Vectorizer
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])
            
            # Compute Cosine Similarity
            self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            
            # Reverse mapping of indices and movie titles
            self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title'].str.lower()).drop_duplicates()
        else:
            print(f"Data file {self.data_path} not found.")

    def get_recommendations(self, title: str, num_recommendations: int = 6):
        title_lower = title.lower()
        if self.indices is None or title_lower not in self.indices:
            return {"error": "Movie not found"}
        
        idx = self.indices[title_lower]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Skip the movie itself (index 0)
        sim_scores = sim_scores[1:num_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = self.movies_df.iloc[movie_indices][['title', 'genre', 'overview']].to_dict(orient='records')
        return recommended_movies
