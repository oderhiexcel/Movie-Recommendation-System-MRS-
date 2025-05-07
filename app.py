import pandas as pd
from surprise import Dataset, Reader, SVD
import streamlit as st

# Load ratings from u.data
ratings = pd.read_csv('Data/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Load movie titles from u.item
movies = pd.read_csv('Data/ml-100k/u.item', sep='|', names=['movie_id', 'title'], usecols=[0, 1], encoding='latin-1')

# Prepare data for scikit-surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

# Build the full training set
trainset = data.build_full_trainset()

# Train an SVD model
model = SVD()
model.fit(trainset)

# Function to get recommendations
def get_recommendations(user_id, num_recommendations=10):
    all_movie_ids = movies['movie_id'].tolist()
    predictions = []
    for movie_id in all_movie_ids:
        predicted_rating = model.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = predictions[:num_recommendations]
    top_movies = []
    for movie_id, rating in top_recommendations:
        title = movies[movies['movie_id'] == movie_id]['title'].iloc[0]
        top_movies.append((title, rating))
    return top_movies

# Set up the Streamlit web page
st.title("Movie Recommendation System")
user_id = st.text_input("Enter your User ID (1 to 943):", "1")

# Show recommendations
if user_id:
    try:
        user_id = int(user_id)
        if 1 <= user_id <= 943:
            st.write(f"Top 10 Recommendations for User {user_id}:")
            recommendations = get_recommendations(user_id)
            for i, (title, rating) in enumerate(recommendations, 1):
                st.write(f"{i}. {title} (Predicted Rating: {rating:.2f})")
        else:
            st.write("Please enter a User ID between 1 and 943.")
    except:
        st.write("Please enter a valid number.")