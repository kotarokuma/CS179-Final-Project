import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

plt.style.use('seaborn-v0_8-whitegrid')

# Load datasets
try:
    ratings_df = pd.read_csv("ratings.csv")
    movies_df = pd.read_csv("movies.csv")
    print("Successfully loaded MovieLens dataset")
except FileNotFoundError:
    print("ERROR: Dataset files not found.")
    exit()

# Basic dataset statistics
print("\n=== DATASET OVERVIEW ===")
n_users = ratings_df['userId'].nunique()
n_movies_rated = ratings_df['movieId'].nunique()
n_ratings = len(ratings_df)
n_movies_total = movies_df['movieId'].nunique()

print(f"Users: {n_users}")
print(f"Movies (total): {n_movies_total}")
print(f"Movies (rated): {n_movies_rated}")
print(f"Total ratings: {n_ratings}")
print(f"Sparsity: {1 - n_ratings / (n_users * n_movies_rated):.4f}")
print(f"Avg ratings per user: {n_ratings/n_users:.1f}")
print(f"Avg ratings per movie: {n_ratings/n_movies_rated:.1f}")

# Rating distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
rating_counts = ratings_df['rating'].value_counts().sort_index()
plt.bar(rating_counts.index, rating_counts.values, color='skyblue', alpha=0.8)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

# User activity distribution
plt.subplot(1, 2, 2)
user_activity = ratings_df.groupby('userId').size()
plt.hist(user_activity, bins=30, color='coral', alpha=0.7, edgecolor='black')
plt.title('User Activity Distribution')
plt.xlabel('Number of Ratings per User')
plt.ylabel('Number of Users')
plt.yscale('log')  # Log scale due to heavy tail

plt.tight_layout()
plt.savefig('basic_distributions.png', dpi=150, bbox_inches='tight')

# Movie popularity analysis
movie_popularity = ratings_df.groupby('movieId').agg({
    'rating': ['count', 'mean', 'std']
}).round(2)
movie_popularity.columns = ['num_ratings', 'avg_rating', 'rating_std']
movie_popularity = movie_popularity.reset_index()

# Merge with movie titles for interpretability
movie_popularity = movie_popularity.merge(movies_df[['movieId', 'title']], on='movieId')

# print(f"\n=== MOVIE POPULARITY ANALYSIS ===")
# print("Most rated movies:")
# print(movie_popularity.nlargest(10, 'num_ratings')[['title', 'num_ratings', 'avg_rating']])

# print("\nHighest rated movies (min 50 ratings):")
# popular_movies = movie_popularity[movie_popularity['num_ratings'] >= 50]
# print(popular_movies.nlargest(10, 'avg_rating')[['title', 'num_ratings', 'avg_rating']])

# Visualize rating vs popularity relationship
plt.figure(figsize=(10, 6))
scatter = plt.scatter(movie_popularity['num_ratings'], movie_popularity['avg_rating'],
                      c=movie_popularity['num_ratings'], cmap='viridis',
                      alpha=0.7, s=20, edgecolor='white', linewidth=0.5)
plt.colorbar(scatter, label='Number of Ratings')
plt.xlabel('Number of Ratings (Popularity)')
plt.ylabel('Average Rating')
plt.title('Movie Popularity vs Average Rating')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('popularity_vs_rating.png', dpi=150, bbox_inches='tight')

# User rating behavior analysis
user_stats = ratings_df.groupby('userId').agg({
    'rating': ['count', 'mean', 'std']
}).round(2)
user_stats.columns = ['num_ratings', 'avg_rating', 'rating_std']

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(user_stats['avg_rating'], bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title('User Average Rating Distribution')
plt.xlabel('Average Rating Given by User')
plt.ylabel('Number of Users')

plt.subplot(1, 3, 2)
plt.hist(user_stats['rating_std'].dropna(), bins=20, color='orange', alpha=0.7, edgecolor='black')
plt.title('User Rating Variance')
plt.xlabel('Std Dev of Ratings Given by User')
plt.ylabel('Number of Users')

plt.subplot(1, 3, 3)
plt.scatter(user_stats['num_ratings'], user_stats['avg_rating'], alpha=0.6, s=15)
plt.xlabel('Number of Ratings Given')
plt.ylabel('User Average Rating')
plt.title('User Activity vs Rating Bias')
plt.xscale('log')

plt.tight_layout()
plt.savefig('user_behavior_analysis.png', dpi=150, bbox_inches='tight')

# Genre analysis
movies_df['genres_list'] = movies_df['genres'].str.split('|')
all_genres = []
for genres in movies_df['genres_list']:
    all_genres.extend(genres)

genre_counts = Counter(all_genres)
top_genres = dict(genre_counts.most_common(10))

plt.figure(figsize=(12, 6))
plt.bar(top_genres.keys(), top_genres.values(), color='purple', alpha=0.8)
plt.title('Top 10 Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('genre_distribution.png', dpi=150, bbox_inches='tight')

# Data preparation insights for filtering
print(f"\n=== FILTERING INSIGHTS ===")
print(f"Matrix dimensions: {n_users} users × {n_movies_rated} movies")
print(f"Matrix sparsity: {(1 - n_ratings/(n_users * n_movies_rated))*100:.2f}%")

# Identify power users and popular movies for potential bias issues. We can adjust these parameters however we want. I just set some random values for demo purposes.
power_users = user_stats.nlargest(20, 'num_ratings').index.tolist()
popular_movies_ids = movie_popularity.nlargest(50, 'num_ratings')['movieId'].tolist()

print(f"Power users (top 20): {len(power_users)} users account for "
      f"{ratings_df[ratings_df['userId'].isin(power_users)].shape[0]/n_ratings*100:.1f}% of ratings")

print(f"Popular movies (top 50): {len(popular_movies_ids)} movies account for "
      f"{ratings_df[ratings_df['movieId'].isin(popular_movies_ids)].shape[0]/n_ratings*100:.1f}% of ratings")


print(f"\n=== SUMMARY ===")
print("Generated visualizations:")
print("- basic_distributions.png: Rating patterns & user activity")
print("- popularity_vs_rating.png: Movie popularity bias analysis")
print("- user_behavior_analysis.png: User rating behavior & biases")
print("- genre_distribution.png: Content distribution")