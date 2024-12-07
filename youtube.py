# video_recommender.py

import requests
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import numpy as np

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Extract Topics
def extract_topics(text, n_topics=5):
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    text_vectorized = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(text_vectorized)

    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topic_keywords = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
        topics.append(" ".join(topic_keywords))
    return topics

# Step 2: Search YouTube Videos using SerpApi
def search_youtube_for_topics(topics, max_results=5):
    video_suggestions = {}
    for topic in topics:
        response = requests.get(
            f"https://serpapi.com/search.json?engine=youtube&search_query={topic}&num={max_results}&api_key= cannot_show_api_publically"
        )
        search_response = response.json()

        videos = []
        for item in search_response.get('video_results', []):
            videos.append({
                'title': item['title'],
                'description': item['description'],
                'url': item['link'],
                'thumbnail': item['thumbnail']
            })
        video_suggestions[topic] = videos
    return video_suggestions

# Step 3: Embed Topics and Cluster Videos
def cluster_topics_and_videos(topics, video_suggestions, n_clusters=3):
    topic_embeddings = embedding_model.encode(topics, convert_to_tensor=True)
    video_embeddings = []

    video_data = []
    for topic, videos in video_suggestions.items():
        for video in videos:
            video_text = f"{video['title']} {video['description']}"
            video_embedding = embedding_model.encode(video_text, convert_to_tensor=True)
            video_embeddings.append(video_embedding)
            video_data.append({'topic': topic, 'video': video})

    # Perform clustering
    embeddings = np.vstack([topic_embeddings.cpu().numpy()] + [ve.cpu().numpy() for ve in video_embeddings])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Assign clusters
    clustered_videos = {f"Cluster {i}": [] for i in range(n_clusters)}
    for idx, label in enumerate(labels[len(topics):]):
        clustered_videos[f"Cluster {label}"].append(video_data[idx]['video'])

    return clustered_videos