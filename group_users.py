from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast

# Load existing user data
users = pd.read_csv("C:/Users/abhin/OneDrive/Documents/Recommendation-system/student_recommendations.csv")

# Fix the 'Recommended_Workshops' column: Convert string representation to lists
users['Recommended_Workshops'] = users['Recommended_Workshops'].apply(ast.literal_eval)

# Preprocess the new user's data
#this preprocessing may change wrt to the data
def preprocess_user(new_user):
    new_user['Branch_Interests'] = new_user['Branch'] + ', ' + new_user['Interests']
    if 'Recommended_Workshops' not in new_user:
        new_user['Recommended_Workshops'] = []
    return new_user

# Function to recommend workshops
def recommend_workshops(new_user, users, top_n=2):
    # Preprocess new user
    new_user = preprocess_user(new_user)
    
    # Combine new user data with existing users for TF-IDF vectorization
    combined_data = users['Branch_Interests'].tolist() + [new_user['Branch_Interests']]
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined_data)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Find top N similar users
    similar_user_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    # Collect workshops from similar users
    recommended_workshops = set()
    for idx in similar_user_indices:
        recommended_workshops.update(users.iloc[idx]['Recommended_Workshops'])
    
    # Exclude workshops already attended by the new user
    recommended_workshops.difference_update(new_user['Recommended_Workshops'])
    

# Example new user
new_user = {
    "Name": "Ram",
    "Reg_No": "Student069",
    "Branch": "Computer Science",
    "Interests": "Artificial Intelligence, Data Science, Web Development",
}

# Get recommendations
recommendations = recommend_workshops(new_user, users)
print("Recommended Workshops:", recommendations)