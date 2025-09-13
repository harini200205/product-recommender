import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------
# Generate Sample Data
# ----------------------
def load_data():
    users = [f"user{i}" for i in range(1, 11)]
    items = pd.DataFrame({
        "item_id": [f"item{i}" for i in range(1, 11)],
        "name": [f"Product {i}" for i in range(1, 11)],
        "category": np.random.choice(["Electronics", "Clothing", "Books"], 10)
    })
    interactions = pd.DataFrame(
        np.random.randint(0, 2, size=(len(users), len(items))),
        index=users,
        columns=items["item_id"]
    )
    return users, items, interactions

# ----------------------
# Recommendation Logic
# ----------------------
def recommend_products(user_id, user_item_matrix, similarity_df, items, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    # Get items interacted by user
    user_ratings = user_item_matrix.loc[user_id]
    interacted_items = user_ratings[user_ratings > 0].index.tolist()

    # Cold start: no interactions → fallback
    if len(interacted_items) == 0:
        popular_items = items.sample(top_n)
        return popular_items.to_dict(orient="records")

    # Collaborative filtering with cosine similarity
    scores = similarity_df[interacted_items].mean(axis=1).sort_values(ascending=False)
    recommended_items = scores.drop(interacted_items).head(top_n).index.tolist()

    # If still empty → fallback
    if not recommended_items:
        popular_items = items.sample(top_n)
        return popular_items.to_dict(orient="records")

    return items[items["item_id"].isin(recommended_items)].to_dict(orient="records")

# ----------------------
# Streamlit UI
# ----------------------
st.title("🛒 E-Commerce Product Recommender")

users, items, interactions = load_data()

# Compute similarity
user_item_matrix = interactions
similarity = cosine_similarity(user_item_matrix.T)
similarity_df = pd.DataFrame(similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Sidebar user selection
selected_user = st.sidebar.selectbox("Select User", users)

# Recommend
recommendations = recommend_products(selected_user, user_item_matrix, similarity_df, items)

st.subheader(f"Recommended Products for {selected_user}:")
if recommendations:
    for rec in recommendations:
        st.write(f"**{rec['name']}** ({rec['category']})")
else:
    st.warning("No recommendations found. Try another user.")
