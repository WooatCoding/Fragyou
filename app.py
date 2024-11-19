import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Cargar el dataset
file_url = "https://raw.githubusercontent.com/WooatCoding/Fragyou/main/FYDT.csv"
data = pd.read_csv(file_url, encoding='UTF-8')

data = data.drop(columns=['Image URL'])
data = data.drop_duplicates(subset='Name', keep='first')

data['Name'] = data['Name'].str.strip()
data['Brand'] = data['Brand'].str.strip()
data['Description'] = data['Description'].str.strip()
data['Notes'] = data['Notes'].str.strip()

data = data.dropna(subset=['Notes'])

data['Name'] = data['Name'].str.lower()
data['Brand'] = data['Brand'].str.lower()
data['Description'] = data['Description'].str.lower()
data['Notes'] = data['Notes'].str.lower()

# Preprocesar las descripciones y las notas
data['Notes'] = data['Notes'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Vectorización de las descripciones utilizando TF-IDF
tfidf_description = TfidfVectorizer(stop_words='english')
description_matrix = tfidf_description.fit_transform(data['Description'])

# Vectorización de las notas utilizando TF-IDF
tfidf_notes = TfidfVectorizer(stop_words='english')
notes_matrix = tfidf_notes.fit_transform(data['Notes'].fillna(''))

# Combinar ambas matrices (descripción + notas)
final_matrix = hstack([description_matrix, notes_matrix])

# Función para hacer recomendaciones basadas en las preferencias del usuario
def recommend_perfume(user_preferences, k=10):
    user_description_vector = tfidf_description.transform([user_preferences['Description']])
    user_notes_vector = tfidf_notes.transform([', '.join(user_preferences['Notes'])])
    user_features = hstack([user_description_vector, user_notes_vector])
    
    similarity_scores = cosine_similarity(user_features, final_matrix)
    
    similar_perfumes_indices = similarity_scores.argsort()[0][-k:][::-1]
    recommendations = data.iloc[similar_perfumes_indices][['Name', 'Brand', 'Description']]
    
    return recommendations

# Título de la aplicación
st.title('Frag You')

# Ingreso de preferencias del usuario
st.header('Cuéntanos que te gusta')

description_input = st.text_area("Descripción de perfumes que le gustan")
notes_input = st.text_input("Notas que le gustan (separadas por coma, ej. vainilla, almendra, cítrico)")

# Procesar las preferencias del usuario
if st.button('Obtener Recomendaciones'):
    if description_input and notes_input:
        user_preferences = {
            'Description': description_input,
            'Notes': notes_input.split(', ')  # Convertir las notas ingresadas a una lista
        }
        
        # Obtener las recomendaciones para el usuario
        recommendations = recommend_perfume(user_preferences)
        
        # Mostrar las recomendaciones
        st.write('**Recomendaciones para ti:**')
        for idx, row in recommendations.iterrows():
            st.write(f"**{row['Name']}** - {row['Brand']}")
            st.write(f"Descripción: {row['Description']}")
            st.write('---')
    else:
        st.error("Por favor, ingrese tanto la descripción como las notas.")
