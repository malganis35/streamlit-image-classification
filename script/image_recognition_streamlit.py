# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:17:39 2023

@author: cao-tri.do
"""

import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2
import urllib.request
import os

#%% Charger le modèle depuis le fichier pickle

url = r'https://github.com/malganis35/streamlit-image-classification/blob/master/script/model.pkl?raw=true'    
model_tmp_filename = "model_tmp.pkl"

if os.path.isfile("model.pkl")==False:
    print("Le modèle n'existe pas en local. Téléchargement du modèle")
    urllib.request.urlretrieve(url, model_tmp_filename)
else:
    print("Le modèle est déjà disponible")
    model_tmp_filename = 'model.pkl'
    
model = pickle.load(open(model_tmp_filename, 'rb'))      


#%% Définition de fonctions pour l'application
def classify_image(x_val):
    """
    Fonction permettant de réaliser la classification d'une image
    """
    
    labels = ['Voiture', 'Avion']
    predictions = model.predict(x_val)
    max_indices = np.argmax(predictions, axis=1)
    class_data = [labels[i] for i in max_indices]
    
    return class_data[0]

def load_image(image, img_size = 224 ):
    """
    Fonction permettant de réaliser le chargement de l'image en utilisant une 
    résolution de 224p et gardant la même structure de code que l'apprentissage
    """
    
    # initialise la matrice
    data = []
    # convertit en array
    img_arr = np.array(image)
    # resize la matrice
    resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
    # transforme en un array compréhensible pour la suite
    data.append([resized_arr, 0])
    val = np.array(data)
    
    # boucle pour convertir dans le format de l'apprentisage
    x_val = []
    y_val = []
    
    for feature, label in val:
      x_val.append(feature)
      y_val.append(label)
    
    # Normalize the data
    x_val = np.array(x_val) / 255
    
    # Resize les matrices
    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)
    
    return x_val

#%%

# Configure la page streamlit
st.set_page_config(page_title="Classification d'image", page_icon=":camera:", layout="wide")

# Logo de l'entreprise
st.image("../images/logo_mazars.png", width=200)

# Trait en gris clair
st.markdown("---", unsafe_allow_html=True)

# Titre
st.title("Classification d'image")
st.text("Author: Mazars Data Services - Cao Tri DO, PhD")
st.markdown("*Exemple d'une démonstration pour classifier des images d'avions et de voitures*")
# Trait en gris clair
st.markdown("---", unsafe_allow_html=True)
st.empty()

# Bouton de chargement d'image
uploaded_file = st.file_uploader("Choisissez une image à classer", type=["jpg", "jpeg", "png"])

# Si le fichier n'est pas vide
if uploaded_file is not None:
    
    col1, col2 = st.columns(2)
    
    # Charge l'image et l'affiche à l'écran
    image = Image.open(uploaded_file)
    col2.image(image, caption='Image à classifier', use_column_width=False)

    # Convertit au format du modèle
    x_val = load_image( image )

    # Classification de l'image
    prediction = classify_image(x_val)

    # Affiche le résultat de la prédiction
    # st.write("Prédiction:", prediction)
    original_title = '<p style="font-family:Arial; color:Black; font-size: 20px;">Prediction</p>'
    col1.markdown(original_title, unsafe_allow_html=True)
    
    html_str = f"""
    <style>
    p.a {{
      font: bold 30px Arial;
      color:Red; 
    }}
    </style>
    <p class="a">{prediction}</p>
    """
    col1.markdown(html_str, unsafe_allow_html=True)
    