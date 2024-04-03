import config
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import cv2

from importlib import reload
from sklearn.metrics import f1_score
from importlib import reload


from src.utils import results
from src.utils import scrapper
from src.utils.visualize import plot_weighted_text
from src.features.text.transformers.text_merger import TextMerger
from src.features.text.transformers.translators import TextTranslator
from src.features.text.pipelines.cleaner import CleanTextPipeline
from src.features.text.pipelines.corrector import CleanEncodingPipeline

# chargement des Ressources
# DATAFRAMES
df = pd.read_csv("../data/raw/X_train.csv")
df_train_clean = pd.read_csv("../data/clean/df_train_index.csv")
index_column = "Unnamed: 0"
df_train_raw = pd.read_csv("../data/raw/X_train.csv")
df_train_raw.set_index(index_column, inplace=True)
ytrain = pd.read_csv("../data/raw/Y_train.csv")


# chargement images
schema_dataframe = "images/schema_dataframe.png"
schema_image = "images/schema_images.png"
schema_dataframe_Y = ("images/schema_dataframe_Y.png")
schema_prepro_txt = "images/schema_prepro_txt.jpg"
schema_prepro_img = "images/schema_prepro_img.jpg"
schema_objectifs = ("images/schema_objectifs.jpg")
graf_isnaPrdt = "images/graf_isnaPrdtypecode.png"
graf_txtLong = "images/graf_boxplot.png"
graf_lang = "images/lang.jpg"
graf_corr = "images/corr.jpg"
graf_WC = "images/maskWC.png"
img_rakuten_website = "images/rakuten_website.png"
corr = 'images/corr_cat.jpg'
image_BERT = 'images/image_BERT.png'
image_ResNet152 = 'images/image_ResNet152.png'
image_ViT = 'images/image_ViT.png'
image_simpleVoting = 'images/image_simpleVoting.png'
image_fusionTF = 'images/image_fusionTF.png'
image_metaVoting = 'images/image_metaVoting.png'
image_bestmetaVoting = 'images/fusion-contribs.jpg'
image_yanniv = 'images/yanniv.jpg'
image_axalia = 'images/axalia.jpg'
image_aida = 'images/aida.jpg'


# dossier images
wc_folder = "images/wc_visuels"

# models specifics
best_voting_models = [
    'fusion/camembert-base-vit_b16_TF6',
    'text/xgboost_tfidf',
    'text/camembert-base-ccnet',
    'image/vit_b16',
    'text/flaubert_base_uncased',
    'image/ResNet152'
]
fusion_model = 'fusion/camembert-base-vit_b16_TF6'

# css custom pour les typo / les bocs etc
custom_css = """
<style>
    /* Styles pour spécifier la taille du texte */
    body {
        font-size: 16px; /* Taille de la police pour tout le texte */
        font-family: 'Roboto', sans-serif; /* Utiliser Roboto pour tout le texte */
        background-color: #eee;
    }
    h1 {
        font-size: 40px; /* Taille de la police pour les titres de niveau 1 */
        font-family: 'Roboto', sans-serif; /* Utiliser Roboto pour tout le texte */
    }
    h2 {
        font-size: 28px; /* Taille de la police pour les titres de niveau 2 */
        font-family: 'Roboto Light', sans-serif; /* Utiliser Roboto pour tout le texte */
    }
    p {
        font-size: 16px; /* Taille de la police pour les paragraphes */
        font-family: 'Roboto Light', sans-serif; /* Utiliser Roboto pour tout le texte */
    }
    
    /* Styles pour les images */
    img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    }

    .img-scale-small {
        width: 200px; /* Définir la largeur de l'image */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    .img-scale-medium {
        width: 400px; /* Définir la largeur de l'image */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    .img-scale-large {
        width: 600px; /* Définir la largeur de l'image */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Styles pour les blocs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:20px;
    }

    .container {
    display: flex;
    justify-content: center;
    align-items: center;
    }
    
    .expander {
    display: flex;
    justify-content: center;
    align-items: center;
    }
    
    .expander-content {
        font-size: 10px; /* Taille de police pour le contenu de l'expander */
    }

    .stTabs [data-baseweb="tab-list"] {
		gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
		height: 70px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 10px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #bf0203;
        color: white;
	}

    .stTabs-content {
        font-size: 10px;
    }
    
    .streamlit-tabs {
        font-size: 20px;
    }
    
    div.st-emotion-cache-16txtl3 {
        padding: 2rem 2rem;
    }

    .block-container {
        padding-top: 1rem;
    }
    
    
   
</style>
"""
st.set_page_config(layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)


# DEBUT CODE STREAMLIT************************************************************************************************************

# SOMMAIRE
st.sidebar.image("images/rakuten.png", use_column_width=True)
st.sidebar.title("Sommaire")
pages = ["Présentation", "Exploration", "DataViz", 'Préprocessing', "Modélisation texte",
         "Modélisation images", "Modélisation fusion", "Test du modèle", "Conclusion"]
page = st.sidebar.radio("Aller vers", pages)
st.sidebar.header("Auteurs")
st.sidebar.markdown("[Thibaud Benoist](link)")
st.sidebar.markdown("[Julien Chanson](link)")
st.sidebar.markdown("[Julien Fournier](link)")
st.sidebar.markdown("[Alexandre Mangwa](link)")


@st.cache_resource
def get_results_manager():
    try:
        if res:
            print('res already loaded')
    except NameError:
        res = results.ResultsManager(config)
        res.add_result_file(config.path_to_results +
                            '/results_benchmark_cbow.csv', 'text')
        res.add_result_file(config.path_to_results +
                            '/results_benchmark_skipgram.csv', 'text')
        res.add_result_file(config.path_to_results +
                            '/results_benchmark_tfidf.csv', 'text')
        res.add_result_file(config.path_to_results +
                            '/results_benchmark_bert.csv', 'bert')
        res.add_result_file(config.path_to_results +
                            '/results_benchmark_img.csv', 'img')
        res.add_result_file(
            config.path_to_results+'/results_benchmark_fusion_TF.csv', 'fusion')
        res.add_result_file(
            config.path_to_results+'/results_benchmark_fusion_meta.csv', 'fusion')

    return res


# res = get_results_manager()
# page 0############################################################################################################################################
if page == pages[0]:
    st.title("PRÉSENTATION DU PROJET")
    tab1, tab2, tab3 = st.tabs(["Contexte", "Objectifs", "Résultats"])

    with tab1:
        col1, col2, col3 = st.columns([3, 1, 2])
        with col1:
            st.image("images/rakuten.png", width=200)
            st.markdown(
                """
    La marketplace Rakuten est une plateforme de vente en ligne ouverte à de nombreux vendeurs.
    """)

            st.markdown("""
                        Un des enjeux majeurs de la marketplace est de permettre aux **acheteurs de trouver facilement les produits qu’ils recherchent.**

                        Pour cela, il est essentiel que les produits soient bien classés dans des catégories pertinentes.
                        
                        Le challenge Rakuten est disponible [en ligne](https://challengedata.ens.fr/participants/challenges/35/)
    """)

            st.markdown("""
                    ####      - 80 000 produits
                    ####      - 27 catégories à distinguer
                    ####      - Description textuelle multilangue
                        """)

        with col3:
            st.header("> Puériculture")
            st.write("Porte bébé Violet et rouge Trois-en-un mère multifonctions Kangourou fermeture à glissière Hoodie Taille: XL Poitrine: 104-109 cm 84-88 cm Hanche: 110-116 cm clair + 1. Marque nouvelle et de haute qualité. 2. Détachable conception pratique et attentionnée. 3. Parfait pour les mères qui allaitent. 4. Anti-vent chaud et style kangourou multifonctionnel haut de gamme. 5. Sac de couchage multifonction amovible de la mère européenne. Spécification: Les types Fermez Buste104-109cm Encolure Sweat à capuche Les hanches110-116cm Tailles disponiblesXLMatériel Coton")
            st.divider()
            st.image("images/image_sample.jpg", width=300)

    with tab2:
        st.markdown(
            """
            
             - **Produire un modèle capable de classifier précisément (au sens du weighted f1-score) chacun des produits.**
             - **Produire un modèle robuste**
            - **Produire un modèle multi-modal (texte + image).**
            
            """)
        st.image("images/process.jpg",
                 caption="Processus de classification multi-modale des produits")
    with tab3:
        st.balloons()
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(
                """
            ### Meilleur modèle
            
            VOTING CLASSIFIER basé sur :
            - Un transformer cross-attentionnel CamemBERT + ViT
            - Un CamemBERT
            - Un FlauBERT
            - Un XGBoost (TF-IDF)
            - Un ViT
            - Un ResNET152

            La fusion a été pondérée par les scores f1 des modèles individuels.

            """)
        with col2:
            st.image('images/models.jpg', width=1200,
                     caption="f1 scores des modèles combinés")


