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
reload(results)
reload(scrapper)


# chargement des Ressources
# DATAFRAMES
df = pd.read_csv("../data/raw/X_train.csv")
df_train_clean = pd.read_csv("../data/clean/df_train_index.csv")
ytrain = pd.read_csv("../data/raw/Y_train.csv")


# chargement images
schema_dataframe = "images/schema_dataframe.png"
schema_image = "images/schema_images.png"
schema_dataframe_Y = ("images/schema_dataframe_Y.png")
schema_prepro_txt = "images/schema_prepro_txt.jpg"
schema_prepro_img = "images/schema_prepro_img.jpg"
schema_objectifs= ("images/schema_objectifs.jpg")
graf_isnaPrdt = "images/graf_isnaPrdtypecode.png"
graf_txtLong = "images/graf_boxplot.png"
graf_lang = "images/lang.jpg"
graf_corr = "images/corr.jpg"
graf_WC = "images/maskWC.png"
img_rakuten_website = "images/rakuten_website.png"
corr = 'images/corr_cat.jpg'

# dossier images
wc_folder = "Images/wc_visuels"



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
    
    
   
</style>
"""
st.set_page_config(layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)


# info streamlit
# pour entrer un graph ou image spécifier use_column_width=True ou container width


# DEBUT CODE STREAMLIT************************************************************************************************************

# LOGO RAKUTEN // toutes pages
# col1, col2, col3 = st.columns([1, 1, 1])
# with col1:
#     st.write("")
# with col2:
#     print('toto', os.getcwd())
#     image_path = "images/rakuten.png"  # Example image path
#     image = Image.open(image_path)
#     resized_image = image.resize((int(image.width * 1), int(image.height * 1)))
#     st.image(resized_image)
# with col3:
#     st.write("")


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


def get_results_manager():
    res = results.ResultsManager(config)
    res.add_result_file(config.path_to_results +
                        '/results_benchmark_sklearn.csv', 'text')
    res.add_result_file(config.path_to_results +
                        '/results_benchmark_sklearn_tfidf.csv', 'text')
    res.add_result_file(config.path_to_results +
                        '/results_benchmark_bert.csv', 'bert')
    res.add_result_file(config.path_to_results +
                        '/results_benchmark_img.csv', 'img')
    res.add_result_file(
        config.path_to_results+'/results_benchmark_fusion_TF.csv', 'fusion')
    res.add_result_file(
        config.path_to_results+'/results_benchmark_fusion_meta.csv', 'fusion')
    return res


# page 0############################################################################################################################################
if page == pages[0]:
    tab1, tab2, tab3 = st.tabs(["Contexte", "Objectifs", "Résultats"])

    with tab1:
        st.image("images/rakuten.png", width=200)
        st.markdown(
            """
 ### La marketplace Rakuten est une plateforme de vente en ligne ouverte à de nombreux vendeurs.
""")

        st.markdown("""
                    ### Un des enjeux majeurs de la marketplace est de permettre aux acheteurs de trouver facilement les produits qu’ils recherchent.

                    ### Pour cela, il est essentiel que les produits soient bien classés dans des catégories pertinentes.
                    
                    ### Le challenge Rakuten est disponible [en ligne](https://challengedata.ens.fr/participants/challenges/35/)
""")
        col1, col2 = st.columns([1, 10])
        with col2:
            st.markdown("""
                    ###      - 80 000 produits
                    ###      - 27 catégories à distinguer
                    ###      - Description textuelle multilangue
                        """)

    with tab2:
        st.markdown(
            """
            ###
            ### - Produire un modèle capable de classifier précisément (au sens du weighted f1-score) chacun des produits.
            ### - Produire un modèle robuste
            ### - Produire un modèle multi-modal (texte + image).
            ###
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
    # col1, col2, col3 = st.columns([1, 1, 1])
    # with col1:
    #     st.write("")
    # with col2:
    #     st.markdown("""
    #         <div style="text-align:center;">
    #             <h1>PRESENTATION</h1>
    #         </div>
    #     """, unsafe_allow_html=True)
    # with col3:
    #     st.write("\n")

    # st.header("Contexte :")
    # st.markdown(
    #     """
    # Le concept fondamental d’une marketplace réside dans sa capacité à mettre en relation vendeurs et acheteurs par le biais
    # d’une plateforme en ligne unique, simplifiant ainsi le processus d’achat et de vente d’une vaste gamme de produits.
    # Pour qu’elle soit efficace, il est crucial que les produits soient aisément identifiables, que les utilisateurs bénéficient
    # d’une navigation fluide tout au long de leur parcours d’achat et que la plateforme offre des recommandations personnalisées alignées
    # sur le comportement des utilisateurs.
    # Un aspect essentiel du bon fonctionnement d’une marketplace est donc l’organisation méthodique des produits dans des catégories précises,
    # facilitant la recherche et la découvrabilité des produits.
    # """)
    # col1, col2, col3 = st.columns([1, 1, 1])
    # with col1:
    #     st.write("")
    # with col2:
    #     st.image(img_rakuten_website)
    # with col3:
    #     st.write("\n")

    # st.markdown(
    #     """

    # Ce processus de classification des produits requiert l’application de techniques de machine learning (ML) pour plusieurs raisons essentielles :
    # les vendeurs pourraient (pour diverses raisons) ne pas assigner les nouveaux produits aux catégories pertinentes,
    # introduisant des erreurs dans l’organisation du catalogue
    # le classement manuel des produits peut s'avérer particulièrement fastidieux et inefficace lors de l’ajout massif d’articles.
    # les catégories peuvent être amenées à être modifiées pour améliorer l'expérience d’achat des utilisateurs,
    # impliquant une mise à jour sur l’ensemble du catalogue.

    # L’utilisation de technique de machine learning permet de surmonter ces problèmes éventuels
    # en automatisant tout ou partie de la catégorisation des produits sur la base des descriptions et images fournies par les vendeurs.

    # Le dataset utilisé dans ce projet est issu d’un challenge proposé par le groupe Rakuten, l’un des acteurs majeurs du marketplace B2B2C.
    # Il se compose d’un catalogue d’environ 80.000 produits, répartis selon 27 catégories distinctes, accompagnés de leurs descriptions textuelles
    # et images correspondantes. Voici le lien du concours [lien vers concours](https://challengedata.ens.fr/challenges/59)
    #     """
    # )
    # st.header("Objectifs :")
    # st.markdown(
    #     """
    # L’objectif principal est de développer le **meilleur modèle capable de classifier précisément (au sens du f1-score)**
    # chacun des produits en se basant sur les descriptions et images fournies.
    # Cela induit de développer plusieurs modèles et de les benchmarker afin de sélectionner le modèle le plus performant et le plus robuste.

    # L’ensemble des membres du groupe projet étant débutant dans le domaine du **NLP (Natural Language Processing)** et de la **CV (Computer Vision)**.
    # Nous avons à comprendre et appliquer ces techniques au projet.

    #     """
    # )


# page 1 ##########################################################################################################################
if page == pages[1]:
    tab1, tab2 = st.tabs(["Dataframes", "Images",])
    with tab1:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.write("")
        with col2:
            st.markdown("""
                <div style="text-align:center;">
                    <h1>EXPLORATION</h1>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.write("\n")

        
        col1, col2, col3 = st.columns([2, 4, 1])
        with col1:
            # Problématique
            st.header("Ressources fournies : dataframes")
            st.markdown(
                """
            Nous avons plusieurs elements fournis pour le test :
            - 1 dataframe X_train composé de 84916 produits
            - 1 dataframe X_test composé de 13812 produits
            - 1 dataframe y_train : avec les code produits
            
            - 1 dossier d'images train avec 84916 images
            - 1 dossier d'images test avec 13812 images
            
            - 27 codes produits
            """
            )
        with col2:
            container = st.container()
            with st.container():
                st.image(schema_dataframe,
                        output_format='auto', use_column_width=True)
        
        with col3:
            st.image(schema_dataframe_Y, use_column_width=True,
                    caption="", output_format='auto')

    

        
        col1, col2, col3 = st.columns([1, 7, 1])
        with col1:
            st.write('')
        with col2:
            st.markdown("""
                        Sur les données concernant les dataframes nous avons de nombreux NaN dans la colonnes description, 
                        plusieurs langues, des balises HTML, des mauvais encodings
                        """
                        )
        with col3:
            st.write("")
    with tab2:
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.header("Ressources fournies : images")
            st.markdown(
                """
            Pour les images :
            - toutes en 500*500
            - que des jpeg
            
            - certaines avec un contour blanc tres important
            """
            )
        with col2:
            st.image(schema_image, width=600, use_column_width=False,
                    caption="", output_format='auto')

    # Objectifs
    
    
    
    st.header("Etapes suivantes")
    col1, col2, col3 = st.columns([1, 7, 1])
    with col1:
        st.write("")
    with col2:
        st.image(schema_objectifs, caption="")
    with col3:
        st.write("")


 # page 2 ##########################################################################################################################
if page == pages[2]:
    # Dataviz
    st.title("DATAVIZ")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Produits par catégories", "Articles sans descriptions par catégories",
                                                 "Longeur des textes par catégories", "Langues par catégories", 'Corrélation entre catégories', 'Wordclouds'])

    with tab1:
        st.header("Produits par catégories")
        st.markdown(
            """
        **Déséquilibre de classes** : Les catégories de produit du jeu de données affichent un déséquilibre notable, 
        allant de moins de 100 articles pour certaines catégories telles que figurines, confiserie ou vêtements pour enfants, 
        jusqu'à plusieurs milliers d’articles pour des catégories comme le mobilier ou les accessoires de piscine
        """
        )
        
        
        

        df_train_clean['categorie'] = df_train_clean['prdtypefull'].str.split(
            ' - ').str[1]
        nb_categories = df_train_clean['categorie'].value_counts()
        nb_categories_sorted = nb_categories.sort_values()
        fig = px.bar(
            x=nb_categories_sorted.index,
            y=nb_categories_sorted.values,
            title="Nombre de produits par catégorie (trié)",
            labels={"x": "Catégories", "y": "Nombre de produits"},
            color=nb_categories_sorted,
            color_discrete_sequence=px.colors.sequential.Viridis,
            width=1400,  # spécifier la largeur du graphique
            height=600,  # spécifier la hauteur du graphique
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=15))
        st.plotly_chart(fig)

        

    with tab2:
        st.header("Articles sans descriptions par catégories")

        st.markdown(
            """
        **Déséquilibre de classes** : Certaines produits n'ont pas de descriptions, on retrouve ici majoritairement les livres BD, les magazines d'occasion, les cartes de jeux.
        """
        )

        

        # Compter le nombre de produits par catégorie
        cat_count = df_train_clean["prdtypefull"].value_counts()
        df = df_train_clean.loc[df_train_clean['description'].isna()]
        df['prdtypefull'] = df_train_clean['prdtypefull'].str.split(
            ' - ').str[1]
        df_count = df['prdtypefull'].value_counts().sort_values()

        # Créer un graphique à barres avec plotly express
        fig = px.bar(
            x=df_count.index,
            y=df_count.values,
            title="Categories avec le plus de valeurs manquantes en description",
            labels={"x": "Catégories", "y": "Nombre de produits"},
            color=df_count.values,
            color_continuous_scale='viridis',
            width=1400,
            height=600,
        )

        # Mettre à jour les étiquettes de l'axe x
        fig.update_xaxes(tickangle=45, tickfont=dict(size=15))

        # Afficher le graphique
        st.plotly_chart(fig)

        

    with tab3:
        st.header("Longeur des textes par catégories")

        st.markdown(
            """
        **Déséquilibre de textes** : La longueur des descriptions textuelles montrent une variabilité importante. 
        Les descriptions des livres d’occasion ou des cartes de jeux sont généralement brèves 
        (quelques dizaines de mots, champ description absent), tandis que celles des jeux vidéo pour PC 
        s'étendent souvent sur plusieurs centaines de mots.
        """
        )

        
        
       

        df_train_clean['longeur'] = (
            df_train_clean["designation_translated"] + df_train_clean["description_translated"]).astype(str)
        df_train_clean['longeur_val'] = df_train_clean['longeur'].apply(
            lambda x: len(x))
        df_train_clean['prdtypefull'] = df_train_clean['prdtypefull'].str.split(
            ' - ').str[1]

        # Créer un graphique à barres avec plotly express
        fig = px.box(df_train_clean,
                        x='prdtypefull',
                        y='longeur_val',
                        title="Longeur des textes par catégories",
                        labels={'prdtypefull': "Catégories",
                                'longeur_val': "Nombre de mots"},

                        width=1400,
                        height=600,
                        )

        # Mettre à jour les étiquettes de l'axe x
        fig.update_xaxes(tickangle=45, tickfont=dict(size=15))

        # Afficher le graphique
        st.plotly_chart(fig)

        

    with tab4:
        st.header("Langues par catégories")

        st.markdown(
            """
        **Variabilité des langues**: Bien que nous ayons fait le choix de traduire l’ensemble du jeu de données vers une langue unique 
        (francais), on remarque que la langue varie significativement selon la catégorie de produit. 
        """
        )

        

        

        # Compter le nombre de fois que chaque produit apparaît pour chaque langue
        counts = df.groupby(['prdtypefull', 'language']
                            ).size().reset_index(name='Nombre')

        # Création du graphique
        fig = go.Figure()

        for langue in counts['language'].unique():
            df_langue = counts[counts['language'] == langue]
            fig.add_trace(go.Bar(
                x=df_langue['prdtypefull'],
                y=df_langue['Nombre'].sort_values().to_numpy(),
                name=langue
            ))

        # Mise en forme du graphique
        fig.update_layout(
            title='Nombre de propositions par type de produit et par langue',
            xaxis=dict(title='Catégories'),
            yaxis=dict(title='Nombre de produits'),
            barmode='stack',  # Barres superposées

            width=1400,
            height=600
        )

        # Affichage du graphique
        st.plotly_chart(fig)

        

        st.header("Langues présentes")
        st.markdown(
            """
            **Variabilité des langues** : les textes sont majoritairement rédigés en français (environ 80 %). Certains textes sont en anglais ou en allemand.
            """
        )

    with tab5:
        st.header("Corrélation des catégories")
        st.markdown(
            """
            **Séparabilité des catégories** : Certaines catégories ont un chevauchement lexical notable (par exemple, les consoles de jeu et les jeux vidéo), 
            comme on peut le remarquer dans les wordclouds ou dans la matrice de corrélation entre vecteurs de fréquence des mots.
            
            On peut voir que les catégories livres neufs, livres d'occasion et Magazines d'occasion sont tres correlées, nous avons également la même analyse mais dans une moindre mesure pour les consoles de jeux,
            jeux de societés, jeux vidéo d'occasion, équipement pour jeux video
            """
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write("")
        with col2:
            st.image(corr, use_column_width=True)

        with col3:
            st.write("")

    with tab6:
        st.header("Wordclouds")
        st.markdown(
            """
            Quelques représentations visuelles de wordclouds. Les worldcloud servent avant tout a représenter les mots les plus fréquents des catégories. Plus un mot est présent plus il est grand
            """
        )

        # Chemin du dossier contenant les images
        images_folder = 'images/wc_visuels/'

        image_files = os.listdir(images_folder)

        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])

        def display_image_with_text(image_path, text):
            with st.container():
                st.image(image_path, use_column_width=True)
                st.markdown(f"<center>{text}</center>", unsafe_allow_html=True)

        for i, col in enumerate([col2, col4]):
            image_path = os.path.join(images_folder, image_files[i])
            with col:
                display_image_with_text(image_path, f"{image_files[i][:-4]}")

        for i, col in enumerate([col2, col4]):
            image_path = os.path.join(images_folder, image_files[i+2])
            with col:
                display_image_with_text(image_path, f"{image_files[i+2][:-4]}")


# page 3  #############################################################################################################################################################################
if page == pages[3]:
    st.title("PREPROCESSING")
    tab1, tab2 = st.tabs(
        ['Traitement sur le texte', 'Traitement sur les images'])

    with tab1:

        st.header("Traitement sur le texte")
        st.markdown(
            """
        Pour reprendre les objectifs par rapport au texte nous allons nettoyer nos colones 
        Voici un schéma explicatif de la procédure :
        """
        )

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.write("")

        with col2:

            st.image(schema_prepro_txt, use_column_width=True)

        with col3:
            st.write("")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            dataframe orginal            
            """)
            st.write(df.head())

        with col2:
            st.markdown("""
            dataframe preprocessé            
            """)
            st.write(df_train_clean.head())

    with tab2:
        st.header("Traitement sur les images")
        st.markdown(
            """
            Concernant les images le padding est ajusté pour n'avoir que de l'information utile dans notre image
            """
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.write("")

        with col2:

            st.image('images/padding_img.png', use_column_width=True)

        with col3:
            st.write("")


# Page4 ############################################################################################################################################
if page == pages[4]:

    st.title("Modélisation : texte")

    tab1, tab2, tab3 = st.tabs(
        ["Synthèse", "Benchmark des modèles", "Détail des performances par modèle"])

    with tab1:

        st.markdown("""
                    
Dans le contexte de la classification de produits sur la base du texte seul, nous avons commencé par
examiner différentes techniques de vectorisation (**Bag-of-Words avec TF-IDF et Word2Vec avec Skipgram
et CBOW**) associées à des méthodes de classification classiques (SVM, régression logistique, arbres de
décision, etc).
                    
Nous avons poursuivi notre stratégie de classification textuelle en
entraînant des transformers de type **BERT (Bidirectional Encoder Representations from Transformers**).
Plusieurs versions de transformers pré-entraînés sur divers corpus français ont été comparées:
**CamemBERT-base**, **CamemBERT-ccnet** et **FlauBERT**.
                    
## Meilleurs résultats par modèle
                    """)

        col11, col12 = st.columns([1, 1])
        with col11:
            st.markdown("""
### Modèles standards
| Modèle  | f1 score | Durée fit (s) |
| :--------------- |---------------:| -----:|
| Linear SVC  |   0.824 |  6 |
| XGBoost  | 0.819 |   3 840 |
| Logistic Regression  | 0.813 |    179 |
| SVC  | 0.784 |    2 993 |
| Random Forest  | 0.776 |    2 344 |
| Multinomial NB  | 0.771 |    0.45 |
                    
""")

        with col12:
            st.markdown("""
### Modèles transformers
| Modèle  | f1 score | Durée fit (s) |
| :--------------- |---------------:| -----:|
| CamemBERT  |   0.886 |  16 955 |
| XGBoost  |   0.885 |  17 225 |
| Logistic Regression  |   0.878 |  15 138 |

                        """)

        st.markdown("""
## Vectorisation
                    
Les modèles transformers (BERT) embarquent leur propres mécanismes de tokenisation et de vectorisation.
Pour les modèles 'standards', nous avons comparé les performances de la vectorisation Bag-of-Words (TF-IDF) avec Word2Vec (SKIP-GRAM et CBOW).
                    
                    
| Vectorisation  | Description | Hyper-paramètres optimaux |
| :--------------- |:---------------| :-----|
| Bag of words (TF-IDF)  | Conversion des textes en vecteur de valeurs TF-IDF a partir de l’ensemble d'entraînement. Pas de limite de taille de vecteur. Valeur de TF-IDF normalisés par la norme euclidienne pour chaque entrée. |  Paramètres par défaut |
| Word2Vec (Skip-gram)  | Modèle visant à prédire un contexte de mot en fonction d'un mot en particulier | window = 10, vector_size = 500, min_count=2 |
| Word2Vec (CBOW)  | Modèle visant à prédire un mot en fonctio d'un contexte | window = 10, vector_size = 300, min_count = 3 |

> _A noter que l'ajustement des hyper-paramètres de Word2Vec dépend de la modélisation appliquée ensuite, d'où la nécessité de faire des GridSearchCV combinés si on souhaite optimiser ces paramètres._
## Méthodologie et benchmark
- Entrainement sur 80% des donnéees
- Evaluation des performances sur les 20% restants
- Optimisation des hyper-paramètres via **GridSearchCV** avec validation croisée à 5 folds sur l'ensemble d'entraînement

                            """)

    with tab2:
        res = get_results_manager()
        fig = res.build_fig_f1_scores(filter_package=['bert', 'text'])
        fig.update_xaxes(range=[0.6, 0.9])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        res = get_results_manager()
        models_paths = res.get_model_paths(filter_package=['bert', 'text'])
        models_paths = np.sort(models_paths)

        option_selected = st.selectbox(
            "Choisissez un modèle pour afficher la matrice de confusion  :", models_paths, format_func=lambda model_path: res.get_model_label(model_path) + ' - ' + str(round(res.get_f1_score(model_path), 3)))

        col1, col2 = st.columns([1, 1])

        with col1:
            plt_matrix = res.get_fig_confusion_matrix(
                option_selected, model_label=res.get_model_label(option_selected))
            st.pyplot(plt_matrix, use_container_width=True)

            st.markdown("""
Les matrices de confusion révèlent la difficulté de ces modèles à différencier des catégories
sémantiquement proches, telles que :
- "Livres d'occasion", "Livres neufs", "Magazines d'occasion", "Bandes dessinées et magazines"
- "Maison Décoration", "Mobilier de jardin", "Mobilier", "Outillage de jardin", "Puériculture"
- "Figurines et jeux de rôle", "Figurines et objets pop culture", "Jouets enfants", "Jeux de société pour
enfants"
- "Jeux vidéo d'occasion", "CDs et équipements de jeux vidéo", "Accessoires gaming"
                    """)

        with col2:
            st.dataframe(
                pd.DataFrame(res.get_f1_scores_report(option_selected)).T,
                use_container_width=True,
                height=1200
            )
# Page5 ############################################################################################################################################
if page == pages[5]:
    st.title("Modélisation : images")
    tab1, tab2, tab3 = st.tabs(
        ["Synthèse", "Benchmark des modèles", "Détail des performances par modèle"])

    with tab1:

        st.markdown("""
                    
Dans le domaine de la classification d'images, l'adoption de réseaux de deep learning est incontournable.
Les réseaux de **neurones convolutifs (CNN)** sont particulièrement performants mais plus récemment, les
modèles basés sur des architectures de transformer, comme le modèle **Vision Transformer (ViT)** se sont
aussi révélés efficaces.
Pour classifier les produits sur la base des images associées, nous avons donc utilisé différents réseaux
convolutifs **(ResNet, EfficientNet et VGG)** ainsi que le transformer **ViT (Vision Transformer)**, tous
pré-entraînés sur la base de données ImageNet.
                    
## Meilleurs résultats par modèle
                    """)

        st.markdown("""

| Modèle  | f1 score | Durée fit (s) |
| :--------------- |---------------:| -----:|
| ViT_b16  |   0.675 |  10 572 |
| ResNet152  | 0.658 |   6 894 |
| ResNet101  | 0.656 |   6 754 |
| EfficientNetB1  | 0.655 |    6 657 |
| Random Forest  | 0.653 |    6 720 |
| Multinomial NB  | 0.620 |    6 054 |
                    

> Les F1-scores mesurés sur l'ensemble de test révèlent une supériorité marquée du modèle Vision
Transformer **(ViT, F1-score = 0.675)** comparativement au meilleur modèle CNN testé **(ResNet152,
F1-score = 0.658)**. Ces modèles image restent cependant beaucoup moins performants que les
modèles texte, illustrant la complexité inhérente à la classification de produits sur la base exclusive
d'images. Néanmoins, il est intéressant de noter que les catégories les plus fréquemment confondues par
les modèles dédiés aux images correspondent presque exactement à celles posant des difficultés dans la
classification de texte.
""")

    with tab2:
        res = get_results_manager()
        fig = res.build_fig_f1_scores(filter_package=['img'])

        fig.update_xaxes(range=[0.5, 0.8])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        res = get_results_manager()
        models_paths = res.get_model_paths(filter_package=['img'])
        models_paths = np.sort(models_paths)

        option_selected = st.selectbox(
            "Choisissez un modèle pour afficher la matrice de confusion  :", models_paths, format_func=lambda model_path: res.get_model_label(model_path) + ' - ' + str(round(res.get_f1_score(model_path), 3)))

        col1, col2 = st.columns([1, 1])

        with col1:
            plt_matrix = res.get_fig_confusion_matrix(
                option_selected, model_label=res.get_model_label(option_selected))

            st.pyplot(plt_matrix, use_container_width=True)

            st.markdown("""
On retrouve des clusters de catégories difficiles à distinguer assez similaires à ceux des modèles texte :
- "Livres d'occasion", "Livres neufs", "Magazines d'occasion", "Bandes dessinées et magazines"
- "Maison Décoration", "Mobilier de jardin", "Mobilier", "Outillage de jardin", "Puériculture"
- "Figurines et jeux de rôle", "Figurines et objets pop culture", "Jouets enfants", "Jeux de société pour
enfants"
- "Jeux vidéo d'occasion", "CDs et équipements de jeux vidéo", "Accessoires gaming"
                    """)

        with col2:
            st.dataframe(
                pd.DataFrame(res.get_f1_scores_report(option_selected)).T,
                use_container_width=True,
                height=1200
            )


# Page6 ############################################################################################################################################
if page == pages[6]:
    st.title("Modélisation : fusion")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Synthèse", "Benchmark des modèles", "Détail des performances par modèle", "Simulateur de fusion"])

    with tab1:

        st.markdown("""
                    
Nous avons retenu les deux meilleurs architectures obtenues sur le texte et les images
(i.e. camemBERT et ViT). Plusieurs architectures de modèles effectuant la fusion entre ces deux
transformers ont ensuite été testées et comparées:
- Approches d’ensemble: classifiers de type voting ou stacking (par régression logistique) opérant sur
les logits de sortie des modèles spécialisés pré-entraînés. _Le poids attribué à chaque modèle dans le
voting classifier est défini par le rapport des F1-scores des modèles spécialisés (par exemple:
camemBERT: F1-scoreBERT / (F1-scoreBERT + F1-scoreViT) = 0.57; ViT: F1-scoreViT / (F1-scoreBERT +
F1-scoreViT) = 0.43)_.                   
Pour éviter tout leakage des F1-scores utilisés comme poids du modèle, la
performance du voting classifier est estimée par validation croisée à 5 folds sur l’ensemble de test.
- Approche transformer (TF): fusion des sorties des derniers blocs de transformer de camemBERT et
ViT par l'intermédiaire d’un bloc transformer cross-attentionnel, suivi d’un nombre variable de blocs
de transformer classiques avec self-attention (TF: 1, 3 ou 6 blocs).

Après analyse approfondie des résultats obtenus, une troisième approche de fusion a été testée : **l'approche hybride**.
Cette approche consiste à intégrer dans un modèle voting classifier une combinaison de plusieurs modèles.
Le meilleur de ces modèles hybrides combine les classifiers suivants: TF6 (hybride),
camembert-base-ccnet (texte), flaubert-base-uncased (texte), xgboost_tfidf (texte), vit_b16 (image),
ResNet152 (image). Ce modèle permet de gagner plus d’un point de F1-score par rapport au modèle
multimodale simple de type transformer **(F1-score = 0.911)**
## Meilleurs résultats par modèle
                    """)
        col11, col12 = st.columns([1, 1])
        with col11:
            st.markdown("""

### Modèles fusion basés sur CamemBERT-ccnet et ViT 
| Modèle  | f1 score | Durée fit (s) |
| :--------------- |---------------:| -----:|
| TF6  |   0.899 |  28 874 |
| TF3  | 0.897 |   24 375 |
| TF1  | 0.899 |   20 773 |
| Voting  | 0.892 |   NA |
| Stacking  | 0.891 |    1 311 |

""")
        with col12:
            st.markdown("""
### Modèles fusion hybrides
                        
| Modèle  | f1 score |
| :--------------- |---------------:|
| **TF6, CamemBERT, FlauBERT, XGBoost (TF-IDF), ViT, ResNet152**  |   **0.911** |
| TF6, CamemBERT, ViT | 0.909 |
| TF6, FlauBERT, ResNet152  | 0.907 |
| CamemBERT, FlauBERT, ViT, ResNet152  | 0.902 |
| CamemBERT, FlauBERT, XGBoost (TF-IDF), ViT | 0.900 |
| SVC (Skip-gram), LinearSVC (TF-IDF), XGBoost (TF-IDF), ViT | 0.852 |


                        """)
    with tab2:
        res = get_results_manager()

        # scores = scores[['serie_name', 'score_test',
        #                  'vectorizer']].reset_index()
        # sorted_scores = scores.sort_values(by='score_test', ascending=False)

        # # plot
        # if title is None:
        #     title = 'Benchmark des f1 scores'
        # fig = uplot.get_fig_benchs_results(
        #     sorted_scores,
        #     'serie_name',
        #     'score_test',
        #     'model',
        #     'f1 score',
        #     color_column='vectorizer',
        #     title=title,
        #     figsize=figsize
        # )
        fig = res.build_fig_f1_scores(filter_package=['fusion'])
        fig.update_xaxes(range=[0.8, 0.92])

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        res = get_results_manager()
        models_paths = res.get_model_paths(filter_package=['fusion'])
        models_paths = np.sort(models_paths)

        option_selected = st.selectbox(
            "Choisissez un modèle pour afficher la matrice de confusion  :", models_paths, format_func=lambda model_path: res.get_model_label(model_path) + ' - ' + str(round(res.get_f1_score(model_path), 3)))

        col1, col2 = st.columns([1, 1])

        with col1:
            plt_matrix = res.get_fig_confusion_matrix(
                option_selected, model_label=res.get_model_label(option_selected))
            st.pyplot(plt_matrix, use_container_width=True)

            st.markdown("""
On retrouve des clusters de catégories difficiles à distinguer assez similaires à ceux des modèles texte :
- "Livres d'occasion", "Livres neufs", "Magazines d'occasion", "Bandes dessinées et magazines"
- "Maison Décoration", "Mobilier de jardin", "Mobilier", "Outillage de jardin", "Puériculture"
- "Figurines et jeux de rôle", "Figurines et objets pop culture", "Jouets enfants", "Jeux de société pour
enfants"
- "Jeux vidéo d'occasion", "CDs et équipements de jeux vidéo", "Accessoires gaming"
                    """)

        with col2:
            st.dataframe(
                pd.DataFrame(res.get_f1_scores_report(option_selected)).T,
                use_container_width=True,
                height=1200
            )

    with tab4:
        res = get_results_manager()
        models_paths = res.get_model_paths()
        models_paths = np.sort(models_paths)

        options_selected = st.multiselect(
            "Choisissez plusieurs modèles pour afficher la matrice de confusion  :", models_paths, format_func=lambda model_path: res.get_model_label(model_path) + ' - ' + str(round(res.get_f1_score(model_path), 3)))

        col1, col2 = st.columns([1, 1])

        with col1:
            if len(options_selected) > 1:
                plt_matrix = res.get_voting_confusion_matrix(
                    options_selected, model_label="fusion personnalisée")
                st.pyplot(plt_matrix, use_container_width=True)
            else:
                st.write("Sélectionnez au moins deux modèles")

        with col2:
            if len(options_selected) > 1:
                st.dataframe(
                    pd.DataFrame(res.get_voting_f1_scores_report(
                        options_selected)).T,
                    use_container_width=True,
                    height=1200
                )
# Page7 ############################################################################################################################################
if page == pages[7]:
    st.header("Classification à partir d'images ou de texte")
    options = ["html", "data", 'Démo "poussette"',
               'Démo "livre Dune"', 'Démo "jeu Dune"']
    option_selected = st.selectbox("Page rakuten ou données :", options)

    if option_selected == "html":

        input_html = st.text_area(
            "Collez ici le contenu html de la page produit de Rakuten", value="")
    elif option_selected == 'Démo "poussette"':
        with open(config.path_to_project + '/data/demo/demo_poussette.html', 'r') as file:
            input_html = file.read()
    elif option_selected == 'Démo "livre Dune"':
        with open(config.path_to_project + '/data/demo/demo_dune_livre.html', 'r') as file:
            input_html = file.read()
    elif option_selected == 'Démo "jeu Dune"':
        with open(config.path_to_project + '/data/demo/demo_dune_jeu.html', 'r') as file:
            input_html = file.read()
    else:
        input_image_url = st.text_input(
            "URL de l'image", value="")
        input_designation = st.text_area(
            "Description du produit", value="")

    if st.button("Valider"):
        res = get_results_manager()
        true_cat = 'nc'
        if input_html:
            scrap = scrapper.RakutenScrapper()
            des, desc, img, true_cat = scrap.get_rakuten_product_infos(
                input_html)
            if desc != 'not found':
                designation = des + ' ' + desc
            else:
                designation = des
            image_url = img
        else:
            designation = input_designation
            image_url = input_image_url

        pred = res.predict(
            models_paths=['fusion/camembert-base-vit_b16_TF6'],
            # model_path='text/camembert-base-ccnet',
            text=designation,
            img_url=image_url
        )

        col1, col2 = st.columns([2, 4])
        with col1:
            tab11, tab12 = st.tabs(["Données brutes", "GradCam"])
            with tab11:
                st.write(designation)
                st.image(image_url, use_column_width=True)
            with tab12:
                icam = pred['icam']
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(icam.image)
                ax[0].axis('off')
                ax[0].set_title('original')
                ax[1].imshow(icam.image_masked)
                ax[1].axis('off')
                ax[1].set_title('deepCAM')
                st.pyplot(fig)

                fig, ax = plt.subplots()
                plot_weighted_text(0, 0.7, icam.text, icam.text_masked*0+0.4, base_font_size=80,
                                   char_per_line=80, title='Original', title_color='blue', title_fontsize=100, ax=ax)
                plot_weighted_text(0, ax.get_ylim()[0], icam.text, icam.text_masked*5, base_font_size=60,
                                   char_per_line=100, title='deepCAM', title_color='purple', title_fontsize=100, ax=ax)
                st.pyplot(fig)
        with col2:
            st.write("<p>Classe prédite : <strong>{}</strong></p>".format(
                pred['pred_labels'][0]), unsafe_allow_html=True)
            st.write("<p>Classe réelle : <strong>{}</strong></p>".format(
                true_cat), unsafe_allow_html=True)
            st.write("<p>Probabilité : <strong>{:.2f}</strong></p>".format(
                np.max(pred['pred_probas'][0])), unsafe_allow_html=True)
            st.write("<p>Temps : <strong>{:.2f} s</strong></p>".format(
                pred['time']), unsafe_allow_html=True)

            # Créer les données pour le graphique
            categories = pred['labels']
            prediction = pred['pred_probas'][0]

            # Créer le graphique à barres
            fig = go.Figure(data=[
                go.Bar(name='Prédiction 1', x=categories, y=prediction),

            ])

            # Personnaliser le layout
            fig.update_layout(
                title='Prédictions pour chaque catégorie',
                xaxis=dict(title='Catégorie'),
                yaxis=dict(title='Probabilité de prédiction'),
                barmode='group'
            )

            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig, use_container_width=True)

    elif option_selected == "Texte":
        col1, col2 = st.columns([2, 4])
        with col1:

            import requests

            # Champ pour saisir l'URL à scrapper
            url_input = st.text_input("Entrez l'URL à scrapper :")

            # Bouton pour lancer le scrapping
            if st.button("Scrapper"):
                # Vérifier si une URL a été saisie
                if url_input:
                    # Récupérer le contenu HTML de l'URL
                    response = requests.get(url_input)
                    # Vérifier si la requête a réussi
                    if response.status_code == 200:
                        # Afficher le texte récupéré
                        st.write(response.text)
            else:
                st.error("Erreur lors de la récupération du contenu de l'URL")

        with col2:
            # Créer les données pour le graphique
            categories = ['Catégorie 1', 'Catégorie 2',
                          'Catégorie 3', 'Catégorie 4', 'Catégorie 5']
            prediction = [0.78, 0.21]

            # Créer le graphique à barres
            fig = go.Figure(data=[
                go.Bar(name='Prédiction 1', x=categories, y=prediction),

            ])

            # Personnaliser le layout
            fig.update_layout(
                title='Prédictions pour chaque catégorie',
                xaxis=dict(title='Catégorie'),
                yaxis=dict(title='Probabilité de prédiction'),
                barmode='group'
            )

            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig, use_container_width=True)

            classe_predite = "Classe A"
            classe_reelle = "Classe B"
            estimation = 0.78
            st.write("<p style='text-align:center;'><strong>Classe prédite: {}</p>".format(
                classe_predite), unsafe_allow_html=True)
            st.write("<p style='text-align:center;'><strong>Classe réelle: {}</p>".format(
                classe_reelle), unsafe_allow_html=True)
            st.write("<p style='text-align:center;'><strong>Estimation: {:.2f}</p>".format(
                estimation), unsafe_allow_html=True)

# Page8 ############################################################################################################################################
if page == pages[8]:
    st.title("Conclusion")
    st.header("Résumé")
    st.markdown("""
                
                Dans ce projet, nous avons évalué l'efficacité de différentes méthodes de classification pour prédire les catégories de produits à partir de données textuelles et visuelles. 
                Nous avons d'abord établi des benchmarks pour des modèles spécialisés dans le traitement des données textuelles et des images, puis fusionné les modèles les plus performants pour former des classificateurs hybrides. 

                En combinant plusieurs modèles spécialisés (**BERT, XGBoost, ViT et ResNet**) avec un modèle hybride, nous avons atteint un score de **0.911** après un entraînement sur 80 % des données. 
                
                """)
    st.markdown("""
    <p style='font-size:12px;'><i>Les performances de nos modèles ont été évaluées par le calcul du f1-score pondéré (weighted F1-score), i.e. le F1-score calculé par catégorie puis moyenné selon la proportion de chaque classe dans le jeu de données. Ce score a été systématiquement calculé grâce à la fonction sklearn.metrics.f1_score de sklearn afin d'éviter toute ambiguïté. Pour analyser plus en détails les performances, nous avons également calculé les matrices de confusion, exprimées en pourcentage du nombre d'articles par catégorie (normalisation par colonne).</i></p>
""", unsafe_allow_html=True)
    st.header("Pistes d'amélioration")
    st.markdown("""
                * Exploration de l'impact des stop words et des hyperparamètres des réseaux de neurones
                * Gestion du déséquilibre des classes. 
                * Autres modéles images.
                * Exploration de différentes architectures pour le modèle hybride.
                """)
   