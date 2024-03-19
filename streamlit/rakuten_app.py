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


# chargement des Ressources
# DATAFRAMES
df = pd.read_csv("../data/raw/X_train.csv", index_col=0)
df_train_clean = pd.read_csv("../data/clean/df_train_index.csv", index_col=0)
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


@st.cache_resource
def get_results_manager():
    try:
        if res:
            print('res already loaded')
    except NameError:
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
        col1, col2, col3 = st.columns([4, 1, 2])
        with col1:
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
            st.markdown("""
                    ####      80 000 produits
                    ####      27 catégories à distinguer
                    ####      Description textuelle multilangue
                        """)

        with col3:
            st.header("> Puériculture")
            st.write("Porte bébé Violet et rouge Trois-en-un mère multifonctions Kangourou fermeture à glissière Hoodie Taille: XL Poitrine: 104-109 cm 84-88 cm Hanche: 110-116 cm clair + 1. Marque nouvelle et de haute qualité. 2. Détachable conception pratique et attentionnée. 3. Parfait pour les mères qui allaitent. 4. Anti-vent chaud et style kangourou multifonctionnel haut de gamme. 5. Sac de couchage multifonction amovible de la mère européenne. Spécification: Les types Fermez Buste104-109cm Encolure Sweat à capuche Les hanches110-116cm Tailles disponiblesXLMatériel Coton")
            st.divider()
            st.image("images/image_sample.jpg", use_column_width=True)

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
    tab1, tab2, tab3 = st.tabs(["Dataframes", "Images", "Catégories"])
    with tab1:
        st.header("Exploration des données texte")
        st.markdown("""
            ### - `X_train.csv` composé de 84916 produits
            ### - `X_test.csv` composé de 13812 produits
            ### - `y_train.csv` : avec les codes produits
""")

        col1, col2 = st.columns([2, 1])
        with col1:
            container = st.container()
            with st.container():
                st.image(schema_dataframe,
                         output_format='auto', use_column_width=True)
        with col2:
            st.image(schema_dataframe_Y, width=None)

    with tab2:
        st.header("Exploration des données images")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(
                """
            Pour les images :
            - toutes en 500*500
            - que des jpeg
            
            - certaines avec un contour blanc tres important
            """
            )
        with col2:
            st.image(schema_image, width=600, use_column_width=True,
                     caption="", output_format='auto')
    with tab3:
        # Chemin du dossier contenant les images
        images_folder = 'images/wc_visuels/'

        image_files = os.listdir(images_folder)

        col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
        with col1:
            df_produits = pd.read_csv(
                config.path_to_project + '/data/prdtype.csv')
            df_produits.prdtypecode = df_produits.prdtypecode.astype(str)
            st.dataframe(
                df_produits,
                use_container_width=True,
                height=1000
            )

        def display_image_with_text(image_path, text):
            with st.container():
                st.image(image_path, use_column_width=True)
                st.markdown(f"<center>{text}</center>", unsafe_allow_html=True)

        for i, col in enumerate([col3, col5]):
            image_path = os.path.join(images_folder, image_files[i])
            with col:
                display_image_with_text(image_path, f"{image_files[i][:-4]}")

        for i, col in enumerate([col3, col5]):
            image_path = os.path.join(images_folder, image_files[i+2])
            with col:
                display_image_with_text(image_path, f"{image_files[i+2][:-4]}")

 # page 2 ##########################################################################################################################
if page == pages[2]:
    # Dataviz
    st.title("DATAVIZ")

    tab1, tab2, tab3, tab4 = st.tabs(["**Déséquilibre de classes**", "**Déséquilibre de texte**",
                                      "**Langues par catégories**", '**Corrélation entre catégories**'])

    with tab1:
        st.markdown(
            """
        Les catégories de produit du jeu de données affichent un déséquilibre notable, 
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
            title="Nombre de produits par catégorie",
            labels={"x": "Catégories", "y": "Nombre de produits"},
            color=nb_categories_sorted,
            color_discrete_sequence=px.colors.sequential.Viridis,
            width=1200,  # spécifier la largeur du graphique
            height=600,  # spécifier la hauteur du graphique
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=15))
        st.plotly_chart(fig)

    with tab2:
        st.markdown(
            """
        Certaines produits n'ont pas de champ descriptions. 
        La longueur des descriptions textuelles montrent aussi une variabilité importante.
        Les descriptions des livres d’occasion ou des cartes de jeux sont généralement brèves 
        (quelques dizaines de mots, champ description absent), tandis que celles des jeux vidéo pour PC 
        s'étendent souvent sur plusieurs centaines de mots.
        """
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            # Compter le nombre de produits par catégorie
            df = df_train_clean.loc[df_train_clean['description'].isna()]
            df['prdtypefull'] = df['prdtypefull'].str.split(
                ' - ').str[1]
            df_count = df['prdtypefull'].value_counts().sort_values()
            print(df_count.index)
            # Créer un graphique à barres avec plotly express
            fig = px.bar(
                x=df_count.index,
                y=df_count.values,
                title="Nombre de produits sans champ description",
                labels={"x": "Catégories", "y": "Nombre de produits"},
                color=df_count.values,
                color_continuous_scale='viridis',
                width=700,
                height=600,
            )

            # Mettre à jour les étiquettes de l'axe x
            fig.update_xaxes(tickangle=45, tickfont=dict(size=14))

            # Afficher le graphique
            st.plotly_chart(fig)

        with col2:
            df_train_clean['longeur'] = (
                df_train_clean["designation_translated"] + df_train_clean["description_translated"]).astype(str)
            df_train_clean['longeur_val'] = df_train_clean['longeur'].apply(
                lambda x: len(x))
            df_train_clean['prdtypefull'] = df_train_clean['prdtypefull'].str.split(
                ' - ').str[1]
            print(df_train_clean.index)
            idx_order = df_count.index.tolist()
            all_cat = df_train_clean['prdtypefull'].unique()
            missing_cat = [idx for idx in all_cat if idx not in idx_order]
            idx_order = missing_cat + idx_order

            # Créer un graphique à barres avec plotly express
            fig = px.box(df_train_clean,
                         x='prdtypefull',
                         y='longeur_val',
                         title="Longeur des textes",
                         labels={'prdtypefull': "Catégories",
                                 'longeur_val': "Nombre de mots"},
                         width=700,
                         height=600,
                         )

            # Mettre à jour les étiquettes de l'axe x
            fig.update_xaxes(tickangle=45, tickfont=dict(size=14),
                             categoryorder='array', categoryarray=idx_order)

            # Afficher le graphique
            st.plotly_chart(fig)

    with tab3:
        st.markdown(
            """
        Les textes sont majoritairement rédigés en français (environ 80 %). Certains textes sont en anglais ou en allemand.
        Bien que nous ayons fait le choix de traduire l’ensemble du jeu de données vers une langue unique 
        (francais), on remarque que la langue varie significativement selon la catégorie de produit. 
        """
        )

        counts = pd.crosstab(df['language'], df['prdtypefull'],
                             normalize='columns').sort_values('fr', axis=1, ascending=False)
        # counts = pd.crosstab(['prdtypefull', 'language']
        #                     ).size().reset_index(name='Nombre')

        # Création du graphique
        fig = go.Figure()

        # for langue in counts['language'].unique():
        #     df_langue = counts[counts['language'] == langue]
        #     fig.add_trace(go.Bar(
        #         x=df_langue['prdtypefull'],
        #         y=df_langue['Nombre'].sort_values().to_numpy(),
        #         name=langue
        #     ))

        for lang in counts.index:
            fig.add_trace(go.Bar(x=counts.columns,  y=counts.loc[lang, :]*100,
                                 name=lang))

        # Mise en forme du graphique
        fig.update_layout(
            title='Pourcentage des langues par categorie',
            xaxis=dict(title='Catégories'),
            yaxis=dict(title='Pourcentage de produits'),
            barmode='stack',  # Barres superposées

            width=1200,
            height=600
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=14))

        # Affichage du graphique
        st.plotly_chart(fig)

    with tab4:
        st.markdown(
            """
            Certaines catégories ont un chevauchement lexical important (par exemple, les consoles de jeu et les jeux vidéo), 
            comme on peut le remarquer dans les wordclouds ou dans la matrice de corrélation entre vecteurs de fréquence des mots.
            """
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write("")
        with col2:
            st.image(corr, use_column_width=True)

        with col3:
            st.write("")

    # with tab6:
    #     st.header("Wordclouds")
    #     st.markdown(
    #         """
    #         Quelques représentations visuelles de wordclouds. Les worldcloud servent avant tout a représenter les mots les plus fréquents des catégories. Plus un mot est présent plus il est grand
    #         """
    #     )

    #     # Chemin du dossier contenant les images
    #     images_folder = 'images/wc_visuels/'

    #     image_files = os.listdir(images_folder)

    #     col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])

    #     def display_image_with_text(image_path, text):
    #         with st.container():
    #             st.image(image_path, use_column_width=True)
    #             st.markdown(f"<center>{text}</center>", unsafe_allow_html=True)

    #     for i, col in enumerate([col2, col4]):
    #         image_path = os.path.join(images_folder, image_files[i])
    #         with col:
    #             display_image_with_text(image_path, f"{image_files[i][:-4]}")

    #     for i, col in enumerate([col2, col4]):
    #         image_path = os.path.join(images_folder, image_files[i+2])
    #         with col:
    #             display_image_with_text(image_path, f"{image_files[i+2][:-4]}")


# page 3  #############################################################################################################################################################################
if page == pages[3]:
    st.title("PREPROCESSING")
    tab1, tab2 = st.tabs(
        ['**Texte**', '**Images**'])

    with tab1:

        st.markdown(
            """
        **Pipeline de pre-processing du texte**:
        """
        )
        st.image(schema_prepro_txt, use_column_width=True)

        options = df_train_clean.loc[~df_train_clean['description'].isna(
        ), 'productid'].to_list()

        idx_selected = st.selectbox("Selectionnez un produit :", options)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            **Texte orginal**            
            """)
            st.write('. '.join([df.loc[df['productid'] == idx_selected, 'designation'].values[0],
                     df.loc[df['productid'] == idx_selected, 'description'].values[0]]))

        with col2:
            st.markdown("""
            **Texte preprocessé**            
            """)
            st.write('. '.join([df_train_clean['designation_translated'][df_train_clean['productid'] == idx_selected].values[0],
                                df_train_clean['description_translated'][df_train_clean['productid'] == idx_selected].values[0]]))

    with tab2:
        st.markdown(
            """
            Redimensionnement des images pour optimizer le padding, en conservant le rapport d'asspect.
            """
        )

        col1, col2, col3 = st.columns([1, 2, 1])

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
        ["**Approches**", "**Benchmark des modèles**", "**Détail des performances par modèle**"])

    with tab1:
        st.markdown("""
                        
    Classification de produits sur la base du texte seul en deux temps: 
    - approche
    combinant différentes techniques de vectorisation à des méthodes de classification classiques.
    - transformers de type **BERT (Bidirectional Encoder Representations from Transformers**).
                        """)
        tab1_1, tab1_2, tab1_3 = st.tabs(["ML standard", "BERT", "Methode"])
        with tab1_1:
            st.markdown("""
    ### Classifieurs
    **SVM** (LinearSVC), **boosted trees** (xgboost), **regression logistique**, **bayesien** (MultinomialNB), etc
    ### Vectorisation
                       
    | Vectorisation  | Description | Hyper-paramètres optimaux |
    | :--------------- |:---------------| :-----|
    | **Bag of words (TF-IDF)**  | Vecteur de valeurs TF-IDF a partir de l’ensemble d'entraînement, sans de limite de taille de vecteur |  Paramètres par défaut |
    | **Word2Vec (Skip-gram)**  | Vecteur d'embedding de taille predefini | window = 10, vector_size = 500, min_count=2 |
    | **Word2Vec (CBOW)**  | Vecteur d'embedding de taille predefini | window = 10, vector_size = 300, min_count = 3 |

    > _A noter que l'ajustement des hyper-paramètres de Word2Vec dépend de la modélisation appliquée ensuite, d'où la nécessité de faire des GridSearchCV combinés si on souhaite optimiser ces paramètres._

       """)

        with tab1_2:
            st.markdown("""
    Different transformers, pré-entraînés sur divers corpus français, ont été comparés (**CamemBERT-base**, **CamemBERT-ccnet** et **FlauBERT**)""")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.write("")
            with col2:
                st.image(image_BERT, use_column_width=True)
            with col3:
                st.write("")

            st.markdown("""
    _Les modèles transformers (BERT) embarquent leur propres mécanismes de tokenisation et de vectorisation._""")

        with tab1_3:
            st.markdown("""
    ## Méthodologie et benchmark
    - Entrainement sur 80% des donnéees
    - Evaluation des performances sur les 20% restants
    - Optimisation des hyper-paramètres via **GridSearchCV** avec validation croisée à 5 folds sur l'ensemble d'entraînement
    - Transformers fine-tuned sur 8 epoques, learning rate de 5e-5, decroissant de 20% a chaque epoque""")

    with tab2:
        col1, col2 = st.columns([3, 1])

        with col1:
            res = get_results_manager()
            fig = res.build_fig_f1_scores(filter_package=['bert', 'text'])
            fig.update_xaxes(range=[0.6, 0.9])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
### Modèles transformers
| Modèle  | f1 score | Durée fit (s) |
| :--------------- |---------------:| -----:|
| CamemBERT  |   0.886 |  16 955 |
| XGBoost  |   0.885 |  17 225 |
| Logistic Regression  |   0.878 |  15 138 |

                        """)

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
        ["**Approches**", "**Benchmark des modèles**", "**Détail des performances par modèle**"])

    with tab1:

        st.markdown("""
Classification de produits sur la base des images seules par deux approches: 
- Convolutional Neural Networks (**CNN**).
- Vision transformers **ViT**.
                    """)

        tab1_1, tab1_2, tab1_3 = st.tabs(["CNN", "ViT", "Methode"])
        with tab1_1:
            st.markdown("""
    ### CNN
    **VGG16**, **ResNet** (ResNet50, ResNet101, ResNet152), **EfficientNet** (EfficientNetB1), tous pré-entrainés sur ImageNet
    ### Exemple d'architecture CNN (ResNet152)
                        """)
            st.image(image_ResNet152, use_column_width=True)

        with tab1_2:
            st.markdown("""
    Vision transformer **ViT**, pré-entraînés sur ImageNet""")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.write("")
            with col2:
                st.image(image_ViT, use_column_width=True)
            with col3:
                st.write("")

            st.markdown("""
    _Les Vision transformers ont une architecture similaire aux BERT, si ce n'est l'etape d'embedding wui s'effectue
    a partir d'une segmentation de l'image en patches qui sont ensuite traités par l'encodeur comme une sequence de tokens._""")

        with tab1_3:
            st.markdown("""
    ## Méthodologie et benchmark similaire a celle utilisée pour le texte
    - Entrainement sur 80% des donnéees
    - Evaluation des performances sur les 20% restants
    - Transformers fine-tuned sur 8 epoques, learning rate de 5e-5, decroissant de 20% a chaque epoque""")

    with tab2:
        col1, col2 = st.columns([3, 1])

        with col1:
            res = get_results_manager()
            fig = res.build_fig_f1_scores(filter_package=['img'])

            fig.update_xaxes(range=[0.5, 0.8])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
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
                        """)

        st.markdown("""
    > Supériorité marquée du modèle Vision
    Transformer **(ViT, 0.675)** comparativement au meilleur modèle CNN testé **(ResNet152, 0.658)**. 
    
    > Ces modèles image restent cependant beaucoup moins performants que les
    modèles texte, illustrant la complexité inhérente à la classification de produits sur la base exclusive
    d'images.
                    """)

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
        ["**Approches**", "**Benchmark des modèles**", "**Détail des performances par modèle**", "**Simulateur de fusion**"])

    with tab1:

        st.markdown("""
                    
Comparaison de differentes approches pour la classification sur la base du texte et des images:
- Classifiers d’ensemble simples de type voting ou stacking opérant sur
les logits de sortie des modèles spécialisés pré-entraînés.    
- Modele hybride de type transformer (TF).
- Meta-ensemble combinant modeles hybrides et specialises par voting
                    """)

        tab1_1, tab1_2, tab1_3, tab1_4 = st.tabs(
            ["Ensembles simples", "Transformer hybride", "Meta Voting", "Methodologie"])
        with tab1_1:
            st.markdown("""
    **Voting** ou **Stacking** opérant sur les logits de sortie des meilleurs modèles spécialisés pré-entraînés (**camemBERT** + **ViT**).
                        """)
            st.write("")  # st.image(image_simpleVoting, use_column_width=True)

        with tab1_2:
            st.markdown("""
    Transformer fusionnant les sorties des derniers blocs de transformer de **camemBERT** et
**ViT** par l'intermédiaire d’un bloc transformer **cross-attentionnel** (*query*: texte; *key*, *value*: image), suivi d’un nombre variable de blocs
de transformer classiques (TF: 1, 3 ou 6 blocs). Tetes attentionnelles de 12 couches par bloc""")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.write("")
            with col2:
                st.write("")  # st.image(image_fusionTF, use_column_width=True)
            with col3:
                st.write("")

        with tab1_3:
            st.markdown("""
    Meta ensemble combinant differents modeles hybrides et spécialisés dans un Voting classifier (ex: **TF6** (hybride),
**camembert-base-ccnet** (texte), **flaubert-base-uncased** (texte), **xgboost_tfidf** (texte), **vit_b16** (image),
**ResNet152** (image)).
> Le **poids** attribué a chaque modèle est défini par le **rapport des F1-scores** (poids du
modelA: F1-modelA / (F1-modelA + F1-modelB + ...). Les performances ont été **cross-validées sur l'ensemble de test** (5 folds)
                        """)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.write("")
            with col2:
                # st.image(image_metaVoting, use_column_width=True)
                st.write("")
            with col3:
                st.write("")

        with tab1_4:
            st.markdown("""
- Entrainement sur 80% des donnéees
- Evaluation des performances sur les 20% restants
- Transformers fine-tuned sur 8 epoques, learning rate de 5e-5, decroissant de 20% a chaque epoque
- **Poids des voting classifiers cross-validées sur l'ensemble de test (5 folds)**""")

    with tab2:

        col11, col12 = st.columns([2, 1])
        with col11:
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
            fig.update_yaxes(showticklabels=False)

            st.plotly_chart(fig, use_container_width=True)

        with col12:
            st.markdown("""

#### Modèles fusion basés sur CamemBERT-ccnet et ViT 
| Modèle  | f1 score | Durée fit (s) |
| :--------------- |---------------:| -----:|
| TF6  |   0.899 |  28 874 |
| TF3  | 0.897 |   24 375 |
| TF1  | 0.899 |   20 773 |
| Voting  | 0.892 |   NA |
| Stacking  | 0.891 |    1 311 |

""")

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

        col1, col2 = st.columns([1, 1])

        with col1:
            options_selected = st.multiselect(
                "Choisissez plusieurs modèles pour afficher la matrice de confusion  :", models_paths, format_func=lambda model_path: res.get_model_label(model_path) + ' - ' + str(round(res.get_f1_score(model_path), 3)))

            if len(options_selected) > 1:
                plt_matrix = res.get_voting_confusion_matrix(
                    options_selected, model_label="fusion personnalisée")
                st.pyplot(plt_matrix, use_container_width=True)
            else:
                st.write("Sélectionnez au moins deux modèles")

            if len(options_selected) > 1:
                st.dataframe(
                    pd.DataFrame(res.get_voting_f1_scores_report(
                        options_selected)).T,
                    use_container_width=True,
                    height=1200
                )
        with col2:
            st.markdown("""
### Benchmark des modèles fusion hybrides
                        
| Modèle  | f1 score |
| :--------------- |---------------:|
| **TF6, CamemBERT, FlauBERT, XGBoost (TF-IDF), ViT, ResNet152**  |   **0.911** |
| TF6, CamemBERT, ViT | 0.909 |
| TF6, FlauBERT, ResNet152  | 0.907 |
| CamemBERT, FlauBERT, ViT, ResNet152  | 0.902 |
| CamemBERT, FlauBERT, XGBoost (TF-IDF), ViT | 0.900 |
| SVC (Skip-gram), LinearSVC (TF-IDF), XGBoost (TF-IDF), ViT | 0.852 |
                        """)


# Page7 ############################################################################################################################################
if page == pages[7]:
    st.header("Classification à partir d'images ou de texte")
    options = ["html", "data", 'Démo "poussette"', 'Démo "hameçons"',
               'Démo "livre Dune"', 'Démo "jeu Dune"']
    option_selected = st.selectbox("Page rakuten ou données :", options)

    if option_selected == "html":

        input_html = st.text_area(
            "Collez ici le contenu html de la page produit de Rakuten", value="")
    elif option_selected == 'Démo "poussette"':
        with open(config.path_to_project + '/data/demo/demo_poussette.html', 'r') as file:
            input_html = file.read()
    elif option_selected == 'Démo "hameçons"':
        with open(config.path_to_project + '/data/demo/demo_hamecon.html', 'r') as file:
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

        tab11, tab12 = st.tabs(
            ["Résultats", "Données / Gradcam"])
        with tab11:
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
        with tab12:
            icam = pred['icam']
            col1, col2 = st.columns([1, 1])
            with col1:
                st.header('Texte original')
                st.write(designation)
                st.header('DeepCAM')
                fig, ax = plt.subplots()
                plot_weighted_text(0, 0.7, icam.text, icam.text_masked*5, base_font_size=60,
                                   char_per_line=100, title='', title_color='purple', title_fontsize=100, ax=ax)
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(icam.image)
                ax[0].axis('off')
                ax[0].set_title('original')
                ax[1].imshow(icam.image_masked)
                ax[1].axis('off')
                ax[1].set_title('deepCAM')
                st.pyplot(fig)


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
