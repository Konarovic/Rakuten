import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

from importlib import reload
from sklearn.metrics import f1_score
from importlib import reload
from results import ResultsManager
import config
import src.utils.results as results


# chargement des Ressources
#DATAFRAMES
df=pd.read_csv("../data/X_train.csv")
df_train_clean=pd.read_csv("../data/clean/df_train_index.csv")
xtrain=pd.read_csv("../data/X_train.csv")
ytrain=pd.read_csv("../data/Y_train.csv")


#chargement images
schema_dataframe="images/Schema_dataframe.png"
schema_image="images/schema_images.png"
schema_dataframe_Y="images/schema_dataframe_Y.png"
schema_prepro_txt="images/schema_prepro_txt.jpg"
schema_prepro_img="images/schema_prepro_img.jpg"
graf_isnaPrdt="images/graf_isnaPrdtypecode.png"
graf_txtLong ="images/graf_boxplot.png"
graf_lang ="images/lang.jpg"
graf_corr ="images/corr.jpg"
graf_WC ="images/maskWC.png"
img_rakuten_website = "images/rakuten_website.png"
corr = 'images/corr_cat.jpg'

#dossier images
wc_folder = "../WC"


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
        font-size: 20px; /* Taille de la police pour les paragraphes */
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
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)



#info streamlit
# pour entrer un graph ou image spécifier use_column_width=True ou container width





#DEBUT CODE STREAMLIT************************************************************************************************************

# LOGO RAKUTEN // toutes pages
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.write("")
with col2:
    image_path = "images/rakuten.png"  # Example image path
    image = Image.open(image_path)
    resized_image = image.resize((int(image.width * 1), int(image.height * 1)))
    st.image(resized_image)
with col3:
    st.write("")


# SOMMAIRE
st.sidebar.title("Sommaire")
pages=["Presentation", "Exploration", "DataViz", 'Préprocessing', "Modélisation texte", "Modélisation images", "Modélisation fusion", "Bonus", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.header("Auteurs")
st.sidebar.markdown("[Thibaud Benoist](link)")
st.sidebar.markdown("[Julien Chanson](link)")
st.sidebar.markdown("[Julien Fournier](link)")
st.sidebar.markdown("[Alexandre Mangwa](link)")



# page 0############################################################################################################################################
if page == pages[0]:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("")
    with col2:
        st.markdown("""
            <div style="text-align:center;">
                <h1>PRESENTATION</h1>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.write("\n")
        
    st.header("Contexte :")    
    st.markdown(
        """
    Le concept fondamental d’une marketplace réside dans sa capacité à mettre en relation vendeurs et acheteurs par le biais
    d’une plateforme en ligne unique, simplifiant ainsi le processus d’achat et de vente d’une vaste gamme de produits. 
    Pour qu’elle soit efficace, il est crucial que les produits soient aisément identifiables, que les utilisateurs bénéficient 
    d’une navigation fluide tout au long de leur parcours d’achat et que la plateforme offre des recommandations personnalisées alignées 
    sur le comportement des utilisateurs. 
    Un aspect essentiel du bon fonctionnement d’une marketplace est donc l’organisation méthodique des produits dans des catégories précises, 
    facilitant la recherche et la découvrabilité des produits.
    """)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("")
    with col2:
        st.image(img_rakuten_website)
    with col3:
        st.write("\n")
    

    st.markdown(
        """

    Ce processus de classification des produits requiert l’application de techniques de machine learning (ML) pour plusieurs raisons essentielles : 
    les vendeurs pourraient (pour diverses raisons) ne pas assigner les nouveaux produits aux catégories pertinentes, 
    introduisant des erreurs dans l’organisation du catalogue
    le classement manuel des produits peut s'avérer particulièrement fastidieux et inefficace lors de l’ajout massif d’articles.
    les catégories peuvent être amenées à être modifiées pour améliorer l'expérience d’achat des utilisateurs, 
    impliquant une mise à jour sur l’ensemble du catalogue.

    L’utilisation de technique de machine learning permet de surmonter ces problèmes éventuels 
    en automatisant tout ou partie de la catégorisation des produits sur la base des descriptions et images fournies par les vendeurs. 

    Le dataset utilisé dans ce projet est issu d’un challenge proposé par le groupe Rakuten, l’un des acteurs majeurs du marketplace B2B2C. 
    Il se compose d’un catalogue d’environ 80.000 produits, répartis selon 27 catégories distinctes, accompagnés de leurs descriptions textuelles 
    et images correspondantes. Voici le lien du concours [lien vers concours](https://challengedata.ens.fr/challenges/59)
        """
    )
    st.header("Objectifs :")
    st.markdown(
        """
    L’objectif principal est de développer le **meilleur modèle capable de classifier précisément (au sens du f1-score)** 
    chacun des produits en se basant sur les descriptions et images fournies. 
    Cela induit de développer plusieurs modèles et de les benchmarker afin de sélectionner le modèle le plus performant et le plus robuste.

    L’ensemble des membres du groupe projet étant débutant dans le domaine du **NLP (Natural Language Processing)** et de la **CV (Computer Vision)**. 
    Nous avons à comprendre et appliquer ces techniques au projet.

        """
    )






# page 1 ##########################################################################################################################
if page == pages[1]:
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
    
    
    col1, col2 = st.columns([1, 3])
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
        """
    )    
    with col2:
        container = st.container()
        with st.container():
            st.image(schema_dataframe, width=1000, output_format='auto', use_column_width=True)
        
    
    
        
    col1, col2 = st.columns([1, 3])
    with col1:
    # Problématique
        
        st.markdown(
        """
        27 codes produits au total
        """
    )    
    with col2:
        st.image(schema_dataframe_Y, use_column_width=False, caption="", output_format='auto')
   
    
    st.markdown("""
                Sur les données concernant les dataframes nous avons de nombreux NaN dans la colonnes description, 
                plusieurs langues, des balises HTML, des mauvais encodings
                """
                )
    
    st.write("")
    st.write("")
    
    
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
        st.image(schema_image, use_column_width=False, caption="", output_format='auto')
        
        
    # Objectifs
    st.header("Etapes suivantes")
    st.markdown(
        """ 
        **0. Exploration de la data**
        
        **1. Nettoyer les données pour les rendre exploitables** 
        
        **2. Entraîner différents modèles de machine learning et deep learning sur les données textes et images**
        
        **3. Créer un modèle fusion à partir de nos résultats pour valider une généralisation du modèle**
        
        **4. Développer une API pour importer des textes et images et tester la classification avec notre modèle fusion** 
        """
    )
          
  
 # page 2 ##########################################################################################################################
if page == pages[2] :
    #Dataviz
    st.title("DATAVIZ")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Produits par catégories", "Articles sans descriptions par catégories", "Longeur des textes par catégories", "Langues par catégories"])

    with tab1:
        st.header("Produits par catégories")
        st.markdown(
        """
        **Déséquilibre de classes** : Les catégories de produit du jeu de données affichent un déséquilibre notable, 
        allant de moins de 100 articles pour certaines catégories telles que figurines, confiserie ou vêtements pour enfants, 
        jusqu'à plusieurs milliers d’articles pour des catégories comme le mobilier ou les accessoires de piscine
        """
    )
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.write("")
        with col2:
    
            df_train_clean['categorie'] = df_train_clean['prdtypefull'].str.split(' - ').str[1]
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
        
        with col3:
            st.write("")
            

    with tab2:
        st.header("Articles sans descriptions par catégories")
        
        st.markdown(
        """
        **Déséquilibre de classes** : Certaines produits n'ont pas de descriptions, on retrouve ici majoritairement les livres BD, les magazines d'occasion, les cartes de jeux.
        """
    )
        
        
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.write("")
        with col2:
            
    
            # Compter le nombre de produits par catégorie
            cat_count = df_train_clean["prdtypefull"].value_counts()
            df = df_train_clean.loc[df_train_clean['description'].isna()]
            df['prdtypefull'] = df_train_clean['prdtypefull'].str.split(' - ').str[1]
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
            
            
        
        with col3:
            st.write("")

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
        
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.write("")
        with col2:
                
        
                df_train_clean['longeur'] = (df_train_clean["designation_translated"] + df_train_clean["description_translated"]).astype(str)
                df_train_clean['longeur_val'] = df_train_clean['longeur'].apply(lambda x:len(x))
                df_train_clean['prdtypefull'] = df_train_clean['prdtypefull'].str.split(' - ').str[1]
                
                # Créer un graphique à barres avec plotly express
                fig = px.box(df_train_clean,
                    x='prdtypefull',
                    y='longeur_val',
                    title="Longeur des textes par catégories",
                    labels={'prdtypefull': "Catégories", 'longeur_val': "Nombre de mots"},
                    
                    width=1400,
                    height=600,
                )

                # Mettre à jour les étiquettes de l'axe x
                fig.update_xaxes(tickangle=45, tickfont=dict(size=15))

                # Afficher le graphique
                st.plotly_chart(fig)
            
        with col3:
                st.write("")
            
    
    with tab4:
        st.header("Langues par catégories")
        
        st.markdown(
        """
        **Variabilité des langues**: Bien que nous ayons fait le choix de traduire l’ensemble du jeu de données vers une langue unique 
        (francais), on remarque que la langue varie significativement selon la catégorie de produit. 
        """
)
    
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.write("")
            

        with col2:
                
        
                # Compter le nombre de fois que chaque produit apparaît pour chaque langue
                counts = df.groupby(['prdtypefull', 'language']).size().reset_index(name='Nombre')

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
            
        with col3:
                st.write("")
    
        st.header("Langues présentes")
        st.markdown(
            """
            **Variabilité des langues** : les textes sont majoritairement rédigés en français (environ 80 %). Certains textes sont en anglais ou en allemand.
            """
        )
        
    
    

    
    
    
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
    
    st.header("Wordclouds")
    st.markdown(
        """
        Quelques représentations visuelles de wordclouds. Les worldcloud servent avant tout a représenter les mots les plus fréquents des catégories. Plus un mot est présent plus il est grand
        """
    )
    
    images = []
    for fichier in os.listdir(wc_folder):
        if fichier.endswith(".jpg") or fichier.endswith(".png"):
            images.append(os.path.join(wc_folder, fichier))
            
    
    

    # Chemin du dossier contenant les images
    images_folder = 'images/wc_visuels/'
    # Liste des noms de fichiers d'images
    image_files = os.listdir(images_folder)
    # Créer les colonnes
    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1]) 
    # Fonction pour afficher une image et du texte centré
    def display_image_with_text(image_path, text):
        with st.container():
            st.image(image_path, use_column_width=True)
            st.markdown(f"<center>{text}</center>", unsafe_allow_html=True)
    # Afficher les images et les textes
    for i, col in enumerate([col2, col4]):
        image_path = os.path.join(images_folder, image_files[i])
        with col:
            display_image_with_text(image_path, f"{image_files[i][:-4]}")
    # Afficher les images et les textes
    for i, col in enumerate([col2, col4]):
        image_path = os.path.join(images_folder, image_files[i+2])
        with col:
            display_image_with_text(image_path, f"{image_files[i+2][:-4]}")


    
    

    

  
#page 3  #############################################################################################################################################################################
if page == pages[3] :
    st.title("PREPROCESSING")
    

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
        
    col1, col2= st.columns([1, 1]) 
    
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
        

#Page4 ############################################################################################################################################ 
if page == pages[4]:
    st.title("MODELISATION : texte")
    st.subheader("approche")
    
    st.markdown("""
                Classification des images
CNN et Vision transformer (Figure 11 et 12)
Dans le domaine de la classification d'images, l'adoption de réseaux de deep learning est incontournable. 
Les réseaux de neurones convolutifs (CNN) sont particulièrement efficaces pour cette tâche mais plus récemment, 
les modèles basés sur des architectures de transformer, comme le modèle Vision Transformer (ViT) 
se sont aussi révélés efficaces pour la classification d'images. 

Pour classifier les produits sur la base des images associées, nous avons donc utilisé différents réseaux convolutifs 
(ResNet, EfficientNet et VGG) ainsi que le transformer ViT (Vision Transformer), tous pré-entraînés sur la base de données ImageNet. 

Tout comme les modèles BERT pour le traitement de texte, chaque modèle de classification d'images a été doté d'une tête de classification 
comprenant une couche dense de 128 unités suivie d'un dropout de 20%, menant à la couche de classification finale. 
L'entraînement a suivi une démarche similaire à celle employée pour les modèles BERT: entraînement sur 80% des données, 
évaluation sur les 20% restants (même partition que pour le texte), avec un fine-tuning des poids sur 8 époques d'entraînement 
et un taux d'apprentissage initial de 5e-5, réduit de 20% à chaque époque.

Les f1-scores mesurés sur l'ensemble de test révèlent une supériorité marquée du modèle Vision Transformer (ViT, f1-score = 0.675) 
comparativement au meilleur modèle CNN testé (ResNet152, f1-score = 0.658). Ces modèles image restent cependant beaucoup moins performant 
que les modèles texte, illustrant la complexité inhérente à la classification de produits sur la base exclusive d'images. Néanmoins, 
il est intéressant de noter que les catégories les plus fréquemment confondues par les modèles dédiés aux images correspondent presque exactement 
à celles posant des difficultés dans la classification de texte.
                        """)
    
    
    
    
    col1, col2 = st.columns([2, 3]) 
    
    with col1:
        res = ResultsManager(config)
        res.add_result_file('data/results_benchmark_sklearn.csv', 'text')
        res.add_result_file('data/results_benchmark_bert.csv', 'bert')
        res.add_result_file('data/results_benchmark_img.csv', 'img')
        res.add_result_file('data/results_benchmark_fusion_TF.csv', 'fusion')
        fig= res.plot_f1_scores(filter_package=['bert','text'])  
        
        st.plotly_chart(fig, use_container_width=True)
        
        # fig=res.plot_confusion_matrix('text/LinearSVC_tfidf', model_label='LinearSVC (TF-IDF)')
        # st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.title("")
    
        options = ["model1", "model2", "model3"]
        
        option_selected = st.selectbox("Choisissez un model pour afficher la matrice de confusion  :", options)
        
        if option_selected == "Model1":
            st.write("Vous avez sélectionné le model1.")
        elif option_selected == "Model2":
            st.write("Vous avez sélectionné le model2.")
        elif option_selected == "Model3":
            st.write("Vous avez sélectionné le model3.")

#Page5 ############################################################################################################################################   
if page == pages[5]:
    st.title("MODELISATION : images")
    st.subheader("approche")
    
    st.markdown("""
                Classification du texte
        Dans le contexte de la classification de produits sur la base du texte seul, nous avons commencé par examiner différentes techniques de vectorisation 
        (Bag-of-Words avec TF-IDF et Word2Vec avec Skipgram et CBOW) associées à des méthodes de classification classiques (SVM, régression logistique, arbres de décision, etc). 
        L'entraînement s'est effectué sur 80% des données, avec une évaluation des performances sur les 20% restants.
        
        Nous avons optimisé les hyper-paramètres (e.g. taille du vecteur d’embedding pour Word2Vec ou paramètres de régularisation pour SVM, etc) via une recherche exhaustive avec validation croisée à 5 folds sur l'ensemble d'entraînement. 
        Ce premier benchmark indique que la vectorisation Bag-of-Words (TF-IDF) combiné à LinearSVC ou xgBoost surpasse les méthodes Word2Vec, avec un f1-score de 0.824 pour LinearSVC basé sur TF-IDF (Figure 7).
        
        Les matrices de confusion révèlent la difficulté de ces modèles à différencier des catégories sémantiquement proches, telles que (Figure 8):
        "Maison Décoration", "Mobilier de jardin", "Mobilier", "Outillage de jardin", "Puériculture"
        "Figurines et jeux de rôle", "Figurines et objets pop culture", "Jouets enfants", "Jeux de société pour enfants"
        "Livres d'occasion", "Livres neufs", "Magazines d'occasion", "Bandes dessinées et magazines"
        "Jeux vidéo d'occasion", "CDs et équipements de jeux vidéo", "Accessoires gaming"

    
        L'emploi de modèles basés sur les transformers dans la résolution de problèmes de classification de texte est devenu incontournable. 
        Nous avons donc poursuivi notre stratégie de classification textuelle en entraînant des transformers de type BERT (Bidirectional Encoder Representations from Transformers). 
        Plusieurs versions de transformers pré-entraînés sur divers corpus français ont été comparées: CamemBERT-base, CamemBERT-ccnet et FlauBERT. 
        Chaque modèle a été complété par une tête de classification comprenant une couche dense de 128 unités suivie d'un dropout de 20%, avant d'arriver à la couche finale de classification. 
        Les modèles ont été entraînés sur 80% des données et testés sur les 20% restants (même partition que mentionnée précédemment).
        
        L'examen des matrices de confusion révèle que l'utilisation de modèles transformers réduit le taux d'erreur sur les catégories sémantiquement proches. 
        Néanmoins, ce sont les mêmes catégories qui posent toujours problème. 
        L'analyse de cas spécifiques de classifications incorrectes met en lumière la complexité inhérente à cette tâche: 
        il est difficile de déterminer à la simple lecture du texte la catégorie associée à ces produits (Figure XX).
                        """)
    
    
    
    
    col1, col2 = st.columns([2, 3]) 
    
    with col1:
        res = ResultsManager(config)
        res.add_result_file('data/results_benchmark_sklearn.csv', 'text')
        res.add_result_file('data/results_benchmark_bert.csv', 'bert')
        res.add_result_file('data/results_benchmark_img.csv', 'img')
        res.add_result_file('data/results_benchmark_fusion_TF.csv', 'fusion')
        fig= res.plot_f1_scores(filter_package=['bert','text'])  
        
        st.plotly_chart(fig, use_container_width=True)
        
        # fig=res.plot_confusion_matrix('text/LinearSVC_tfidf', model_label='LinearSVC (TF-IDF)')
        # st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.title("")
    
        options = ["model1", "model2", "model3"]
        
        option_selected = st.selectbox("Choisissez un model pour afficher la matrice de confusion  :", options)
        
        if option_selected == "Model1":
            st.write("Vous avez sélectionné le model1.")
        elif option_selected == "Model2":
            st.write("Vous avez sélectionné le model2.")
        elif option_selected == "Model3":
            st.write("Vous avez sélectionné le model3.")        




#Page6 ############################################################################################################################################   
if page == pages[6]:
    st.title("MODELISATION : fusion")
    st.subheader("approche")
    
    st.markdown("""
                composition et la mise en page avant impression. Le Lorem Ipsum est le faux texte standard de l'imprimerie depuis les années 1500, 
                quand un imprimeur anonyme assembla ensemble des morceaux de texte pour réaliser un livre spécimen de polices de texte. 
                
                Il n'a pas fait que survivre cinq siècles, mais s'est aussi adapté à la bureautique informatique, 
                sans que son contenu n'en soit modifié. Il a été popularisé dans les années 1960 grâce à la vente de feuilles Letraset 
                contenant des passages du Lorem Ipsum, et, plus récemment, par son inclusion dans des applications de mise en page de texte, 
                comme Aldus PageMaker.
                        """)
    
    
    
    
    col1, col2 = st.columns([2, 3]) 
    
    with col1:
        res = ResultsManager(config)
        res.add_result_file('data/results_benchmark_sklearn.csv', 'text')
        res.add_result_file('data/results_benchmark_bert.csv', 'bert')
        res.add_result_file('data/results_benchmark_img.csv', 'img')
        res.add_result_file('data/results_benchmark_fusion_TF.csv', 'fusion')
        fig= res.plot_f1_scores(filter_package=['bert','text'])  
        
        st.plotly_chart(fig, use_container_width=True)
        
        # fig=res.plot_confusion_matrix('text/LinearSVC_tfidf', model_label='LinearSVC (TF-IDF)')
        # st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.title("")
    
        options = ["model1", "model2", "model3"]
        
        option_selected = st.selectbox("Choisissez un model pour afficher la matrice de confusion  :", options)
        
        if option_selected == "Model1":
            st.write("Vous avez sélectionné le model1.")
        elif option_selected == "Model2":
            st.write("Vous avez sélectionné le model2.")
        elif option_selected == "Model3":
            st.write("Vous avez sélectionné le model3.")


#Page7 ############################################################################################################################################   
if page == pages[7]:
    st.title("TEST du modele")
    st.header("Classification à partir d'images ou de texte")
    
    st.subheader("image ou texte")

    options = ["Image", "Texte"]
    
    option_selected = st.selectbox("Image ou texte  :", options)
    
    if option_selected == "Image":
        col1, col2 = st.columns([2, 4])
        with col1: 
            
            # Champ de saisie pour l'URL de l'image
            image_url = st.text_input("Entrez l'URL de l'image", "https://fiverr-res.cloudinary.com/images/q_auto,f_auto/gigs/313370685/original/81a70456ffda906bfb8763cb2cb549e1359b98a7/create-web-app-with-steamlit.png")

            # Affichage de l'image à partir de l'URL
            st.image(image_url, caption="Image téléchargée depuis un lien URL")
            
            
            uploaded_file = st.file_uploader("Télécharger une image", type=['jpg', 'png'])
            # Vérifier si un fichier a été téléchargé
            if uploaded_file is not None:
                # Afficher l'image téléchargée dans un petit cadre
                st.image(uploaded_file, caption='Image téléchargée')
                
        with col2:
                # Créer les données pour le graphique
                    categories = ['Catégorie 1', 'Catégorie 2', 'Catégorie 3', 'Catégorie 4', 'Catégorie 5']
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
                    st.write("<p style='text-align:center;'><strong>Classe prédite: {}</p>".format(classe_predite), unsafe_allow_html=True)
                    st.write("<p style='text-align:center;'><strong>Classe réelle: {}</p>".format(classe_reelle), unsafe_allow_html=True)
                    st.write("<p style='text-align:center;'><strong>Estimation: {:.2f}</p>".format(estimation), unsafe_allow_html=True)
                        
            
            
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
                    categories = ['Catégorie 1', 'Catégorie 2', 'Catégorie 3', 'Catégorie 4', 'Catégorie 5']
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
                    st.write("<p style='text-align:center;'><strong>Classe prédite: {}</p>".format(classe_predite), unsafe_allow_html=True)
                    st.write("<p style='text-align:center;'><strong>Classe réelle: {}</p>".format(classe_reelle), unsafe_allow_html=True)
                    st.write("<p style='text-align:center;'><strong>Estimation: {:.2f}</p>".format(estimation), unsafe_allow_html=True)