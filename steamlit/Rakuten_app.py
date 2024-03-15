import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


custom_css = """
<style>
    /* Styles pour spécifier la taille du texte */
    body {
        font-size: 16px; /* Taille de la police pour tout le texte */
        font-family: 'Roboto', sans-serif; /* Utiliser Roboto pour tout le texte */
        background-color: #eee;
    }
    h1 {
        font-size: 36px; /* Taille de la police pour les titres de niveau 1 */
        font-family: 'Roboto', sans-serif; /* Utiliser Roboto pour tout le texte */
    }
    h2 {
        font-size: 24px; /* Taille de la police pour les titres de niveau 2 */
        font-family: 'Roboto Light', sans-serif; /* Utiliser Roboto pour tout le texte */
    }
    p {
        font-size: 20px; /* Taille de la police pour les paragraphes */
        font-family: 'Roboto Light', sans-serif; /* Utiliser Roboto pour tout le texte */
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)



# chargement des dataframes
df=pd.read_csv("train.csv")
df_train_clean=pd.read_csv("data/clean/df_train_index.csv")
xtrain=pd.read_csv("data/X_train.csv")
ytrain=pd.read_csv("data/Y_train.csv")

# Logo avec centrage
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.write("")
with col2:
    st.image("rakuten.png")
with col3:
    st.write("")



#chargement images
schema_dataframe="images/Schema_dataframe.png"
schema_image="images/Schema_images.png"
schema_dataframe_Y="images/Schema_dataframe_Y.png"
schema_prepro_txt="images/schema_prepro_txt.jpg"
schema_prepro_img="images/schema_prepro_img.jpg"
graf_isnaPrdt="images/graf_isnaPrdtypecode.png"
graf_txtLong ="images/graf_boxplot.png"
graf_lang ="images/lang.jpg"
graf_corr ="images/corr.jpg"
graf_WC ="images/maskWC.png"


# sommaire
st.sidebar.title("Sommaire")
pages=["Presentation", "Exploration", "DataViz", 'Préprocessing', "Modélisation texte", "Modélisation images", "Modélisation fusion", "Bonus", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.header("Auteurs")
st.sidebar.markdown("[Thibaut Benoist](link)")
st.sidebar.markdown("[Julien Chanson](link)")
st.sidebar.markdown("[Julien Fournier](link)")
st.sidebar.markdown("[Alexandre Mangwa](link)")


# page 0----------------------------------------------------------------------------------------------------------------------------------------------
if page == pages[0]:
    # Titre principal
    st.title("Classification de produits")

    # Description du projet
    st.markdown(
        """
        L'objectif principal de ce projet est de développer un ou plusieurs modèles prédictifs capables de classifier précisément (au sens du f1-score)
        chacun des produits en se basant sur les descriptions et images fournis.
        """
    )

    st.title("1. EXPLORATION")
    
    # Problématique
    st.header("1.1 Problématique")
    st.markdown(
        """
        La catégorisation des produits pour les catalogues en ligne est essentielle pour les places de marché électroniques. 
        Elle englobe la classification des titres et des images, ce qui a un impact majeur sur divers aspects tels que la recherche, 
        les recommandations personnalisées et la compréhension des requêtes.

        Dans cet environnement, l'intégration de méthodes de machine learning multimodales devient cruciale. 
        Ces approches exploitent la combinaison des données textuelles et visuelles des produits pour améliorer l'efficacité et la précision des résultats. 
        Pour les entreprises de commerce électronique, cette stratégie offre une opportunité significative d'optimisation et de performance
        """
    )

    
    st.subheader("Dataframes & Images")
    st.markdown("""
                ##### Dataframe X_train
                """
                )
    st.image(schema_dataframe)
    st.markdown("""
                ##### Dataframe Y_train
                """
                )
    st.image(schema_dataframe_Y)
    st.markdown(
        """
        Nous pouvons noter plusieurs choses sur le dataframe :
        - beaucoup de Nan
        - plusieurs langues
        - des balises HTML
        - des mauvais encodings
        
        Nous avons également 84 916 produits pour le dataframe de train, le test en comprend 13 812
        """
    )
    st.image(schema_image)
    st.markdown(
        """
        Nous pouvons noter plusieurs choses sur les images :
        - toutes en 500-500
        - Certaines avec des paddings blanc tout autour
        
        Nous avons le même nombre de produits
        """
    )
    
    # Objectifs
    st.header("1.2 Objectifs")
    st.markdown(
        """ 
        A partir de la nous allons donc procéder à:
        
        **1. Nettoyer les données pour les rendre exploitables** 
        
        **2. Entraîner différents modèles de machine learning et deep learning sur les données textes et images**
        
        **3. Créer un modèle fusion à partir de nos résultats pour valider une généralisation du modèle**
        
        **4. Développer une API pour importer des textes et images et tester la classification avec notre modèle fusion** 
        """
    )
          
  
 #page1 ----------------------------------------------------------------------------------------------------------------------------------------------
if page == pages[1] :
    #Dataviz
    st.title("DATAVIZ")
    
    tab1, tab2, tab3 = st.tabs(["Produits par catégories", "Articles sans descriptions par catégories", "Longeur des textes par catégories"])

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
                labels={"x": "Catégorie", "y": "Nombre de produits"},
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
                labels={"x": "Catégorie", "y": "Nombre de produits"},
                color=nb_categories_sorted,
                color_discrete_sequence=px.colors.sequential.Viridis,
                width=1400,  # spécifier la largeur du graphique
                height=600,  # spécifier la hauteur du graphique
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=15))
            st.plotly_chart(fig)
        
        with col3:
            st.write("")

    with tab3:
        st.header("Longeur des textes par catégories")
        
        st.markdown(
        """
        **Déséquilibre de textes** : La longueur des descriptions textuelles des produits montrent une variabilité importante. 
        Les descriptions des livres d’occasion ou des cartes de jeux sont généralement brèves (quelques dizaines de mots, champ description absent), 
        tandis que celles des jeux vidéo pour PC s'étendent souvent sur plusieurs centaines de mots.
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
                labels={"x": "Catégorie", "y": "Nombre de produits"},
                color=nb_categories_sorted,
                color_discrete_sequence=px.colors.sequential.Viridis,
                width=1400,  # spécifier la largeur du graphique
                height=600,  # spécifier la hauteur du graphique
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=15))
            st.plotly_chart(fig)
        
        with col3:
            st.write("")
    
    
    st.header("Analyses du nombre de produits par classes")
    
        
        
    #st.image(graf_isnaPrdt)
    
    st.header("2.2 Longueur des textes par catégories")
    st.image(graf_txtLong)

    st.header("2.3 Langues présentes")
    st.markdown(
        """
        **Variabilité des langues** : les textes sont majoritairement rédigés en français (environ 80 %). Certains textes sont en anglais ou en allemand.
        """
    )
    st.image(graf_lang)
    
    st.header("2.4 Corrélation des catégories")
    st.markdown(
        """
        **Séparabilité des catégories** : Certaines catégories ont un chevauchement lexical notable (par exemple, les consoles de jeu et les jeux vidéo), 
        comme on peut le remarquer dans les wordclouds ou dans la matrice de corrélation entre vecteurs de fréquence des mots.
        
        On peut voir que les catégories livres neufs, livres d'occasion et Magazines d'occasion sont tres correlées, nous avons également la même analyse mais dans une moindre mesure pour les consoles de jeux,
        jeux de societés, jeux vidéo d'occasion, équipement pour jeux video
        """
    )
    st.image(graf_corr)
    
    st.header("2.5 Wordclouds")
    st.markdown(
        """
        Quelques représentations visuelles de wordclouds. Les worldcloud servent avant tout a représenter les mots les plus fréquents des catégories. Plus un mot est présent plus il est grand
        """
    )
    st.image(graf_WC)
    
    

    

  
# page 2  
if page == pages[2] : 
    st.write("### Modélisation")
    st.header("Traitement sur le texte")
    st.markdown(
        """
        Pour reprendre les objectifs par rapport au texte nous allons nettoyer nos colones 
        Voici un schéma explicatif de la procédure :
        """
    )
    st.image(schema_prepro_txt)
    st.header("Traitement sur les images")
    st.markdown(
        """
        Concernant les images le padding va etre ajusté
        """
    )
    st.image(schema_prepro_img)
  
  
