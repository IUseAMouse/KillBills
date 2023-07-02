import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn import metrics
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from loader import load_data
from processing import format_data, train_word2vec, extract_embeddings, get_name_embedding


def random_set():
    """
    Associe aléatoirement une classe train ou test à une entrée
    """
    val = random.randint(0, 10)
    if val < 8:
        return "train"
    return "test"


def create_clusters_GMM(number_of_clusters,points):
    """
    Créé des clusters en mélange de gaussiennes 
    à partir des embeddings d'un Word2Vec
    """
    gmm = GaussianMixture(n_components=number_of_clusters, random_state=0, n_init=10).fit(points)
    labels = gmm.predict(X)
    l_array = np.array([[label] for label in labels])
    clusters = np.append(points,l_array,axis=1)
    return clusters


def create_clusters_dbscan(points):
    """
    Créé des clusters par analyse de densité
    à partir des embeddings d'un Word2Vec
    """
    db = DBSCAN(eps=3).fit(points)
    l_array = np.array([[label] for label in db.labels_])
    clusters = np.append(points,l_array,axis=1)
    n_clusters = len(set(db.labels_))
    return clusters, n_clusters


def train_classifier(train, clf):
    """
    Entraine n'importe quel classifieur sklearn
    """
    X = np.array([list(arr) for arr in train['Embedding']])
    y = np.array([cluster for cluster in train['GMM_Cluster']])

    clf.fit(X, y)

    return clf


def predict(name, emb_model, clf):
    embedding = get_name_embedding(name, emb_model)
    embedding = np.array([embedding])
    out_classes = clf.predict(embedding)

    return out_classes


if __name__ == '__main__':
    #####################################################################
    # Chargement data et des embeddings des noms de produits
    #####################################################################
    print(30*'-')
    print('Chargement des données...')
    data = load_data(sample_size=50000, local_save=False)
    data = format_data(data)
    emb_model = train_word2vec(data)
    data = extract_embeddings(data, emb_model)

    # Définition d'un clustering set, training set et testing set
    data['Set'] = data['Name'].apply(lambda x: random_set())
    X = np.array([list(arr) for arr in data['Embedding']])

    #####################################################################
    # Visualisation ACP pour choisir un type de clustering
    #####################################################################
    print(30*'-')
    print("Chargement de l'ACP...")
    pca = PCA(n_components=2)
    pca.fit(X)

    X_pca = pca.transform(X)

    pca_df = pd.DataFrame(X_pca, columns = ['x', 'y'])

    fig, ax = plt.subplots()
    ax.scatter(
        x=pca_df['x'],
        y=pca_df['y']
    )
    plt.plot()
    plt.show()

    n_clusters = int(input("Après ACP, quel nombre de clusters choisir : "))
    
    #####################################################################
    # Application d'un DBSCAN/KMeans
    #####################################################################
    print(30*'-')
    print("Recherche d'un clustering fonctionnel...")

    max_dist = 20
    clusters, n_clusters_dbs = create_clusters_dbscan(X)

    if n_clusters_dbs >= n_clusters :
        print(f"Solution possible avec DBSCAN")
        clusters = clusters[:, clusters.shape[1] - 1]
        data['DBSCAN_Cluster'] = clusters

        fig, ax = plt.subplots()
        ax.scatter(
            x=pca_df['x'],
            y=pca_df['y'],
            c=clusters
        )
        plt.plot()
        plt.show()
        
    
    print(f"Calcul de solution alternative en mélange de gaussiens")
    clusters = create_clusters_GMM(n_clusters, X)

    clusters = clusters[:, clusters.shape[1] - 1]
    data['GMM_Cluster'] = clusters

    fig, ax = plt.subplots()
    ax.scatter(
        x=pca_df['x'],
        y=pca_df['y'],
        c=clusters
    )
    plt.plot()
    plt.show()

    
    #####################################################################
    # Entrainement d'un classifieur
    #####################################################################
    print(30*'-')
    print("Entrainement d'un classifieur...")
    train = data[data['Set'] == 'train']
    test = data[data['Set'] == 'test']
    X_test = np.array([list(arr) for arr in test['Embedding']])
    y_test = np.array([cluster for cluster in test['GMM_Cluster']])

    clf_model = RandomForestClassifier()
    clf_model = train_classifier(train, clf_model)

    y_pred = clf_model.predict(X_test)
    model_precision = metrics.precision_score(y_test, y_pred, average='macro')
    print(f"Le classifieur performe avec une précision de {model_precision}")

    print(f"Test de la fonction d'inférence du classifieur sur le mot 'Pain au chocolat':")
    test_article = "Pain au chocolat"
    pred_class = predict(test_article, emb_model, clf_model)
    print(f"Pain au chocolat est de classe {pred_class} selon les patterns trouvés par le clustering")

    with open("embedding_model.pkl", "wb") as file:
        pkl.dump(emb_model, file)

    with open("classifier_model.pkl", "wb") as file:
        pkl.dump(clf_model, file)






