import pickle as pkl
import pandas as pd 
import numpy as np 

import gensim
from gensim.utils import tokenize
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

from loader import load_data

# with open("data.pkl", "rb") as source:
#     data = pkl.load(source)
    

def format_data(data: list):
    """
    Conserve le nom et le prix d'un item,
    nettoie le nom de l'item en cas de présence
    de caractères parasites.

    Paramètres
    data (list): List des données brutes

    Retour:
    data (pd.DataFrame): DataFrame des données nettoyées
    
    """
    new = list()
    for entry in data:
        try:
            name = entry[0].split(',')[4]
            price = float(entry[1])
            
            new += [[name, price]]
        except:
            pass

    # Passage à numpy pour faciliter l'indexage
    new = np.array(new)
    data = pd.DataFrame({'Name': list(new[:, 0]), 'Price': list(new[:, 1])}).drop_duplicates()

    # Nouvel index
    data['index'] = range(len(data))
    data = data.set_index('index')

    # Nettoyage des noms
    data['Name'] = data['Name'].apply(
        lambda name: name.strip('\"')
    )

    return data


def train_word2vec(data):
    train_names = [list(tokenize(name, deacc=True, lower=True)) for name in data['Name']]
    model = Word2Vec(sentences=train_names, vector_size=100, window=3, min_count=1, workers=4)

    return model


def get_vect(word: str, model: gensim.models.Word2Vec):
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros((model.vector_size,))


def get_name_embedding(sentence: str, model: gensim.models.Word2Vec):
    name_embedding = sum(get_vect(w, model) for w in sentence)
    if type(name_embedding) == int:
        name_embedding = np.zeros((model.vector_size,))
    return name_embedding


def extract_embeddings(data: pd.DataFrame, model: gensim.models.Word2Vec):
    data['Embedding'] = data['Name'].apply(lambda name: get_name_embedding(name, model))
    return data