import pickle as pkl
import numpy as np
from processing import get_name_embedding

with open("embedding_model.pkl", "rb") as file:
    emb_model = pkl.load(file)

with open("classifier_model.pkl", "rb") as file:
    clf_model = pkl.load(file)

def predict(name, emb_model, clf):
    embedding = get_name_embedding(name, emb_model)
    embedding = np.array([embedding])
    out_classes = clf.predict(embedding)

    return out_classes


while True:
    input_name = input("De quel article voulez-vous chercher la classe ? : ")
    pred = predict(input_name, emb_model, clf_model)

    print(f"La classe associée est {pred}")