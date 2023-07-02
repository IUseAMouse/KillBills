import os
from dotenv import load_dotenv

import psycopg2
import pickle as pkl

load_dotenv()


def load_data(sample_size: int, local_save="False"):
    try:
        handler = psycopg2.connect(
            host = os.environ['HOST'],
            user = os.environ['user_name'],
            password = os.environ['password'],
            database = os.environ['dbname'],
            port = os.environ['port']
        )

        table = os.environ['table']

        cursor = handler.cursor()
        # Il y a 8.886.767 éléments dans la table, gardons seulement
        # un subset pour établir un premier modèle

        cursor.execute(f"select items, amount from {table} limit {sample_size}")
        data = cursor.fetchall()

        if local_save:
            with open("./data.pkl", "wb") as output:
                pkl.dump(data, output)
        
        handler.close() 
        return data

    except:
        print("Couldn't load database")
        return -1


    

    
