# KillBills - Clustering et Classification d'articles de caisse, Yvann Vincent

Le répertoire GitHub contient une solution qui se décompose en plusieurs modules pour le traitement des données. Voici une description plus détaillée de chaque module :

## Module Loader

Le module Loader est responsable de la connexion à la base de données POSTGRE et de la collecte des données nécessaires pour le traitement ultérieur. Il charge les informations de connexion à la base de données à partir d'une variable d'environnement située dans le fichier .env. Cette approche permet de ne pas coder ces informations en dur dans le code source, évitant ainsi qu'elles n'apparaissent publiquement sur le répertoire GitHub.

En utilisant les informations de connexion extraites de la variable d'environnement, le module Loader établit une connexion sécurisée à la base de données POSTGRE et récupère les données requises pour le reste du traitement. Cela garantit que les informations sensibles ne sont pas exposées dans le code source et que les paramètres de connexion peuvent être facilement modifiés en fonction de l'environnement d'exécution.

**Il n'est pas nécessaire de faire appel à ce module pour lancer le projet**

## Module Processing

Le module Processing joue un rôle crucial dans le traitement des données brutes collectées. Il effectue différentes étapes de transformation sur ces données afin de les rendre exploitables pour d'autres tâches. Parmi les transformations effectuées, on retrouve l'utilisation d'un modèle d'extraction syntaxique/sémantique d'embeddings. Ce modèle permet de capturer les informations contextuelles des données, ce qui facilite leur utilisation ultérieure.

**Il n'est pas nécessaire de faire appel à ce module pour lancer le projet**

## Module Demo

Le module Demo a pour objectif de présenter une démonstration complète du fonctionnement du système de clustering mis en place dans cette solution. Pour cela, il implémente deux algorithmes de clustering populaires :

- DBSCAN 
- GMM 

En utilisant ces algorithmes, le module Demo effectue le clustering des données préalablement transformées. De plus, il offre des visualisations par ACP pour aider à choisir le nombre optimal de clusters. Enfin, ce module propose également l'implémentation d'un classifieur Random Forest, qui a été entraîné spécifiquement pour ce problème de clustering. Ce classifieur atteint une precision/recall/accuracy de 90% en moyenne d'une itération à l'autre.

Pour lancer la démo : python ./demo.py

## Module Predict

Le module Predict permet d'effectuer l'inférence du classifieur Random Forest sur des termes spécifiques choisis par l'utilisateur. Pour cela, il utilise les fichiers sauvegardés du modèle d'embedding pour obtenir les représentations vectorielles des termes sélectionnés. Ensuite, le classifieur Random Forest est utilisé pour prédire les clusters correspondants à ces termes. Cette fonctionnalité permet à l'utilisateur d'explorer et de comprendre comment les termes choisis sont regroupés par le système de clustering.

Pour lancer la prédiction en sandbox : python ./predict.py


## Choix de méthode : Extraction sémantique des noms d'articles
Dans le but de regrouper les articles de caisse par catégories, j'ai choisi d'extraire des informations à partir des noms d'articles. L'objectif est de rendre la proximité sémantique entre les articles apparente dans leur représentation numérique. Pour cela, un modèle Word2Vec a été utilisé, entraîné spécifiquement sur un corpus composé des noms d'articles issus des tickets de caisses.

Cette approche présente plusieurs avantages :

- En utilisant un modèle Word2Vec, qui repose sur la distribution des mots dans un corpus, les similarités sémantiques entre les mots sont capturées. Ainsi, les articles de caisse partageant des termes sémantiquement proches seront plus susceptibles d'être regroupés ensemble dans les résultats de clustering.

- Étant basée sur des représentations vectorielles des mots, cette approche est indépendante de la langue utilisée dans les noms d'articles. Elle peut donc être appliquée à des corpus de différentes langues, ce qui la rend adaptable à des contextes multilingues. Elle est donc extensible à des jeux de données en anglais, par exemple.

- Contrairement à une approche de clustering basée sur des catégories prédéfinies, l'extraction sémantique des noms d'articles permet d'identifier des regroupements naturels dans les données, sans dépendre d'une catégorisation préalable. Cela peut révéler des associations inattendues entre les articles et permettre d'explorer des regroupements potentiels non envisagés initialement.

## Confirmation de la méthode par ACP

L'avantage de l'ACP appliquée sur les embeddings extraits par Word2Vec réside dans le fait qu'elle révèle des clusters et des patterns facilement identifiables à l'œil nu. En réduisant la dimensionnalité des données, l'ACP permet de visualiser les relations entre les articles de manière plus concise et compréhensible.

Les clusters qui se forment dans l'espace réduit sont souvent évidents lorsque la quantité de données devient importante, ce qui facilite l'interprétation des résultats de clustering.

Afin que vous puissiez le voir je me suis permis de joindre une image, "pca_example.png", qui montre un exemple d'ACP ou des patterns et des clusters évidents apparaissent à partir des embeddings extraits par Word2Vec.


