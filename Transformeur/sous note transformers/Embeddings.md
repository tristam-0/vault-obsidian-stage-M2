Les embeddings transforment une séquence de tokens (mots/sous-mots) en **vecteurs** numériques denses qui capturent leur **signification sémantique** et leurs **relations contextuelles**. Cela permet au modèle de traiter le texte comme des données mathématiques.


## Pipeline complet : texte → embeddings
Texte d'entrée : "I like pizza!"
↓
Tokenisation : ["<i>", "<like>", "<pizza>", "<!>", "<unk>"]
↓
IDs de tokens : [5, 1247, 4589, 2, 0]   (via lookup table)
↓
Embeddings : lookup dans matrice (vocab_size × dim)
↓
Matrice finale : [batch, seq_len, embedding_dim] = [1, 5, 512]

























## Comment implémenter un embedding
Word2Vec repose sur un concept simple mais puissant : entraîner un réseau de neurones élémentaire pour réaliser l’une des deux tâches suivantes :

- **CBOW (Continuous Bag of Words)** : cette méthode prédit un mot cible à partir d’un ensemble de mots de contexte qui l’entourent.
- **Skip-gram** : à l’inverse, cette méthode vise à prédire les mots de contexte à partir d’un mot cible donné.
- Word2Vec repose sur un concept simple mais puissant : entraîner un réseau de neurones élémentaire pour réaliser l’une des deux tâches suivantes :
![[Pasted image 20260430152123.webp]]
- **CBOW (Continuous Bag of Words)** : cette méthode prédit un mot cible à partir d’un ensemble de mots de contexte qui l’entourent.
- **Skip-gram** : à l’inverse, cette méthode vise à prédire les mots de contexte à partir d’un mot cible donné.
- - **Input** : le réseau reçoit en entrée des vecteurs de mots sous forme de one-hot encoding.
- **Encodeur** : une première couche, fonctionnant comme une projection, compresse ces vecteurs dans un espace de dimension réduite, enrichissant ainsi les vecteurs de caractéristiques sémantiques.
- **Décodeur** : les vecteurs compressés sont ensuite projetés vers un espace de dimension plus grande, similaire à celle de l’entrée.
- **Softmax** : enfin, une fonction softmax est appliquée pour transformer ces vecteurs en probabilités, indiquant la chance de chaque mot d’être le mot cible ou de figurer dans le contexte du mot cible, selon le modèle utilisé.
