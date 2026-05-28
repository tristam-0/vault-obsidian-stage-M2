
## Idée Générale
L'objectif est d'ajouter une couche de généralisation de domaine de type **MoE (Mixture of Experts)** juste après le module d'auto-attention (Self-Attention) et avant le FFN standard dans un bloc Transformer (par exemple, dans l'encodeur d'un DETR). 

Contrairement à un MoE classique qui remplace le FFN, cette approche conserve le FFN pré-entraîné (gelé) et utilise des **adaptateurs légers (lightweight adapters)** en parallèle comme experts, activés dynamiquement par un **Domain Router**. Cela permet d'adapter les caractéristiques (features) de l'image en fonction de son style, de son environnement ou de son domaine.

### Flux de données dans un bloc d'encodeur :
$$\text{Features} \rightarrow \text{Self-Attention} \rightarrow \text{Router} \rightarrow \begin{cases} \text{ViT Adapter A (Domaine 1)} \\ \text{ViT Adapter B (Domaine 2)} \\ \text{Conv Adapter A (Domaine 3)} \\ \text{Conv Adapter B (Domaine 4)} \end{cases} \rightarrow \text{FFN}$$

![[Customizing Domain Adapters for Domain Generalization-1779957815056.webp]]

---
## 1. Le Router (Domain Router)

### Fonctionnement :
Le routeur utilise une approche de **Soft Routing** via une fonction **Softmax** (Équation 4 du papier). calcule une distribution de poids pour chaque image.
$$\text{Sortie} = \sum_{i=1}^{N} \text{Softmax}_i(Wz) \cdot E_i(z)$$
* Chaque image active **tous** les adaptateurs en même temps.
* La sortie finale est une combinaison (somme pondérée) des contributions de chaque expert. Par exemple, une image pourra être traitée à 70% par le ViT Adapter A et à 30% par le Conv Adapter B.
* **Avantage :** Cela permet au DETR de combiner les forces de plusieurs domaines si une image de test se trouve "entre deux" styles connus.
* *déaventage* : des conv adapteur persque inutile veux come meme étre utiliser (ex conv adpter C avec 0.01%) (idée solution techolde + resofmax ? probléme posible ?)
---
## 2. Le ViT Adapter (Vision Transformer Adapter)

### Caractéristiques :
* Structure linéaire classique en goulot d'étranglement (Bottleneck).
* Prend l'entrée $h$ (de dimension canal $q$) et applique la transformation suivante :
$$h_{out} = h + f(h W_{down}) W_{up}$$

Où :
* $W_{down} \in \mathbb{R}^{q \times r}$ est la projection basse (réduit la dimension de $q$ à $r$, avec q>>r; ex : $r=48$).
* $f(\cdot)$ est la fonction d'activation non-linéaire **GELU**.
* $W_{up} \in \mathbb{R}^{r \times q}$ est la projection haute (remonte la dimension de $r$ à $q$).
* Fonctionne directement sur les tokens au format 1D. Très léger et facile à paralléliser.

### Question : Est-il utilisable dans le Décodeur ?
> [!NOTE] Compatibilité Décodeur
> Bien que l'article ne l'applique que sur l'encodeur, le ViT Adapter manipule des tokens purement séquentiels (1D). Dans le décodeur d'un DETR, les *Object Queries* sont également des tokens 1D. Un ViT Adapter peut donc être inséré dans le décodeur sans aucune modification structurelle. Cela permettrait d'adapter la sémantique des requêtes d'objets selon le domaine. mais faix mon de sens que dans l'encodeur

---

## 3. Le Conv Adapter (Convolutional Adapter)

### Caractéristiques :
* Inspiré des blocs résiduels goulots d'étranglement de ResNet.
* Applique la transformation mathématique suivante :

$$h_{out} = h + \text{Conv}_{1\times1}(f(\text{Conv}_{3\times3}(f(\text{Conv}_{1\times1}(h)))))$$

Où :
* $\text{Conv}_{1\times1}$ (basse) réduit le nombre de canaux de $q$ à $c$ (avec $c=8$).
* $\text{Conv}_{3\times3}$ applique la convolution sur le voisinage 2D pour capturer le biais inductif local.
* $\text{Conv}_{1\times1}$ (haute) ré-augmente les canaux de $c$ à $q$.
* $f(\cdot)$ est l'activation **GELU**.
* Nécessite de remodeler temporairement (*reshape*) les tokens 1D en une grille spatiale 2D and zero paddings avant la convolution $3\times3$, puis de les ré-aplatir en 1D.

### Question : Est-il utilisable dans le Décodeur ?
> [!WARNING] Incompatibilité fondamentale avec le Décodeur 
> **Oui, le Conv Adapter est fondamentalement incompatible avec le Décodeur d'un DETR.** > Pour appliquer une convolution $3\times3$, les tokens doivent posséder une **cohérence spatiale** (c'est-à-dire que des tokens côte à côte dans la matrice doivent représenter des pixels voisins dans l'image réelle).
> * **Dans l'Encodeur :** Les tokens correspondent à des patchs d'images juxtaposés. Le passage en grille 2D est donc mathématiquement et visuellement correct.
> * **Dans le Décodeur :** Les tokens sont des *Object Queries* (des requêtes abstraites, ex: *"cherche le premier objet"*, *"cherche le deuxième objet"*). Ces requêtes n'ont **aucune topologie spatiale de grille 2D**. Faire un *reshape* 2D sur des Object Queries pour leur appliquer un filtre de convolution $3\times3$ n'a aucun sens et  risque de détruire complètement l'apprentissage.

## entrainement

l'aticle conseill d'utiliser un modele par entréner avec pois geler