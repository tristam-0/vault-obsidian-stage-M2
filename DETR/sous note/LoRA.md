https://arxiv.org/pdf/2106.09685
LoRA = Incremental Low-Rank Adaptation 
## 1. Objectif de LoRA

Le but de **LoRA** est de réaliser une mise à jour des poids d'un grand modèle (comme un Transformer) de manière efficace, **sans modifier l'intégralité de ses paramètres d'origine**. Les poids d'origine ($W$) restent gelés (frozen).

## 2. Le fonctionnement mathématique (Le "Low-Rank")

### Exemple avec une matrice de taille $10\,000 \times 10\,000$

Si on voulait mettre à jour directement la matrice d'origine, il faudrait créer une matrice de modification $\Delta W$ de la même taille, soit **100 millions de paramètres** à entraîner.

À la place, LoRA décompose cette matrice de modification $\Delta W$ en **deux matrices beaucoup plus petites** (de faible rang $r$, où souvent $r = 8$ ou $16$) :
- La matrice $B$ de taille **$10\,000 \times r$** (ici $10\,000 \times 8$)
- La matrice $A$ de taille **$r \times 10\,000$** (ici $8 \times 10\,000$)
$$\Delta W = B \times A$$
Dimensions : $(10\,000 \times 8) \times (8 \times 10\,000) = 10\,000 \times 10\,000$.
Le produit redonne exactement la dimension de la matrice d'origine.
### Calcul du nombre de paramètres à entraîner :
- Matrice $B$ : $10\,000 \times 8 = 80\,000$ paramètres.
- Matrice $A$ : $8 \times 10\,000 = 80\,000$ paramètres.    
- **Total à entraîner** : $80\,000 + 80\,000 = 160\,000$ paramètres au lieu de $100$ millions (soit **99,84 % d'économie**).

### initialisation
Dans l'article de Hu et al., les auteurs stipulent que la matrice $A$ est initialisée avec une distribution aléatoire normale (Gaussienne), tandis que la matrice $B$ est initialisée avec des zéros exacts.
Mécaniquement, au premier pas d'entraînement ($t=0$) :
$$B \times A = 0 \times A = 0$$
L'activation de la couche devient :
$$h = Wx + \Delta Wx = Wx + 0 = Wx$$

## 3. Comment combine-t-on les deux ?
pour obtenir la matrice finale adaptée à ta tâche ($W'$), on réalise simplement une **addition** entre la matrice originale gelée et notre modification Low-Rank.

Lors de l'entraînement (la passe avant ou _forward pass_), quand un vecteur d'entrée $x$ arrive dans la couche, le calcul se fait en parallèle :
$$h = Wx + \Delta Wx = Wx + \frac{\alpha}{r}(B \times A)x$$
L'hyperparamètre $\alpha$ est une constante fixée (souvent égale à la première valeur de $r$ que l'on teste). Le terme $\frac{\alpha}{r}$ agit comme un **coefficient de normalisation de la variance**.
Si on décide d'augmenter le rang pour donner plus de capacité d'apprentissage au modèle (par exemple, on passe de $r=8$ à $r=32$), la somme contient soudainement 4 fois plus d'éléments. La magnitude (et la variance) des valeurs dans $\Delta W$ va mécaniquement augmenter.

En intégrant $\frac{\alpha}{r}$, on met à l'échelle les activations. Si $r$ double, le diviseur double aussi. **L'amplitude des mises à jour reste mathématiquement constante.** Cela garantit que le _learning rate_ que tu as trouvé pour $r=8$ fonctionnera presque parfaitement pour $r=32$ ou $r=64$.