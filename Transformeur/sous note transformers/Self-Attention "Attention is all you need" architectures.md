ici on parle de l'architectures décrite dans le papier Attention is all you need mais il existe des varient / modification [[self-Attention variantes liste]]
On reçoit des [[Embeddings]] à l'entrée du bloc.
On calcule le score d'attention sur plusieurs têtes (chacune avec 3 matrices $W_Q$ $W_K$ $W_V$ entraînées), puis on les combine avec une matrice finale $W_O$ qu'on entraîne aussi.
![[Pasted image 20260427154025.webp]]
## Detaille.
### One head Attention
Dans une tête de self-attention, on a 3 matrices $W_Q$ $W_K$ $W_V$ qui sont entraînées avec les [[Transformers base (NLP)]].
Elles sont utilisées pour obtenir les matrices $Q$, $K$ et $V$ à partir de la matrice de tokens $X$.
On calcule ensuite le score d’attention avec la formule suivante :
![[Pasted image 20260427142600.webp]]
#### Calcule score d'attention. 
Ici, $x_1$ et $x_2$ proviennent d’une [[tokenisation]] précédente ; dans l'exemple ils représentent les mots delicious et pizza.
Normalement, on assemble $x_1$ et $x_2$ pour former la matrice $X$. Mais ici, pour mieux comprendre, on va utiliser $x_1$ et $x_2$ à la place.

Nous allons multiplier $X_1$ avec chaque matrice pour trouver 3 vecteurs de plus petite dimension.
$q_1$, $k_1$ et $v_1$ : c’est en utilisant ces vecteurs que l’on va calculer le score d’attention.

Note : valeur usuelle $x_1$ vecteur de taille 512 alors que q,k,v sont de taille 64. C’est un choix architectural pour rendre le calcul plus petit et plus rapide.
Note 2: si on avais utiliser la matrice X alors on aurais les matrice Q, K, V
Note 3: Q = Queries, K Keys, V =  Values
![[Pasted image 20260427135759.webp]]

Dans un premier temps, on calcule les **scores d’alignement** de $q_1$ avec chaque clé, pour mesurer ==la similarité entre cette query et les clés des autres tokens.== 

puis on va diviser par $\sqrt d_k$  pour préparé utilisation du softmax et évité une variance qui explose ou un softmax peu discriminant.  mais la valeur est un **choix d'architecture**
>exemple  :
		Sans scaling  : softmax([98,80,5])               = [0.999999985, 1.52e-08, 4e-41] 
		Avec √64=8   : softmax([12.25,10,0.625]) = [0.905, 0.095, 8e-6] 
		Avec 64          : softmax([1.53,1.25,0.078]) = [0.503, 0.380, 0.118]

on fois fais le softmax on a des poids qui dise **à quel point le token $x_j$ est pertinent pour le token $x_i$**.
Le nouveau vecteur pour le token i est une combinaison linéaire des **tous les vecteurs Values** de la séquence, pondérée par leur **pertinence relative**.

note : formule softmax : $\sigma(s_i) = \frac{e^{s_i}}{\sum_{j=1}^n e^{s_j}} \quad \text{pour} \quad i = 1, \dots, n$ 
![[Pasted image 20260427150754.webp]]
autre explication des calcule : 
- https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3/
- https://medium.com/@sachinsoni600517/cross-attention-in-transformer-f37ce7129d78
### Multi head Attention
avec plusieurs head chaque head a 3 matrices différence pour chaque ex head 1 $W1_Q$ $W1_K$ $W1_V$.
![[Pasted image 20260427152236.webp]]on fais le calcule d'attention pour chaque head
![[Pasted image 20260427152303.webp|697]]puis on les multiplie avec un matrice $WO$ pour avoir un résulta qui capture des information de tous les head d'attention.
![[Pasted image 20260427152525.webp]]
et on retrouve finalement la taille original de la matrice X sous la forme de la matrice Z résultat de la self-attention.

