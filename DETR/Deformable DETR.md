Cet article propose des modification de architecture [[DETR]]
## Amélioration par rapport à DETR. 
amélioration des performances sur les petits objets et meilleures performances obtenues en 10 fois moins d'entraînement 

## origine de l'idee
![[Deformable convolution]]
## Deformable DETR — Attention
propose une modification du calcule de l'attantion
de

### Deformable DETR — Attention Classique
Le papier définit la Multi-Head Attention classique de la manière suivante :
$\LARGE \mathrm{MultiHeadAttn}(z_q, x) = \sum_{m=1}^{M} W_m \left[ \sum_{k \in \Omega_k} A_{mqk} \cdot W'_m x_k \right]$
Pour simplifier, on peut isoler le calcul d'une seule tête ($m$) :

$\LARGE \mathrm{OneHeadAttn}(z_q, x) = \sum_{k \in \Omega_k} A_{mqk} \cdot W'_m x_k$
Les poids d'attention représentent la matrice de similarité normalisée (Softmax) :

$$A_{mqk} \propto \exp \left\{ \frac{(U_m z_q)^T (V_m x_k)}{\sqrt{C_v}} \right\} = \exp \left\{ \frac{z^T_q U^T_m V_m x_k}{\sqrt{C_v}} \right\}$$

- **$\propto$ (Proportionnel à)** : Signifie que la somme de tous les $A_{mqk}$ pour un $q$ donné est égale à 1. C'est l'équivalent mathématique de la fonction **Softmax**.
#### Lexique des notations
##### Les indices et ensembles
- **$m$** : Indice de la tête d'attention ($m \in \{1, \dots, M\}$ avec $M$ le nombre total de têtes).
- **$q$** : Indice de l'élément de la requête (_Query_), associé au vecteur de caractéristiques $z_q$.
- **$k$** : Indice de l'élément de la clé/valeur (_Key/Value_), associé au vecteur de caractéristiques $x_k$.
- **$\Omega_k$** : L'ensemble des indices des clés (représente l'ensemble des pixels de l'image ou des tokens)
##### Les variables et dimensions
- **$C$** : Dimension totale de l'embedding (le nombre de canaux/features de l'image ou du modèle).
- **$C_v$** : Dimension interne de chaque tête d'attention. Elle est définie par $C_v = \frac{C}{M}$ (souvent notée $d_k$ dans d'autres papiers).
##### Les matrices de projection
- **$W_m \in \mathbb{R}^{C \times C_v}$** : Matrice de projection de sortie pour la tête $m$. Elle prend les résultats de dimension $C_v$ et les ramène dans la dimension d'origine $C$ pour pouvoir faire la somme sur les $M$ têtes.
- **$W'_m \in \mathbb{R}^{C_v \times C}$** : Matrice de projection pour générer la **Value**. Elle projette les features d'entrée $x_k \in \mathbb{R}^C$ dans l'espace de la tête d'attention de dimension $C_v$.
- **$U_m \in \mathbb{R}^{C_v \times C}$** : Matrice utilisée pour projeter la Query $z_q$. Dans la formule, on utilise sa transposée $U_m^T$.
- **$V_m \in \mathbb{R}^{C_v \times C}$** : Matrice utilisée pour projeter la Key $x_k$.

## Attention déformable
L'idée centrale est de passer d'un calcul **global** (regarder partout HW) à un calcul **local et adaptatif** (regarder uniquement quelques points pertinents N<<HW).

idée est inspirée de la convolution déformable :
![[Pasted image 20260504145939.webp]]
le posésuse est le suivent :
1. Convolution Classique (Grille rigide 3x3) 
2. Détermination des Offsets (on a un tenseur 2N canaux pour avoir $\Delta x , \Delta x$) 
3. Grille Déformée (Interpolation bilinéaire pour trouvais quelle pixel utiliser)  
4. Convolution Finale (Sur les pixels décalés de la feature map original)

**Dans l'Attention Déformable :** C'est la **Query ($z_q$)** qui va prédire elle-même où elle veut regarder (les offsets) et avec quelle importance (les poids d'attention).

### Deformable Attention Module
$\LARGE \mathrm{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^{M} W_m \left[ \sum_{k=1}^{K} A_{mqk} \cdot W'_m x\bigl(p_q + \Delta p_{mqk}\bigr) \right]$Pour une seule head d'attention ($m$) :
$\LARGE \mathrm{OneDeformAttn}(z_q,p_q, x) = \sum_{k=1}^{K} A_{mqk} \cdot W'_m x\bigl(p_q + \Delta p_{mqk}\bigr)$
#### 1. Les Entrées
- **$z_q$** : Le vecteur de la requête (Query).
- **$x$** : La carte de caractéristiques d'entrée (taille $H \times W \times C$).
- **$p_q$** : Le point de référence (coordonnées 2D $[x, y]$ normalisées dans $[0, 1]$).
    - **Comment est-il déterminé ?**
        - **Dans l'Encodeur :** Ce sont des coordonnées fixes générées à partir d'une grille régulière. Le point $p_q$ correspond au centre de chaque pixel/cellule de la feature map.
        - **Dans le Décodeur :** Il est prédit dynamiquement à partir de l' *Object Query* par une couche linéaire suivie d'une fonction `Sigmoid`
			- [[Anchor DETR]] possible d'utiliser une grille comme dans anchore DETR.
#### 2. Le mécanisme d'échantillonnage local ($K$)
- **$K$** : Le nombre total de points d'échantillonnage par tête (avec $K \ll HW$). Le modèle ne va regarder que $K$ pixels au lieu de toute l'image.
- **$\Delta p_{mqk}$** : L'offset 2D (déplacement en $\Delta x, \Delta y$).
	- **Comment l'offset est-il calculé ?**
        - On applique **une seule projection linéaire** sur le vecteur de la requête $z_q$. 
        - Cette projection génère directement un vecteur de dimension $3 \times M \times K$  et on utilise 2MK channels pour tous les couples $(\Delta x, \Delta y)$ pour l'ensemble des $M$ têtes et des $K$ points d'échantillonnage. le 1MK restant est utilisé pour obtenir les poids d'attention $A_{mqk}$ (après passage dans un Softmax).
	- **Comment éviter que l'offset sorte de l'image ?**
        - **Mécanisme technique :** Géométriquement, si le point $p + \Delta p$ sort des limites ($H \times W$), l'implémentation CUDA (`ms_deform_attn_cuda`) applique un **zero-padding implicite**. La valeur échantillonnée pour ce point devient `0`.
        - **Impact sur l'attention :** Sa contribution finale dans la somme pondérée de l'attention est donc nulle.
        - **Filiation historique :**
            - **DCNv1 (2017) :** Introduction de l'échantillonnage par interpolation bilinéaire. Les coordonnées hors-bornes renvoient une valeur nulle.
            - **DCNv2 (2018) :** Ajout d'un poids de modulation (masque entre 0 et 1) permettant au réseau d'apprendre à "ignorer" activement les offsets aberrants ou hors-cadre.
            - **Deformable DETR (2021) :** Reprend le zero-padding et le poids d'attention $A_{mqk}$ fait office de modulateur et subit un softmax et par Backpropagation ou réduire le poids d'attention $A_{mqk}$ vers 0 ou corriger l'offset $\Delta p_{mqk}$.
            - - **DCNv3 (2022/2023) :** Introduit un **Softmax** sur les masques de modulation (inspiré des Transformers) pour stabiliser l'entraînement à grande échelle. C'est aussi l'introduction du partage des canaux par **groupes** (similaires aux têtes d'attention).
	        - **DCNv4 (2024) :** Retire le Softmax car il créait un goulot d'étranglement mémoire (Memory Bound) trop lourd sur le GPU. La stabilisation de la somme des poids n'est plus faite *dans* l'opérateur, mais déléguée aux couches de normalisation externes (**LayerNorm / RMSNorm**) situées entre les blocs de l'architecture.
	- **Comment fonctionne la Backpropagation avec des coordonnées floues ?**
		* **Le problème :** Les offsets ($\Delta x, \Delta y$) sont des nombres réels (ex: `1.42`). Les pixels sont discrets. Un simple arrondi (`round`) bloquerait les gradients (dérivée = 0).
		- **La solution :** L'**interpolation bilinéaire**. La valeur échantillonnée est une moyenne pondérée des 4 pixels les plus proches.
- **$x(p_q + \Delta p_{mqk})$** : La valeur du pixel à la position décalée. Comme le décalage $\Delta p$ est continu (ex: `+1.4` pixel), on utilise une **interpolation bilinéaire** pour calculer la valeur exacte de la feature à cet endroit précis.
	- formule mathématique **interpolation bilinéaire** :
		- Pour un point de coordonnées continues $p = (x, y)$, on définit ses 4 voisins entiers :
		- $x_1, x_2$ (les entiers encadrant $x$, avec $x_2 = x_1 + 1$)
		- $y_1, y_2$ (les entiers encadrant $y$, avec $y_2 = y_1 + 1$)
		- La formule générale de l'interpolation bilinéaire est une somme pondérée des valeurs des 4 pixels $Q$, notées $f(Q)$ :
		- $$f(x,y) = \frac{1}{(x_2-x_1)(y_2-y_1)} \begin{bmatrix} x_2-x & x-x_1 \end{bmatrix} \begin{bmatrix} f(x_1,y_1) & f(x_1,y_2) \\ f(x_2,y_1) & f(x_2,y_2) \end{bmatrix} \begin{bmatrix} y_2-y \\ y-y_1 \end{bmatrix}$$
		- Puisque la distance entre deux pixels adjacents est égale à 1 ($x_2 - x_1 = 1$), la formule se développe directement en une somme pondérée des 4 pixels environnants :
		- $$x(p_q + \Delta p_{mqk}) = (x_2 - x)(y_2 - y) \cdot f(x_1,y_1) + (x - x_1)(y_2 - y) \cdot f(x_2,y_1) + (x_2 - x)(y - y_1) \cdot f(x_1,y_2) + (x - x_1)(y - y_1) \cdot f(x_2,y_2)$$
#### 3. Les poids d'attention ($A_{mqk}$)
Dans l'attention classique, $A_{mqk}$ était calculé par un produit scalaire Query-Key géant. Ici, c'est beaucoup plus simple :
- Les poids $A_{mqk}$ et les offsets $\Delta p_{mqk}$ sont **tous les deux prédits directement** en appliquant une simple projection linéaire (une couche Dense/Linéaire) sur la Query $z_q$ . Cette projection génère directement un vecteur de dimension $3 \times M \times K$  et on utilise 1MK pour obtenir les poids d'attention $A_{mqk}$ (après passage dans un Softmax)
- La somme de ces poids d'attention $A_{mqk}$ pour les $K$ points est égale à 1 (normalisée via un Softmax).

### Multi-scale Deformable Attention Module