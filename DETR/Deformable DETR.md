Cet article propose des modifications de l'architecture [[DETR]]
https://arxiv.org/pdf/2010.04159 
## Rappel : Attention Classique (DETR)
### Formulations Mathématiques Attention Classique
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

l'idée est inspirée de la convolution déformable :
![[Pasted image 20260504145939.webp]]
le processus est le suivant :
1. Convolution Classique (Grille rigide 3x3) 
2. Détermination des Offsets (on a un tenseur 2N canaux pour avoir $\Delta x , \Delta x$) 
3. Grille Déformée (Interpolation bilinéaire pour trouver quel pixel utiliser)  
4. Convolution Finale (Sur les pixels décalés de la feature map original)

**Dans l'Attention Déformable :** C'est la **Query ($z_q$)** qui va prédire elle-même où elle veut regarder (les offsets) et avec quelle importance (les poids d'attention).

### Déformable Attention Module (Single-Scale)
![[Deformable DETR-1779277411685.webp]]
$\LARGE \mathrm{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^{M} W_m \left[ \sum_{k=1}^{K} A_{mqk} \cdot W'_m x\bigl(p_q + \Delta p_{mqk}\bigr) \right]$Pour une seule head d'attention ($m$) :
$\LARGE \mathrm{OneDeformAttn}(z_q,p_q, x) = \sum_{k=1}^{K} A_{mqk} \cdot W'_m x\bigl(p_q + \Delta p_{mqk}\bigr)$
#### 1. Les Entrées
- **$z_q$** : Le vecteur de la requête (Query).
- **$x$** : La carte de caractéristiques d'entrée (taille $H \times W \times C$).
- **$p_q$** : Le point de référence (coordonnées 2D $[x, y]$ normalisées dans $[0, 1]$).
    - **Comment est-il déterminé ?**
        - **Dans l'Encodeur :** Ce sont des coordonnées fixes générées à partir d'une grille régulière. Le point $p_q$ correspond au centre de chaque pixel/cellule de la feature map.
        - **Dans le Décodeur :** Il est prédit dynamiquement à partir de l' *Object Query* par une couche linéaire suivie d'une fonction `Sigmoid`
			- Possible d'utiliser une grille comme dans [[Anchor DETR]].
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
            - **Deformable DETR (2021) :** Reprend le zero-padding et le poids d'attention $A_{mqk}$ fait office de modulateur et subit un softmax et par Backpropagation ou réduire le poids d'attention $A_{mqk}$ vers 0 et/ou corriger l'offset $\Delta p_{mqk}$.
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
	- **questionnement** : Dans DCNv3 (amélioration de DCNv2 qui a inspiré l'approche), ils ont ajouté un Softmax qui a ensuite été enlevé dans DCNv4 (car il créait un goulot d'étranglement mémoire), déléguant ainsi la normalisation à des couches de normalisation externes. Possibilité de faire de même ?

### Multi-scale Deformable Attention Module
L'idée est de modifier le Deformable Attention Module pour pouvoir exploiter plusieurs multi-scale feature maps
![[Deformable DETR-1779280813270.webp]]

$$\mathrm{MSDeformAttn}(z_q, \hat{p}_q, \{x_l\}_{l=1}^L) = \sum_{m=1}^{M} W_m \left[ \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot W'_m x_l\bigl(\phi_l(\hat{p}_q) + \Delta p_{mlqk}\bigr) \right]$$
Pour une seule head d'attention ($m$) :
$$\mathrm{OneMSDeformAttn}(z_q, \hat{p}_q, \{x_l\}_{l=1}^L) =  \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot W'_m x_l\bigl(\phi_l(\hat{p}_q) + \Delta p_{mlqk}\bigr) $$
#### lexique
- **$L$** : Représente le nombre de niveaux de résolutions (couches de features du backbone, souvent $L=4$).
- **$x_l \in \mathbb{R}^{C \times H_l \times W_l}$** : La feature map du niveau $l$ .
- **$\hat{p}_q$** : Le point de référence exprimé en coordonnées **normalisées** (entre 0 et 1) pour être indépendant de la taille des différentes cartes de caractéristiques.
- **$\phi_l(\hat{p}_q)$** : Une fonction de mise à l'échelle (un-normalization) qui réajuste les coordonnées de $\hat{p}_q$ à la résolution réelle (pixels réels) de la feature map du niveau $l$.
#### Différences :
- Dans la version multi-échelle, le Softmax est appliqué **globalement sur tous les niveaux et tous les points en même temps**. Pour une tête d'attention donnée ($m$), le modèle dispose de $L \times K$ points au total (par exemple $4 \text{ niveaux} \times 4 \text{ points} = 16 \text{ points}$). Le Softmax répartit l'importance (le budget d'attention de 100 %) sur ces 16 points.
Dans l'encoder :
- **Scale-Level Embedding :** il y a un besoin d'avoir une représentation du niveau de l'échelle $e_l$.
    - Celle-ci est initialisée aléatoirement avec le réseau et est apprise.
    - Elle est ajoutée à l'entrée de l'encodeur.
        - **Critique/Questionnement :** Dans _End-to-End Object Detection with Transformers_, il a été montré qu'ajouter le _spatial positional encoding_ directement à l'attention arrivait à de meilleurs résultats. Possibilité que la même chose soit possible pour le _Scale-Level Embedding_ ?
**Dans le décodeur :**
- **Attention :** Seule la _cross-attention_ est modifiée pour devenir déformable. La _self-attention_ reste identique à celle du DETR original (_End-to-End Object Detection_).
**Dans la classification finale :**
- **Prédiction des boîtes :** Modification de la fonction de détermination des _bounding boxes_ 
**Dans l'input :** 
![[Deformable DETR-1779280869522.webp]]
les couches exploitées sont obtenues à partir des différentes couches du CNN

## bounding box prediction in deformable DETR

- idée : deformable DETR utiliser des Point de référence ($\hat{p}_q$) prédit à partir de son _query embedding_ via un réseau linéaire simple (FFN) suivi d'une fonction sigmoïde. donc la head de détection de Deformable DETR ne prédit que des **offsets relatifs (des décalages)** par rapport à un point de référence.
Le module de classification finale utilise un Feed-Forward Network (FFN) à 3 couches pour prédire la boîte finale sous la forme suivante :
$$\text{Box} = \{ \sigma(\Delta x + \sigma^{-1}(\hat{p}_{qx})), \sigma(\Delta y + \sigma^{-1}(\hat{p}_{qy})), \sigma(\Delta w), \sigma(\Delta h) \}$$
- **$\hat{p}_{qx}, \hat{p}_{qy}$** : Les coordonnées normalisées du point de référence initial.
- **$\Delta x, \Delta y$** : Les offsets (décalages) prédits par le FFN pour recadrer le centre de la boîte.
- **$\Delta w, \Delta h$** : Les tailles (largeur/hauteur) prédites, relatives au point.
- **$\sigma$ (Sigmoïde) :** Appliquée à la fin pour s'assurer que toutes les coordonnées finales de la boîte restent bien comprises entre $0$ et $1$ (normalisées par rapport à l'image).
###  Iterative Bounding Box Refinement
Pour améliorer encore la précision, Deformable DETR propose une variante où chaque couche du décodeur affine itérativement les prédictions de la couche précédente.
- **Formulation mathématique :** Pour une couche de décodeur $d$ (allant de $1$ à $D$) :
    - Le point de référence fourni est la boîte de la couche précédente : $\hat{b}^{d-1} = \{\hat{x}^{d-1}, \hat{y}^{d-1}, \hat{w}^{d-1}, \hat{h}^{d-1}\}$.
    - La couche $d$ prédit de nouveaux deltas de modification : $\Delta b^d = \{\Delta x^d, \Delta y^d, \Delta w^d, \Delta h^d\}$.
La nouvelle boîte affinée est calculée ainsi :    $\hat{b}^d = \{ \sigma(\Delta x^d + \sigma^{-1}(\hat{x}^{d-1})), \sigma(\Delta y^d + \sigma^{-1}(\hat{y}^{d-1})), \sigma(\Delta w^d + \sigma^{-1}(\hat{w}^{d-1})), \sigma(\Delta h^d + \sigma^{-1}(\hat{h}^{d-1})) \}$
## two-stage deformable DETR

## Computational complexity
### Déformable Attention Module
selon le papier le coûp de Déformable Attention Module est
$$\mathcal{O}(N_q C^2 + min(HWC²,N_qKC^2)+5N_q K C)$$
on reprend les valeurs utilisées pour le même calcul dans [[DETR]]
image $800\times1333$ -> feature map $25\times41$ -> $HW=1025$
- $C$ : Dimension d'embedding (équivalent à $d = 256$)
- $N_q$ : Nombre de requêtes (queries)
- $HW$ : Taille de la feature map ($1025$)
- $K$ : Nombre de points clés échantillonnés par requête ($4$)

1. Dans l'Encodeur ($N_q = HW = 1025$)
**Calcul numérique :**
Partie projections et clés : $2 \times 1025 \times 256^2 = 2 \times 1025 \times 65536 = 134\,348\,800$
Partie échantillonnage (linéaire) : $5 \times 1025 \times 4 \times 256 = 5\,248\,000$
**Total Encodeur Déformable = $139\,596\,800$ opérations**
VS DETR 336M

2. Dans la Cross-Attention du Décodeur ($N_q = 100$)
**Calcul numérique :**
Projection des requêtes : $100 \times 256^2 = 6\,553\,600$
Valeurs échantillonnées : $100 \times 4 \times 256^2 = 26\,214\,400$
Coordonnées et poids d'attention : $5 \times 100 \times 4 \times 256 = 512\,000$
**Total Décodeur (Cross-Attention) = $33\,280\,000$ opérations**
VS DETR 99M

### Déformable Attention Module multi-scale
(A revérifier)
La formule de complexité s'ajuste en multipliant l'échantillonnage par le nombre de niveaux $L$ (ici, $L=4$) :
$$\mathcal{O}(N_q C^2 + \min(\sum_{l=1}^{L} H_l W_l C^2, N_q L K C^2) + 5 N_q L K C)$$
Posons d'abord les résolutions de nos feature maps à partir d'une image d'origine de $800 \times 1333$. Les niveaux de réduction classiques (strides) d'un ResNet sont $/8, /16, /32, /64$ :
- **Niveau 1 ($/8$)** : $100 \times 166 \approx 16\,600$ pixels ($H_1 W_1$)
- **Niveau 2 ($/16$)** : $50 \times 83 \approx 4\,150$ pixels ($H_2 W_2$)
- **Niveau 3 ($/32$)** : $25 \times 41 = 1\,025$ pixels ($H_3 W_3$)
- **Niveau 4 ($/64$)** : $13 \times 21 \approx 273$ pixels ($H_4 W_4$)
**Total des pixels multi-échelles ($\sum HW$)** = $16\,600 + 4\,150 + 1\,025 + 273 = \mathbf{22\,048}$

**Paramètres**
- $C = 256$
- $K = 4$ points par échelle
- $L = 4$ échelles
- $L \times K = 16$ points échantillonnés au total par requête.

1. Multi-Scale dans l'Encodeur
Dans l'encodeur multi-échelle, **chaque pixel de chaque échelle** génère une requête. Le nombre total de requêtes est donc la somme de tous les pixels : $N_q = \sum HW = 22\,048$.

**Calcul numérique :**
- Partie projections et clés : $2 \times 22\,048 \times 256^2 = 2 \times 22\,048 \times 65\,536 = \mathbf{2\,889\,940\,992}$
- Partie échantillonnage ($5 N_q L K C$) : $5 \times 22\,048 \times 4 \times 4 \times 256 = \mathbf{451\,543\,040}$
- **Total Encodeur Multi-Scale = $3\,341\,484\,032$ opérations (~3.34 Milliards)**


## Modification du HungarianMatcher : Intégration de la Focal Loss
> [!info] Liens utiles
> * Concept théorique : [[Focal Loss]]
> * Code source : `models/matcher.py` (Deformable DETR / Anchor DETR)

### Ce qui change :
Dans les évolutions de DETR, le calcul du coût de classification du Matcher passe d'un `softmax` classique à une **Focal Loss** basée sur des `sigmoid` indépendantes .
> [!warning] Changement architectural implicite (Softmax vs Multi-Sigmoïdes)
> L'utilisation de la Focal Loss impose un changement radical dans la manière dont le modèle prédit les classes, bien que ce ne soit pas détaillé explicitement dans le texte du papier :
> 
> * **DETR Original (Softmax) :** Le modèle utilise une unique fonction `softmax` globale. Toutes les classes (y compris la classe "fond") sont en compétition directe au sein d'une distribution de probabilité unique qui doit obligatoirement sommer à $1.0$.
> * **Modèles Modifiés (Multi-Sigmoïdes indépendantes) :** On jette le Softmax. À la place, **chaque classe possède sa propre activation `sigmoid` indépendante**. La détection devient une suite de questions binaires parallèles (ex: *"Est-ce une voiture ? 0.90"*, *"Est-ce un piéton ? 0.01"*). 
> 
> **Les deux conséquences majeures :**
> 1. **Changement de dimension des tenseurs :** On passe d'une couche de sortie de taille **$X + 1$** classes (où le $+1$ était la classe fond dans le DETR original) à une taille de **$X$** classes seulement. 
> 2. **Le fond devient un état implicite :** La classe "fond" (background) n'existe plus en tant que variable ou canal de sortie dans le réseau. Le fond correspond désormais au moment exact où **toutes les sigmoïdes de toutes les classes sont simultanément proches de 0**.
### Pourquoi cette modification ?
* **Stabilisation de l'entraînement :** En calculant le coût via la formule `pos_cost_class - neg_cost_class`, le Matcher évalue l'impact net d'une requête sur la perte globale. Si le modèle est trop incertain, la classification génère une pénalité (coût positif).
* **Élimination du bruit d'assignation :** Tant que le modèle est incertain sur les classes (notamment au début de l'entraînement), le coût positif de la Focal Loss neutralise les micro-fluctuations des probabilités (les bruits à $0.01$ ou $0.02$). L'algorithme hongrois se base alors uniquement sur la **géométrie pure ($L_{\text{box}}$)** pour faire ses choix obligatoires, évitant de sauter d'une requête à l'autre et de déstabiliser la classe "fond" (background).