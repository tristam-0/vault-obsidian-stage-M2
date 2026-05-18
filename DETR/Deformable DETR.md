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
idée : au lieux de faire la le calcule du sore de l'atantion pour tout les K. ne le faire que pour ne partie des Key. (besoin de parlé rapidement deformable convolution pour introdure idée point de réf et ofsete. j'ai un image d'ilustration (convolution clasique -> détermination offsets(comment ?) -> utilise offsets pour changer pixclle utilise -> convolution final))
$\LARGE \mathrm{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^{M} W_m \left[ \sum_{k=1}^{K} A_{mqk} \cdot W'_m x\bigl(p_q + \Delta p_{mqk}\bigr) \right]$
pour une seul head
$\LARGE \mathrm{OneDeformAttn}(z_q,p_q, x) = \sum_{k=1}^{K} A_{mqk} \cdot W'_m x\bigl(p_q + \Delta p_{mqk}\bigr)$
- K est l'index de la key et K est le nombre total de key $K<<HW$
- $P_q$ : point de référance pour une Query
- $\Delta P_{mqk}$ ofset qui dépent de m la head Q la query et k la key. (besoin de comment détreminer et enténer)