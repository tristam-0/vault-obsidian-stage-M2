https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DA-DETR_Domain_Adaptive_Detection_Transformer_With_Information_Fusion_CVPR_2023_paper.**pdf
https://openaccess.thecvf.com/content/CVPR2023/supplemental/Zhang_DA-DETR_Domain_Adaptive_CVPR_2023_supplemental.pdf
rolle diférance CNN / Transformer

- **Le backbone CNN** extrait des caractéristiques locales et spatiales (bords, contours, localisation précise).
- **La tête Transformer** capture les relations globales entre les pixels et l'information sémantique de haut niveau.

Plutôt que d'aligner séparément les caractéristiques du CNN et du Transformer entre le domaine source et le domaine cible, **DA-DETR** propose de **fusionner ces deux types d'informations** pour former une représentation unifiée avant l'alignement.

### Les composants clés de DA-DETR

L'innovation centrale de l'article est un module nommé **CTBlender** (_CNN-Transformer Blender_), qui associe deux mécanismes de fusion :

1. **Split-Merge Fusion (SMF) :** Les caractéristiques du Transformer viennent moduler celles du CNN. Le CNN divise ses caractéristiques en groupes sémantiques guidés par le Transformer, puis les fusionne avec un mélange de canaux (_channel shuffling_) pour faire communiquer les informations spatiales et sémantiques.
![[DA-DETR-1784794352609.webp]]
### 1. Les Entrées (Échelle $l$)
Pour un niveau d'échelle $l$ donné (ex. un niveau de résolution dans le FPN/Multi-scale) :
- **$f_l \in \mathbb{R}^{C \times H_l \times W_l}$** : la carte de caractéristiques issues du backbone **CNN** (contient les détails spatiaux et de localisation fins).
- **$p_l \in \mathbb{R}^{C \times H_l \times W_l}$** : la carte de caractéristiques issues de l'encodeur du **Transformer** (contient les relations globales et le contexte sémantique haut niveau).
Les deux tenseurs ont exactement la même dimension (Canaux $C$, Hauteur $H_l$, Largeur $W_l$)
### 2. Étape 1 : Le Découpage (_Split_)
On divise le canal de dimensions $C$ de $f_l$ et de $p_l$ en **$K$ groupes distincts** le long de l'axe des canaux :
$$f_l = [f_l^1, f_l^2, \dots, f_l^K], \quad \text{où } f_l^k \in \mathbb{R}^{\frac{C}{K} \times H_l \times W_l}$$
$$p_l = [p_l^1, p_l^2, \dots, p_l^K], \quad \text{où } p_l^k \in \mathbb{R}^{\frac{C}{K} \times H_l \times W_l}$$
### 2A. Étape 2A : Fusion par Canal (_Channel-wise Fusion_)
Pour capturer l'importance relative de chaque canal (les dépendances inter-canaux), le Transformer génère un vecteur de poids de canal $p_{l,k}^c \in \mathbb{R}^{\frac{C}{K} \times 1 \times 1}$ pour le groupe $k$ :
$$p_{l,k}^c = f_s\left(w_c \cdot \text{GAP}(p_{l,k}) + b_c\right)$$
- **$\text{GAP}(\cdot)$** : _Global Average Pooling_, qui compresse les dimensions spatiales $H_l \times W_l$ en une valeur moyenne par canal.
- **$w_c$ et $b_c$** : Le vecteur de poids appris (_learnable weight vector_) et le biais appris (_learnable bias vector_).
- **$f_s(\cdot)$** : Une fonction d'activation (ex. Sigmoid) qui contraint les valeurs entre $0$ et $1$.

**Modulation du CNN :**

Ce masque par canal vient moduler la caractéristique CNN correspondante $f_{l,k}$ via un produit élément par élément (avec propagation spatiale / _broadcasting_) :
$$f_{l,k}^{channel} = f_{l,k} \odot p_{l,k}^c$$

### 2B. Étape 2B : Fusion Spatiale (_Spatial-wise Fusion_)

Pour ajuster dynamiquement l'attention sur les zones d'intérêt spatiales (distinguer l'arrière-plan des objets), le Transformer génère une carte d'attention spatiale $p_{l,k}^s \in \mathbb{R}^{\frac{C}{K} \times H_l \times W_l}$ pour le groupe $k$ :
$$p_{l,k}^s = f_s\left(w_s \cdot \text{GN}(p_{l,k}) + b_s\right)$$
- **$\text{GN}(\cdot)$** : _Group Normalization_, appliquée au sous-groupe de canaux du Transformer.
- **$w_s$ et $b_s$** : La carte de poids apprise (_learnable weight map_) et le biais appris (_learnable bias map_).
- **$f_s(\cdot)$** : La fonction d'activation ramenant les poids dans $[0, 1]$.

**Modulation du CNN :**
Le masque spatial vient pondérer localement chaque pixel de la carte de caractéristiques CNN $f_{l,k}$ :
$$\tilde{f}_{l,k} = f_{l,k} \odot p_{l,k}^c \odot p_{l,k}^s$$
### 3. Étape 3 : La Génération des groupes modulés (Group Feature Generation)
À la fin de l'étape 2, pour un groupe $k$ donné, on ne sépare pas le traitement. Le sous-groupe de caractéristiques CNN $f_{l,k}$ est pondéré simultanément par le masque de canal $p_{l,k}^c$ **ET** le masque spatial $p_{l,k}^s$.
$$\tilde{f}_{l,k} = f_{l,k} \odot p_{l,k}^c \odot p_{l,k}^s$$
**Pourquoi avoir découpé en $K$ groupes ?**
L'objectif n'est pas de séparer le spatial et le canal entre les groupes. L'objectif est de permettre au modèle d'apprendre **$K$ représentations sémantiques différentes** (similaire au mécanisme de _Multi-Head Attention_). Par exemple, le groupe 1 pourrait se spécialiser dans les contours des objets, le groupe 2 dans les textures, etc., et chaque groupe bénéficie de sa propre modulation spatiale et par canal.
On obtient ainsi $K$ sous-cartes modulées de mêmes dimensions :
$$\tilde{f}_{l,k} \in \mathbb{R}^{\frac{C}{K} \times H_l \times W_l} \quad \text{pour } k = 1, \dots, K$$
### 4. Étape 4 : Le Mélange et la Reconstitution (Channel Shuffling & Merge)
Maintenant que nous avons $K$ blocs de canaux modulés, il faut reconstruire la carte globale de dimension $C$. Si l'on se contentait de les recoller côte à côte, les canaux resteraient enfermés dans leurs groupes respectifs. L'information apprise par le groupe 1 ne pourrait jamais interagir avec l'information du groupe 2 dans les couches suivantes du réseau.

Pour forcer la communication entre tous les groupes, le SMF utilise le **Channel Shuffling** (Mélange des canaux) :
1. **Concaténation initiale :** Les $K$ sous-cartes $\tilde{f}_{l,k}$ sont recollées le long de l'axe des canaux. On retrouve bien un tenseur à $C$ canaux.
2. **Channel Shuffle :** On réordonne les canaux en les entrelaçant, exactement comme lorsqu'on bat un jeu de cartes (mélange en "fermeture éclair").
    - _Exemple visuel :_ Au lieu d'avoir `[Canaux du Groupe 1 | Canaux du Groupe 2 | ...]`, on aura `[Canal 1 du G1, Canal 1 du G2, ..., Canal 2 du G1, Canal 2 du G2, ...]`.

Cela garantit une fusion riche où l'information globale du Transformer circule à travers toutes les dimensions des caractéristiques du CNN.
**Formule de sortie du module SMF (pour l'échelle $l$) :**
$$\hat{f}_l = \text{Shuffle}\left(\text{Concat}\left([\tilde{f}_{l,1}, \tilde{f}_{l,2}, \dots, \tilde{f}_{l,K}]\right)\right) \in \mathbb{R}^{C \times H_l \times W_l}$$
> **Note :** Le tenseur de sortie $\hat{f}_l$ a exactement la même dimension qu'à l'entrée. Il est maintenant prêt à être envoyé au module suivant, le **Scale Aggregation Fusion (SAF)**, qui va s'occuper de fusionner ces cartes entre les différentes échelles $l$.

### Résumé du flux SMF
$$\begin{matrix} (f_l, p_l) & \xrightarrow{\text{Split}} & (f_{l,k}, p_{l,k})_{k=1}^K \\ & \xrightarrow{\text{Modulation}} & \tilde{f}_{l,k} = f_{l,k} \odot p_{l,k}^{s/c} \\ & \xrightarrow{\text{Shuffle \& Merge}} & \hat{f}_l \quad \text{(Puis transmis au module SAF)} \end{matrix}$$
La sortie $\hat{f}_l$ contient désormais à la fois les détails fins de localisation du backbone CNN et la sémantique globale guidée par l'encodeur Transformer, prête à être agrégée à travers les échelles par la **Scale Aggregation Fusion (SAF)**.



1. **Scale Aggregation Fusion (SAF) :** Il combine les caractéristiques fusionnées à travers plusieurs échelles d'image (_multi-scale_), garantissant que les détails fins de localisation et le contexte global soient préservés à toutes les résolutions.
![[DA-DETR-1784794365082.webp]]
2. **Discriminateur unique :** La représentation riche produite par le CTBlender est envoyée à un unique discriminateur de domaine, qui utilise un apprentissage contradictoire (_adversarial learning_) pour rendre les caractéristiques invariantes au changement de domaine.
![[DA-DETR-1784794194258.webp]]




