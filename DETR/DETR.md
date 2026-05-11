https://arxiv.org/pdf/2005.12872
DETR pour Detection Transformer a pour but d'identifier des objets dans une image. 
Le modèle a deux objectifs  :
- Localiser les objets dans l’image (prédire des bounding boxes).
- Classifier correctement chaque objet détecté.
## Architecture

![[Pasted image 20260429162333.webp]]

Une image  de taille $3×H_0×W_0$ est d’abord passée dans un CNN backbone. Son rôle est d’extraire des features visuelles riches. Dans l’article original, un ResNet-50 est utilisé, en retirant les couches de classification finales.

On obtient une feature map de dimension spatiale H×W avec C canaux.
Les valeurs typiques sont C = 2048 et $H=H_0/32$  , $W=W_0=/32$ 
Ensuite :
- Une convolution 1×1 projette ces features dans une dimension  plus petite d 
- La feature map est aplatie en une séquence de taille d×HW (Malgré le fait qu'on a flatten, il est toujours possible de utiliser les positions (i, j) où elles étaient de base. pour le positionnel en coding ).
- Cette séquence est envoyée dans un transformer encodeur-décodeur.

### Encodeur–Décodeur
![[Pasted image 20260429163348.webp]]
#### Encodeur
L’encodeur suit globalement l’architecture standard des transformers, avec une particularité importante :
- Un encodage positionnel est ajouté aux features pour conserver l’information spatiale ([[Sinusoidal Encoding]] dans l’article).
- Cet encodage est ajouté aux entrées de chaque couche de self attention pour générer les Key et Query.
- La Value, donc, n'a pas de [[Positional Encoder]]. 
#### Decoder
##### Input : Object Queries
Le décodeur prend en entrée un ensemble fixe de $N$ vecteurs appris, appelés _object queries_. Ces vecteurs agissent comme des "slots" de détection spécialisés, où $N$ définit le nombre maximal d'objets détectables dans une image. Chaque _query_ apprend à interroger des régions spécifiques de l'image pour prédire un objet.
#### **Gestion de la position et Invariance**
Par définition, les _object queries_ sont invariantes par permutation (elles ne possèdent pas de notion spatiale intrinsèque). Pour résoudre ce problème et permettre au modèle de localiser les objets, un **embedding positionnel appris** est injecté au niveau des _Queries_ et des _Keys_ dans chaque bloc d'attention :
- **Self-Attention :** L'embedding positionnel est ajouté aux _Queries_ et aux _Keys_. Cela permet aux _queries_ de communiquer entre elles tout en conservant leur identité spatiale, évitant ainsi les prédictions redondantes (ex: deux _queries_ visant le même objet).
- **Cross-Attention :** L'embedding positionnel est réinjecté dans les _Queries_ avant leur interaction avec les features de l'encodeur.
- **Principe de réinjection :** Contrairement à un transformer classique, cette injection n'a pas lieu uniquement en entrée du décodeur. Elle est répétée à **chaque couche** du décodeur (via une opération d'addition).
####  Justification positional encoding. 
Pour plus d'informations, voire section 4.2 subsection Important of positional encoding 
La justification pour laquelle les deux composantes du positional encoding, le spatial et l'output positional en coding, sont ajoutées à attentions, se fait de par une expérimentation qui a été menée et que c'est la configuration qui offre le meilleur résultat. 
par rapport à juste le faire une fois à l'entrée comme dans l'original transformer 
### FFN
La sortie du décodeur est envoyée dans deux réseaux feed-forward distincts :
- Une tête de classification : prédit une classe (ou "no object").
- Une tête de régression : prédit une bounding box (x,y,w,h)
## Loss
DETR produit un nombre fixe N de prédictions en une seule passe, avec N choisi tel que :
$\large N≫$nombre d’objets dans l’image
### 1.Matching 
Pour chaque prédiction, nous tentons de trouver à quelle grand truth celle-ci doit être associée. On résout un problème d’assignation optimale :
$\large \sigma = \arg\min_{\sigma \in S_N} \sum_{i=1}^N L_\text{match}(y_i, \hat{y}_{\sigma(i)})$
où :
- $y_i$ est le i-ème objet réel (classe $c_i$​, boîte $b_i$​)
-  $\hat{y}_{\sigma(i)}$ est la prédiction associée via la permutation optimale $σ$

**Coût de matching**
$\large L_\text{match}(y_i, \hat{y}_{\sigma(i)}) = {1}_{\{c_i \neq \emptyset\}} \left[ -\hat{p}_{\sigma(i)}(c_i) + L_\text{box}(b_i, \hat{b}_{\sigma(i)}) \right]$
Explication des termes :
-  $\large {1}_{\{c_i \neq \emptyset\}}$​ : on ignore les objets "vides" (padding).
-  $\large -\hat{p}_{\sigma(i)}(c_i)$ : favorise les prédictions avec forte probabilité pour la bonne classe.
- $\large L_\text{box}$​ : mesure la qualité de la localisation.
$\large L_\text{box}(b_i, \hat{b}_{\sigma(i)}) = \lambda_\text{iou} L_\text{iou}(b_i, \hat{b}_{\sigma(i)}) + \lambda_\text{L1} \| b_i - \hat{b}_{\sigma(i)} \|_1$
- $L_{iou}$​ : capture le chevauchement global entre les boîtes.
- $L_1$: pénalise les erreurs absolues sur les coordonnées.
-  $\lambda_\text{iou}, \lambda_\text{L1} \in \mathbb{R}^+$ sont des hyperparamètres.
#### Generalized IoU

Le terme $L_{\text{iou}}$ est défini à partir du GIoU :

$$
L_{\text{iou}}(b_i, \hat{b}_{\sigma(i)}) = 1 - \text{GIoU}(b_i, \hat{b}_{\sigma(i)})
$$
$$
\text{GIoU}(b_i, \hat{b}_{\sigma(i)}) =
\frac{|b_i \cap \hat{b}_{\sigma(i)}|}{|b_i \cup \hat{b}_{\sigma(i)}|}
-
\frac{|B(b_i, \hat{b}_{\sigma(i)}) \setminus (b_i \cup \hat{b}_{\sigma(i)})|}
{|B(b_i, \hat{b}_{\sigma(i)})|}
$$
- $\setminus$ **"différence d'ensemble"** (enlève les éléments de B qui sont dans A)
- $|\cdot|$ : aire d'une boîte
- $b_i \cap \hat{b}_{\sigma(i)}$ : intersection
- $b_i \cup \hat{b}_{\sigma(i)}$ : union
- $B(b_i, \hat{b}_{\sigma(i)})$ : plus petite boîte englobante contenant les deux
![[Pasted image 20260507110805.png]]
voir : https://giou.stanford.edu/
### 2. Hungarian loss
Une fois le matching optimal $\hat{\sigma}$ déterminé, on calcule la loss sur **l’ensemble des N prédictions**, en utilisant cette correspondance.
$\large  \mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \sum_{i=1}^{N} \left[ - \log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbf{1}_{\{c_i \neq \emptyset\}} \, \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$
- $- \log \hat{p}_{\hat{\sigma}(i)}(c_i)$ Loss de classification 
-  $\large {1}_{\{c_i \neq \emptyset\}}$​ : La loss de boîte n’est appliquée **que pour les vrais objets**
-  $\large L_\text{box}$​ : Pénalise l’erreur de localisation uniquement pour les prédictions matchées à un objet réel
Le papier recommande que si $\{c_i = \emptyset\}$. Il faut **diviser par 10 la loss obtenue** tenir compte du déséquilibre des classes.  

Le résultat final est que si on a $N = 100$, et qu'il y a trois objets réels dans l'image, alors nous aurons trois box qui seront prédits comme des objets, et 97 qui seront prédits comme n'étant pas un objet. 

### 3.  Auxiliaire decoding loss
durant l'entraînement après chaque décodeur Ils utilisent la prédiction des FFN et hungraian loss. Toutes les prédictions partagent leur paramètre.  
Une couche partagée supplémentaire de normalisation est utilisée pour normaliser L'entrée des FFN après les différents couches de décondeurs. 


## Computational complexity
**Encodeur** : Self-attention → **O(d²HW + d(HW)²)**
**Décodeur** :
- Self-attention : **O(d²N + dN²)** (N=100 queries ≪ HW=~2500)
- Cross-attention : **O(d²(N+HW) + d N HW)** ≪ encodeur

**`d`** = **dimension d'embedding**
### exemple
image 800*1333 -> feature map $25*41$ -> HW=1025
d=256
N = 100

**Encodeur** : Self-attention : 336 134 400
**Décodeur** :
- Self-attention : 9 113 600
- Cross-attention :  99 968 00
- total = 109 081 600
encodeur demande un peu plus de 3 fois plus d’opérations que le décodeur