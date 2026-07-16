
## 1. Vue d'Ensemble & Flux de Données (Architecture Teacher-Student)

Le framework **Probabilistic Teacher (PT)** résout le problème du transfert de domaine sans supervision (_Unsupervised Domain Adaptation - UDA_) pour la détection d'objets.

Les approches de _Self-Training_ classiques filtrent les pseudo-boîtes cibles à l'aide d'un **seuil de confiance rigide** (ex. $p > 0.8$). Cela pose deux problèmes majeurs :

1. **Dépendance au seuil :** Pas de jeu de validation annoté dans le domaine cible pour régler ce hyperparamètre.
    
2. **Ignorance de l'incertitude de localisation :** Une boîte peut avoir une confiance de classe élevée mais des coordonnées très imprécises.
    

PT propose un framework **sans seuil** (_threshold-free_) qui capture l'**incertitude de classification et de localisation** sous forme de distributions de probabilité.

## 2. Adaptation de la Localisation & Pertes Bounding Box (`lost bbox`)

### Problème dans les détecteurs standards

Dans un détecteur classique , la tête de régression prédit un vecteur déterministe pour chaque boîte :
$$\mathbf{b} = [x, y, w, h]$$
Cette approche ne permet pas d'évaluer si le modèle hésite sur les limites précises d'un objet flou ou dégradé dans le domaine cible.
### Solution Probabiliste

Dans PT, la tête de régression est modifiée pour prédire une **distribution gaussienne** indépendante pour chaque coordonnée $k \in \{x, y, w, h\}$ :
$$p(b_k) = \mathcal{N}\left(b_k; \mu_k, \sigma_k^2\right) = \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\left( -\frac{(b_k - \mu_k)^2}{2\sigma_k^2} \right)$$
- $\mu_k$ est la position prédite de la boîte.
- $\sigma_k^2$ est la **variance de localisation**, représentant l'**incertitude** du modèle sur cette coordonnée.
### Perte de Régression / Localisation Cible ($\mathcal{L}_{loc}^{target}$)
Plutôt qu'une perte $L_1$ ou Smooth $L_1$ déterministe, la perte entre la prédiction du Student $(\mu_S, \sigma_S^2)$ et le pseudo-label du Teacher $(\mu_T, \sigma_T^2)$ s'appuie sur la **Negative Log-Likelihood (NLL)** ou la **Divergence KL** :
$$\mathcal{L}_{loc}^{target} = \frac{1}{N_p} \sum_{i=1}^{N_p} \sum_{k \in \{x, y, w, h\}} \left( \frac{\left\vert{}\mu_{i,k}^S - \mu_{i,k}^T\right\vert{}}{2 \left(\sigma_{i,k}^S\right)^2} + \frac{1}{2} \log\left(\left(\sigma_{i,k}^S\right)^2\right) \right)$$
### Signification Intuitives & Physiques
1. Si l'incertitude du Student $(\sigma_S^2)$ est très élevée, le premier terme $\frac{\vert{}\mu_S - \mu_T\vert{}}{2\sigma_S^2}$ est atténué : le gradient de l'erreur d'alignement ne détruit pas les poids du réseau.
2. Le second terme $\frac{1}{2} \log(\sigma_S^2)$ agit comme un régulariseur empêchant $\sigma_S^2 \to \infty$.
3. Le Student apprend ainsi à **ajuster dynamiquement son incertitude** selon la cohérence des prédictions du Teacher.
## 3. Adaptation de la Classification & Entropy Focal Loss (`entropy focal loss` / `lost clasification`)

### Mesure de l'Incertitude par l'Entropie

Pour la branche classification, la prédiction du Teacher pour une boîte $i$ est un vecteur de probabilités $\mathbf{p}_i^T = [p_{i,1}^T, p_{i,2}^T, \dots, p_{i,C}^T]$ sur $C$ classes.

L'incertitude de classification est mesurée par l'**entropie de Shannon** :

$$H(\mathbf{p}_i^T) = -\sum_{c=1}^{C} p_{i,c}^T \log\left(p_{i,c}^T\right)$$

Afin de borner cette valeur entre $0$ et $1$, elle est normalisée par l'entropie maximale $\log(C)$ :

$$\hat{H}_i = \frac{H(\mathbf{p}_i^T)}{\log(C)} \in [0, 1]$$

- Si $\hat{H}_i \to 0$ : Le Teacher est extrêmement confiant dans une classe (Incertitude nulle).
    
- Si $\hat{H}_i \to 1$ : La distribution est uniforme / ambiguë (Incertitude maximale).
    

### Formule Formelle de l'Entropy Focal Loss (EFL)

L'**Entropy Focal Loss (EFL)** module la perte de Cross-Entropie de consistance en fonction du niveau de certitude du Teacher :

$$\mathcal{L}_{cls}^{target} = -\frac{1}{N_p} \sum_{i=1}^{N_p} \left( 1 - \hat{H}_i \right)^\gamma \sum_{c=1}^{C} p_{i,c}^T \log\left(p_{i,c}^S\right)$$

où :

- $p_{i,c}^T$ est le pseudo-label de probabilité généré par le Teacher (souvent affiné par un _Sharpening_ / adoucissement de température).
    
- $p_{i,c}^S$ est la probabilité prédite par le Student sur l'image fortement augmentée.
    
- $\gamma \ge 0$ est le paramètre de focalisation (_focusing parameter_, par exemple $\gamma = 2$).
    
- $(1 - \hat{H}_i)^\gamma$ est le **facteur de modulation basé sur l'entropie**.
    

### Pourquoi cette formule est construite ainsi ?

- **Remplacement du seuil rigide :** Au lieu de rejeter brutalement un pseudo-label si $p < 0.8$, l'EFL pondère **continûment** chaque échantillon.
    
- **Comportement dynamique :**
    
    - Si le Teacher est sûr ($\hat{H}_i \approx 0$), le poids $(1 - 0)^\gamma = 1$ : la perte s'applique pleinement.
        
    - Si le Teacher hésite ($\hat{H}_i \approx 1$), le poids $(1 - 1)^\gamma = 0$ : l'échantillon incertain est ignoré par le gradient du Student.
        

## 4. Anchor Adaptation (`anchor adaptation`)

### Qu'est-ce que c'est et où l'utiliser ?

- **Domaine d'application :** Uniquement dans les détecteurs **basés sur des ancres** (Anchor-based) tels que Faster R-CNN (Region Proposal Network - RPN) ou RetinaNet.
    
- **Problème résolu :** Les ancres par défaut (tailles/ratios d'aspect) sont définies manuellement sur le domaine source. Lors d'un changement de domaine (ex. images de synthèse vers images réelles en grand angle), la distribution géométrique des objets change radicalement.
    

### Mécanisme

Puisque l'ancre peut être vue comme un paramètre apprenable ou ajustable :

1. PT enregistre les statistiques des boîtes cibles prédites par le Teacher au cours de l'entraînement.
    
2. Les tailles/ratios des ancres du Student/Teacher sont mis à jour progressivement (via K-Means ou mise à jour adaptative par moyenne glissante) pour s'aligner sur la distribution géométrique du domaine cible.
    

## 5. Augmentations de Données : Faibles vs Fortes (`weak` vs `strong augmentation`)

L'entraînement par consistance repose sur une **asymétrie d'augmentation** entre le Teacher et le Student.

|**Propriété**|**Weak Augmentation (Teacher)**|**Strong Augmentation (Student)**|
|---|---|---|
|**Rôle**|Générer des pseudo-labels stables et de haute qualité.|Forcer le Student à apprendre des représentations invariantes et robustes.|
|**Transformations géométriques**|Flipping horizontal aléatoire, redimensionnement simple.|Identiques au Teacher (pour aligner les coordonnées spatiales).|
|**Transformations d'apparence**|Aucune (ou très légères).|Color Jitter (Luminosité, Constraste), Grayscale, Gaussian Blur, CutMix / Random Erasing.|
|**Où l'appliquer dans le code ?**|Appliqué à l'image cible $x^t \to x^{t,w}$ injectée dans le **Teacher**.|Appliqué à la même image cible $x^t \to x^{t,s}$ injectée dans le **Student**.|

## 6. Adaptation de la Méthode à une Architecture de type DETR (Transformer)

L'utilisateur souhaite adapter la philosophie de PT à une architecture de type **DETR / Deformable-DETR**. Voici les ponts méthodologiques nécessaires :

### 1. Remplacement d'Anchor Adaptation par Query Adaptation

DETR est un détecteur **sans ancres** (_anchor-free_), basé sur un jeu de $N$ **Object Queries** apprises dans le Decoder Transformer.

- **Adaptation pour DETR :** L'Anchor Adaptation n'a pas lieu d'être sous sa forme RPN. En revanche, on met en place une **Query Adaptation** :
    
    - Les _Object Queries_ capturent des prioris de position et d'échelle.
        
    - On adapte les requêtes spatiales (_Positional Queries_) en injectant des prototypes de requêtes mis à jour par le domaine cible (méthode type **DA-DETR** ou **EW-DETR**).
        

### 2. Tête de Bounding Box Probabiliste dans DETR

Dans DETR standard, le Feed-Forward Network (FFN) de régression prédit $[x_{center}, y_{center}, w, h] \in [0, 1]^4$.

**Modification :** Le FFN de régression doit prédire 8 sorties :

$$\left[ \mu_x, \mu_y, \mu_w, \mu_h, \log(\sigma_x^2), \log(\sigma_y^2), \log(\sigma_w^2), \log(\sigma_h^2) \right]$$

### 3. Matching Bipartite Probabiliste (Algorithme Hongrois)

DETR utilise l'algorithme hongrois pour associer les $N$ prédictions du réseau aux Ground Truths / Pseudo-labels.

Pour le domaine cible non annoté :

1. Le **Teacher** produit $M$ pseudo-boîtes probabilistes sur $x^{t,w}$.
    
2. Le **Student** produit $N$ prédictions probabilistes sur $x^{t,s}$.
    
3. La matrice de coût pour le matching hongrois entre le pseudo-label $j$ du Teacher et la query $i$ du Student est construite en incluant l'incertitude :
    

$$\mathcal{C}_{i,j} = \lambda_{cls} \mathcal{C}_{cls}\left(p_i^S, p_j^T\right) + \lambda_{L1} \left\Vert{} \mu_i^S - \mu_j^T \right\Vert{}_1 + \lambda_{giou} \mathcal{C}_{giou}\left(\mu_i^S, \mu_j^T\right) + \lambda_{unc} \sum_k \sigma_{i,k}^2$$

### 4. Application des Pertes Cibles dans DETR

```
Queries filtrées par le Matching Hongrois
                 |
                 +--> Classification -> Appliquer Entropy Focal Loss (EFL) sur p_i^S avec \hat{H}(p_j^T)
                 |
                 +--> Bounding Box   -> Appliquer la Perte NLL Gaussienne / GIoU pondérée par \sigma_S^2
```

## 7. Tableau Récapitulatif : RPN / Faster R-CNN (Papier PT) vs DETR (Adaptation Proposée)

|**Composant**|**PT Original (Faster R-CNN)**|**PT Adapté à DETR**|
|---|---|---|
|**Backbone & Features**|CNN (ResNet + FPN)|CNN/Swin + Transformer Encoder|
|**Prioris Géométriques**|Ancres fixes adaptées par **Anchor Adaptation**|**Object Queries** adaptées par Query Alignment|
|**Tête de Régression**|Offset par rapport aux ancres $+\ \sigma^2$|Coordonnées sigmoid $[0,1]$ $+\ \sigma^2$ (via MLP)|
|**Assignation des labels**|IoU Threshold (RPN)|**Bipartite Hungarian Matching** probabiliste|
|**Loss Classification**|EFL sur régions de propositions|EFL sur les queries appariées (+ classe `no-object`)|
|**Loss Bounding Box**|NLL Gaussienne sur offsets|NLL Gaussienne + GIoU pondérée par variance|