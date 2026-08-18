s## 1. Vue d'Ensemble & Flux de Données (Architecture Teacher-Student)
Le framework **Probabilistic Teacher (PT)** résout le problème du transfert de domaine sans supervision (_Unsupervised Domain Adaptation - UDA_) pour la détection d'objets.

Les approches de _Self-Training_ classiques filtrent les pseudo-boîtes cibles à l'aide d'un **seuil de confiance rigide** (ex. $p > 0.8$). Cela pose deux problèmes :

1. **Dépendance au seuil :** Pas de jeu de validation annoté dans le domaine cible pour régler ce hyperparamètre.
2. **Ignorance de l'incertitude de localisation :** Une boîte peut avoir une confiance de classe élevée mais des coordonnées très imprécises.

PT propose un framework **sans seuil** (_threshold-free_) qui capture l'**incertitude de classification et de localisation** sous forme de distributions de probabilité.

## 2. Adaptation de la Localisation & Pertes Bounding Box ($\mathcal{L}_{bbox}$)

### 2.1 Solution Probabiliste au Niveau de la Boîte
Au lieu de prédire un vecteur déterministe $\mathbf{b} = [x, y, w, h]$, la tête de régression prédit une **distribution Gaussienne indépendante** pour chaque coordonnée $k \in \{x, y, w, h\}$ :
$$b_k \sim \mathcal{N}\left(\mu_k, \sigma_k^2\right)$$
$$p(b_k) = \mathcal{N}\left(b_k; \mu_k, \sigma_k^2\right) = \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\left( -\frac{(b_k - \mu_k)^2}{2\sigma_k^2} \right)$$
- **$\mu_k$ (Moyenne) :** La coordonnée spatiale prédite.
- **$\sigma_k^2$ (Variance) :** L'incertitude estimée par le réseau.
- **Contrainte de domaine :** La variance $\sigma_k^2$ est contrainte dans $]0, 1[$ via une activation Sigmoïde appliquée sur la sortie brute du réseau :$$\sigma_k^2 = \text{sigmoid}(s_k)$$
### 2.2 Formulation Générale de la Perte ($\mathcal{L}_{bbox}$ - Équation 1)
La régression est formalisée comme la **Cross-Entropie** $\mathcal{H}$ entre la distribution cible (Dirac en $t_i^{GT}$) et la Gaussienne prédite $t_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$ :
$$\mathcal{L}_{bbox} = \frac{1}{N_{bbox}} \sum_{i} \mathbb{I}_{fg}(t_i) \mathcal{H}\left(t_i^{GT}, t_i\right) = -\frac{1}{N_{bbox}} \sum_{i} \mathbb{I}_{fg}(t_i) \log\left( \mathcal{N}\left(t_i^{GT}; \mu_i, \sigma_i^2\right) \right)$$
En développant le logarithme de la densité Gaussienne, on obtient l'expression intuitive de la perte de vraisemblance négative (_Negative Log-Likelihood - NLL_) :
$$\mathcal{L}_{bbox} = \frac{1}{N_{bbox}} \sum_{i} \mathbb{I}_{fg}(t_i) \left[ \frac{(t_i^{GT} - \mu_i)^2}{2\sigma_i^2} + \frac{1}{2}\log(\sigma_i^2) + \frac{1}{2}\log(2\pi) \right]$$
> **Analyse physique de l'équation :**
> - Le premier terme $\frac{(t_i^{GT} - \mu_i)^2}{2\sigma_i^2}$ pondère l'erreur quadratique par l'inverse de la variance. Si l'incertitude $\sigma_i^2$ est grande, l'impact d'une erreur d'alignement est atténué.
> - Le second terme $\frac{1}{2}\log(\sigma_i^2)$ agit comme une **pénalité de régularisation** : il empêche le réseau de prédire systématiquement une variance infinie pour annuler la perte.

### Adaptation sur le Domaine Cible ($\mathcal{L}_{T-box}$ - Équation 6)
Pour le domaine cible non annoté, la cible de régression devient la distribution pseudo-label $t_i^{PL}$ issue du Teacher sur l'image faiblement augmentée ($x_{t,w}$), affûtée par la fonction $S_{bbox}$ :
$$\mathcal{L}_{T-box} = \frac{1}{N_{bbox}} \sum_{i} \mathbb{I}_{fg}(t_i) \mathcal{H}\left( S_{bbox}\left(t_i^{PL}, \tau_{bbox}\right), t_i \right)$$
Pour la régression des boîtes, l'entropie d'une distribution Gaussienne ne dépend que de sa variance $\sigma^2$ (plus la variance est grande, plus l'incertitude/entropie est grande).

Pour "affûter" la prédiction de la boîte de détection du Teacher, on n'a pas de SoftMax. La fonction $S_{bbox}$ consiste simplement à réduire artificiellement la variance prédite en la multipliant par une température $\tau_{bbox} < 1$ :

$$\sigma^2 \leftarrow \sigma^2 * \tau_{bbox}$$

**Comment ça marche physiquement ?**
- Le Teacher prédit une coordonnée avec une certaine incertitude $\sigma^2$.
- En multipliant cette variance par un facteur inférieur à $1$ (ex: $0.5$), on "écrase" la courbe de Gauss. La cloche devient beaucoup plus étroite et pointue autour de la moyenne $\mu$.
- On force ainsi le Teacher à générer un pseudo-label de localisation avec une "fausse" confiance accrue, ce qui donne une cible plus nette et moins ambiguë (low-entropy) pour entraîner le Student.
## 3. Adaptation de la Classification ($\mathcal{L}_{T-cls}$) & Entropy Focal Loss (EFL)

Dans une architecture à deux étapes (Faster R-CNN), la classification non supervisée sur le domaine cible s'effectue à la fois au niveau du **RPN** et de la **RoI-Head**.

### 3.1 Pertes de Classification Target ($\mathcal{L}_{T-cls}^{RPN}$ et $\mathcal{L}_{T-cls}^{ROI}$ — Équation 5 du Papier)
La perte d'adaptation de classification $\mathcal{L}_{T-cls}$ se décompose en deux termes de Cross-Entropie $\mathcal{H}$ calculés entre les prédictions du Teacher (sur l'image cible $x_{t,w}$) et du Student (sur l'image cible $x_{t,s}$) :
$$\mathcal{L}_{T-cls}^{RPN} = \frac{1}{N_{cls}^{RPN}} \sum_{i} \mathcal{H}\left( M\left(S_{cls}\left(p_i^{PL}, \tau_{cls}\right)\right), \, p_i^{RPN} \right)$$
$$\mathcal{L}_{T-cls}^{ROI} = \frac{1}{N_{cls}^{ROI}} \sum_{i} \mathcal{H}\left( S_{cls}\left(p_i^{PL}, \tau_{cls}\right), \, p_i^{ROI} \right)$$
#### Définition des variables :
- **$p_i^{PL}$ :** La distribution de probabilité de classification prédite par le Teacher pour la $i$-ème proposition.
- **$p_i^{RPN}$ et $p_i^{ROI}$ :** Les distributions de probabilité de classification prédites respectivement par le RPN et la RoI-Head du Student.
- **$S_{cls}(\cdot, \tau_{cls})$ :** La fonction d'affûtage (_Sharpening Function_) avec le facteur de température $\tau_{cls}$.
- **$M(\cdot)$ (Merging Operation) :** Une opération de fusion qui additionne toutes les probabilités des classes de premier plan (_foreground_) afin d'obtenir une distribution binaire premier-plan / arrière-plan ($fg/bg$) nécessaire au guidage du RPN.
- **$N_{cls}^{RPN}$ et $N_{cls}^{ROI}$ :** Les tailles de batch (nombre de candidats) dans le RPN et la RoI-Head.
## L'affûtage de la Classification ($S_{cls}$ - Équation 7)

Au lieu de modifier les probabilités après coup, l'article applique une **température** directement dans la fonction SoftMax du Teacher.
Si $z_i$ est le "logit" (le score brut) pour la classe $i$, le SoftMax avec température $\tau_{cls}$ est :
$$S_{cls}(\mathbf{z}, \tau_{cls}) = SoftMax(.,\tau_{cls)=}) \frac{\exp(z_i / \tau_{cls})}{\sum_j \exp(z_j / \tau_{cls})}$$
**Comment ça marche physiquement ?**
L'article précise qu'ils fixent $\tau < 1$ (par exemple $\tau = 0.5$).
- Diviser les logits par $0.5$ revient à les multiplier par $2$.
- Les écarts entre le logit dominant et les autres sont amplifiés de manière exponentielle.
- Résultat : la probabilité de la classe dominante se rapproche de $1$ (Dirac), et celle des autres se rapproche de $0$. L'entropie (l'incertitude) chute drastiquement.
### 3.2 Formulation Générale de l'Entropy Focal Loss (EFL — Équation 11)

Bien que l'affûtage réduise l'entropie des prédictions, les pseudo-boîtes cibles contiennent inévitablement du bruit sous _domain shift_. PT introduit l'**Entropy Focal Loss (EFL)** pour pondérer par l'incertitude :

$$\mathcal{H}_{EFL}(\cdot, \cdot) = \left( 1 - \frac{E}{E_{norm}} \right)^\lambda \mathcal{H}(\cdot, \cdot)$$

#### Définition des variables :

- **$\mathcal{H}(\cdot, \cdot)$ :** La perte de Cross-Entropie de base (définie à l'Équation 5 pour la classification ou à l'Équation 1 pour la régression).
- **$\lambda$ :** Un hyperparamètre de focalisation (analogue au $\gamma$ de la Focal Loss standard).
- **$E$ :** L'entropie de la prédiction du Teacher sur la cible (mesurant l'incertitude de la classe ou des coordonnées).
- **$E_{norm}$ :** Le terme de normalisation correspondant à l'**entropie maximale théorique** :
    - **Pour la classification ($E_{norm}^{cls}$) :** $E_{norm} = \log(n + 1)$, où $n$ est le nombre de classes de premier plan (avec $+1$ pour la classe _background_).
    - **Pour la localisation/régression ($E_{norm}^{box}$) :** $E_{norm} = \frac{1}{2}\log(2\pi) + \frac{1}{2}$ (dérivé de l'entropie d'une distribution Gaussienne univariée).
### 3.3 Signification Intuitive de l'EFL

Au lieu d'un filtrage binaire par seuil ($p > \text{threshold}$), l'EFL module de manière continue la contribution de chaque proposition :
- **Faible incertitude ($E \to 0$) :**$$\left( 1 - \frac{E}{E_{norm}} \right)^\lambda \to 1$$
    La prédiction du Teacher est très sûre : la perte $\mathcal{H}$ s'applique à plein régime pour entraîner le Student.
- **Forte incertitude ($E \to E_{norm}$) :**
    $$\left( 1 - \frac{E}{E_{norm}} \right)^\lambda \to 0$$
    La prédiction du Teacher est très bruitée / indécise : le poids tend vers $0$, ce qui empêche les fausses détections ou boîtes mal alignées de perturber le Student.


## 4. Augmentation Forte & Alignement Intra-Domaine (_Intra-Domain Alignment_)
![[learning Domain Adaptive object Detection with Probabilist teacher-1784298004541.webp]]
Dans la section 5, les auteurs mettent en lumière un problème souvent négligé dans la littérature de l'UDA-OD (_Unsupervised Domain Adaptation for Object Detection_) : l'**Intra-Domain Gap**.

### 4.1. Le Phénomène d'Intra-Domain Gap (Section 5.1)

Les travaux classiques d'UDA se focalisent sur l'**Inter-Domain Gap** (le décalage global de distribution entre le domaine Source $A$ et le domaine Cible $B$, par exemple temps clair vs brouillard).

Cependant, en analysant les Vrais Positifs (TP) et Faux Négatifs (FN) prédits sur le domaine cible, les auteurs observent une **forte disparité de performance au sein même du domaine cible** :

- **Objets faciles :** Les grands objets bien visibles conservent de bonnes prédictions.
- **Objets difficiles :** Les objets petits, très flous ou partiellement occultés (_occluded_) subissent une dégradation massive des performances.

> **Définition (Intra-Domain Gap) :** L'écart de difficulté d'adaptation qui existe _à l'intérieur_ du domaine cible entre les objets clairs/volumineux et les objets petits/flous/occultés.

### 4.2. Alignement Intra-Domaine via Augmentation Forte (Section 5.2)

Pour combler cet _intra-domain gap_ sans ajouter de sous-réseau complexe, PT exploite l'**augmentation de données forte** (_Strong Data Augmentation_) comme stratégie d'alignement implicite.

![[learning Domain Adaptive object Detection with Probabilist teacher-1784298192421.webp|chéma de fonctionement]]

#### Mécanisme pas à pas :

1. **Génération par le Teacher :** Sur l'image cible faiblement augmentée ($x_{t,w}$), le Teacher prédit des pseudo-labels avec une **faible entropie** (haute confiance) sur les objets faciles.
2. **Transformation pour le Student :** L'image transmise au Student ($x_{t,s}$) subit des augmentations fortes (_random resizing_, flou gaussien, _color jitter_). Ces transformations dégradent artificiellement les objets faciles pour qu'ils ressemblent aux objets difficiles (petits, flous ou occultés).
3. **Guidage & Alignement :** Le Student est forcé d'apprendre des représentations robustes sur ces versions dégradées, guidé par les pseudo-labels fiables du Teacher.

> **Conclusion :** L'augmentation forte agit comme un **alignement intra-domaine implicite** : elle force le réseau à transférer la confiance acquise sur les grands objets nets vers les objets dégradés du même domaine.


### Flux de Données & Entraînement EMA
- **Student ($\theta_S$) :** Entraîné par rétropropagation du gradient sur les images sources annotées ($x_s$) et les images cibles fortement augmentées ($x_{t,s}$).
- **Teacher ($\theta_T$) :** Génère les pseudo-labels probabilitaires à partir des images cibles faiblement augmentées ($x_{t,w}$). Ses poids sont mis à jour via une moyenne mobile exponentielle (_Exponential Moving Average_) :$$\theta_T \leftarrow \alpha \theta_T + (1 - \alpha) \theta_S$$
## 5. Fonction d'Objectif Globale & Synthèse des Pertes ($\mathcal{L}_{total}$)

La perte globale combine la supervision sur la source et les adaptations probabilistes sur la cible :
![[learning Domain Adaptive object Detection with Probabilist teacher-1784469354686.webp|697]]
![[learning Domain Adaptive object Detection with Probabilist teacher-1784469394425.webp]]![[learning Domain Adaptive object Detection with Probabilist teacher-1784469440605.webp|700x289]]
