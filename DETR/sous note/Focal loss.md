---
tags:
  - deep-learning
  - object-detection
  - loss-function
  - detr
---

# Focal Loss for Dense Object Detection (Lin et al., 2017)

##  Contexte & Problématique
https://arxiv.org/pdf/1708.02002 
La **Focal Loss** a été introduite initialement pour les modèles de détection *One-Stage* (comme RetinaNet), mais elle est reprise dans les évolutions de DETR (**Deformable DETR**, **Anchor DETR**) au niveau du calcul des pertes et du *Hungarian Matcher*.

Son but principal est de résoudre le problème du **déséquilibre extrême des classes (Class Imbalance)** entre l'arrière-plan (le fond) et les objets à détecter (ratio typique de $1:1000$). 

> [!danger] Le problème dans DETR
> Dans la structure de DETR, on utilise par exemple 300 requêtes (queries) par image. Si une image ne contient que 2 objets, nous nous retrouvons avec **298 prédictions "pas d'objet" (background)** et seulement **2 objets réels**. Si on utilise une perte classique, la somme des petites pertes des prédictions "background" va complètement submerger (noyer) la perte des objets réels, empêchant le modèle de bien apprendre.
---
## 🛠️ Fonctionnement Mathématique
La Focal Loss ajuste le poids des exemples dynamiquement en combinant deux mécanismes : l'équilibrage des classes ($\alpha$) et la focalisation sur les exemples difficiles ($\gamma$).
### 1. La formule générale
Pour une probabilité prédite $p$ (comprise entre 0 et 1) :
* **Pour un exemple Positif (Objet réel) :**
$$\text{Loss}_{\text{pos}} = - \alpha \cdot (1 - p)^\gamma \cdot \log(p)$$
* **Pour un exemple Négatif (Fond / Background) :** $$\text{Loss}_{\text{neg}} = - (1 - \alpha) \cdot p^\gamma \cdot \log(1 - p)$$### 2. Le rôle des hyperparamètres
* **$\gamma$ (Gamma) — Facteur de focalisation (ex: `gamma = 2.0`) :**
  Il réduit drastiquement la contribution des exemples "faciles" (ceux dont le modèle est déjà sûr).
  * *Exemple :* Si le modèle est sûr à 99% qu'une boîte est vide ($p = 0.01$ pour la classe objet), alors $(0.01)^2 = 0.0001$. La perte pour cet exemple facile devient quasiment nulle. Le modèle concentre son énergie sur les prédictions incertaines.
* **$\alpha$ (Alpha) — Équilibrage des classes (ex: `alpha = 0.25`) :**
  Il donne un poids de base aux positifs ($\alpha$) et aux négatifs ($1 - \alpha$). 
  * *Contre-intuition importante :* Bien que les positifs soient rares, on utilise $\alpha = 0.25$ (et donc $0.75$ pour les négatifs). C'est parce que le paramètre $\gamma = 2.0$ fait déjà un travail tellement puissant pour écraser la perte des négatifs faciles, qu'un $\alpha$ trop grand en faveur des positifs déstabiliserait l'apprentissage. La combinaison $(\alpha=0.25, \gamma=2.0)$ est le standard empirique universel. (source Focal Loss for Dense Object Detection https://arxiv.org/pdf/1708.02002 page 6 table b)
---

## 💻 Application dans le Hungarian Matcher (Deformable / Anchor DETR)
Dans le code du Matcher, la Focal Loss est convertie en coût d'association. Pour chaque requête, on calcule la différence de coût entre "considérer la prédiction comme un objet" et "la considérer comme du fond" :

```python
alpha = 0.25
gamma = 2.0

# Perte si on classifie la requête comme du fond (Négatif)
neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())

# Perte si on classifie la requête comme l'objet cible (Positif)
pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

# Le coût final est la différence : impact sur la perte globale
cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
```


## 📝 Exemples Concrets de Calcul (DETR vs Deformable DETR)

Imaginons une requête spécifique ($\hat{y}_i$) évaluée face à un objet réel ($y_j$) de la classe **"Voiture"**.
Pour cet exemple, le modèle a trouvé 2 **excellente boîte géométrique**, ce qui nous donne un coût de boîte très bas :
* $L_{\text{box}} = 0.15$

Voyons comment les deux Matchers réagissent si le modèle est encore très hésitant sur la classe (début d'entraînement).

---

### Cas 1 : Le Matcher original de DETR (Softmax)
Le modèle utilise un `softmax` global. Écrase par la classe fond, le modèle ne prédit que $p = 0.02$ pour la classe "Voiture" (et donc $0.98$ pour le fond).

* **Formule :** $L_{\text{match}} = -\hat{p}(c_i) + L_{\text{box}}$
* **Calcul :** $L_{\text{match}} = -0.02 + 0.15 = \mathbf{+0.13}$

> [!danger] Le problème du Cas 1
> Le coût total est inférieur à la boîte seule ($0.13 < 0.15$) car le score de $-0.02$ vient "aider" à faire baisser le coût global. Le Matcher va sélectionner cette requête. 
>
> **Conséquence sur la Loss :** La Loss va recevoir cette association et voir une prédiction "Voiture" à seulement $0.02$. Elle va générer un gradient massif pour détruire les $0.98$ de confiance du fond de cette requête. Si au tour d'après, une autre requête a une boîte similaire et un score de $0.03$, le Matcher va sauter sur l'autre requête, créant une instabilité chronique où le modèle passe son temps à basculer la classe fond d'une requête à l'autre.
---
### Cas 2 : Le Matcher modifié (Focal Loss avec Sigmoid)
Ici, on utilise la Focal Loss avec $\alpha = 0.25$ et $\gamma = 2.0$. Reprenons exactement la même situation : le modèle est très incertain sur la classe ($p = 0.20$ pour "Voiture") mais la boîte est excellente ($L_{\text{box}} = 0.15$).

#### Étape A : Calcul du coût de classification Focal
* $\text{Loss}_{\text{pos}} = -0.25 \times (1 - 0.20)^2 \times \log(0.20) \approx +0.256$
* $\text{Loss}_{\text{neg}} = -(1 - 0.25) \times 0.20^2 \times \log(1 - 0.20) \approx +0.0066$
* $\text{Coût}_{\text{class}} = \text{Loss}_{\text{pos}} - \text{Loss}_{\text{neg}} = 0.256 - 0.0066 = \mathbf{+0.2494}$ (La mauvaise classification agit comme une pénalité positive).

#### Étape B : Calcul du coût total de Matching
* **Formule :** $L_{\text{match}} = \text{Coût}_{\text{class}} + L_{\text{box}}$
* **Calcul :** $L_{\text{match}} = 0.2494 + 0.15 = \mathbf{+0.3994}$

>[!success] Analyse du Cas 2 (Corrigée)
>
>Même si la boîte est excellente ($0.15$), le coût total grimpe à **$+0.3994$** car l'incertitude sur la classe agit comme une pénalité (coût positif).
>
Comme l'algorithme hongrois est **obligé** d'associer une requête à l'objet réel, il choisira quand même cette requête si elle est la "moins pire" des disponibles.
>
Cependant, le fait que le coût de classification soit devenu positif pour toutes les requêtes incertaines neutralise le bruit : l'algorithme n'est plus tenté de choisir une mauvaise boîte plutôt qu'une bonne juste parce qu'elle avait une probabilité de classe de $0.03$ au lieu de $0.01$. C'est la **géométrie pure ($L_{\text{box}}$)** qui dicte le choix du match tant que le modèle n'est pas sûr de la classe. Cela stabilise l'entraînement.

> [!warning] Pourquoi ne pas ajouter une sigmoïde dédiée au "Fond" ?
> Face à cette structure, on pourrait intuitivement vouloir créer une sigmoïde supplémentaire qui prédirait explicitement si la boîte est vide ("Est-ce du fond ? Oui/Non"). C'est une erreur conceptuelle et mathématique pour deux raisons :
> 
> 1. **Cela détruirait l'intérêt de la Focal Loss :** Sur les 300 requêtes de l'image, environ 298 sont du vide. Si on crée une couche "Fond", ces 298 requêtes deviendraient des cibles positives faciles à détecter. La Focal Loss accumulerait alors une énorme quantité de pertes sur ces exemples positifs ultra-majoritaires, recréant exactement le problème de déséquilibre massif des classes que l'on cherchait à fuir.
> 2. **Incohérence des sigmoïdes indépendantes :** Contrairement au Softmax où les probabilités s'ajustent mutuellement pour faire $1.0$, des sigmoïdes indépendantes ne communiquent pas entre elles. Sans contrainte, le modèle pourrait très bien prédire simultanément *Voiture = 0.90* ET *Fond = 0.90*, ce qui est absurde. 
> 
> **La règle d'or :** Dans cette architecture, le fond n'est pas une classe, c'est **l'état implicite par défaut** (le vide) lorsque toutes les sigmoïdes des vraies objets sont proches de 0.