

![[EW-DETR-figure1.webp]]
le parpier 
 - incremental learning 
 - domain adaptation

![[EW-DETR-figure3.webp]]
# Incremental LoRA adapters
## 1. Concept Général : Incremental LoRA
Pour éviter l'oubli catastrophique (catastrophic forgetting) lors de l'apprentissage de nouvelles tâches, le modèle attache des **Incremental Low-Rank Adaptation ([[LoRA]]) adapters** aux FFN de l'encodeur et du décodeur du Transformer.
- **Avantage :** Fournit une mémoire compacte des tâches passées.
- **Bonus :** Ne nécessite de stocker _aucune_ donnée des tâches précédentes.
## 2. Architecture des Poids (Low-Rank Adaptation)
![[EW-DETR-1783602617354.webp]]
Pour chaque couche ciblée à la tâche $t$, les poids de base du modèle initial ($W_0$) sont **gelés**. Le modèle utilise deux types d'adaptateurs en parallèle :
1. **Aggregate LoRA Adapter ($\Delta W^{t-1}_{agg}$)** : Un buffer _non-entraînable_ (pendant la tâche $t$) qui accumule et stocke les connaissances de toutes les tâches précédentes.
2. **Task-Specific LoRA Adapter ($\Delta W^{t}_{task}$)** : Les paramètres _entraînables_ dédiés uniquement à la tâche actuelle pour capturer les spécificités des nouvelles classes. Il est réinitialisé à chaque nouvelle tâche.
Pour rappel, chacune de ces deux matrices sont constituées de deux matrices de taille plus petites. voir notes [[LoRA]] pour plus d'informations 
## 3. Gestion du déséquilibre des données (Data Imbalance)
![[EW-DETR-1783602716293.webp]]
Lorsqu'on passe à une nouvelle tâche, on doit fusionner les nouvelles connaissances avec les anciennes. Mais si la nouvelle tâche contient énormément de données par rapport au passé, elle risque d'écraser la mémoire.
pour résoudre Le problème, l'article propose d'utiliser data-aware merging coefficient $β_t$, limiter par $β_{min}$ and $β_{max}$, et pondéré par le nombre d'images des tâches actuelles et précédentes. 
$$\beta_t = \left\{ \begin{array}{ll} 1, & t = 1, \\[4pt] \beta_{\max} - (\beta_{\max} - \beta_{\min}) \frac{N_t}{N_{1:t-1}}, & t \geq 2. \end{array} \right.$$
- $N_t$ = Nombre d'images dans la tâche actuelle.
- $N_{1:t-1}$ = Nombre total d'images vues dans **toutes** les tâches précédentes.
valeur suggéré $(β_{min}, β_{max}) = (0.2 , 0.8)$
- ## 4. Fusion et Compression (SVD)
À la fin de l'entraînement de la tâche $t$, on doit mettre à jour la "mémoire globale". On fusionne l'adaptateur spécifique avec l'adaptateur global :
$$\Delta W^{t}_{merged} = (1 - \beta_t) \Delta W^{t-1}_{agg} + \beta_t \Delta W^{t}_{task}$$
Pour éviter que cette nouvelle matrice mémoire devienne trop grosse (qu'elle perde son statut "Low-Rank" après l'addition), on la recompresse en utilisant une **SVD tronquée (Singular Value Decomposition)** :
- Décomposition : $\Delta W^{merged}_t = U \Sigma V^\top$
- On garde les $r$ composantes principales : $\Delta W^{merged}_t \approx U_r \Sigma_r V_r^\top$
on découpe le résultat pour recréer nos fameuses deux petites matrices pour la mémoire future :
- $B^{agg}_t = U_r \Sigma_r$
- $A^{agg}_t = V_r^\top$
ainsi , la mémoire de la tache suivante devient : $\Delta W^{agg}_t = B^{agg}_t A^{agg}_t$

# Query-Norm Objectness Adapter

Le **Query-Norm Objectness Adapter** a pour but d'aider le modèle à détecter des objets "inconnus" (classes qu'il n'a pas encore apprises) et à résister aux changements de domaine (ex: passer de photos réelles à des dessins). Pour cela, il modifie la sortie du décodeur en séparant deux informations fondamentales :
1. **La sémantique (Le "Quoi") :** La direction du vecteur de caractéristiques (qui indique la classe de l'objet).
2. **La magnitude (Le "Est-ce un objet ?") :** La norme (longueur) du vecteur, qui agit comme un indice universel de présence d'objet (_class-agnostic objectness_).
##  2. Explication Mathématique (Étape par étape)
![[EW-DETR-1783602651612.webp]]
### Étape A : La Normalisation (Isoler la Direction)
Soit $h_i$ le vecteur de caractéristiques issu de la **dernière couche du décodeur** pour la requête (query) $i$.
$$h_{norm} = \frac{\text{LN}(h_i)}{\|\text{LN}(h_i)\|_2}$$

> [!info] 💡 Explication de l'équation 7 
> 
> Un vecteur possède toujours deux choses : une **direction** et une **longueur** (la norme).
> 
> - **$\text{LN}$** applique une Layer Normalization classique.
>     
> - **$\|\cdot\|_2$** calcule la longueur géométrique du vecteur (Norme L2).
>     
> - **Pourquoi diviser par la norme ?** En divisant un vecteur par sa propre longueur, on force sa longueur à devenir exactement égale à 1. Le vecteur résultant $h_i^{norm}$ se retrouve projeté sur une " unit sphere".
>     
> - **L'intérêt analytique :** En forçant la longueur à 1, on détruit l'information de magnitude. Le vecteur $h_i^{norm}$ ne contient plus _que_ la direction pure (la sémantique de la classe). Ainsi, si le domaine change (covariate shift) et modifie l'amplitude des signaux, cette représentation reste parfaitement stable.
### Étape B : La Combinaison Convexe (Classification)
Pour obtenir la caractéristique finale de classification, le modèle ne garde pas uniquement le vecteur normalisé. Il crée un mélange (_convex combination_) entre le vecteur original $h_i$ et le vecteur normalisé $h_i^{norm}$, géré par un paramètre apprenable $\alpha_{mix}$ :
$$h_{cls} = (1 - \alpha_{mix})h_i + \alpha_{mix}h_{norm}$$
Ensuite, une couche linéaire classique transforme ce vecteur hybride en prédictions de classes (logits) :
$$z_{cls} = W_{cls}h_{cls} + b_{cls}$$
### Étape C : L'évaluation de la "Présence d'Objet" (Objectness)
Dans les architectures basées sur DETR, il y a un phénomène empirique connu : **les requêtes qui trouvent un vrai objet développent une norme (longueur de vecteur) beaucoup plus grande** que les requêtes qui tombent sur du fond (background). (démontre dans l'article).
Le module exploite directement cette propriété scalaire $\|h_i\|_2$ (la longueur qu'on avait enlevée à l'étape A). Il la fait passer par un petit réseau de neurones ($f_{obj}$) puis applique un ajustement de température ($\tau$) :
$$z^{obj}_i = \frac {f_{obj}{(\|h_i\|_2})}{\tau + \epsilon} $$
Note : $\epsilon$ est juste une toute petite constante (ex: $10^{-5}$) ajoutée pour éviter les erreurs de calcul mathématique (stabilité numérique).
## 🚀 3. Avantage architectural clé (Zéro surcoût)

L'un des arguments majeurs de ce module est sa légèreté lors de l'entraînement :
**QNorm-Obj n'introduit aucune fonction de perte (loss) supplémentaire.**
Il n'y a pas besoin de dire explicitement au modèle "apprends cette norme". Les vecteurs de classification normalisés ($h^{cls}_i$) et le réseau d'objectness ($f_{obj}$) apprennent implicitement à s'ajuster grâce à la fonction de perte standard de détection d'objets (la _detection loss_ classique de DETR).


# Entropy-Aware Unknown Mixing (EUMix)

## 1. Objectif du Module
Les objets "inconnus" ne sont jamais étiquetés dans les jeux de données (ce qui imite la vraie vie). Le module **EUMix** sert à calibrer le score de la classe "Inconnue" en fusionnant deux signaux indépendants :
1. **L'incertitude du classifieur :** Le modèle n'arrive pas à assigner une classe connue avec certitude.
2. **Objectness:** Le modèle "sent" qu'il y a un objet à cet endroit précis.
## 2. Dérivation Mathématique (Étape par étape)
Soit une requête $i$. Le modèle produit des prédictions brutes (logits) pour les classes connues ($z_i^{known}$) et un logit spécifique pour la classe inconnue ($z_i^{unk}$). Il reçoit aussi le score d'objectness ($z_i^{obj}$) calculé par le module _QNorm-Obj_.

### Étape A : Mesurer "l'espace disponible" (Le Gap)
Le but de cette étape est de regarder la sortie du classifieur pour voir si le modèle est certain d'avoir reconnu une classe qu'il connaît déjà, ou s'il est complètement perdu.

Pour cela, le modèle isole d'abord le sous-vecteur des classes connues $z_{i}^{known}$ (en laissant totalement de côté le logit inconnu $z_{i}^{unk}$ pour l'instant). Il cherche ensuite la classe connue pour laquelle il a obtenu le score le plus élevé :
$$p_i^{known,max} = \max_{c \in K_t} \sigma(z_{i,c}^{known})$$
-  Précision sur le Sigmoïde ($\sigma$) le model utilise la [[Focal loss]] 

Il calcule ensuite l'**écart de confiance** (_Gap_ $g_i$) :
$$g_i = (1 - p_i^{known,max})^\gamma$$
- **Logique :** Si $p_i^{known,max}$ est faible (le modèle doute), l'écart $g_i$ s'approche de 1. Cela libère de la "masse de probabilité" pour dire que c'est un inconnu. Le paramètre $\gamma$ (apprenable) sert de température pour rendre cette transition plus ou moins abrupte.
### Étape B : La probabilité guidée par l'Objectness
On multiplie la probabilité qu'il y ait un objet (via $z_i^{obj}$) par ce fameux Gap ($g_i$) :
$$p_{obj,i}^{unk} = \sigma(z_i^{obj}) \cdot g_i$$
- **Logique :** Cette valeur est forte **uniquement** si le modèle voit  un objet (Objectness fort) **ET** qu'aucune classe connue ne correspond (Gap fort).
### Étape C : La probabilité guidée par le Classifieur
En parallèle, on convertit le logit "inconnu" brut en probabilité, en y ajoutant un biais apprenable ($b_{obj}$) :

$$p_{cls,i}^{unk} = \sigma(z_i^{unk} + b_{obj})$$
Logique et rôle du biais ($b_{obj}$) :
- **Le problème du Matcheur (Biais négatif) :** Pendant l'entraînement, les objets inconnus ne sont jamais annotés. L'algorithme d'appariement (_Hungarian Matching_) force donc le modèle à les associer au fond (_Background_). La Focal Loss punit continuellement ce canal, écrasant le logit $z_i^{unk}$ vers des valeurs fortement négatives (ex: $-5$), ce qui bloque mathématiquement la Sigmoïde à $0$.
- **La solution (Re-positivation) :** Le biais $b_{obj}$ est un scalaire global que l'optimiseur va apprendre à rendre **positif** (ex: $+4$).
	- (L'occasion du gradient : L'optimiseur fait monter ce biais lorsque le modèle fait une erreur en tentant d'absorber un objet inconnu dans une classe connue. La perte punit violemment cette fausse alerte sur la classe connue, ce qui ouvre le Gap et force l'optimiseur à utiliser le biais $b_{obj}$ pour rediriger l'énergie vers la classe inconnue afin de réduire la perte globale de la requête).
- **Effet mathématique :** Il  décale la sensibilité de la courbe Sigmoïde vers le haut. Il compense la pénalisation du matcheur en permettant à un très faible signal positif du réseau ($z_i^{unk} = -5 \implies -5 + 4 = -1$) de réactiver la Sigmoïde ($\sigma(-1) \approx 0.27$) pour valider la détection.
### Étape D : Le Mixage Final
Les deux probabilités sont fusionnées via un paramètre apprenable $\alpha \in (0, 1)$ :
$$p_{final,i}^{unk} = \alpha \cdot p_{cls,i}^{unk} + (1 - \alpha) \cdot p_{obj,i}^{unk}$$
Enfin, on reconvertit cette probabilité en logit pour la passer à la fonction de perte finale :
$$z_{final,i}^{unk} = \text{logit}(p_{final,i}^{unk})$$
> [!info] 💡 La dynamique de $\alpha$
> 
> Au début de l'entraînement, $\alpha$ int favorise l'Objectness ($p_{obj,i}^{unk}$), car c'est un signal physique robuste. Plus le classifieur devient intelligent et fiable sur les caractéristiques visuelles des inconnus, plus l'optimiseur fait naturellement monter la valeur de $\alpha$.
## 🛠️ 3. Protocole d'Entraînement
Le protocole est conçu pour empêcher la triche et imiter une véritable IA en conditions réelles :
- **Ce qui est annoté :** Uniquement les classes connues de la tâche actuelle ($T_t$).
- **Ce qui n'est PAS annoté :** Les classes des tâches précédentes (pour tester l'oubli) ET les objets véritablement nouveaux (pour tester la détection d'inconnus).
- **Les "Hold-out classes" :** Certaines classes (ex: les camions) sont volontairement effacées de tous les jeux de données d'entraînement. Elles servent de vérité terrain absolue lors de l'évaluation pour vérifier si l'IA arrive bien à les détecter en tant que "classe inconnue".
## 4. Dynamique et Interaction des Gradients (Le flux de l'optimisation)

Puisqu'il n'y a **aucune étiquette "inconnu"** pendant l'entraînement, la cible (_Ground Truth_) de la Focal Loss pour le canal inconnu est **toujours égale à 0**. L'apprentissage des paramètres $\alpha$ et $b_{obj}$ repose entièrement sur un jeu de balance de gradients entre trois cas de figure :

### 🟢 Cas 1 : Un objet inconnu est confondu avec une classe connue (L'occasion d'apprentissage)
- **Situation :** L'image contient un objet non annoté (ex: un chien). Le modèle s'y intéresse et fait monter par erreur le logit d'une classe connue (ex: $z_{chat} = +3$).
- **Cible de la Loss :** Le matcheur n'ayant pas d'annotation pour cette zone, la cible pour _toutes_ les classes (connues et inconnues) est fixée à `0` (Background).
- **Comportement des gradients :** 1. La Focal Loss voit une probabilité de $95\%$ sur la classe `chat` alors que la cible est `0`. Elle génère un **gradient négatif massif** sur le canal `chat` pour écraser son logit.
    2. En écrasant le logit connu, $p^{known,max}$ chute vers $0$, ce qui **ouvre instantanément le Gap** ($g_i \to 1$).
    3. L'encodeur-décodeur (l'attention) maintient un score d'Objectness très fort ($z_{obj}$ haut) car l'objet est physiquement présent.
    
- **Résultat sur EUMix :** L'optimiseur (AdamW) cherche à minimiser la perte globale de la requête. Il "comprend" que pour éliminer la terrible pénalité liée au faux positif (`chat`), il doit utiliser la valve de sécurité : il ajuste $\alpha$ et augmente le biais positif $b_{obj}$. L'énergie de la fausse alerte est transférée vers le canal inconnu. Même si le canal inconnu subit une petite pénalité (car sa cible est aussi à 0), la réduction de la perte sur la classe connue est tellement immense que l'optimiseur valide ce choix.
    

### 🔴 Cas 2 : Du vrai bruit de fond (Le vide complet) est détecté comme inconnu (Faux Positif)

- **Situation :** Le modèle s'active sur une zone vide de l'image (un morceau de ciel ou de route) et fait monter le logit inconnu par erreur.
- **Cible de la Loss :** Cible à `0` pour tout le monde (Background).
- **Comportement des gradients :**
    1. Comme c'est du vide, les couches géométriques du réseau ne détectent aucune forme. Le score d'Objectness est proche de zéro ($z_{obj} \to \text{très bas} \implies \sigma(z_{obj}) \approx 0$).
    2. La branche de l'Objectness ($p^{unk}_{obj}$) est donc mathématiquement verrouillée à $0$ par la géométrie.
    3. Le seul moyen pour que le modèle ait prédit un inconnu ici est que le biais $b_{obj}$ soit trop agressif ou que le logit brut $z^{unk}$ ait drifté.
- **Résultat sur EUMix :** La Focal Loss applique une pénalité directe sur le canal inconnu. Ne pouvant pas modifier l'Objectness (qui est déjà à 0), l'optimiseur n'a pas d'autre choix que d'envoyer un **gradient négatif direct sur $b_{obj}$** pour le diminuer, et ajuste $\alpha$ pour réduire la sensibilité globale du classifieur dans le vide. Cela empêche le modèle de voir des inconnus partout.
    

### 🔵 Cas 3 : Un objet de classe connue est correctement détecté (Vrai Positif connu)

- **Situation :** Le modèle voit une vraie voiture, parfaitement annotée dans le dataset.
- **Cible de la Loss :** Cible à `1` pour la classe `voiture`, et `0` pour toutes les autres (y compris l'inconnu).
- **Comportement des gradients :**
    1. Le modèle prédit correctement la voiture : $z_{voiture}$ est haut, donc $p^{known,max} \approx 1$.
    2. Le Gap mathématique se ferme immédiatement : $g_i = (1 - 1)^\gamma = 0$.
    3. Comme $g_i = 0$, la branche d'Objectness de l'inconnu est multipliée par zéro : $p^{unk}_{obj} = \sigma(z_{obj}) \cdot 0 = 0$.
- **Résultat sur EUMix :** Le flux de gradient provenant de la détection réussie est **totalement bloqué** par le Gap fermé. EUMix est mathématiquement isolé. Cela garantit que l'apprentissage des classes connues ne vient pas perturber ou dérégler les paramètres de détection des objets inconnus.