improtent de reconert des ovject non vue comme "inconu"


olving World Object Detection (EWOD),

First, Incremental LoRA Adapters (Section 3.3) employ a dual-adapter architecture: an aggregate adapter that accumulates compressed knowledge from all previous tasks and a task-specific adapter that captures current task updates. Through data-aware merging guided by pertask sample ratios and low-rank projection via truncated SVD, we achieve stable knowledge consolidation without storing any exemplars while explicitly addressing data imbalance. Second, Query-Norm Objectness Adapter (Section 3.4) decouples query semantics from magnitude by normalizing decoder features, yielding domain-invariant classagnostic representations that enable robust unknown detection even under severe domain shifts—crucially, without requiring any auxiliary supervision or additional losses, unlike [7, 36]. Third, Entropy-Aware Unknown Mixing (Section 3.5) calibrates unknown predictions by combining classification uncertainty with objectness evidence, ensuring that high-objectness, high-uncertainty queries are correctly identified as unknowns rather than spuriously absorbed into known classes or background.


![[EW-DETR-figure1.webp]]

but de EWOD

(i) detect all known classes Kt across all seen domains {D1, . . . , Dt}; (ii) detect all unseen objects as “unknown” without explicit unknown supervision; (iii) incrementally learn a subset of unknowns as knowns when their labels are revealed in later tasks
(iv) achieve these objectives in an exemplar-free manner without storing any previous data



3 modulles intruduis
![[EW-DETR-figure3.webp]]
# Incremental LoRA adapters
## 1. Concept Général : Incremental LoRA
Pour éviter l'oubli catastrophique (catastrophic forgetting) lors de l'apprentissage de nouvelles tâches, le modèle attache des **Incremental Low-Rank Adaptation ([[LoRA]]) adapters** aux FFN de l'encodeur et du décodeur du Transformer.
- **Avantage :** Fournit une mémoire compacte des tâches passées.
- **Bonus :** Ne nécessite de stocker _aucune_ donnée des tâches précédentes.
## 2. Architecture des Poids (Low-Rank Adaptation)

Pour chaque couche ciblée à la tâche $t$, les poids de base du modèle initial ($W_0$) sont **gelés**. Le modèle utilise deux types d'adaptateurs en parallèle :
1. **Aggregate LoRA Adapter ($\Delta W^{t-1}_{agg}$)** : Un buffer _non-entraînable_ (pendant la tâche $t$) qui accumule et stocke les connaissances de toutes les tâches précédentes.
2. **Task-Specific LoRA Adapter ($\Delta W^{t}_{task}$)** : Les paramètres _entraînables_ dédiés uniquement à la tâche actuelle pour capturer les spécificités des nouvelles classes. Il est réinitialisé à chaque nouvelle tâche.
Pour rappel, chacune de ces deux matrices sont constituées de deux matrices de taille plus petites. voir notes [[LoRA]] pour plus d'informations 
## 3. Gestion du déséquilibre des données (Data Imbalance)
Lorsqu'on passe à une nouvelle tâche, on doit fusionner les nouvelles connaissances avec les anciennes. Mais si la nouvelle tâche contient énormément de données par rapport au passé, elle risque d'écraser la mémoire.
pour résoudre Le problème, l'article propose d'utiliser data-aware merging coefficient $β_t$, limiter par $β_{min}$ and $β_{max}$, et pondéré par le nombre d'images des tâches actuelles et précédentes. 
$$\beta_t = \left\{ \begin{array}{ll} 1, & t = 1, \\[4pt] \beta_{\max} - (\beta_{\max} - \beta_{\min}) \frac{N_t}{N_{1:t-1}}, & t \geq 2. \end{array} \right.$$
- $N_t$ = Nombre d'images dans la tâche actuelle.
- $N_{1:t-1}$ = Nombre total d'images vues dans **toutes** les tâches précédentes.
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


# entropy Aware Unknown Mixing module
both objectness and classification uncertainty to modulate the final class scores in a balanced manner.