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


Calibrating the unknown class in EWOD is challenging because unknown instances are never directly labelled, and the detector is trained only on the current known classes. As a result, unknown evidence tends to be either underconfident or spuriously absorbed into nearby known classes by the softmax, especially under domain shift
(ici j'ai un proplble du softmax en luimeme ou plus généralement de l'attention ?)

EUMix combines these two signals into: (1) an objectness-driven unknown probability that is high when the detector believes there is an object but all known classes are uncertain, and (2) a classifier-driven unknown probability derived from the learned unknown logit.  These two estimates are blended through a single learnable mixing weight α ∈ (0, 1),  punk  final,i = α punk  cls,i + (1 − α) punk  obj,i, (11)


prople comme on fais pour dans l'entrénement ne se marche pas sur les piere ?

 dans le cas d'un classe unconu on a pas d'anotation donc on va normalement de macher avec avec le banckgond donc envoiller les point pas dans la bone direction

A. EUMix Mathmematical Formulation  For completeness, we detail the formulation of the EntropyAware Unknown Mixing (EUMix) module in this section. For a given query i at task t, let  zcls  i = zknown  i , zunk  i (15)  denote the raw classification logits over the |Kt| known classes and the single unknown class, where zknown  i∈  R|Kt| and zunk  i ∈ R. We write zknown  i,c for the logit of known  class c ∈ Kt. In addition, let zobj  i be the objectness logit produced by the Query-Norm Objectness Adapter for the same query.  We first compute the maximum known-class confidence  pknown,max  i = max  c∈Kt σ zknown  i,c , (16)  where σ(·) denotes the logistic sigmoid. If the model is highly confident in some known class, pknown,max  i is close to 1, whereas ambiguous or out-of-distribution objects yield lower values. We convert this observation into a calibrated gap gi that measures how much probability mass is available for the unknown class:  gi = 1 − pknown,max  i  γ, γ = softplus(θγ), (17)  where θγ is a learned scalar and γ > 0 acts as a temperature on the gap. When γ > 1 the gap is sharpened, so that only strongly uncertain predictions yield a significant gi; when γ < 1 the transition is smoother, which is beneficial if known-class logits are noisy.  We interpret the product of objectness and gi as an objectness-derived unknown probability:  punk  obj,i = σ zobj  i gi, (18)  which is high exactly when the model believes there is an object at the query location but no known class explains it confidently. In parallel, we convert the learned unknown logit into a probability, allowing a learnable bias bobj to compensate for the fact that the unknown logit rarely sees positive supervision:  punk  cls,i = σ zunk  i + bobj . (19)  EUMix combines these two estimates through a learnable mixing coefficient  α = σ(θα), (20)  where θα is a scalar parameter. The final unknown probability is  punk  final,i = α punk  cls,i + (1 − α) punk  obj,i, (21)  which is then converted back to a logit:  zunk  final,i = logit punk  final,i , (22)  where logit denotes the inverse of the logistic sigmoid. The mixing weight α is initialised to favour the classifier and is learned end-to-end; if the classifier becomes reliable on unknowns, α naturally increases, but in early tasks or under strong domain shift the model can lean more heavily on the objectness–gap signal.  The final logits fed into the detection loss are then  zfinal  i = zknown  final,i , zunk  final,i . (23)  All parameters in this module, θγ, θα, θλ, bobj, are trained jointly with the rest of the network using exactly the same detection loss as the base detector, without any explicit supervision on the unknown category. Their role is purely to reshape the logit space so that unknown evidence coming from objectness and classification uncertainty is translated into calibrated unknown scores.

Training Protocol. During training on task Tt, only the current task’s known classes Kt receive supervision through bounding box annotations. Crucially, instances of previously learned classes {K1 ∪ . . . ∪ Kt−1} and truly novel objects remain unlabeled in the training set, mimicking realworld scenarios where exhaustive annotation is impractical. Additionally, certain classes are intentionally withheld from training across all tasks (e.g., truck in Table S1) to serve as consistent unknown objects throughout the evaluation.