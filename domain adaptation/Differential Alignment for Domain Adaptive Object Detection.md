**Lien :** [arXiv:2412.12830](https://arxiv.org/pdf/2412.12830) **Tags :** #DomainAdaptation #ObjectDetection #DETR #TeacherStudent #DeepLearning
![[Differential Alignment for Domain Adaptive Object Detection-figure_2.webp]]
## 🎯Objectif et Idée Centrale de l'Article

L'objectif de l'article est d'améliorer l'adaptation de domaine (Domain Adaptation) en identifiant et en alignant spécifiquement les zones de l'image qui contiennent le plus d'informations spécifiques au domaine (celles qui subissent le plus de "domain shift").

L'approche repose sur deux concepts fondamentaux :

## 2. PDFA (Prediction-Discrepancy Feedback Alignment)
- **L'idée :** Mesurer la différence (discrepancy) entre les prédictions générées par le modèle **Teacher** et le modèle **Student**.
- **L'observation clé (Figure 3) :** Là où se trouvent les perturbations liées au domaine (ex: brouillard, flou), l'écart de prédiction entre le Teacher et le Student est beaucoup plus grand.
![[Differential Alignment for Domain Adaptive Object Detection-figure_3.webp]]
- **Conclusion :** Cet écart permet d'identifier précisément les régions où se crée le décalage de domaine. Le modèle "Student" peut ainsi être guidé pour concentrer son apprentissage et son alignement sur ces zones difficiles.
### Équation du PDFA
-**Équation 1 : Écart de prédiction (Prediction Discrepancy)**
    On calcule la différence carrée entre les prédictions du Teacher ($P_T$) et du Student ($P_S$) sur les instances de la cible :    $$P_{div} = \text{Square}(P_T - P_S)$$     
     _Signification :_ Plus $P_{div}$ est grand sur une région, plus le Student est déstabilisé par le changement de domaine (ex: zone avec du brouillard, éclairage difficile).
#### Équation 2 : Poids brut d'incertitude par instance ($w_{ins}$)
Pour transformer cette carte de divergence globale $P_{div}$ en un score d'incertitude spécifique à une boîte/instance donnée, on calcule la valeur moyenne de la divergence sur l'ensemble des $C$ classes pour l'instance concernée :
$$w_{ins} = \frac{1}{C} \sum_{c=1}^{C} P_{div}(:, c)$$
- **D'où vient $w_{ins}$ et pourquoi ?**
    - **$\frac{1}{C}$ :** On divise par le nombre total de classes $C$ pour obtenir une moyenne.
	- **$\sum_{c=1}^{C}$ :** On fait la somme des écarts de prédiction Teacher-Student de la classe $1$ jusqu'à la classe $C$ pour l'instance en question.
	- **Pourquoi ?** Cela permet de condenser la divergence de classification vectorielle brute en un **score d'incertitude scalaire unique** mesurant à quel point cet objet spécifique subit du _domain shift_.
#### Équation 2b : Normalisation du poids ($\tilde{w}_{ins}$)
Pour pouvoir utiliser ce poids sans déstabiliser l'entraînement, on applique une normalisation Min-Max sur l'ensemble des instances du batch :
$$\tilde{w}_{ins} = \text{Normalize}(w_{ins}) = \frac{w_{ins} - \min(w_{ins})}{\max(w_{ins}) - \min(w_{ins})}$$
- **Pourquoi normaliser ?** Cela garantit que $\tilde{w}_{ins} \in [0, 1]$, évitant ainsi l'explosion des gradients tout en créant une distribution de poids comparables entre les différents objets de l'image.
#### Équation 3 : Perte d'alignement au niveau instance ($L_{ins}^{adv}$)
Au lieu de donner le même poids à toutes les instances, la perte adversariale finale est pondérée par le score d'incertitude $\tilde{w}_{ins}$ :

$$L_{ins}^{adv} = \Vert{}\tilde{w}_{ins} \odot f_{ins}\Vert{}_1$$
- **Comprendre la perte brute $f_{ins}$ :** C'est la perte adversariale classique calculée par le **Discriminateur d'Instances ($D_{ins}$)**. Ce discriminateur est un petit sous-réseau (couches _fully-connected_ / MLP) qui prend en entrée le vecteur de caractéristiques de l'objet ($F_{ins}$). Son but est de deviner si l'objet provient de la Source ou de la Cible ($D_{ins}(F_{ins}) \to 1$ pour la Source, $0$ pour la Cible). Elle se calcule via une entropie croisée binaire :
- $$f_{ins} = - d \cdot \log\left(D_{ins}(F_{ins})\right) - (1 - d) \cdot \log\left(1 - D_{ins}(F_{ins})\right)$$
    _(où $d \in \{0, 1\}$ est le label réel du domaine : 1 pour Source, 0 pour Cible)._
- **Pourquoi faire cela ?** Le but est de forcer le _backbone_ à devenir **"aveugle au domaine"** (_domain-invariant_) : les caractéristiques extraites doivent être identiques, peu importe l'environnement de l'image (ex. beau temps vs brouillard).
- **Rôle du produit terme à terme ($\odot$) :** En multipliant la perte brute $f_{ins}$ par $\tilde{w}_{ins}$, les instances qui affichent un fort écart entre le Teacher et le Student (donc subissant un lourd _domain shift_) reçoivent un poids très élevé lors de la rétropropagation. Le réseau est ainsi mathématiquement contraint de concentrer ses efforts d'alignement sur les objets les plus problématiques.
## 2. UFOA (Uncertainty-based Foreground-Oriented Alignment)

- **Le contexte :** On ne possède pas de données annotées sur le domaine CIBLE. Le modèle Teacher est donc utilisé pour générer des pseudo-labels.
- **La méthode :** Pour éviter que le modèle ne perde du temps à s'aligner sur l'arrière-plan (_background_), l'article propose de générer un **masque des zones importantes (_foreground_)** à partir des boîtes de détection, afin de séparer le traitement des objets et du fond.
![[Differential Alignment for Domain Adaptive Object Detection-1784208254864.webp|700x297]]
### Équations de l'UFOA 
#### Équation 4 : Masque Premier Plan vs Arrière-Plan

À partir des pseudo-boîtes transmises par le Teacher, on construit un masque binaire $M$ ($M=1$ sur les zones d'objets, $M=0$ sur le fond). On l'applique sur la carte de caractéristiques $F_{img}$ :
$$F_{fg}^{img} = M \odot F_{img} \quad \text{et} \quad F_{bg}^{img} = (1 - M) \odot F_{img}$$
> ⚠️ **Note importante sur $F_{img}$ :** > $F_{img}$ est la carte de caractéristiques issues du **Backbone CNN** (ex: ResNet).
> 
> _Réflexion pour un Transformer/DETR :_ Dans un détecteur classique, $F_{img}$ va directement aux têtes de détection. Dans DETR, cette carte issue du backbone est d'abord projetée puis envoyée dans le _Transformer Encoder_. Faire l'alignement à la sortie du backbone pose la question de savoir si le Transformer réussira à exploiter cet alignement
#### Équation 5 : Alignement d'image pondéré ($\mathcal{L}_{img}^{adv}$)
$$\mathcal{L}_{img}^{adv} = \gamma \mathcal{L}_{fg}^{adv} + (1 - \gamma) \mathcal{L}_{bg}^{adv}$$
- **Explication détaillée des fonctions de Loss ($\mathcal{L}_{fg}^{adv}$ et $\mathcal{L}_{bg}^{adv}$) :**
    - $\mathcal{L}_{fg}^{adv}$ est la **loss d'alignement adversariale** calculée par le discriminateur d'images appliquée **uniquement sur le premier plan** ($F_{fg}^{img}$). Le discriminateur cherche à différencier le premier plan Source du premier plan Cible ; cette loss force le _backbone_ à rendre les caractéristiques des objets indépendantes du domaine :    $$\mathcal{L}_{fg}^{adv} = - d \cdot \log(D(F_{fg}^{img})) - (1 - d) \cdot \log(1 - D(F_{fg}^{img}))$$ 
    - $\mathcal{L}_{bg}^{adv}$ est la même loss, mais calculée spécifiquement **sur l'arrière-plan** ($F_{bg}^{img}$).
- **Rôle du paramètre $\gamma$ :** $\gamma \in [0, 1]$ (généralement fixé $> 0.5$) donne intentionnellement un poids supérieur à $\mathcal{L}_{fg}^{adv}$ par rapport à $\mathcal{L}_{bg}^{adv}$. Cela garantit que le modèle privilégie l'alignement visuel des objets (_foreground_) plutôt que celui du décor ambiant (_background_).
## Fonctionnement Global : Architecture Teacher-Student & Entraînement
Maintenant que les modules PDFA et UFOA sont définis, voici comment fonctionne l'architecture globale d'entraînement.
### 🔹 D'où viennent le Teacher et le Student ? Pourquoi sont-ils différents ?
À la base, il s'agit du **même modèle**, mais leurs poids évoluent différemment :
1. **Étape 1 (Pré-entraînement Source A) :** On entraîne un modèle initial de façon classique et totalement supervisée sur le **Domaine Source A** (avec ses annotations exactes $Y_S$).
2. **Étape 2 (Duplication) :** On duplique ce modèle en deux instances identiques au départ 
    - Le **Student** ($\theta_S$)
    - Le **Teacher** ($\theta_T$)
3. **Mise à jour et divergence entre les deux modèles :**
    - **Le Student ($\theta_S$)** est mis à jour à chaque itération par **rétropropagation directe du gradient** sur les pertes supervisées, non-supervisées et adversariales. Il s'adapte rapidement mais ses prédictions sont plus instables et sensibles au bruit.
    - **Le Teacher ($\theta_T$)** ne calcule **aucun gradient**. Ses poids sont mis à jour de manière passive via une **Moyenne Mobile Exponentielle (EMA)** des poids du Student :        $$\theta_T \leftarrow \alpha \theta_T + (1 - \alpha) \theta_S \quad (\text{avec } \alpha \approx 0.9996)$$
    - **Pourquoi cette différence est cruciale :** Le Teacher constitue une version "lissée" et temporelle de l'évolution du Student. Sur une image difficile du domaine Cible, le Student réagit de manière instable, tandis que le Teacher fournit des pseudo-labels et des cartes de confiance plus stables. **C'est précisément l'écart entre ces deux réactions qui produit $P_{div}$ (Équation 1).**
### Flux des Données pendant l'Entraînement
À chaque itération d'entraînement, le mini-batch contient deux flux de données simultanés :
![[Differential Alignment for Domain Adaptive Object Detection-1784205218437.webp]]
1. **Données Source A ($X_S$ avec labels $Y_S$) :**
    - Passées uniquement dans le **Student**.
    - Permet de calculer la perte supervisée classique $L_{sup}$ pour conserver les performances de détection.
2. **Données Cible B ($X_T$ SANS labels) :**
    - Passées dans le **Teacher** $\rightarrow$ génère les pseudo-labels de boîtes et la carte de prédiction $P_T$.
    - Passées dans le **Student** $\rightarrow$ génère les prédictions $P_S$ et les cartes de caractéristiques $F_{img}$.
    - **Calcul des modules :**
        - $P_T$ et $P_S$ permettent de calculer $P_{div}$ $\rightarrow$ alimente le module **PDFA** ($L_{ins}^{adv}$).
        - Les pseudo-boîtes du Teacher créent le masque $M$ sur $F_{img}$ $\rightarrow$ alimente le module **UFOA** ($L_{img}^{adv}$).
    - Le Student s'entraîne aussi de manière non supervisée ($L_{unsup}$) en utilisant les pseudo-labels du Teacher comme vérité terrain.
## Comment adapter cette idée à une architecture DETR ?
### 1. Adaptation du Module UFOA (Foreground-Oriented Alignment)

#### ❌ Le problème avec l'UFOA classique sur DETR
- Dans l'article original, le masque $M$ est construit en faisant l'**union géométrique de milliers de boîtes candidates** générées par un RPN (Faster R-CNN).
- **DETR** utilise un mécanisme d'assignation bipartite (_Hungarian Matching_) avec un nombre fixe et restreint d'_object queries_ (ex: 100 ou 300). Il n'y a pas de forte densité de boîtes superposées pour créer un masque géométrique spatial propre au niveau des pixels.
- De plus, $F_{img}$ dans l'article est extrait directement du **Backbone CNN**. Dans DETR, la carte de caractéristiques passe ensuite dans l'**Encoder Transformer**, mais on ne peux pas utiliser le masque apprée ....

#### ✅ La solution proposée pour DETR

- **Inspiration EW-DETR :** Remplacer le masque spatial binaire $M$ basé sur l'union de boîtes par une sélection de boites basée sur la **norme des features de l'Encoder**. (voir EW-DETR)

## PDFA
### ❌ Le Problème : Incompatibilité entre les cartes 2D (CNN) et les Object Queries (DETR)
1. **Dans l'article (Architectures CNN / Dense) :**
    - Le modèle génère des cartes de prédiction denses $P_S(x,y)$ et $P_T(x,y)$ sur une grille spatiale (ex: $80 \times 80$).
    - Il est très simple de faire la soustraction pixel par pixel : $P_{div} = (P_T - P_S)^2$.
2. **Dans DETR :**
    - DETR n'a pas de grille 2D mais $N$ _Object Queries_ globales (ex: 100 requêtes).
    - **Pas d'alignement direct :** La Query #3 du Teacher ($q_3^T$) peut prédire un objet en haut à gauche, tandis que la Query #3 du Student ($q_3^S$) prédit un objet en bas à droite. On ne peut pas les soustraire directement !

- **La Proposition  :**
    1. Passer l'image cible dans le Teacher ($\rightarrow$ 100 prédictions $P_T$) et dans le Student ($\rightarrow$ 100 prédictions $P_S$).
    2. Utiliser l'**algorithme de Hungarian Matching** pour réassocier chaque boîte du Student $P_S^{\text{match}(i)}$ à la boîte correspondante du Teacher $P_T^i$.
    3. Calculer la divergence d'instance uniquement entre les paires associées :
    $$P_{div}(i) = \text{Square}\left(P_T^i - P_S^{\text{match}(i)}\right)$$
    4. Extraire le poids d'incertitude $\tilde{w}_{ins}(i)$ à partir de cette divergence pour pondérer la loss adversariale appliquée sur les embeddings des queries réassociées.