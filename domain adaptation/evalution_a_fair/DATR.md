DATR: Unsupervised Domain Adaptive Detection

Transformer with Dataset-Level Adaptation and

Prototypical Alignment

#### A. Alignement par prototypes par classe (_Class-wise Prototypes Alignment - CPA_)

Plutôt que d'aligner aveuglement toutes les caractéristiques, le module CPA prend en compte la catégorie des objets (_class-aware_). Il crée des **prototypes** (représentations moyennes) pour chaque classe d'objets et s'assure qu'une classe du domaine source (ex: "voiture") soit alignée spécifiquement avec la même classe dans le domaine cible.
![[DATR-1784813527532.webp|694]]
#### 2. L'extraction des prototypes par classe

Un **prototype** est la représentation vectorielle moyenne de tous les objets appartenant à une même classe au sein d'un lot d'images (_batch_) :

- DATR réalise cette agrégation via un calcul matriciel direct et très rapide.
    
- Il génère ainsi, pour un même _batch_, un prototype pour la classe "voiture" source et un prototype pour la classe "voiture" cible.
    

#### 3. L'alignement contradictoire (_Adversarial Alignment_)

Une fois les prototypes extraits par catégorie :

- Un **discriminateur de domaine** essaie de deviner si le prototype d'une classe provient du domaine source ou du domaine cible.
    
- Le détecteur est entraîné de manière contradictoire (_adversarial learning_) pour "tromper" le discriminateur. Cela force les représentations d'une même classe à devenir indifférenciables entre les deux domaines.

#### B. Schéma d'alignement à l'échelle du jeu de données (_Dataset-level Alignment Scheme - DAS_)

Pour dépasser la vision restreinte des petits lots d'images (_batch-level_), le schéma DAS conserve une mémoire des prototypes sur l'ensemble du jeu de données. Grâce à un **apprentissage contrastif** (_contrastive learning_) :

- **Il rapproche** les représentations d'une même classe entre les deux domaines.
    
- **Il éloigne** les représentations de classes différentes pour éviter les confusions (ex: éviter de confondre un piéton et un poteau).
![[DATR-1784813924834.webp]]#### 1. La mémoire statistique globale

Au lieu de jeter les caractéristiques extraites à chaque batch, DAS accumule les représentations vectorielles moyennes des objets au fil de l'entraînement :

- Le modèle conserve en mémoire l'historique des prototypes calculés.
    
- En calculant la moyenne statistique stricte de ces éléments accumulés, DAS construit de véritables **prototypes globaux** représentatifs de tout le jeu de données pour chaque classe, côté source et côté cible.
    

#### 2. L'apprentissage contrastif (_Contrastive Learning_)

Sur ces prototypes globaux, DAS applique une perte contrastive (_contrastive loss_) basée sur deux forces complémentaires :

- **Attraction (Paires positives) :** Il attire le prototype global de la classe "voiture" du domaine source vers le prototype global "voiture" du domaine cible.
    
- **Répulsion (Paires négatives) :** Il repousse activement les prototypes de **classes différentes** (ex: il éloigne le prototype "voiture" du prototype "piéton"), peu importe leur domaine d'origine.
    

#### 3. La séparabilité inter-classes (_Inter-class Distinguishability_)

Cette répulsion résout un piège majeur du _domain adaptation_ : le chevauchement des catégories. En forçant une frontière distante entre des classes distinctes, DAS empêche le réseau de confondre deux objets physiquement ou sémantiquement proches (comme un panneau et un poteau) lorsqu'il passe au domaine cible.



$$L_{total} = L_{det} + \lambda_1 L_{adv} + \lambda_2 L_{contrast}$$

Voici comment chacune de ces pertes fonctionne et comment elle s'articule directement avec le module correspondant.

### 1. La perte contradictoire ($L_{adv}$) $\rightarrow$ Liée au module CPA

Le module **CPA** (_Class-wise Prototypes Alignment_) extrait la représentation moyenne (prototype) de chaque classe présente dans un mini-lot (_batch_).

- **Type de perte :** Entraînement contradictoire (_Adversarial Loss_ / Entropie croisée binaire).
    
- **Comment elle fonctionne :**
    
    Un **discriminateur de domaine** prend en entrée le prototype d'une classe (par exemple le prototype "voiture") et tente de deviner s'il provient de l'image source ou cible.
    
- **Objectif :**
    
    Le réseau de détection cherche à **tromper** ce discriminateur en alignant au maximum la distribution des caractéristiques de chaque classe.
    
- **Spécificité :** Un système de masquage annule la perte pour les catégories d'objets qui ne sont pas présentes dans le _batch_ courant.
    

### 2. La perte contrastive ($L_{contrast}$) $\rightarrow$ Liée au module DAS

Le module **DAS** (_Dataset-level Alignment Scheme_) conserve une mémoire statistique pour construire la vision globale de chaque classe sur l'ensemble du jeu de données.

- **Type de perte :** Apprentissage contrastif supervisé (_Contrastive Loss_ de type InfoNCE).
    
- **Comment elle fonctionne :**
    
    Cette perte agit directement sur les prototypes globaux stockés en mémoire.
    
- **Objectif (double action) :**
    
    1. **Attraction intra-classe :** Elle minimise la distance entre le prototype global "source" et le prototype global "cible" d'une même catégorie.
        
    2. **Répulsion inter-classes :** Elle maximise la distance entre les prototypes de classes différentes (ex: séparer la classe "voiture" de la classe "piéton"), qu'ils viennent de la source ou de la cible.
        

### 3. La perte de détection ($L_{det}$) $\rightarrow$ Liée au détecteur de base & Mean-Teacher

Cette perte ne sert pas à l'alignement de domaine mais garantit que le modèle continue d'apprendre à détecter des objets avec précision.

- **Type de perte :** Perte classique des architectures DETR.
    
    - **Classification :** Perte de classification (_Focal Loss_).
        
    - **Localisation :** Perte de régression des boîtes englobantes (Combinaison $L_1$ et $GIoU$).
        
- **Comment elle fonctionne :**
    
    - Sur le domaine **source**, elle utilise les vraies annotations.
        
    - Sur le domaine **cible**, elle utilise les **pseudo-étiqu**