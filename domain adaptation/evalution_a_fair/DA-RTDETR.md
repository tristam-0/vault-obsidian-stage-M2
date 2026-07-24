# DA-RTDETR: domain-adaptive RT-DETR with feature fusion and category-level constraints

#### A. Discriminateur amélioré par fusion (_FFED — Feature Fusion Enhanced Discriminator_)

- **Objectif :** Aligner les caractéristiques visuelles extraites par le _backbone_ et l'encodeur.
- **Fonctionnement :** Il combine deux processus :
    - **HFA (_Hierarchical Feature Adaptation_) :** adapte les caractéristiques échelle par échelle.
	    - - **Le problème :** Un détecteur comme RT-DETR extrait des caractéristiques à plusieurs niveaux de résolution ($S_3, S_4, S_5$). Le _domain shift_ (l'écart d'apparence entre la source et la cible) ne se manifeste pas de la même façon selon la taille des objets :
		    - Le brouillard ou la nuit masque très fort les petits détails au loin (niveau $S_3$).
		    - En revanche, la forme globale des grands objets proches (niveau $S_5$) reste souvent identifiable.
		- **Le fonctionnement de HFA :** Au lieu d'appliquer une seule règle d'adaptation globale sur toute l'image, HFA applique une adaptation **hiérarchique** (découpée échelle par échelle).
		- **L'avantage :** Chaque niveau de résolution apprend à corriger ses propres déformations dues au changement d'environnement, sans brouiller les informations des autres niveaux.
    - **FF (_Feature Fusion_) :** fusionne les informations spatiales (bas niveau) et sémantiques (haut niveau).
	    - - **Le problème :** Si le discriminateur ne regarde que les niveaux séparés :
    
	    - Sur les couches basses ($S_3$), il risque de bloquer sur du simple bruit visuel ou de la texture (ex: les gouttes de pluie).
        
	    - Sur les couches hautes ($S_5$), il sait _ce que c'est_ (une voiture), mais perd la précision des contours exacts de la boîte englobante.
        
	- **Le fonctionnement de FF :** FF prend les caractéristiques ajustées par HFA et **fusionne les informations spatiales** (bords, contours fins de $S_3$) avec les **informations sémantiques** (identité de l'objet de $S_5$).
    
	- **L'avantage :** Le discriminateur évalue une représentation complète. Il ne demande pas juste _"Est-ce que cette texture vient du brouillard ?"_, mais _"Est-ce que l'association [forme précise + catégorie d'objet] correspond au domaine cible ?"_.
    ![[DA-RTDETR-1784880083687.webp|700x485]]![[DA-RTDETR-1784879146403.webp]]
- **Effet :** Force le réseau à extraire des représentations visuelles invariantes d'un domaine à un autre.
![[DA-RTDETR-1784878585135.webp]]
![[DA-RTDETR-1784879146403.webp]]
### 1. AIFI (_Attention-based Intra-scale Feature Interaction_)

- **Rôle :** Traiter les relations complexes au sein d'une **seule niveau d'échelle**.
    
- **Comment il fonctionne :** Il applique un mécanisme d'**auto-attention (Self-Attention)** uniquement sur la carte de caractéristiques la plus profonde ($S_5$), extraite par le backbone.
    
- **Pourquoi uniquement sur $S_5$ ?**
    
    - $S_5$ contient les informations sémantiques les plus riches (compréhension globale des objets).
        
    - En raison de sa résolution spatiale réduite ($1/32$ de l'image originale), le calcul de l'attention y est très rapide.
        
    - Appliquer l'attention sur des échelles à plus haute résolution ($S_3, S_4$) alourdirait inutilement les calculs.
        

### 2. CCFF (_CNN-based Cross-scale Feature Fusion_)

- **Rôle :** Fusionner les informations visuelles à travers les **différentes échelles** ($S_3, S_4$ et $S_5$).
    
- **Comment il fonctionne :** Plutôt que d'utiliser un Transformer complexe pour croiser les niveaux, il utilise des **blocs convolutifs (CNN)** rapides (comme les blocs RepC3) configurés selon une structure de pyramide de caractéristiques.
    
- **Pourquoi utiliser des convolutions ?**
    
    - La fusion multi-échelle sert à combiner des détails géométriques fins (bas niveau) avec du contexte sémantique (haut niveau).
        
    - Les CNN excellent dans ce traitement spatial local tout en étant nettement plus rapides et légers en mémoire que les Transformers.


#### B. Contrainte non linéaire sur les jetons de catégorie (_NFC — Nonlinear Feature Constraint_)
- **Objectif :** Garantir un alignement précis **par catégorie d'objets** dans le décodeur.
- **Fonctionnement :** NFC étend la **perte Deep CORAL** (_Correlation Alignment_) aux jetons (_tokens_) de catégories du décodeur. Elle réduit l'écart entre les statistiques de second ordre (les matrices de covariance) du domaine source et du domaine cible.
- **Effet :** Empêche la confusion entre des classes d'objets proches lorsque l'environnement change.
### 1. Quelle est la fonction de perte utilisée ?

C'est la **perte CORAL** (_Correlation Alignment_). Elle ne compare pas directement chaque objet un par un (ce qui est impossible sans annotations sur le domaine cible), mais compare la **forme de la distribution des données** entre la source et la cible.

### 2. Pourquoi utiliser les statistiques de second ordre ?

- **1er ordre (la moyenne) :** Aligne uniquement le "centre" des représentations. C'est souvent insuffisant car deux classes d'objets différentes (ex: un _camion_ et un _bus_) peuvent avoir des moyennes très proches.
    
- **2e ordre (la matrice de covariance) :** Décrit la **forme**, la **dispersion** et les **relations** entre les caractéristiques. Réduire l'écart de covariance garantit que la structure interne de la catégorie est préservée d'un domaine à l'autre.
    

### 3. Comment est-elle calculée ? (Étape par étape)

Pour chaque classe $k$ (ex: _piéton_, _voiture_, _panneau_), le décodeur extrait les jetons d'objets pour le domaine source ($F_S$) et le domaine cible ($F_T$).

#### Étape 1 : Calcul des matrices de covariance ($C_S$ et $C_T$)

On calcule la matrice de covariance $C_S$ pour la source et $C_T$ pour la cible :

$$C_S = \frac{1}{n_S - 1} \left( F_S^\top F_S - \frac{1}{n_S} (\mathbf{1}^\top F_S)^\top (\mathbf{1}^\top F_S) \right)$$

$$C_T = \frac{1}{n_T - 1} \left( F_T^\top F_T - \frac{1}{n_T} (\mathbf{1}^\top F_T)^\top (\mathbf{1}^\top F_T) \right)$$

- $n_S$ et $n_T$ : nombre d'échantillons de jetons pour cette catégorie dans la source et la cible.
    
- $F_S$ et $F_T$ : matrices contenant les vecteurs de caractéristiques des jetons.
    

#### Étape 2 : Calcul de la perte CORAL ($L_{\text{CORAL}}$)

On mesure la distance entre ces deux matrices de covariance grâce à la **norme de Frobenius au carré** :

$$L_{\text{CORAL}} = \frac{1}{4d^2} \Vert{} C_S - C_T \Vert{}_F^2$$

- $d$ : dimension des jetons de caractéristiques (ex: 256 dans RT-DETR).
    
- $\Vert{} \cdot \Vert{}_F^2$ : somme des carrés des différences pour chaque élément de la matrice.
    

### 4. Pourquoi parle-t-on de contrainte "Non Linéaire" (NFC) ?

Les jetons du décodeur subissent des transformations **fortement non linéaires** dans le décodeur Transformer (attention croisée multi-têtes + réseaux _Feed-Forward_ / FFN).

Appliquer la perte CORAL **directement sur ces jetons en sortie du décodeur** (au lieu de le faire uniquement sur le backbone) impose une contrainte dans cet espace de représentation non linéaire.

> **En résumé :** FFED (le discriminateur) aligne l'apparence globale au niveau des convolutions, tandis que NFC ($L_{\text{CORAL}}$) aligne la sémantique fine de chaque classe d'objet au niveau du décodeur.