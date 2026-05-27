https://arxiv.org/pdf/2404.04452

les domain adaptation methods sont categorise en 4
- feature-level
- instance-level
- model-level
- hybrid approsaches


domain generalization,
- multi-domain learning, 
- meta-learning, 
- regularization techniques
- data augmentation strategies.


domain adaptation seeks to minimize the discrepancy between specific source and target domains,


domain generalization, strives to create a model that remains effective across various unseen domains by utilizing the diversity of multiple source domains during training.

vision transformer utiliser un vision transforme comme feature extractore (multi-modal )





# domain adaptation 

## Feature-Level Adaptation (ViT)
La **Feature-Level Adaptation** (adaptation au niveau des caractéristiques) en UDA (*Unsupervised Domain Adaptation*) vise à aligner les distributions de caractéristiques entre le **domaine source** (données étiquetées) et le **domaine cible** (données non étiquetées). L'objectif est de transformer l'espace latent pour que les features apprises sur la source restent discriminantes et applicables sur la cible, résolvant ainsi le problème de *domain shift*.
### 1. Domain-Oriented Transformer (DOT) [90]
*   **Problématique :** Les méthodes traditionnelles biaisent souvent le classifieur vers les données sources, ce qui détruit la séparabilité des classes dans le domaine cible.
*   **Solution :** DOT aligne les caractéristiques à travers **deux espaces distincts** (un pour chaque domaine).
*   **Mécanisme :**
    *   Utilise des *classification tokens* (tokens CLS) et des classifieurs séparés pour chaque domaine.
    *   Combine un alignement basé sur le contrastif (*contrastive-based alignment*) et des pseudo-labels guidés par la source.
    *   **Résultat :** Préserve la séparabilité spécifique à chaque domaine tout en capturant les features invariantes.
### 2. TRANS-DA [93]
*   **Mécanisme :** Génère des pseudo-labels avec un faible niveau de bruit et réentraîne le modèle en créant de nouvelles images composées de patchs mélangés (source et cible).
*   **Loss :** Intègre une *cross-domain alignment loss* pour faire correspondre les centroïdes des patchs étiquetés (source) et pseudo-étiquetés (cible).

### 3. Domain-Transformer [94]
*   **Mécanisme :** Hybride CNN-Transformer. Introduit un mécanisme d'**attention au niveau du domaine** (*domain-level attention*) de type "plug-and-play" et une régularisation de variété (*manifold regularization*).
*   **Focus :** Met l'accent sur les caractéristiques transférables en garantissant une consistance sémantique locale d'un domaine à l'autre, plutôt que de se focaliser uniquement sur les interactions locales de patchs.
### 4. Spectral UDA (SUDA) [95]
*   **Mécanisme :** Aligne les domaines directement dans l'**espace spectral** (fréquentiel) via un *Spectrum Transformer (ST)*.
*   **Focus :** Apprentissage multi-vues pour capturer des représentations cibles diversifiées et invariantes au domaine. Économe et efficace pour la classification, segmentation et détection.

### 5. Semantic Aware Message Broadcasting (SAMB) [96]
*   **Problématique :** L'utilisation d'un seul token CLS global dans les ViTs est insuffisante pour un alignement de domaine précis.
*   **Solution :** Introduit des **group tokens** dédiés à la diffusion de messages (*message broadcasting*) vers différentes régions sémantiques.
*   **Stratégie :** Entraînement en deux étapes combinant un alignement de caractéristiques par méthode adversariale et du *self-training* par pseudo-labels.
### Cas Particulier : Test-Time Adaptation (TTA)
*L'adaptation se fait en ligne, directement au moment de l'inférence sur le domaine cible.*
*   **DePT (Data-efficient Prompt Tuning) [97] :** Combine des invites visuelles (*visual prompts*) dans le ViT avec un fine-tuning des prompts initialisés par la source. Utilise une banque de mémoire pour le pseudo-étiquetage en ligne et une régularisation hiérarchique auto-supervisée. Très efficace même avec peu de données.
*   **CTTA (Continual TTA) [98] :** Approche légère basée sur des invites au niveau de l'image (*image-level prompts*). Les modifications sont appliquées **sur l'image d'entrée** pour l'adapter au modèle source, sans modifier l'architecture ni les poids du modèle. Cela évite l'accumulation d'erreurs et l'oubli catastrophique.

# Glossaire du papier 

- **DG** = Domain Generalization *(Généralisation de domaine)*
- **DA** = Domain Adaptation *(Adaptation de domaine)*
- **ViT** = Vision Transformer
- **CNN** = Convolutional Neural Network *(Réseau de neurones convolutif)*
- **UDA** = Unsupervised Domain Adaptation *(Adaptation de domaine non supervisée)*
- **MLP** = Multi-Layer Perceptron *(Perceptron multicouche)*
- **MSA** = Multi-Head Self-Attention *(Mécanisme d'auto-attention à têtes multiples)*
- self-supervised learning (SSL)
- Recurrent Neural Networks (RNNs)