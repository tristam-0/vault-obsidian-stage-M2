https://arxiv.org/pdf/2412.12830

![[Differential Alignment for Domain Adaptive Object Detection-figure_2.webp]]

## Prediction-Discrepancy Feedback Alignment
"to automatically recog-
nize instances with rich domain-specific information, "

musure les diférance étudident ensigion
![[Differential Alignment for Domain Adaptive Object Detection-figure_3.webp]]la figure 3 monque bien que la ou il y a des brouller -> plus des décale de domener -> plus écra édutiend enségnient 

(ou est crée l'écare)

## Uncertain-based Foreground-Oriented Alignment
on a pas de donner sur le donner CIBLE -> UTILISE DU MODELE TEACH POUR CRÉE DES LABLE

aprédite des de l'union des diférance boite crée un masque des zone improtante 
note : comme en DETR avec le hungraian on a une logique 1 objet une détection on peux pas utiliser d'union pour touver les zone importante


**Lien :** [arXiv:2412.12830](https://arxiv.org/pdf/2412.12830) **Tags :** #DomainAdaptation #ObjectDetection #DETR #TeacherStudent #DeepLearning

## 🎯Objectif et Idée Centrale de l'Article

L'objectif de l'article est d'améliorer l'adaptation de domaine (Domain Adaptation) en identifiant et en alignant spécifiquement les zones de l'image qui contiennent le plus d'informations spécifiques au domaine (celles qui subissent le plus de "domain shift").

L'approche repose sur deux concepts fondamentaux :

## 2. PDFA (Prediction-Discrepancy Feedback Alignment)
- **L'idée :** Mesurer la différence (discrepancy) entre les prédictions générées par le modèle **Teacher** et le modèle **Student**.
- **L'observation clé (Figure 3) :** Là où se trouvent les perturbations liées au domaine (ex: brouillard, flou), l'écart de prédiction entre le Teacher et le Student est beaucoup plus grand.
![[Differential Alignment for Domain Adaptive Object Detection-figure_3.webp]]
- **Conclusion :** Cet écart permet d'identifier précisément les régions où se crée le décalage de domaine. Le modèle "Student" peut ainsi être guidé pour concentrer son apprentissage et son alignement sur ces zones difficiles.
### Équation
- **Équation 1 : Écart de prédiction (Prediction Discrepancy)**
    On calcule la différence carrée entre les prédictions du Teacher ($P_T$) et du Student ($P_S$) sur les instances de la cible :    $$P_{div} = \text{Square}(P_T - P_S)$$     
     _Signification :_ Plus $P_{div}$ est grand sur une région, plus le Student est déstabilisé par le changement de domaine (ex: zone avec du brouillard, éclairage difficile).
- **Équation 2 : Poids d'alignement de l'instance ($w_{ins}$)**
    On dérive un poids $w_{ins} \in [0, 1]$ à partir de $P_{div}$ pour chaque instance : (besoinde plus d'info d'ou vien w_{ins} est porquoi ?expluqe equoi 3)
$$\tilde{w}_{ins} = \text{Normalize}(w_{ins})$$
- **Équation 3 : Perte d'alignement au niveau instance ($L_{ins}^{adv}$)**
    
    Au lieu de donner le même poids à toutes les instances, la perte adversariale est pondérée par $\tilde{w}_{ins}$ :
    besoin d'explique équoi 5 en plus 
    $$L_{ins}^{adv} = \Vert{}\tilde{w}_{ins} \odot f_{ins}\Vert{}_1$$
    
    - _Où $f_{ins}$_ est la perte adversariale brute donnée par le discriminateur d'instances.
        
    - _Signification :_ Les instances avec un fort écart Teacher/Student reçoivent une forte pénalité d'alignement, forçant le réseau à prioriser leur adaptation.
### 2. UFOA (Uncertainty-based Foreground-Oriented Alignment)

- **Le contexte :** On ne possède pas de données annotées sur le domaine CIBLE. Le modèle Teacher est donc utilisé pour générer des pseudo-labels.
    
- **La méthode de l'article :** Pour éviter de s'aligner sur le fond (background), l'article propose de faire l'union des différences entre les boîtes prédites pour générer un **masque des zones importantes (foreground)**. Cela force le modèle à se concentrer sur les objets.
### equoition 
- **Équation 4 : Masque Premier Plan vs Arrière-Plan**
    
    À partir des pseudo-boîtes du Teacher, on génère un masque binaire $M$ (1 sur les objets, 0 sur le fond) pour séparer les features de l'image globale $F_{img}$ entre le foreground et le backgonde :
imrpotant ici     F_{img} est une des fitu map du cnn backbone 
note : donc si on le fais pour le transformeur il va faloir le frans a=ausiis dans le back bonne
    $$F_{fg}^{img} = M \odot F_{img} \quad \text{et} \quad F_{bg}^{img} = (1 - M) \odot F_{img}$$
    
- **Équation 5 : Alignement d'image pondéré ($L_{img}^{adv}$)**
    
    $$L_{img}^{adv} = \gamma L_{fg}^{adv} + (1 - \gamma) L_{bg}^{adv}$$
    prsion d'exliquer les 2 quoition L_{fg}^{adv} pour elle sont comme elle sont
    - _Où $L_{fg}^{adv}$ et $L_{bg}^{adv}$_ sont les pertes adversariales calculées séparément par le discriminateur d'image sur le premier plan et le fond.
        
    - _Signification :_ Le paramètre $\gamma$ permet d'accorder plus d'importance à l'alignement des objets (foreground) qu'à celui du décor (background).

## 🚀 Ma Note : Comment adapter cette idée à une architecture DETR ?

### ❌ Le problème avec DETR et l'UFOA

La méthode de masque par union de boîtes (UFOA) n'est pas applicable directement sur une architecture de type **DETR**.

- **Pourquoi ?** DETR utilise le **Hungarian Matching** (assignation bipartite), qui impose une logique stricte de **1 query = 1 objet**. Contrairement aux architectures classiques (ex: RPN), on ne peut pas générer une forte densité de boîtes et utiliser leur union pour isoler et cartographier les zones importantes.
    

### ✅ La solution : S'inspirer de EW-DETR

Puisque le masque d'union est inutilisable, on peut exploiter les propriétés des _object queries_ elles-mêmes.

- **L'apport de EW-DETR :** Il a été démontré dans _EW-DETR (Evolving World Object Detection)_ que les objets inconnus ou subissant un fort décalage de domaine possèdent **une norme (feature norm) beaucoup plus importante**.
    
- **Stratégie d'adaptation :** Au lieu de créer un masque géométrique à partir des boîtes, il est possible d'utiliser la **norme des features des queries** de DETR. Les requêtes (queries) présentant une norme exceptionnellement élevée peuvent agir comme indicateur des zones de "domain shift" (zones importantes / incertaines), remplaçant ainsi efficacement l'idée du masque de l'article original tout en respectant l'architecture DETR.