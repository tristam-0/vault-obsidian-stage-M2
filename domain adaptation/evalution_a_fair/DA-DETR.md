https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DA-DETR_Domain_Adaptive_Detection_Transformer_With_Information_Fusion_CVPR_2023_paper.**pdf

rolle diférance CNN / Transformer

- **Le backbone CNN** extrait des caractéristiques locales et spatiales (bords, contours, localisation précise).
    
- **La tête Transformer** capture les relations globales entre les pixels et l'information sémantique de haut niveau.
    

Plutôt que d'aligner séparément les caractéristiques du CNN et du Transformer entre le domaine source et le domaine cible, **DA-DETR** propose de **fusionner ces deux types d'informations** pour former une représentation unifiée avant l'alignement.

### Les composants clés de DA-DETR

L'innovation centrale de l'article est un module nommé **CTBlender** (_CNN-Transformer Blender_), qui associe deux mécanismes de fusion :

1. **Split-Merge Fusion (SMF) :** Les caractéristiques du Transformer viennent moduler celles du CNN. Le CNN divise ses caractéristiques en groupes sémantiques guidés par le Transformer, puis les fusionne avec un mélange de canaux (_channel shuffling_) pour faire communiquer les informations spatiales et sémantiques.
![[DA-DETR-1784794352609.webp]]
2. **Scale Aggregation Fusion (SAF) :** Il combine les caractéristiques fusionnées à travers plusieurs échelles d'image (_multi-scale_), garantissant que les détails fins de localisation et le contexte global soient préservés à toutes les résolutions.
![[DA-DETR-1784794365082.webp]]
3. **Discriminateur unique :** La représentation riche produite par le CTBlender est envoyée à un unique discriminateur de domaine, qui utilise un apprentissage contradictoire (_adversarial learning_) pour rendre les caractéristiques invariantes au changement de domaine.
![[DA-DETR-1784794194258.webp]]