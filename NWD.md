# A Normalized Gaussian Wasserstein Distance for Tiny Object Detection

https://github.com/jwwangchn/NWD
https://www.sciencedirect.com/science/article/pii/S0924271622001599?dgcid=author
https://arxiv.org/pdf/2206.13996

## Problème à résoudre

key observation is that the Intersection over Union (IoU) metric and its extensions are very sensitive to the location deviation of the tiny objects, which drastically deteriorates the quality of label assignment when used in anchor-based detector
![[NWD-1789981890338.webp]]

> [!danger] Pourquoi IoU est inadapté aux tiny objects
> - Sur la figure, pour une voiture de $5\times8$ pixels, un décalage de **1 pixel** (boîte B) fait chuter l'IoU à $0.54$, et un décalage de **3 pixels** (boîte C) à $0.14$. Pour un objet normal ($200\times320$), le même décalage ne change presque rien ($0.97 \to 0.91$).
> - IoU ne dit rien quand $|P\cap G| = 0$ (aucun recouvrement) ou quand $|P\cap G| = P$ ou $G$ (inclusion) : la valeur sature et le gradient disparaît.
> - En label assignment, un anchor peut donc basculer de positif à négatif à cause d'un seul pixel de décalage. Résultat : très peu d'anchors positifs par objet (en moyenne $0.72$ avec IoU, $0.71$ avec GIoU, $0.19$ avec DIoU/CIoU) et un entraînement sous-supervisé.

> [!info] Flash : c'est quoi un "anchor" ?
> Un **anchor** est une **boîte de référence prédéfinie** (une position + une taille + un ratio d'aspect, c'est-à-dire un rapport largeur/hauteur : $1:1$, $1:2$, $2:1$, etc.) que le détecteur fait glisser sur l'image ou la feature map. Le réseau ne prédit pas les boîtes de zéro : pour chaque anchor, il répond à deux questions.
> 1. **Est-ce un objet ?** (classification objet/fond, ou quelle classe)
> 2. **Comment le décaler / le redimensionner** pour coller à l'objet (régression des offsets).
>
> En **label assignment**, on décide quel anchor "joue" quel objet : un anchor suffisamment proche d'une gt devient **positif** (il apprend à se transformer en cet objet), les autres sont **négatifs** (fond). Le critère de proximité est habituellement l'IoU — c'est précisément ce critère que NWD vient remplacer.
>
> À ne pas confondre avec [[DETR]] : dans DETR il n'y a pas d'anchors, ce rôle est joué par les *object queries*. (Variante : certains détecteurs n'utilisent qu'un simple **anchor point** $(x,y)$, voir [[Anchor DETR]].)

# Normalized Gaussian Wasserstein Distance

## 1. Modéliser la bounding box comme une gaussienne 2D

L'idée : au lieu de voir la boîte comme un rectangle qu'on intersecte, on la voit comme un **ovale / une ellipse**, avec un poids maximal au centre qui décroît vers les bords. C'est particulièrement adapté aux tiny objects, car leur boîte contient souvent plus de background que d'objet.

Pour une boîte horizontale $R = (cx, cy, w, h)$ (centre, largeur, hauteur), l'équation de son ellipse inscrite est :

$$\frac{(x-\mu_x)^2}{\sigma_x^2} + \frac{(y-\mu_y)^2}{\sigma_y^2} = 1$$

où $(\mu_x, \mu_y)$ est le centre de l'ellipse et $\sigma_x, \sigma_y$ les longueurs des demi-axes. Ici :

$$\mu_x = cx, \quad \mu_y = cy, \quad \sigma_x = \frac{w}{2}, \quad \sigma_y = \frac{h}{2}$$

Cette ellipse est exactement une **courbe de niveau** de la densité d'une gaussienne 2D :

$$f(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\intercal \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)}{2\pi|\boldsymbol{\Sigma}|^{1/2}}$$

avec $\mathbf{x} = (x,y)$, $\boldsymbol{\mu}$ le vecteur moyenne et $\boldsymbol{\Sigma}$ la matrice de covariance. C'est vrai lorsque $(\mathbf{x}-\boldsymbol{\mu})^\intercal \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) = 1$.

Donc la boîte horizontale $R = (cx,cy,w,h)$ se modélise en une gaussienne $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ :

$$\boldsymbol{\mu} = \begin{bmatrix} cx \\ cy \end{bmatrix}, \qquad \boldsymbol{\Sigma} = \begin{bmatrix} \frac{w^2}{4} & 0 \\[4pt] 0 & \frac{h^2}{4} \end{bmatrix}$$

La similarité entre deux boîtes $A$ et $B$ devient alors une **distance entre deux distributions**.

## 2. L'équation qui remplace IoU : NWD

On mesure l'écart entre les deux gaussiennes avec la **distance de Wasserstein du 2nd ordre** ($W_2$), issue de la théorie du transport optimal. Pour $\mu_1 = \mathcal{N}(\mathbf{m}_1, \boldsymbol{\Sigma}_1)$ et $\mu_2 = \mathcal{N}(\mathbf{m}_2, \boldsymbol{\Sigma}_2)$ :

$$W_2^2(\mu_1, \mu_2) = \|\mathbf{m}_1 - \mathbf{m}_2\|_2^2 + \left\|\boldsymbol{\Sigma}_1^{1/2} - \boldsymbol{\Sigma}_2^{1/2}\right\|_F^2$$

Pour deux boîtes $A = (cx_a, cy_a, w_a, h_a)$ et $B = (cx_b, cy_b, w_b, h_b)$, comme les covariances sont diagonales, tout se simplifie en une simple distance euclidienne entre des vecteurs de taille 4 :

$$W_2^2(\mathcal{N}_a, \mathcal{N}_b) = (cx_a - cx_b)^2 + (cy_a - cy_b)^2 + \frac{(w_a - w_b)^2}{4} + \frac{(h_a - h_b)^2}{4}$$

Cette distance vit dans $[0, +\infty[$, contrairement à IoU qui est dans $[0,1]$. On la normalise donc via une exponentielle pour retomber sur une valeur comparable à l'IoU :

$$\text{NWD}(\mathcal{N}_a, \mathcal{N}_b) = \exp\left(-\frac{\sqrt{W_2^2(\mathcal{N}_a, \mathcal{N}_b)}}{C}\right)$$

- $C$ = constante liée au dataset (échelle de référence). Dans le papier : $C = 12.8$ pour AI-TOD (taille absolue moyenne des objets) et $C = 35.8$ pour VisDrone.
- $\text{NWD} \in (0, 1]$, et $\text{NWD} = 1$ si et seulement si les deux boîtes sont identiques (comme IoU $= 1$).

> [!info] Lecture intuitive
> Pour deux boîtes de même taille, $\text{NWD} = \exp(-d / C)$ où $d$ est la distance entre les centres : c'est un **noyau gaussien sur la distance des centres**, donc une décroissance douce, sans la "falaise" d'IoU.
> $C$ joue le rôle de température : petit $C$ = pénalité agressive, grand $C$ = tolérance plus large. Le papier montre que le choix de $C$ est robuste autour de la taille moyenne des objets du dataset.

Propriétés clés face à IoU :
1. **Invariance à l'échelle** : les courbes NWD-décalage se superposent quelles que soient les tailles des boîtes (IoU non).
2. **Lisse au décalage** : pas de chute brutale d'un pixel à l'autre.
3. **Toujours défini** : sans recouvrement ou en inclusion, NWD reste strictement positive et continue à mesurer la relation de position (IoU sature à 0 ou 1, voire devient non différentiable).

## 3. Comment déterminer $C$ ?

$C$ n'est pas un seuil arbitraire : c'est l'**échelle de référence** de la distance, exprimée en pixels. Il fixe le "rayon de tolérance" de la métrique.

### Recette du papier

$C$ = **taille absolue moyenne des objets du train set**, où la taille absolue d'une boîte est $AS = \sqrt{w \times h}$ (définition AI-TOD, reprise aussi dans Dot Distance) :

$$C = \frac{1}{N}\sum_{i=1}^{N}\sqrt{w_i \, h_i}$$

| Dataset | Taille absolue moyenne | $C$ utilisé |
|---|---|---|
| AI-TOD | $12.8$ px | $12.8$ |
| AI-TOD-v2 | $12.7 \pm 5.6$ px | $12.7$ (robuste de $8$ à $24$) |
| VisDrone2019 | $35.8 \pm 32.8$ px | $35.8$ |


> [!warning] Pièges d'échelle
> - Si les coordonnées sont normalisées (DETR : boîtes dans $[0,1]$), il faut aussi normaliser $C$ : $C_{\text{norm}} = C_{\text{px}} / \text{taille image}$ (ex. $12.8/800 \approx 0.016$ pour AI-TOD en $800\times800$).
> - Si on change la résolution des images ou le dataset (domain adaptation, multi-scale), il faut **ré-estimer** $C$ : c'est une propriété des données, pas du modèle.
> 
> Ne pas confondre $C$ et un seuil : $C$ règle la *forme* de la décroissance, $\theta_p/\theta_n$ (ou Top-$k$) règlent la *décision* positif/négatif.

## 4. NWD comme fonction de perte

Comme pour IoU/GIoU, on transforme la similarité en perte :

$$\mathcal{L}_{\text{NWD}} = 1 - \text{NWD}(\mathcal{N}_p, \mathcal{N}_g)$$

où $\mathcal{N}_p$ est la gaussienne de la boîte prédite et $\mathcal{N}_g$ celle de la ground truth. Sous cette forme, la loss fournit un gradient **même** quand $|P\cap G| = 0$ ou $|P\cap G| = P$ ou $G$, contrairement à la IoU-Loss.

Dans le papier, NWD remplace IoU à trois endroits : le **label assignment**, le **NMS** (seuil $0.85$ dans le code officiel) et la **loss de régression** (avec `loss_weight=10.0`). Il peut aussi remplacer le terme IoU du Hungarian matcher d'un détecteur type DETR.

## 5. Exemples : quoi est positif, quoi est négatif ?

NWD sert à décider quels anchors deviennent des échantillons **positifs** et lesquels deviennent **négatifs** :

- **Version à seuils** (papier original) : positif si $\text{NWD} > \theta_p$, négatif si $\text{NWD} < \theta_n$ pour toutes les gt, les anchors entre les deux sont ignorés. Même logique que Faster R-CNN avec $(0.7, 0.3)$, mais la valeur NWD remplace l'IoU.
- **Version RKA** (RanKing-based Assigning, version ISPRS 2022) : pour chaque gt, on trie les anchors par score NWD décroissant et on prend le **Top-$k$** comme positifs, le reste en négatif. Plus de seuil à régler, et chaque gt est garanti d'avoir des positifs.
![[NWD-1789984766916.webp]]
Exemple chiffré avec la voiture $5\times8$ de la figure ($C = 12.8$, seuils $\theta_p = 0.7$, $\theta_n = 0.3$) :

| Cas | Décalage centre | IoU | NWD | Label via IoU | Label via NWD |
|---|---|---|---|---|---|
| Anchor identique | $(0,0)$ | $1.00$ | $1.00$ | positif | positif |
| Prédiction à 1 px | $(1,1)$ | $0.54$ | $0.90$ | ignoré | positif |
| Prédiction à 3 px | $(3,3)$ | $0.14$ | $0.72$ | négatif | positif |
| Anchor voisin sans recouvrement | $(6,0)$ | $0.00$ | $0.63$ | négatif | ignoré |
| Anchor éloigné | $(20,20)$ | $0.00$ | $0.11$ | négatif | négatif |

> [!success] Ce que montre l'exemple
> - Dès 3 pixels de décalage, IoU envoie l'anchor en négatif alors que NWD le garde positif : l'objet reçoit enfin de la supervision.
> - Sans recouvrement, IoU $= 0$ pour *tous* les anchors et ne permet pas de les classer ; NWD garde un ordre ($0.63 > 0.11$) et permet au RKA de choisir le "moins mauvais" comme positif.
> - Statistique du papier (mêmes seuils par défaut) : nombre moyen d'anchors positifs par gt = $0.72$ (IoU), $0.71$ (GIoU), $0.19$ (DIoU/CIoU), $\mathbf{1.05}$ (NWD).

## 6. Comparaison avec le GIoU utilisé dans DETR

Dans [[DETR]], le terme de boîte est :

$$\mathcal{L}_{\text{box}}(b_i, \hat{b}) = \lambda_{\text{iou}}\big(1 - \text{GIoU}(b_i, \hat{b})\big) + \lambda_{L1}\|b_i - \hat{b}\|_1$$

$$\text{GIoU}(b_i, \hat{b}) = \text{IoU}(b_i, \hat{b}) - \frac{|B(b_i, \hat{b}) \setminus (b_i \cup \hat{b})|}{|B(b_i, \hat{b})|}$$

où $B$ est la plus petite boîte englobante. GIoU corrige le "zéro gradient" d'IoU en pénalisant la boîte englobante, mais il **reste construit sur l'aire d'intersection** : il hérite donc de la sensibilité au pixel près pour les tiny objects. De plus, quand une boîte contient entièrement l'autre, la pénalité s'annule et GIoU redevient exactement IoU.

| Critère | GIoU (DETR) | NWD |
|---|---|---|
| Principe | aire de recouvrement + pénalité de la boîte englobante | distance entre distributions gaussiennes |
| Boîtes disjointes | défini (pénalité), mais toujours basé sur l'IoU | décroît progressivement, indépendant de l'intersection |
| Inclusion | dégénère en IoU (pénalité nulle) | continue de mesurer (centre + taille) |
| Sensibilité tiny | forte (dérivée de l'aire / du pixel) | lisse (exponentielle) |
| Échelle | non invariant, dépend de la boîte englobante | invariant d'échelle, $C$ fixe l'échelle de référence |
| Plage de valeurs | $[-1, 1]$ → loss $\in [0, 2]$ | $(0, 1]$ → loss $\in [0, 1[$ |
| Hyperparamètre | aucun | $C$ (taille moyenne du dataset) |

Pour l'intégrer dans DETR, on remplace simplement le terme IoU/GIoU du coût de matching et de la loss :

$$\mathcal{L}_{\text{box}} = \lambda_{\text{NWD}}\big(1 - \text{NWD}(\mathcal{N}_b, \mathcal{N}_{\hat{b}})\big) + \lambda_{L1}\|b - \hat{b}\|_1$$

> [!warning] Points d'attention avant de remplacer GIoU par NWD dans DETR
> 1. **DETR n'utilise pas IoU seul** : le terme $L_1$ fournit déjà un gradient en cas de non-recouvrement. L'apport de NWD se joue surtout sur la **stabilité du matching** et la sensibilité au décalage pour les petits objets.
> 2. **Échelle de $C$** : DETR normalise les boîtes dans $[0,1]$, alors que $C = 12.8$ est en pixels. Il faut exprimer $C$ dans la même échelle (pour AI-TOD : $12.8/800 \approx 0.016$ en coordonnées relatives), sinon le rayon d'action de l'exponentielle change complètement.
> 3. **Différentiabilité** : $\sqrt{W_2^2}$ n'est pas dérivable quand deux boîtes coïncident parfaitement → on ajoute un petit $\epsilon$ (c'est ce que fait le code officiel).
> 4. **Coût** : négligeable, c'est une distance euclidienne sur 4 valeurs.
