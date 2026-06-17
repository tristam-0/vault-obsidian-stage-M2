# Configuration & Bonnes Pratiques - Grid'5000 (Projet DETR)

## 1. Connexion et Navigation

### Étape 1 : Accès à la passerelle principale + connexion au site
```bash
ssh <identifiant>@access.grid5000.fr
ssh <site>
```
 liste des site :
 Bordeaux | Grenoble | Lille | Louvain | Luxembourg | Lyon | Nancy | Nantes | Rennes | Sophia | Strasbourg | Toulouse
### Étape 2 : importation du code
crée un dossier pour le projet puis clone le git
```bash
mkdir ~/detr-project
cd ~/detr-project
git clone https://github.com/ton-compte/ton-repo-detr.git .
```

> `rsync`  pourais étre utilise pour copier les configuration ?
### Étape 3 : Configuration du .venv et installation/modification
1. Création de l'environnement virtuel Python + activation :
```bash
python3 -m venv .venv 
source .venv/bin/activate
```
2. Configuration du cache de PIP (indispensable pour éviter de stocker les packages lourds comme PyTorch en double dans le Home) :
```bash
mkdir -p /tmp/$USER/pip_cache
export PIP_CACHE_DIR=/tmp/$USER/pip_cache
```
3. Installation des dépendances du projet :
> warting : il peux i avoir des probléme en fonction des carte sur les serveur posible besoibn de plusieur requirements en fonction de la cible
```bash
pip install -r requirements.txt 
```

### 4. Architecture des Scripts 

Pour que le projet soit propre, modulable on utilise **trois scripts distincts**. 

- **A. Le Chef d'Orchestre (`start_run`)**
    - **Rôle :** C'est le boss. C'est lui qu'on envoie au serveur (à OAR). Il coordonne les autres scripts.
    - **Fonctionnement :** Il prend en argument le nom du script d'entraînement qu'on veut lancer. Il charge l'environnement (`modules.sh` et `.venv`),
    - _Avantage :_ Si demain on crée un 
- **B. Le Gestionnaire de Dataset (ex :`prepare_dataset_coco.sh`)**
    - **Rôle :** gére la préparation du dataset actuellement télécharger au déput de run
- **C. Le Script d'Entraînement (`ex_train_model.sh`)**
    - **Rôle :** Lancer le code Python (PyTorch/DETR).
    - **Fonctionnement :** Il commence par utilise B pour géré le dataset puis récupérer le chemin exact du dataset préparé par B. Ensuite, il lance la commande `python detr/main.py` avec tous les bons hyperparamètres (epochs, batch size, etc.).

### 5. Lancement de l'Entraînement sur Grid'5000

Une fois les scripts prêts et rendus exécutables (`chmod +x start_run scrip/*`), on soumet le travail au gestionnaire de ressources (OAR).

**exemple commande run**
```Bash
oarsub -S "./start_run scrip/ex_train_model.sh" -t night -l /gpu=1,walltime=12:50:00 
```
**Que fait cette commande ?**
1. `"./start_run scrip/ex_train_model.sh"` : La commande que le serveur exécutera. Notre chef d'orchestre (`start_run`) est appelé, et on lui donne en argument la partition de code exacte à jouer (`ex_train_model.sh`).
2. `-l /gpu=1,walltime=14:00:00` : **La configuration matérielle**. 
3. `-t night` : emperche de franchir la ligne 9h - 19h par accident 

**Où lire les résultats (Logs) ?** Une fois le job lancé, OAR va créer deux fichiers directement dans le dossier où tu as tapé la commande (souvent la racine du projet) :
- `OAR.<job_id>.stdout` : Le fichier de sortie standard (tu y verras les `echo` de tes scripts, l'avancement des epochs, etc.).
- `OAR.<job_id>.stderr` : Le fichier d'erreurs (utile pour déboguer si le script plante avant la fin).

## transfer de fichier
pour telecharger un fichier sur le serveur sur lille
on peux everser l'ordeur pour evoiler un fichier sur le serveur.
en local :
```bash
scp -r tgroussa@access.grid5000.fr:lille/detr-project/scrip ~/Bureau/stage_M2/
```


## 2. Gestion de l'Espace de Stockage & Quota (Limite 25 Go)

> ⚠️ **Règle d'or :** Ne **JAMAIS** mettre les gros datasets dans le `home` (`~/`). Le dossier personnel est partagé sur le réseau (NFS), le surcharger ralentit toute l'infrastructure.

### Stratégie pour les gros Datasets (ex: > 20 Go)

1. **Code source :** Reste léger et isolé dans un sous-dossier dédié (ex: `~/detr-project`).
2. **Stockage Local Synchrone :** Une fois le nœud de calcul GPU démarré, télécharger ou extraire le dataset directement dans le stockage local de la machine : `/tmp` ou `/dev/shm`.
3. **Avantages :** - Pas de limite de quota à cet endroit (plusieurs centaines de Go disponibles).
    - Vitesse de lecture SSD/RAM ultra-rapide pour PyTorch.
4. **Code Python :** Pointer le chemin des données directement vers le dossier local (ex: `/tmp/mon_dataset`).

> 🔴 **Attention :** Le dossier `/tmp` d'un nœud est **éphémère**. Il est intégralement effacé dès que la réservation (`OAR`) se termine. Le script d'entraînement doit donc copier/décompresser le dataset au début de chaque job.

## 3. Règle d'Usage de VS Code (Charte Administrateurs)

- ❌ **INTERDIT :** Connecter VS Code directement sur la Frontend (`fnancy`, `flille`). Risque de bannissement automatique des processus.
    
- ❌ **INTERDIT :** Ouvrir tout le répertoire racine (`~`) dans VS Code (l'indexation récursive fait crasher le serveur NFS).
    
- **AUTORISÉ :** Réserver un nœud de calcul via `oarsub -I`, faire un tunnel SSH vers ce nœud spécifique, et ouvrir **uniquement** le sous-dossier du projet (`~/detr-project`).