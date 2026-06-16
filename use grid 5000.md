# Configuration & Bonnes Pratiques - Grid'5000 (Projet DETR)

## 1. Connexion et Navigation

### Étape 1 : Accès à la passerelle principale + connexion au site
```bash
ssh <identifiant>@access.grid5000.fr
ssh <site>
```

### Étape 2 : importation du code
crée un dossier pour le projet puis clone le git
```bash
mkdir ~/detr-project
cd ~/detr-project
git clone https://github.com/ton-compte/ton-repo-detr.git .
```
note pour future : comment copier de site A a site B ?
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
```bash
pip install -r requirements.txt 
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