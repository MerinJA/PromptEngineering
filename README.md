# Reproduction pédagogique DOA-PVO (drone autonome)

Ce projet reproduit en 2D l'idée centrale de **DOA-PVO** : anticiper la position future d'obstacles mobiles avant de choisir une vitesse sûre pour un drone. Il est conçu pour un devoir : les résultats sont calculés par simulation et ne constituent pas une réplication complète du banc d'essai de l'article.

## Contenu

- `run_experiments.py` : simulateur, comparaison d'une baseline réactive / DOA-PVO et export des résultats.
- `requirements.txt` : dépendances Python.
- `outputs/` : créé automatiquement avec les trajectoires, graphiques et tableau de métriques.

## Installation et exécution

```powershell
cd drone_doa_pvo
python -m pip install -r requirements.txt
python run_experiments.py
```

## Méthode étudiée

- **Baseline réactive** : le drone évite uniquement les positions observées à l'instant courant. Ce n'est pas l'algorithme Velocity Obstacle (VO) complet.
- **DOA-PVO (proposé)** : pour chaque obstacle, le simulateur estime sa vitesse par différence finie entre deux mesures de position, puis prédit plusieurs positions futures. Une vitesse candidate est rejetée si elle approche une de ces positions avec une marge de sécurité.

La commande choisit ensuite la vitesse sûre qui progresse le mieux vers le but, tout en limitant les changements de vitesse. Les obstacles ont une trajectoire non linéaire (oscillation latérale) ; c'est donc un cas où l'anticipation doit être visible.

## Figures pour la présentation

- `trajectories.png` : une trajectoire par méthode dans chaque scénario.
- `metrics.png` : taux de réussite, distance minimale et durée du trajet.
- `metrics.csv` : valeurs numériques utilisées pour les figures.
- `classroom_random_trajectories.png` : les six premières rencontres aléatoires lisibles où les deux méthodes réussissent, annotées en anglais. Elles ne sont pas sélectionnées selon le gain de DOA-PVO.
- `classroom_random_summary.png` : comparaison statistique sur 60 scénarios aléatoires reproductibles, en anglais.
- `random_metrics.csv` : données brutes de la dernière campagne ; elle est différente à chaque exécution normale.
- `experiment_history.log` : historique cumulatif. Chaque lancement ajoute un bloc séparé indiquant la date, la graine, le nombre d'itérations, le taux de succès, la marge minimale moyenne et la longueur moyenne de route pour Reactive et DOA-PVO.

Une exécution normale crée une nouvelle campagne aléatoire. Pour répéter exactement une campagne intéressante, relancer avec la graine affichée :

```powershell
python run_experiments.py --seed 123456789 --runs 60
```

## Limites à présenter honnêtement

Le drone est représenté par un disque en mouvement plan. La position mesurée comporte un faible bruit synthétique, mais il ne s'agit pas d'un modèle de caméra/LiDAR ou de contrôle complet d'un quadrirotor. Le projet évalue donc seulement le mécanisme d'évitement local.

Article de référence : Liang et al., *A Dynamic Obstacle Avoidance algorithm for unmanned aerial vehicles based on Predictive Velocity Obstacles*, Robotics and Autonomous Systems, 2026, doi:10.1016/j.robot.2025.105250.
# PromptEngineering
