Mot de passe: Marchesvp.99
Explication de tous les fichiers présents dans le dossier : 

# Script get_polygon_coords.py

Objectif: definir les coordonnées des zones d'interet (ici mangeoires+eau) 
pré-requis: vidéos mp4 d'interêt
cliquer sur  chaque points du polygones puis espace pour passer à un autre polygone, puis entrée quand c'est terminé. 
à la sortie: affichage des coordonnées de tous les points cliqués. Les coordonnées obtenues peuvent être utilisées par la suite dans le script zone_souris

# Script zone_souris

## objectifs: 
- Suivre un point d’intérêt (par ex. le marqueur “Accelerometer” (qui correspond au miniscope) ou le museau de la souris) extrait par DLC,
- Déterminer automatiquement si l’animal se trouve dans des zones spécifiques de la cage (mangeoire, eau),
- Produire une vidéo annotée avec les zones et les trajectoires,
- Exporter un CSV contenant les épisodes (entrées/sorties de zones).
- permet d'analyser une vidéo sur deeplabcut en visualisant les zones d'interet (les mangeoires et l'eau) 

⚠️ utilisation de deeplabcut -> environnement conda :  sur PC, uniquement dans la commande anaconda prompt. 
Sur mac et PC dans la commande avant tout lancement de script:   "conda activate dlc-env"

## données d'entrée: 
- config.yaml : fichier de configuration du projet DeepLabCut
- une vidéo à analyser en .mp4 (risque de prendre du temps à analyser si trop longue)

## données de sortie: 
- deux vidéos : une avec les points d'annotations (nez, miniscope, oreilles, ventre, pattes arrieres) nommée: *nomdelevidéodorigine*DLC_resnet_MeilleureModeleJul23shuffle1_172000_labeled et une avec le point d'interet (le miniscope)+les zones d'intêret nommée *nomdelavideodorigine*_zones_annotée
- Un CSV listant les épisodes de présence dans chaque zone (en frames et en secondes) nommé *nomdelavideodorigine*_zones_entries

## Fonctionnement du script
Étapes automatiques
L’utilisateur renseigne :
- le chemin de la vidéo .mp4,
- le nombre de zones de nourriture (1 ou 2).

Le script lance :
Analyse DeepLabCut (deeplabcut.analyze_videos).
Génération vidéo DLC annotée (deeplabcut.create_labeled_video).
Extraction des coordonnées (x,y) du point choisi (ici c'est "Accelerometer"(qui correspond au miniscope)).

Analyse des zones :
- Définition de polygones fixes représentant les zones (nourriture et eau).
- Détection des entrées/sorties du point dans ces zones.
- Fusion des épisodes proches (< min_gap secondes).

Zones définies (par défaut)
Cas 1 mangeoire :
  Zone Nourriture 1 (polygone vert).
  Zone Eau.
Cas 2 mangeoires :
  Zone Nourriture 1.
  Zone Nourriture 2.
Zone Eau.

Les coordonnées des polygones sont actuellement codées en dur dans le script mais peuvent etre modifiées à l'aide du script get_polygon_coords.py
À ajuster si la configuration de la cage change.

## Paramètres importants
point_name : nom du point suivi dans DLC (par défaut "Accelerometer" qui correspond à miniscope).
fps : fréquence d’échantillonnage vidéo (par défaut 30).
min_gap : seuil de fusion d’épisodes rapprochés (par défaut 1.0s).
zones_config : coordonnées des polygones définissant les zones → à adapter selon la cage.

Adaptabilité :
Pour changer de point (ex. museau) → modifier point_name dans __init__, + point_name
Pour changer de FPS ou zones → modifier fps et zones_config.

Si les colonnes ne contiennent pas le point choisi → un message liste les points disponibles.

Pour la suite: améliorer l'entrainement deeplabcut pour limiter les saut de points, quitte à faire un projet deeplabcut avec un seul point (miniscope).


# Script lecture_csv(1) et lecture_csv_2GPIO(2)
Permet de binariser les signaux IR de detecteurs de présence dans la mangeoire en determinant un seuil au dessus duquel on considère que la présence esr détéctée. 
(1) pour les sessions avec une seule nourriture.
(2) pour les sessions avce deux nourritures différentes.








# Script ARWHMM: 

Le modèle utilisé est un ARWHMM (AutoRegressive Weakly-supervised Hidden Markov Model), implémenté en Python de manière personnalisée.
La particularité est que l’algorithme combine :
- un modèle de Markov caché autorégressif (apprend des états latents de comportement),
- une supervision faible via les signaux IR (pondération W, qui donne plus d’importance aux périodes de nourrissage).

## Organisation du script

- Chargement et prétraitement des données

- Lecture des fichiers IMU (*.imu_relative.csv) et IR (IR_mouse_session*_binaire.csv).

- Nettoyage des signaux (valeurs infinies/NaN).

- Alignement temporel IMU/IR via interpolation + correction par corrélation croisée.

- Extraction de features (accX, accY, accZ, norme, dérivées, orientation si dispo).

- Normalisation robuste (z-score).

- Création de matrices de supervision W (pondération renforcée sur les périodes IR=1).

- Implémentation du modèle ARWHMM (SimpleARWHMM)

  - Initialisation aléatoire des paramètres.

  - Calcul des vraisemblances conditionnelles via un modèle AR.

  - Algorithme forward-backward pondéré (intègre les W).

  - Estimation par EM (M-step : mise à jour des transitions, coefficients AR et covariances).

  - Critère de convergence basé sur la log-vraisemblance.

- Séparation Train/Test:
Une session est tirée au hasard comme test, les autres servent à l’entraînement (hold-out leave-one-session).

- Entraînement du modèle
Le modèle apprend les états latents sur les données train.
La supervision IR renforce la détection du comportement de nourrissage.

- Prédiction et évaluation
  - Application du modèle sur la session test.
  - Identification de l’état latent le plus corrélé au signal IR (ground truth).
  - Évaluation avec métriques standard : F1-score, précision, rappel.
- Visualisation
  - Graphiques comparant prédictions vs. IR (timeline).
  - Vérification visuelle de l’alignement des signaux.

## Utilisation

LIBRARIES: pip install numpy pandas matplotlib scipy scikit-learn

## Paramètres importants 

Pour la plupart des paramètres, c'est beaucoup de test et de recherche, les valeurs présentes ne sont pas du tout forcément les plus adaptées

target_fs : fréquence cible pour le downsampling (par défaut 5 Hz).

ar_order : ordre autorégressif du modèle (par défaut 1).

num_states : nombre d’états cachés (par défaut 5).

SUP_STRENGTH : force de la supervision (plus grand = le modèle colle davantage aux IR).

eps : régularisation numérique pour éviter les divisions par zéro.

## Notes 

Supervision faible :

Les IR ne sont pas directement utilisés comme labels mais comme pondérations W.

Cela permet d’entraîner un modèle non supervisé, mais guidé par les IR, c'est pour ça que c'est un modèle ARWHMM et pas ARHMM (Auto Regressive Hidden Markov Model)

Déséquilibre de classes :

Le nourrissage représente <10% du temps total, ce qui est vraiement peu, 
Le script compense en donnant plus de poids aux événements IR=1 via SUP_STRENGTH.

## À améliorer: 
Pour l'instant, le rappel est généralement élevé (le modèle détecte beaucoup d’épisodes).
La précision est faible (beaucoup de faux positifs).
Cela reflète le déséquilibre de classes (feeding <10% du temps total).

- Implémentation maison (pas de package comme ssm) → code plus pédagogique mais moins optimisé.

- Peu de features → modèle limité en expressivité (basé surtout sur acc norm et orientation).

- le num_states le plus adapté pourrait être trouvé pas cross validation 

# Script ARWHMM LOSO

Même objectif que le script précédent mais modification au niveau du Train/test: on fait un Leave-One_Session-Out: meilleure robustesse statistique: À chaque itération, on retire une session entière (toutes ses données) pour le test, et on entraîne sur toutes les autres.
Mais très long (compter 1 semaine d'entraînement) et très peu précis, pour l'instant, ca reste qu'uns ébauche 

# Script ARWHMM backup matlab: 
Même principe que les précédent mais marche avec les scripts matlab, n'est pas optimisé car trop peu de données matlab utilisables pour faire un entrainement correct, mais si tu as beaucoup plus de données (à présent j'ai uniquement 7 sessions exploitables), cette idée pourrait devenir intéressante
