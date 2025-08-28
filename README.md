Explication de tous les fichiers présents dans le dossier.

### Script get_polygon_coords.py

Objectif: definir les coordonnées des zones d'interet (ici mangeoires+eau) 
pré-requis: vidéos mp4 d'interêt
cliquer sur  chaque points du polygones puis espace pour passer à un autre polygone, puis entrée quand c'est terminé. 
à la sortie: affichage des coordonnées de tous les points cliqués. Les coordonnées peuvent être utilisiées par la suite dans le script zone_souris

### Script Zone_souris

# objectifs: 
- Suivre un point d’intérêt (par ex. le marqueur “Accelerometer” (qui correspond au miniscope) ou le museau de la souris) extrait par DLC,
- Déterminer automatiquement si l’animal se trouve dans des zones spécifiques de la cage (nourriture, eau),
- Produire une vidéo annotée avec les zones et les trajectoires,
- Exporter un CSV contenant les épisodes (entrées/sorties de zones).
- permet d'analyser une vidéo sur deeplabcut en visualisant les zones d'interet (les mangeoires et l'eau) 

⚠️ utilisation de deeplabcut -> environnement conda :  sur PC, dans la commande anaconda prompt
et sur mac et PC dans la commande avant tout lancement de script:   "conda activate dlc-env"

# pré-requis: 
- config.yaml : fichier de configuration du projet DeepLabCut
- une vidéo à analyser en .mp4 (pas trop longue sinon ça risque de prendre du temps)
# donnée de sortie: 
- deux vidéos : une avec les points d'annotations et une avec les point d'interet (le miniscope) et les zones  
- Un CSV listant les épisodes de présence dans chaque zone (en frames et en secondes).

# Fonctionnement du script
Étapes automatiques
L’utilisateur renseigne :

- le chemin de la vidéo .mp4,
- le nombre de zones de nourriture (1 ou 2).

Le script lance :
Analyse DeepLabCut (deeplabcut.analyze_videos).
Génération vidéo DLC annotée (deeplabcut.create_labeled_video).
Extraction des coordonnées (x,y) du point choisi (ici c'est "Accelerometer"(qui correspond au miniscope).

Analyse des zones :
- Définition de polygones fixes représentant les zones (nourriture et eau).
- Détection des entrées/sorties du point dans ces zones.
- Fusion des épisodes proches (< min_gap secondes).

Zones définies (par défaut)
Cas 1 mangeoire :
  Zone Nourriture 1 (polygone vert).
  Zone Eau (orange).
Cas 2 mangeoires :
  Zone Nourriture 1 (vert).
  Zone Nourriture 2 (bleu).
Zone Eau (orange).

Les coordonnées des polygones sont actuellement codées en dur dans le script (pixels de la vidéo).
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

Pour la suite: améliorer l'entrainement deeplabcut pour limiter les saut de points, quitte à faire un projet deeplabcut avec un seul point.


### Script lecture_csv(1) et lecture_csv_2GPIO(2)
permet de binariser les signaux IR de detecteurs de présence dans la mangeoire. 
(1) pour les sessions avec une seule nourriture.
(2) pour les sessions avce deux nourritures différentes.


### Script ARWHMM LOSO
plus robuste et plus prometteur que le script ARWHMM mais très long (compter 1 semaine d'entrainement)
###Méthodologie
1. Prétraitement des données
Nettoyage des fichiers CSV (suppression NaN et inf).
Alignement temporel IMU ↔ IR par interpolation + correction de décalage (cross-corrélation).
Extraction de features (accélération, norme, dérivées, orientation si disponible).
Downsampling des signaux à 5 Hz pour alléger le calcul. (target_fs modifiable)
Normalisation (z-score robuste).

2. Supervision faible (W)
L’IR est utilisé pour guider l’apprentissage mais pas comme vérité stricte.

On construit une matrice W :
Quand IR=0, l’état "non-feeding" est privilégié.
Quand IR=1, l’état "feeding" est privilégié.
    #les infrarouges sont binarisés avec les scripts lecture_csv et lecture_csv_2GPIO si deux portes dans la cage

Paramètre SUP_STRENGTH : contrôle l’influence de cette supervision.

3. Modèle ARWHMM
Modèle de type Hidden Markov Model autorégressif (ARHMM).
Modifié pour intégrer la supervision faible (W).
Nombre d’états num_states ajustable (Ici il est à 5 mais je ne suis pas sûre que ce soit le plus optimal).

4. Validation croisée (LOSO)
Méthode Leave-One-Session-Out :
À chaque itération, une session complète est gardée en test.
Le modèle est entraîné sur toutes les autres.
Évaluation sur la session laissée de côté.
Plus robuste que du simple train/test, et adapté aux données animales (sessions indépendantes).

5. Évaluation
Identification de l’état "feeding" par corrélation avec le signal IR.

Calcul des métriques :
Précision (précision des prédictions positives).
Rappel (capacité à détecter tous les épisodes de nourrissage).
F1-score (compromis entre précision et rappel).

Résultats stockés dans loso_results.csv (résumé par session + macro/micro-moyennes).

##Utilisation

1. Organisation des données
data/imu_files/mXXXX/sessionY/*.imu_relative.csv   # fichiers IMU
gpio_binaire/mXXXX/IR_mXXXX_sessionY_binaire.csv  # fichiers IR

2. Paramètres à ajuster
Dans le script ARWHMM.py :
target_fs : fréquence cible (5 Hz par défaut).
num_states : nombre d’états latents (par ex. 5).
SUP_STRENGTH : poids de la supervision (par ex. 20).

#Le script :
Charge toutes les sessions valides.
Lance la validation croisée LOSO.
Affiche les métriques et sauvegarde un fichier loso_results.csv.

#Résultats typiques
Le rappel est généralement élevé (le modèle détecte beaucoup d’épisodes).
La précision est faible (beaucoup de faux positifs).

Cela reflète le déséquilibre de classes (feeding <10% du temps total).

###Points importants pour la reprise

#Implémentation maison :
Le code ne repose pas sur une librairie comme ssm.
Les calculs sont parfois lents ou sensibles aux paramètres.

#Déséquilibre de classes :
Feeding = minoritaire → précision limitée.
Il faudra tester des pondérations différentes (SUP_STRENGTH, ajustement des poids W).

#Structure des données :
Bien vérifier la correspondance entre fichiers IMU et IR.
Si les noms changent → adapter les regex dans le script.

#Perspectives :
Ajouter des features (spectres fréquentiels, dérivées d’orientation).
Explorer d’autres modèles (par ex. ARHMM non supervisé, ou réseaux récurrents).

###Conclusion

Ce projet a permis de poser une première brique méthodologique pour la détection de comportements alimentaires à partir de signaux IMU.
Le pipeline de prétraitement est automatisé.
L’ARWHMM faible supervision est opérationnel et validé en LOSO.
Le modèle distingue déjà une dynamique liée au nourrissage, mais la précision doit être améliorée.


### Script ARWHMM backup matlab: 
même principe que le précédent mais marche avec les scripts matlab, n'est pas optimisé car trop peu de données matlab utilisables pour faire un entrainement correct

### Script ARWHMM_LOSO: 


