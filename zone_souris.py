import deeplabcut
import cv2
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('animal_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AnimalBehaviorAnalyzer:
    """Analyseur de comportement animal avec DeepLabCut"""
    
    def __init__(self, config_path: str, point_name: str = "Accelerometer", fps: int = 30, min_gap: float = 1.0):
        """
        Initialise l'analyseur
        
        Args:
            config_path: Chemin vers le fichier config.yaml de DLC
            point_name: Nom du point à tracker
            fps: Images par seconde de la vidéo
            min_gap: Seuil pour fusionner des épisodes rapprochés (en secondes)
        """
        self.config_path = Path(config_path)
        self.point_name = point_name
        self.fps = fps
        self.min_gap = min_gap
        
        # Vérification du fichier de config
        if not self.config_path.exists():
            raise FileNotFoundError(f"Fichier de configuration DLC introuvable: {config_path}")
        
        # Définition des zones selon le nombre de nourritures
        self.zones_config = {
            1: {
                "Zone Nourriture 1": Polygon([(778, 323), (849, 275), (868, 344), (782, 387)]),
                "Zone Eau": Polygon([(238, 342), (288, 321), (288, 378)])
            },
            2: {
                "Zone Nourriture 1": Polygon([(768, 161), (801, 131), (809, 185), (775, 224)]),
                "Zone Nourriture 2": Polygon([(779, 484), (819, 477), (825, 553), (785, 549)]),
                "Zone Eau": Polygon([(238, 342), (288, 321), (288, 378)])
            }
        }
        
        self.zone_colors = {
            "Zone Nourriture 1": (0, 255, 0),
            "Zone Nourriture 2": (255, 0, 0),
            "Zone Eau": (0, 165, 255)
        }
    
    def get_user_input(self) -> Tuple[str, int]:
        """Récupère les inputs utilisateur de manière sécurisée"""
        # Chemin vidéo
        while True:
            video_path = input("Quel est le chemin complet vers la vidéo à analyser (.mp4) ? ➤ ").strip()
            video_path = Path(video_path)
            
            if video_path.exists() and video_path.suffix.lower() == '.mp4':
                break
            else:
                print(" Fichier vidéo introuvable ou format incorrect. Veuillez réessayer.")
        
        # Nombre de zones de nourriture
        while True:
            try:
                nb_nourritures = int(input("Combien de points de nourriture dans la cage ? (1 ou 2) ➤ "))
                if nb_nourritures in [1, 2]:
                    break
                else:
                    print(" Saisie invalide. Tapez 1 ou 2.")
            except ValueError:
                print(" Saisie invalide. Tapez un nombre.")
        
        return str(video_path), nb_nourritures
    
    def run_dlc_analysis(self, video_path: Path) -> bool:
        """Exécute l'analyse DeepLabCut avec gestion d'erreurs"""
        try:
            logger.info(" Démarrage de l'analyse DeepLabCut...")
            deeplabcut.analyze_videos(str(self.config_path), [str(video_path)], save_as_csv=True)
            logger.info(" Analyse DeepLabCut terminée avec succès")
            return True
            
        except Exception as e:
            logger.error(f" Erreur lors de l'analyse DeepLabCut: {e}")
            return False
    
    def create_labeled_video(self, video_path: Path) -> bool:
        """Génère la vidéo annotée DLC avec gestion d'erreurs"""
        try:
            logger.info(" Génération de la vidéo annotée DLC...")
            deeplabcut.create_labeled_video(str(self.config_path), [str(video_path)])
            logger.info(" Vidéo annotée DLC générée avec succès")
            return True
            
        except Exception as e:
            logger.error(f" Erreur lors de la génération de la vidéo annotée: {e}")
            return False
    
    def find_csv_file(self, video_path: Path) -> Optional[Path]:
        """Trouve automatiquement le fichier CSV DLC"""
        video_dir = video_path.parent
        video_stem = video_path.stem
        
        csv_candidates = list(video_dir.glob(f"{video_stem}*DLC*.csv"))
        
        if len(csv_candidates) == 0:
            logger.error(" Aucun fichier CSV DLC trouvé pour cette vidéo")
            return None
        elif len(csv_candidates) > 1:
            logger.warning(" Plusieurs fichiers CSV trouvés, utilisation du premier")
        
        csv_path = csv_candidates[0]
        logger.info(f" Fichier CSV trouvé: {csv_path}")
        return csv_path
    
    def load_and_validate_csv(self, csv_path: Path) -> Optional[Tuple[pd.Series, pd.Series, str]]:
        try:
            # Lire le fichier CSV avec des options pour éviter les erreurs de type
            df = pd.read_csv(csv_path, header=[0, 1], index_col=0, low_memory=False)

            # Afficher les colonnes pour débogage
            logger.info(f"Colonnes du CSV: {df.columns}")

            model_name = df.columns.levels[0][0]

            # Vérification de l'existence du point
            if (model_name, self.point_name) not in df.columns:
                logger.error(f"Point '{self.point_name}' introuvable dans le CSV")
                logger.info(f"Points disponibles: {[col[1] for col in df.columns if not col[1].endswith('.1')]}")
                return None

            # Extraire les coordonnées x et y
            x = pd.to_numeric(df[(model_name, self.point_name)], errors='coerce')
            y = pd.to_numeric(df[(model_name, f"{self.point_name}.1")], errors='coerce')

            logger.info(f"CSV chargé: {len(x)} frames, modèle '{model_name}'")
            return x, y, model_name

        except Exception as e:
            logger.error(f"Erreur lors du chargement du CSV: {e}")
            return None

    
    def get_safe_output_paths(self, video_path: Path) -> Tuple[Path, Path]:
        """Génère des chemins de sortie sûrs (évite les conflits)"""
        video_dir = video_path.parent
        video_stem = video_path.stem
        
        # Générer des noms uniques si nécessaire
        counter = 1
        while True:
            suffix = f"_v{counter}" if counter > 1 else ""
            output_video = video_dir / f"{video_stem}_zones_annotee{suffix}.mp4"
            output_csv = video_dir / f"{video_stem}_zones_entries{suffix}.csv"
            
            if not output_video.exists() and not output_csv.exists():
                break
            counter += 1
        
        return output_video, output_csv
    
    def merge_episodes(self, entries: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Fusionne les épisodes rapprochés"""
        if not entries:
            return []
        
        # Convertir en secondes
        episodes = [(start / self.fps, end / self.fps) for start, end in entries]
        merged_episodes = []
        
        current_start, current_end = episodes[0]
        
        for start, end in episodes[1:]:
            if start - current_end <= self.min_gap:
                current_end = end  # fusion
            else:
                merged_episodes.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged_episodes.append((current_start, current_end))  # dernier épisode
        
        return merged_episodes
    
    def analyze_zones_and_create_video(self, video_path: Path, x: pd.Series, y: pd.Series, zones: Dict, output_video: Path, output_csv: Path) -> bool:
        try:
            # Vérification des séries x et y
            if x.empty or y.empty:
                logger.error("Les séries x ou y sont vides.")
                return False

            # Vérification des index
            if len(x) != len(y):
                logger.error(f"Les séries x et y n'ont pas la même longueur: x={len(x)}, y={len(y)}")
                return False

            # Ouverture de la vidéo
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error("Impossible d'ouvrir la vidéo")
                return False

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(str(output_video), fourcc, self.fps, (width, height))
            if not out.isOpened():
                logger.error("Impossible de créer le fichier vidéo de sortie")
                cap.release()
                return False

            # Initialisation du suivi des zones
            frame_idx = 0
            zone_states = {zone: False for zone in zones}
            zone_starts = {zone: None for zone in zones}
            zone_segments = {zone: [] for zone in zones}

            logger.info("Traitement des frames...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_idx >= len(x):
                    break

                # Vérification de l'index
                if frame_idx not in x.index or frame_idx not in y.index:
                    logger.warning(f"Index {frame_idx} manquant dans les séries x ou y.")
                    frame_idx += 1
                    continue

                px, py = x[frame_idx], y[frame_idx]
                point = Point(px, py) if not np.isnan(px) and not np.isnan(py) else None

                status_text = "HORS ZONE"
                status_color = (0, 0, 255)

                # Analyse de chaque zone
                for zone_name, polygon in zones.items():
                    in_zone = point is not None and polygon.contains(point)

                    if in_zone:
                        status_text = zone_name
                        status_color = self.zone_colors.get(zone_name, (255, 255, 0))

                    # Gestion des entrées/sorties de zone
                    if in_zone and not zone_states[zone_name]:
                        zone_states[zone_name] = True
                        zone_starts[zone_name] = frame_idx
                    elif not in_zone and zone_states[zone_name]:
                        zone_states[zone_name] = False
                        start = zone_starts[zone_name]
                        if start is not None:
                            zone_segments[zone_name].append((start, frame_idx))
                        zone_starts[zone_name] = None

                    # Dessiner le polygone
                    pts = np.array(polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=self.zone_colors[zone_name], thickness=2)
                    cv2.putText(frame, zone_name, tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.zone_colors[zone_name], 2)

                # Dessiner le point
                if point is not None:
                    cv2.circle(frame, (int(px), int(py)), 5, status_color, -1)
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                out.write(frame)
                frame_idx += 1

                # Progress log tous les 1000 frames
                if frame_idx % 1000 == 0:
                    logger.info(f"   Traité {frame_idx} frames...")

            cap.release()
            out.release()

            # Traitement final des zones encore actives
            for zone_name in zones:
                if zone_states[zone_name] and zone_starts[zone_name] is not None:
                    zone_segments[zone_name].append((zone_starts[zone_name], frame_idx - 1))

            logger.info(f"Vidéo annotée sauvegardée: {output_video}")

            # Export CSV
            self.export_episodes_csv(zone_segments, output_csv)

            return True

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des zones: {e}")
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            return False

    
    def run_complete_analysis(self) -> bool:
        """Exécute l'analyse complète"""
        try:
            # Récupération des inputs utilisateur
            video_path_str, nb_nourritures = self.get_user_input()
            video_path = Path(video_path_str)
            
            logger.info(f" Vidéo: {video_path}")
            logger.info(f" Zones de nourriture: {nb_nourritures}")
            
            # Sélection des zones
            zones = self.zones_config[nb_nourritures]
            
            # Étape 1: Analyse DLC
            if not self.run_dlc_analysis(video_path):
                return False
            
            # Étape 2: Vidéo annotée DLC
            if not self.create_labeled_video(video_path):
                logger.warning(" Échec de la génération de vidéo annotée DLC, mais poursuite de l'analyse")
            
            # Étape 3: Chargement du CSV
            csv_path = self.find_csv_file(video_path)
            if csv_path is None:
                return False
            
            csv_data = self.load_and_validate_csv(csv_path)
            if csv_data is None:
                return False
            
            x, y, model_name = csv_data
            
            # Étape 4: Génération des chemins de sortie
            output_video, output_csv = self.get_safe_output_paths(video_path)
            
            # Étape 5: Analyse des zones et création de la vidéo
            if not self.analyze_zones_and_create_video(video_path, x, y, zones, output_video, output_csv):
                return False
            
            logger.info(" Analyse terminée avec succès!")
            return True
            
        except KeyboardInterrupt:
            logger.info(" Analyse interrompue par l'utilisateur")
            return False
        except Exception as e:
            logger.error(f" Erreur inattendue: {e}")
            return False


def main():
    """Fonction principale"""
    # Paramètres par défaut - à adapter selon tes besoins
    config_path = input("Chemin vers le fichier config.yaml de DeepLabCut ➤ ").strip()
    
    try:
        analyzer = AnimalBehaviorAnalyzer(
            config_path=config_path,
            point_name="Accelerometer",
            fps=30,
            min_gap=1.0
        )
        
        success = analyzer.run_complete_analysis()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f" Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()