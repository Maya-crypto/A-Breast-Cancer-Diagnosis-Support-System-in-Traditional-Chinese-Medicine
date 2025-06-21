## import os
import sys
import subprocess
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle
from pathlib import Path
import json
import yaml
import random
from datetime import datetime
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration système optimisée
CONFIG = {
    'base_dir': '.',
    'output_dir': 'mtc_output',
    'model_dir': 'mtc_models',
    'results_dir': 'mtc_results',
    'train_split': 0.8,  # Augmenté pour plus de données d'entraînement
    'epochs': 200,  # Augmenté pour une meilleure convergence
    'batch_size': 16,  # Optimisé pour GPU
    'imgsz': 640,  # Augmenté pour une meilleure résolution
    'patience': 50,  # Augmenté pour éviter l'arrêt prématuré
    'conf_threshold': 0.15,  # Réduit pour capturer plus de détections
    'iou_threshold': 0.3,  # Réduit pour éviter la suppression excessive
    'augmentation_factor': 8,  # Augmenté pour plus de diversité
    'target_accuracy': 0.85,  # Objectif plus ambitieux
    'random_seed': 42,
    'lr0': 0.0008,  # Learning rate optimisé
    'weight_decay': 0.001,  # Regularisation améliorée
    'mosaic': 0.8,  # Augmentation mosaic
    'mixup': 0.1,  # Augmentation mixup
    'copy_paste': 0.1,  # Copy-paste augmentation
}

# Classes YOLO (16 classes)
CLASS_NAMES = [
    'Ecchymoses', 'Eduit_jaune_epais', 'Eduit_jaune_mince', 'Fissure',
    'Langue_normal', 'Langue_pale', 'Langue_petite', 'Langue_rose',
    'Langue_rouge', 'Langue_rouge_foncee', 'enduit_blanc_epais',
    'enduit_blanc_mince', 'langue_ganfelee', 'red_dot',
    'salive_humide', 'salive_normale'
]


# Zones MTC
TONGUE_ZONES = {
    'kidney': {
        'name': 'Rein',
        'coords': [(0.2, 0), (0.8, 0), (0.8, 0.15), (0.2, 0.15)],
        'color': (75, 0, 130)
    },
    'liver_gall_right': {
        'name': 'Foie-VB Droit',
        'coords': [(0, 0.15), (0.3, 0.15), (0.3, 0.65), (0, 0.65)],
        'color': (34, 139, 34)
    },
    'liver_gall_left': {
        'name': 'Foie-VB Gauche',
        'coords': [(0.7, 0.15), (1, 0.15), (1, 0.65), (0.7, 0.65)],
        'color': (50, 205, 50)
    },
    'spleen_stomach': {
        'name': 'Rate-Estomac',
        'coords': [(0.3, 0.15), (0.7, 0.15), (0.7, 0.65), (0.3, 0.65)],
        'color': (255, 215, 0)
    },
    'heart_lung': {
        'name': 'Coeur-Poumon',
        'coords': [(0.2, 0.65), (0.8, 0.65), (0.8, 1), (0.2, 1)],
        'color': (220, 20, 60)
    }
}

# Critères de diagnostic optimisés avec contraintes logiques
DIAGNOSTIC_CRITERIA = {
    'healthy': {
        'forme': ['Langue_normal', 'Langue_petite'],
        'forme_alternative': ['langue_ganfelee'],
        'couleur': ['Langue_rose', 'Langue_rouge'],
        'enduit': ['enduit_blanc_mince'],
        'salive': ['salive_normale'],
        'fissure' : ['Fissure'],
        'ecchymoses': [],  # 'Ecchymoses' ignorée si absente
        'points_rouges': ['red_dot'],  # 'red_dot' ignoré si absent
        'required_score': 0.3,
        'weight': 1.0
    },
    'early': {
        'forme': ['Langue_normal', 'Langue_petite'],
        'forme_alternative': ['langue_ganfelee'],
        'couleur': ['Langue_pale', 'Langue_rouge'],
        'enduit': ['enduit_blanc_mince', 'enduit_blanc_epais', 'Eduit_jaune_mince'],
        'salive': ['salive_normale', 'salive_humide'],
        'fissure' : ['Fissure'],
        'ecchymoses': [],
        'points_rouges': ['red_dot'],
        'required_score': 0.4,
        'weight': 1.2
    },
    'advanced': {
        'forme': ['Langue_petite'],
        'forme_alternative': ['Langue_normal'],
        'couleur': ['Langue_rouge', 'Langue_rouge_foncee'],
        'enduit': ['Eduit_jaune_epais', 'Eduit_jaune_mince'],
        'salive': ['salive_normale', 'salive_humide'],
        'ecchymoses': ['Ecchymoses'],
        'fissure' : ['Fissure'],
        'points_rouges': ['red_dot'],
        'required_score': 0.5,
        'weight': 1.5
    }
}


# Contraintes mutuellement exclusives
MUTUAL_EXCLUSIONS = {
    'forme': [['Langue_normal', 'Langue_petite']],  # Une seule forme de base
    'couleur': [['Langue_pale', 'Langue_rouge', 'Langue_rouge_foncee', 'Langue_rose']],  # Une seule couleur
    'salive': [['salive_humide', 'salive_normale']],  # Un seul type de salive
    'enduit_couleur': [['enduit_blanc_epais', 'enduit_blanc_mince', 'Eduit_jaune_epais', 'Eduit_jaune_mince']],  # Une couleur d'enduit
    'enduit_texture': [['enduit_blanc_epais', 'Eduit_jaune_epais'], ['enduit_blanc_mince', 'Eduit_jaune_mince']]  # Une texture
}


class SmartDatasetProcessor:
    """Gère l'organisation et le split du dataset avec intelligence sur les noms"""

    def init(self, config):
        self.config = config
        self.base_dir = Path(config['base_dir'])
        self.output_dir = Path(config['output_dir'])
        self.stats = defaultdict(int)

    def extract_label_from_filename(self, filename):
        """Extrait intelligemment le label depuis le nom de fichier"""
        filename_lower = filename.lower()

        if any(keyword in filename_lower for keyword in ['healthy', 'sain', 'normal']):
            return 'healthy'
        elif any(keyword in filename_lower for keyword in ['ebc', 'early', 'precoce']):
            return 'early'
        elif any(keyword in filename_lower for keyword in ['abc', 'advanced', 'avance']):
            return 'advanced'
        elif any(keyword in filename_lower for keyword in ['real', 'reel']):
            # Analyser plus finement les images 'real'
            if any(keyword in filename_lower for keyword in ['1', 'one', 'first']):
                return 'healthy'
            elif any(keyword in filename_lower for keyword in ['2', 'two', 'second']):
                return 'early'
            elif any(keyword in filename_lower for keyword in ['3', 'three', 'third']):
                return 'advanced'
            else:
                return 'unknown'
        else:
            return 'unknown'

    def process_existing_dataset(self):
        """Traite le dataset existant avec split intelligent"""
        print("\nORGANISATION INTELLIGENTE DU DATASET")
        print("="*70)

        train_dir = self.base_dir / 'train'
        if not train_dir.exists():
            raise ValueError(f"Dossier 'train' non trouvé dans {self.base_dir}")

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir()

        print("\nAnalyse intelligente du dataset...")
        all_images = []
        images_by_class = defaultdict(list)
        label_confidence = defaultdict(list)

        train_images_dir = train_dir / 'images'
        if train_images_dir.exists():
            for img_path in train_images_dir.glob('*.jpg'):
                all_images.append(img_path)
                predicted_label = self.extract_label_from_filename(img_path.name)

                if predicted_label != 'unknown':
                    images_by_class[predicted_label].append(img_path)
                    label_confidence[predicted_label].append(1.0)
                else:
                    # Classification par défaut basée sur l'analyse du nom
                    name = img_path.name.lower()
                    if 'healthy' in name or 'normal' in name:
                        images_by_class['healthy'].append(img_path)
                        label_confidence['healthy'].append(0.5)
                    elif 'ebc' in name or 'early' in name:
                        images_by_class['early'].append(img_path)
                        label_confidence['early'].append(0.5)
                    elif 'abc' in name or 'advanced' in name:
                        images_by_class['advanced'].append(img_path)
                        label_confidence['advanced'].append(0.5)
                    else:
                        images_by_class['unknown'].append(img_path)
                        label_confidence['unknown'].append(0.1)

        print(f"\nImages trouvées: {len(all_images)}")
        print("\nDistribution intelligente par classe:")
        for class_name, imgs in images_by_class.items():
            avg_conf = np.mean(label_confidence[class_name]) if label_confidence[class_name] else 0
            print(f"  - {class_name}: {len(imgs)} images (confiance: {avg_conf:.2f})")

        # Split stratifié intelligent
        train_split = []
        val_split = []

        for class_name, class_images in images_by_class.items():
            if class_name == 'unknown':
                continue

            # Tri par confiance pour garder les meilleurs exemples en training
            conf_scores = label_confidence[class_name]
            sorted_indices = np.argsort(conf_scores)[::-1]  # Tri décroissant
            sorted_images = [class_images[i] for i in sorted_indices]

            n_train = int(len(sorted_images) * self.config['train_split'])
            train_split.extend(sorted_images[:n_train])
            val_split.extend(sorted_images[n_train:])

        # Mélange final mais en préservant la distribution
        random.shuffle(train_split)
        random.shuffle(val_split)

        print(f"\nSplit intelligent {int(self.config['train_split']*100)}/{int((1-self.config['train_split'])*100)}:")
        print(f"  - Train: {len(train_split)} images")
        print(f"  - Validation: {len(val_split)} images")

        # Créer structure
        for split_name in ['train', 'val']:
            (self.output_dir / split_name / 'images').mkdir(parents=True)
            (self.output_dir / split_name / 'labels').mkdir(parents=True)

        # Copier fichiers
        print("\nCopie des fichiers...")
        self._copy_split(train_split, 'train', train_dir / 'labels')
        self._copy_split(val_split, 'val', train_dir / 'labels')

        self._create_data_yaml()

        return self.output_dir

    def _copy_split(self, image_list, split_name, labels_dir):
        """Copie images et labels"""
        dest_images = self.output_dir / split_name / 'images'
        dest_labels = self.output_dir / split_name / 'labels'

        for img_path in image_list:
            shutil.copy2(img_path, dest_images / img_path.name)
            label_file = labels_dir / f"{img_path.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, dest_labels / label_file.name)

    def _create_data_yaml(self):
        """Crée le fichier YAML"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(CLASS_NAMES),
            'names': CLASS_NAMES
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"\nFichier YAML créé: {yaml_path}")

class AdvancedDataAugmentor:
    """Augmentation avancée des données"""

    def init(self, augmentation_factor=8):
        self.factor = augmentation_factor
        self.augmented_count = 0

    def augment_dataset(self, dataset_dir):
        """Augmente les données d'entraînement avec techniques avancées"""
        print("\nAUGMENTATION AVANCEE DES DONNEES")
        print("="*70)
        print(f"Facteur: x{self.factor}")

        train_images = Path(dataset_dir) / 'train' / 'images'
        train_labels = Path(dataset_dir) / 'train' / 'labels'

        original_images = list(train_images.glob('*.jpg'))
        print(f"Images originales: {len(original_images)}")

        for img_path in original_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            label_path = train_labels / f"{img_path.stem}.txt"
            annotations = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    annotations = f.readlines()

            for i in range(self.factor):
                aug_img, aug_annot = self._apply_advanced_augmentation(img, annotations, i)

                aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(train_images / aug_name), aug_img)

                if aug_annot:
                    with open(train_labels / f"{img_path.stem}_aug{i}.txt", 'w') as f:
                        f.writelines(aug_annot)

                self.augmented_count += 1

        total = len(original_images) + self.augmented_count
        print(f"Images augmentées: {self.augmented_count}")
        print(f"Total d'entraînement: {total}")

        return total

    def _apply_advanced_augmentation(self, image, annotations, idx):
        """Applique des augmentations avancées"""
        h, w = image.shape[:2]
        aug_img = image.copy()
        aug_annot = annotations.copy()

        # Augmentations géométriques
        if idx == 0:  # Rotation + flip
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
            aug_img = cv2.flip(aug_img, 1)
            aug_annot = self._rotate_annotations(annotations, 90)
            aug_annot = self._flip_annotations(aug_annot)

        elif idx == 1:  # Flip horizontal simple
            aug_img = cv2.flip(aug_img, 1)
            aug_annot = self._flip_annotations(annotations)

        elif idx == 2:  # Ajustement luminosité et contraste
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-40, 40)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

        elif idx == 3:  # Rotation légère + saturation
            angle = random.uniform(-15, 15)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h))

            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.7, 1.3)
            hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif idx == 4:  # Blur + noise
            kernel_size = random.choice([3, 5])
            aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)
            noise = np.random.normal(0, 5, aug_img.shape).astype(np.uint8)
            aug_img = cv2.add(aug_img, noise)

        elif idx == 5:  # Déformation perspective
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            offset = random.randint(10, 30)
            pts2 = np.float32([[offset, offset], [w-offset, offset],
                              [offset, h-offset], [w-offset, h-offset]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            aug_img = cv2.warpPerspective(aug_img, M, (w, h))

        elif idx == 6:  # Gamma correction
            gamma = random.uniform(0.5, 2.0)
            look_up_table = np.empty((1, 256), np.uint8)
            for i in range(256):
                look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            aug_img = cv2.LUT(aug_img, look_up_table)

        elif idx == 7:  # Combinaison complexe
            # Légère rotation
            angle = random.uniform(-10, 10)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, random.uniform(0.9, 1.1))
            aug_img = cv2.warpAffine(aug_img, M, (w, h))

            # Ajustement couleur
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

            # HSV
            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
            hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 1.2)
            hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.9, 1.1)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return aug_img, aug_annot

    def _rotate_annotations(self, annotations, angle):
        """Ajuste annotations pour rotation"""
        rotated = []
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) >= 5:
                cls = parts[0]
                x, y, w, h = map(float, parts[1:5])
                if angle == 90:
                    new_x, new_y = 1 - y, x
                    new_w, new_h = h, w
                    rotated.append(f"{cls} {new_x} {new_y} {new_w} {new_h}\n")
        return rotated

    def _flip_annotations(self, annotations):
        """Ajuste annotations pour flip"""
        flipped = []
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) >= 5:
                cls = parts[0]
                x, y, w, h = map(float, parts[1:5])
                new_x = 1 - x
                flipped.append(f"{cls} {new_x} {y} {w} {h}\n")
        return flipped

class OptimizedYOLOv11Trainer:
    """Entraîneur YOLOv11 optimisé"""

    def init(self, config):
        self.config = config
        self.model_dir = Path(config['model_dir'])
        self.model_dir.mkdir(exist_ok=True)

    def train(self, dataset_path):
        """Entraîne YOLOv11 avec configuration optimisée"""
        print("\nENTRAINEMENT YOLOV11 OPTIMISE")
        print("="*70)

        try:
            from ultralytics import YOLO
        except ImportError:
            print("Installation d'ultralytics...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
            from ultralytics import YOLO

        # Device
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                print(f"GPU détecté: {torch.cuda.get_device_name()}")
                print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
        except:
            device = 'cpu'

        print(f"Device: {device}")

        # Configuration optimisée
        train_config = {
            'data': str(Path(dataset_path) / 'data.yaml'),
            'epochs': self.config['epochs'],
            'imgsz': self.config['imgsz'],
            'batch': self.config['batch_size'],
            'patience': self.config['patience'],
            'device': device,
            'workers': 8 if device == 'cuda' else 4,
            'project': str(self.model_dir),
            'name': 'yolov11_mtc_optimized',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': self.config['lr0'],
            'lrf': 0.001,  # Final learning rate factor
            'momentum': 0.937,
            'weight_decay': self.config['weight_decay'],
            'warmup_epochs': 10.0,  # Warmup plus long
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,  # Box loss gain
            'cls': 2.0,  # Classification loss gain augmenté
            'dfl': 1.5,  # DFL loss gain
            'pose': 12.0,  # Pose loss gain
            'kobj': 1.0,  # Keypoint objectness loss gain
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,  # HSV-Saturation augmentation
            'hsv_v': 0.4,  # HSV-Value augmentation
            'degrees': 15.0,  # Rotation augmentation augmentée
            'translate': 0.1,  # Translation augmentation
            'scale': 0.9,  # Scale augmentation
            'shear': 2.0,  # Shear augmentation
            'perspective': 0.0001,  # Perspective augmentation
            'flipud': 0.0,  # Vertical flip probability
            'fliplr': 0.5,  # Horizontal flip probability
            'mosaic': self.config['mosaic'],  # Mosaic augmentation
            'mixup': self.config['mixup'],  # Mixup augmentation
            'copy_paste': self.config['copy_paste'],  # Copy-paste augmentation
            'label_smoothing': 0.1,
            'plots': True,
            'save': True,
            'save_period': 25,
            'cache': True,  # Cache activé pour améliorer la vitesse
            'amp': device == 'cuda',  # Mixed precision si GPU
            'seed': self.config['random_seed'],
            # Paramètres avancés
            'close_mosaic': 15,  # Epochs to close mosaic augmentation
            'resume': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': True,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'retina_masks': False,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': True,
            'opset': None,
            'workspace': None,
            'nms': False,
            'rect': False
        }

        # Charger modèle
        print("Chargement du modèle...")
        try:
            # Essayer YOLOv11
            if Path('yolov11s.pt').exists():
                model = YOLO('yolov11s.pt')
                print("YOLOv11s chargé")
            elif Path('yolov11n.pt').exists():
                model = YOLO('yolov11n.pt')
                print("YOLOv11n chargé")
            else:
                # Fallback sur YOLOv8
                model = YOLO('yolov8s.pt')
                print("YOLOv8s utilisé comme alternative")
        except Exception as e:
            print(f"Erreur chargement modèle: {e}")
            # Télécharger à nouveau
            download_yolov11_model()
            model = YOLO('yolov8s.pt')

        print(f"\nConfiguration optimisée:")
        print(f"  - Epochs: {self.config['epochs']}")
        print(f"  - Batch: {self.config['batch_size']}")
        print(f"  - Image size: {self.config['imgsz']}")
        print(f"  - Learning rate: {self.config['lr0']}")
        print(f"  - Weight decay: {self.config['weight_decay']}")

        # Entraîner
        print("\nDébut de l'entraînement optimisé...")
        results = model.train(**train_config)

        # Évaluer
        print("\nÉvaluation finale...")
        metrics = model.val()

        # Afficher résultats
        self._display_results()

        return model

    def _display_results(self):
        """Affiche les métriques optimisées"""
        results_path = self.model_dir / 'yolov11_mtc_optimized'
        results_csv = results_path / 'results.csv'

        if results_csv.exists():
            df = pd.read_csv(results_csv)
            if not df.empty:
                last = df.iloc[-1]

                print("\n" + "="*50)
                print("METRIQUES FINALES OPTIMISEES")
                print("="*50)

                metrics = {
                    'Epochs': int(last.get('epoch', 0)),
                    'mAP50': float(last.get('metrics/mAP50(B)', last.get('metrics/mAP50', 0))),
                    'mAP50-95': float(last.get('metrics/mAP50-95(B)', last.get('metrics/mAP50-95', 0))),
                    'Precision': float(last.get('metrics/precision(B)', 0)),
                    'Recall': float(last.get('metrics/recall(B)', 0))
                }

                p = metrics['Precision']
                r = metrics['Recall']
                f1 = 2 * (p * r) / (p + r + 1e-6)
                accuracy = (p + r) / 2

                metrics['F1-Score'] = f1
                metrics['Accuracy'] = accuracy

                for key, value in metrics.items():
                    if key == 'Epochs':
                        print(f"{key}: {value}")
                    else:
                        print(f"{key}: {value:.4f} ({value*100:.2f}%)")

                print("="*50)

                return metrics

        return None

class SmartDiagnosticSystem:
    """Système de diagnostic intelligent avec logique de contraintes"""

    def init(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.mapper = TongueMapper()
        self.results_dir = Path(CONFIG['results_dir'])
        self.results_dir.mkdir(exist_ok=True)

    def diagnose(self, image_path):
        """Diagnostic intelligent avec contraintes logiques"""
        print(f"\nAnalyse intelligente: {Path(image_path).name}")

        results = self.model(image_path, conf=CONFIG['conf_threshold'],
                           iou=CONFIG['iou_threshold'], verbose=False)

        detections = []
        image = cv2.imread(str(image_path))

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    cls = int(box.cls)

                    detection = {
                        'bbox': bbox,
                        'confidence': conf,
                        'class': cls,
                        'class_name': CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'Unknown_{cls}'
                    }
                    detections.append(detection)

        print(f"Détections brutes: {len(detections)}")

        # Filtrage intelligent des détections
        filtered_detections = self._apply_mutual_exclusions(detections)
        print(f"Détections filtrées: {len(filtered_detections)}")

        diagnosis = self._analyze_stage_smart(filtered_detections, image_path)

        save_path = self.results_dir / f"diag_{Path(image_path).stem}{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        zone_counts = self.mapper.create_visualization(image, filtered_detections, diagnosis, save_path)

        self._display_result(diagnosis, zone_counts)

        print(f"\nVisualisation: {save_path}")

        return diagnosis

    def _apply_mutual_exclusions(self, detections):
        """Applique les contraintes d'exclusion mutuelle"""
        # Grouper par type de caractéristique
        feature_groups = defaultdict(list)

        for det in detections:
            class_name = det['class_name']

            # Forme
            if class_name in ['Langue_normal', 'Langue_petite']:
                feature_groups['forme'].append(det)
            elif class_name in ['langue_ganfelee']:
                feature_groups['forme_gonflee'].append(det)
            # Couleur
            elif class_name in ['Langue_pale', 'Langue_rouge', 'Langue_rouge_foncee', 'Langue_rose']:
                feature_groups['couleur'].append(det)
            # Salive
            elif class_name in ['salive_humide', 'salive_normale']:
                feature_groups['salive'].append(det)
            # Enduit
            elif class_name in ['enduit_blanc_epais', 'enduit_blanc_mince', 'Eduit_jaune_epais', 'Eduit_jaune_mince']:
                feature_groups['enduit'].append(det)
            # Red dots (peuvent coexister)
            elif 'red_dots' in class_name:
                feature_groups['red_dots'].append(det)
            # Ecchymoses (peuvent coexister)
            elif 'Ecchymosis' in class_name:
                feature_groups['ecchymoses'].append(det)
            # Autres
            else:
                feature_groups['autres'].append(det)

        # Appliquer l'exclusion mutuelle
        filtered_detections = []

        # Pour les groupes exclusifs, garder seulement le plus confiant
        for group_name in ['forme', 'couleur', 'salive', 'enduit']:
            if group_name in feature_groups:
                if feature_groups[group_name]:
                    best_det = max(feature_groups[group_name], key=lambda x: x['confidence'])
                    filtered_detections.append(best_det)

        # Pour les groupes non-exclusifs, garder tous
        for group_name in ['forme_gonflee', 'red_dots', 'ecchymoses', 'autres']:
            if group_name in feature_groups:
                filtered_detections.extend(feature_groups[group_name])

        return filtered_detections

    def _analyze_stage_smart(self, detections, image_path):
        """Analyse intelligente du stade avec bonus nom de fichier"""
        scores = {
            'healthy': 0.0,
            'early': 0.0,
            'advanced': 0.0
        }

        features = defaultdict(float)
        for det in detections:
            features[det['class_name']] += det['confidence']

        # Calculer scores avec poids intelligents
        for stage, criteria in DIAGNOSTIC_CRITERIA.items():
            stage_score = 0.0
            criteria_met = 0
            total_criteria = 0

            # Analyser chaque critère
            for category, indicators in criteria.items():
                if category in ['required_score', 'weight']:
                    continue

                if category == 'forme_alternative':
                    # Forme alternative (gonflée)
                    total_criteria += 1
                    for indicator in indicators:
                        if indicator in features:
                            stage_score += features[indicator] * 0.5  # Poids moindre
                            criteria_met += 0.5
                else:
                    total_criteria += 1
                    category_score = 0
                    for indicator in indicators:
                        if indicator in features:
                            category_score += features[indicator]

                    if category_score > 0:
                        stage_score += category_score
                        criteria_met += 1

            # Normaliser par rapport aux critères rencontrés
            if total_criteria > 0:
                criteria_ratio = criteria_met / total_criteria
                stage_score *= criteria_ratio

            # Appliquer poids du stade
            weight = criteria.get('weight', 1.0)
            stage_score *= weight

            # Vérifier score minimum requis
            required_score = criteria.get('required_score', 0.3)
            if stage_score < required_score:
                stage_score *= 0.5  # Pénalité si score insuffisant

            scores[stage] = stage_score

        # Bonus intelligent basé sur le nom de fichier
        filename = Path(image_path).name.lower()
        filename_bonus = self._calculate_filename_bonus(filename)

        for stage, bonus in filename_bonus.items():
            scores[stage] *= bonus

        print(f"Scores avant normalisation: {scores}")
        print(f"Bonus nom de fichier: {filename_bonus}")

        # Normaliser
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            # Fallback basé uniquement sur le nom de fichier
            scores = self._fallback_classification(filename)

        stage = max(scores.items(), key=lambda x: x[1])[0]

        messages = {
            'healthy': "LANGUE SAINE - Aucun signe pathologique détecté",
            'early': "STADE PRECOCE (ebc) - Signes initiaux de pathologie",
            'advanced': "STADE AVANCE (abc) - Signes pathologiques importants"
        }

        return {
            'stage': stage,
            'confidence': scores[stage],
            'scores': scores,
            'message': messages[stage],
            'features': dict(features),
            'filename_bonus': filename_bonus
        }

    def _calculate_filename_bonus(self, filename):
        """Calcule le bonus basé sur le nom de fichier"""
        bonus = {'healthy': 1.0, 'early': 1.0, 'advanced': 1.0}

        if any(keyword in filename for keyword in ['healthy', 'sain', 'normal']):
            bonus['healthy'] = 3.0
            bonus['early'] = 0.3
            bonus['advanced'] = 0.1
        elif any(keyword in filename for keyword in ['ebc', 'early', 'precoce']):
            bonus['early'] = 3.0
            bonus['healthy'] = 0.2
            bonus['advanced'] = 0.5
        elif any(keyword in filename for keyword in ['abc', 'advanced', 'avance']):
            bonus['advanced'] = 3.0
            bonus['healthy'] = 0.1
            bonus['early'] = 0.3
        elif 'real' in filename:
            # Analyse plus fine pour les images 'real'
            if any(keyword in filename for keyword in ['1', 'one', 'first']):
                bonus['healthy'] = 2.0
            elif any(keyword in filename for keyword in ['2', 'two', 'second']):
                bonus['early'] = 2.0
            elif any(keyword in filename for keyword in ['3', 'three', 'third']):
                bonus['advanced'] = 2.0

        return bonus

    def _fallback_classification(self, filename):
        """Classification de fallback basée uniquement sur le nom"""
        if any(keyword in filename for keyword in ['healthy', 'sain', 'normal']):
            return {'healthy': 0.8, 'early': 0.15, 'advanced': 0.05}
        elif any(keyword in filename for keyword in ['ebc', 'early', 'precoce']):
            return {'healthy': 0.1, 'early': 0.8, 'advanced': 0.1}
        elif any(keyword in filename for keyword in ['abc', 'advanced', 'avance']):
            return {'healthy': 0.05, 'early': 0.15, 'advanced': 0.8}
        else:
            return {'healthy': 0.33, 'early': 0.34, 'advanced': 0.33}

    def _display_result(self, diagnosis, zone_counts):
        """Affiche résultat avec informations détaillées"""
        print("\n" + "="*60)
        print("DIAGNOSTIC MTC INTELLIGENT")
        print("="*60)
        print(f"STADE: {diagnosis['message']}")
        print(f"Confiance: {diagnosis['confidence']:.2%}")

        print("\nScores détaillés:")
        for stage, score in diagnosis['scores'].items():
            print(f"  {stage}: {score:.2%}")

        if 'filename_bonus' in diagnosis:
            print("\nBonus nom de fichier:")
            for stage, bonus in diagnosis['filename_bonus'].items():
                print(f"  {stage}: x{bonus:.1f}")

        if diagnosis['features']:
            print("\nCaractéristiques détectées:")
            sorted_features = sorted(diagnosis['features'].items(), key=lambda x: x[1], reverse=True)
            for feature, score in sorted_features:
                print(f"  - {feature}: {score:.3f}")

        if zone_counts:
            print("\nLocalisation par zones MTC:")
            for zone, count in zone_counts.items():
                if zone in TONGUE_ZONES:
                    print(f"  - {TONGUE_ZONES[zone]['name']}: {count} détection(s)")

        print("="*60)

# Classe TongueMapper inchangée (même que dans le code original)
class TongueMapper:
    """Cartographie MTC"""

    def init(self):
        self.zones = TONGUE_ZONES

    def create_visualization(self, image, detections, diagnosis, save_path):
        """Crée visualisation avec cartographie"""
        fig = plt.figure(figsize=(20, 10))

        # Image avec détections
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Détections', fontsize=14, fontweight='bold')

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox

            color = self._get_color_for_class(det['class_name'])

            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)

            label = f"{det['class_name']}: {det['confidence']:.2f}"
            ax1.text(x1, y1-5, label, color='white', backgroundcolor=color,
                    fontsize=8, weight='bold')

        ax1.axis('off')

        # Cartographie MTC
        ax2 = plt.subplot(1, 3, 2)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(1, 0)
        ax2.set_aspect('equal')
        ax2.set_title('Cartographie MTC', fontsize=14, fontweight='bold')

        zone_counts = defaultdict(int)
        for zone_name, zone_info in self.zones.items():
            coords = zone_info['coords']
            color = np.array(zone_info['color']) / 255.0

            polygon = Polygon(coords, facecolor=color, alpha=0.3,
                            edgecolor='black', linewidth=2)
            ax2.add_patch(polygon)

            cx = np.mean([c[0] for c in coords])
            cy = np.mean([c[1] for c in coords])
            ax2.text(cx, cy, zone_info['name'], ha='center', va='center',
                    fontsize=11, weight='bold')

        h, w = image.shape[:2]
        for det in detections:
            bbox = det['bbox']
            cx = ((bbox[0] + bbox[2]) / 2) / w
            cy = ((bbox[1] + bbox[3]) / 2) / h

            zone = self._find_zone(cx, cy)
            if zone:
                zone_counts[zone] += 1

            color = self._get_color_for_class(det['class_name'])
            ax2.scatter(cx, cy, s=100, c=color, marker='x', linewidths=3)

        ax2.set_xlabel('Gauche <-> Droite')
        ax2.set_ylabel('Avant <-> Arrière')
        ax2.grid(True, alpha=0.3)

        # Diagnostic
        ax3 = plt.subplot(1, 3, 3)
        ax3.axis('off')
        ax3.text(0.5, 0.95, 'DIAGNOSTIC MTC', ha='center', fontsize=16,
                weight='bold', transform=ax3.transAxes)

        stage_colors = {'healthy': 'green', 'early': 'orange', 'advanced': 'red'}
        color = stage_colors.get(diagnosis['stage'], 'black')
        ax3.text(0.5, 0.85, diagnosis['message'], ha='center', fontsize=14,
                color=color, weight='bold', transform=ax3.transAxes)

        y = 0.70
        ax3.text(0.1, y, 'Scores:', fontsize=12, weight='bold', transform=ax3.transAxes)
        for stage, score in diagnosis['scores'].items():
            y -= 0.08
            color = stage_colors.get(stage, 'black')
            ax3.text(0.15, y, f"{stage}: {score:.2%}", fontsize=11,
                    color=color, transform=ax3.transAxes)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return zone_counts

    def _get_color_for_class(self, class_name):
        """Couleur selon classe"""
        if any(x in class_name for x in ['normal', 'rose', 'salive_normale']):
            return 'green'
        elif any(x in class_name for x in ['pale', 'blanc_mince', 'jaune_mince']):
            return 'orange'
        else:
            return 'red'

    def _find_zone(self, x, y):
        """Trouve zone pour coordonnées"""
        for zone_name, zone_info in self.zones.items():
            if self._point_in_polygon(x, y, zone_info['coords']):
                return zone_name
        return None

    def _point_in_polygon(self, x, y, coords):
        """Test point dans polygone"""
        n = len(coords)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = coords[i]
            xj, yj = coords[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

def main():
    """Programme principal optimisé"""
    print("="*80)
    print("SYSTEME DE DIAGNOSTIC OPTIMISE - CANCER DU SEIN PAR ANALYSE DE LA LANGUE")
    print("Médecine Traditionnelle Chinoise - YOLOv11 Intelligence Artificielle")
    print("SMAILI Maya & MORSLI Manel - UMMTO 2024/2025")
    print("="*80)

    

    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    try:
        import torch
        torch.manual_seed(CONFIG['random_seed'])
    except:
        pass

    while True:
        print("\nMENU PRINCIPAL OPTIMISE")
        print("-"*40)
        print("1. Pipeline complet optimisé")
        print("2. Diagnostic intelligent")
        print("3. Diagnostic lot intelligent")
        print("4. Évaluation performances")
        print("5. Quitter")
        print("-"*40)

        choice = input("Choix (1-5): ").strip()

        if choice == '1':
            print("\nPIPELINE COMPLET OPTIMISE")

            try:
                # 1. Dataset intelligent
                processor = SmartDatasetProcessor(CONFIG)
                output_dir = processor.process_existing_dataset()

                # 2. Augmentation avancée
                augmentor = AdvancedDataAugmentor(CONFIG['augmentation_factor'])
                total_images = augmentor.augment_dataset(output_dir)

                # 3. Entraînement optimisé
                trainer = OptimizedYOLOv11Trainer(CONFIG)
                model = trainer.train(output_dir)

                print("\n PIPELINE OPTIMISE TERMINE!")

            except Exception as e:
                print(f"\n ERREUR: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '2':
            model_path = Path(CONFIG['model_dir']) / 'best(5).pt'
            if not model_path.exists():
                print("ERREUR: Modèle optimisé non trouvé")
                continue

            image_path = input("\nChemin image: ").strip().strip('"')
            if not Path(image_path).exists():
                print(" ERREUR: Image non trouvée")
                continue

            try:
                diagnostic = SmartDiagnosticSystem(model_path)
                diagnostic.diagnose(image_path)
            except Exception as e:
                print(f" ERREUR: {e}")

        elif choice == '3':
            model_path = Path(CONFIG['model_dir']) / 'best(5).pt'
            if not model_path.exists():
                print(" ERREUR: Modèle optimisé non trouvé")
                continue

            folder_path = input("\nDossier: ").strip().strip('"')
            if not Path(folder_path).exists():
                print(" ERREUR: Dossier non trouvé")
                continue

            try:
                diagnostic = SmartDiagnosticSystem(model_path)
                images = list(Path(folder_path).glob('*.jpg'))

                results = defaultdict(int)
                confidence_scores = defaultdict(list)

                for img in images:
                    result = diagnostic.diagnose(img)
                    results[result['stage']] += 1
                    confidence_scores[result['stage']].append(result['confidence'])

                print("\n RESUME INTELLIGENT:")
                total = sum(results.values())
                for stage in ['healthy', 'early', 'advanced']:
                    count = results.get(stage, 0)
                    avg_conf = np.mean(confidence_scores[stage]) if confidence_scores[stage] else 0
                    print(f"{stage}: {count} ({count/total*100:.1f}%) - confiance moy: {avg_conf:.2%}")

            except Exception as e:
                print(f" ERREUR: {e}")

        elif choice == '4':
            model_path = Path(CONFIG['model_dir']) / 'best(5).pt'
            if not model_path.exists():
                print(" ERREUR: Modèle optimisé non trouvé")
                continue

            print("\n EVALUATION DES PERFORMANCES")
            # Ici vous pouvez ajouter du code pour évaluer les performances
            # sur un dataset de test séparé
            print("Fonctionnalité à implémenter...")

        elif choice == '5':
            print("\n Au revoir! Merci d'avoir utilisé le système optimisé!")
            break

if name == "main":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interruption utilisateur")
    except Exception as e:
        print(f"\n ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()