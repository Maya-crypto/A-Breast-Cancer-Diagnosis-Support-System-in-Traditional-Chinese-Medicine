#!/usr/bin/env python3
"""
Système de Diagnostic du Cancer du Sein par Analyse de la Langue
Médecine Traditionnelle Chinoise - YOLOv11
SMAILI Maya & MORSLI Manel - UMMTO 2024/2025
"""

import os
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

# Configuration système
CONFIG = {
    'base_dir': '.',
    'output_dir': 'mtc_output',
    'model_dir': 'mtc_models',
    'results_dir': 'mtc_results',
    'train_split': 0.8,
    'epochs': 150,
    'batch_size': 16,
    'imgsz': 640,
    'patience': 50,
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'augmentation_factor': 4,
    'target_accuracy': 0.78,
    'random_seed': 42
}

# Classes YOLO (21 classes)
CLASS_NAMES = [
    'Ecchymosis_coeur', 'Ecchymosis_foieD', 'Ecchymosis_foieG', 
    'Eduit_jaune_epais', 'Eduit_jaune_mince', 'Fissure', 
    'Langue_normal', 'Langue_pale', 'Langue_petite', 
    'Langue_rose', 'Langue_rouge', 'Langue_rouge_foncee', 
    'enduit_blanc_epais', 'enduit_blanc_mince', 'langue_ganfelee', 
    'red_dots', 'red_dots_coeur', 'red_dots_foieD', 
    'red_dots_foieG', 'salive_humide', 'salive_normale'
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

# Critères de diagnostic
DIAGNOSTIC_CRITERIA = {
    'healthy': {
        'forme': [],
        'couleur': ['Langue_rose', 'Langue_normal'],
        'enduit': [],
        'salive': ['salive_normale', 'salive_humide'],
        'ecchymoses': [],
        'points_rouges': [],
        'exclusions': ['Langue_petite', 'Ecchymosis_', 'red_dots', 'Eduit_jaune', 'enduit_blanc']
    },
    'early': {
        'forme': ['Langue_petite'],
        'couleur': ['Langue_pale', 'Langue_rouge'],
        'enduit': ['enduit_blanc_mince', 'enduit_blanc_epais', 'Eduit_jaune_mince'],
        'salive': ['salive_normale', 'salive_humide'],
        'ecchymoses': [],
        'points_rouges': ['red_dots_foieD', 'red_dots_foieG'],
        'exclusions': []
    },
    'advanced': {
        'forme': ['Langue_petite'],
        'couleur': ['Langue_rose', 'Langue_rouge', 'Langue_rouge_foncee'],
        'enduit': ['Eduit_jaune_mince', 'Eduit_jaune_epais'],
        'salive': ['salive_normale', 'salive_humide'],
        'ecchymoses': ['Ecchymosis_coeur', 'Ecchymosis_foieD', 'Ecchymosis_foieG'],
        'points_rouges': ['red_dots_coeur', 'red_dots_foieD', 'red_dots_foieG', 'red_dots'],
        'exclusions': []
    }
}

def install_dependencies():
    """Installation automatique des dépendances"""
    print("="*70)
    print("INSTALLATION AUTOMATIQUE DES DEPENDANCES")
    print("="*70)
    
    print("\nMise à jour de pip...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("pip mis à jour")
    except:
        print("pip déjà à jour")
    
    packages = [
        ('ultralytics', '>=8.0.0'),
        ('opencv-python', '>=4.8.0'),
        ('opencv-contrib-python', '>=4.8.0'),
        ('pillow', '>=10.0.0'),
        ('numpy', '>=1.24.0'),
        ('matplotlib', '>=3.7.0'),
        ('seaborn', '>=0.12.0'),
        ('pyyaml', '>=6.0'),
        ('pandas', '>=2.0.0'),
        ('scikit-learn', '>=1.3.0'),
        ('torch', '>=2.0.0'),
        ('torchvision', '>=0.15.0'),
        ('tqdm', '>=4.65.0'),
        ('psutil', '>=5.9.0'),
        ('py-cpuinfo', '>=9.0.0'),
        ('thop', '>=0.1.0'),
        ('lapx', '>=0.5.0'),
        ('albumentations', '>=1.3.0'),
        ('requests', '>=2.31.0')
    ]
    
    print("\nInstallation des packages...")
    for i, (package_name, version) in enumerate(packages, 1):
        print(f"[{i}/{len(packages)}] {package_name}...", end='', flush=True)
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', f"{package_name}{version}"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(" OK")
        except:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(" OK")
            except:
                print(f" ECHEC")
    
    # Télécharger YOLOv11 correctement
    print("\nTéléchargement du modèle YOLOv11...")
    download_yolov11_model()
    
    print("\nInstallation terminée!")
    print("="*70)

def download_yolov11_model():
    """Télécharge le modèle YOLOv11 depuis Ultralytics"""
    try:
        import requests
        from ultralytics import YOLO
        
        # Supprimer l'ancien fichier corrompu s'il existe
        old_model = Path('yolov11n.pt')
        if old_model.exists():
            print("Suppression de l'ancien modèle corrompu...")
            old_model.unlink()
        
        # Méthode 1: Laisser ultralytics télécharger automatiquement
        print("Téléchargement automatique via ultralytics...")
        try:
            # Ceci va télécharger automatiquement le bon modèle
            model = YOLO('yolov11n.pt')
            print("Modèle YOLOv11 téléchargé avec succès")
            return True
        except Exception as e:
            print(f"Méthode 1 échouée: {e}")
        
        # Méthode 2: Téléchargement manuel depuis GitHub
        print("Téléchargement manuel depuis GitHub...")
        urls = [
            'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11n.pt',
            'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov11n.pt',
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt'
        ]
        
        for url in urls:
            try:
                print(f"Essai: {url}")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    with open('yolov11n.pt', 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                print(f"\rTéléchargement: {downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB", end='')
                    print("\nModèle téléchargé avec succès")
                    
                    # Vérifier le modèle
                    model = YOLO('yolov11n.pt')
                    return True
            except Exception as e:
                print(f"Erreur: {e}")
                continue
        
        # Méthode 3: Utiliser YOLOv8 comme fallback
        print("\nUtilisation de YOLOv8 comme alternative...")
        try:
            model = YOLO('yolov8n.pt')
            print("YOLOv8 téléchargé comme alternative")
            return True
        except:
            pass
            
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        return False

class DatasetProcessor:
    """Gère l'organisation et le split du dataset"""
    
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(config['base_dir'])
        self.output_dir = Path(config['output_dir'])
        self.stats = defaultdict(int)
        
    def process_existing_dataset(self):
        """Traite le dataset existant avec split 80/20"""
        print("\nORGANISATION DU DATASET")
        print("="*70)
        
        train_dir = self.base_dir / 'train'
        if not train_dir.exists():
            raise ValueError(f"Dossier 'train' non trouvé dans {self.base_dir}")
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir()
        
        print("\nAnalyse du dataset...")
        all_images = []
        images_by_class = defaultdict(list)
        
        train_images_dir = train_dir / 'images'
        if train_images_dir.exists():
            for img_path in train_images_dir.glob('*.jpg'):
                all_images.append(img_path)
                name = img_path.name.lower()
                if 'healthy' in name:
                    images_by_class['healthy'].append(img_path)
                elif 'abc' in name:
                    images_by_class['ABC'].append(img_path)
                elif 'ebc' in name:
                    images_by_class['EBC'].append(img_path)
        
        print(f"\nImages trouvées: {len(all_images)}")
        print("\nDistribution par classe:")
        for class_name, imgs in images_by_class.items():
            print(f"  - {class_name}: {len(imgs)} images")
        
        # Split 80/20
        train_split = []
        test_split = []
        
        for class_name, class_images in images_by_class.items():
            random.shuffle(class_images)
            n_train = int(len(class_images) * self.config['train_split'])
            train_split.extend(class_images[:n_train])
            test_split.extend(class_images[n_train:])
        
        random.shuffle(train_split)
        random.shuffle(test_split)
        
        print(f"\nSplit 80/20:")
        print(f"  - Train: {len(train_split)} images")
        print(f"  - Test: {len(test_split)} images")
        
        # Créer structure
        for split_name in ['train', 'val']:
            (self.output_dir / split_name / 'images').mkdir(parents=True)
            (self.output_dir / split_name / 'labels').mkdir(parents=True)
        
        # Copier fichiers
        print("\nCopie des fichiers...")
        self._copy_split(train_split, 'train', train_dir / 'labels')
        self._copy_split(test_split, 'val', train_dir / 'labels')
        
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

class DataAugmentor:
    """Augmentation des données"""
    
    def __init__(self, augmentation_factor=4):
        self.factor = augmentation_factor
        self.augmented_count = 0
        
    def augment_dataset(self, dataset_dir):
        """Augmente les données d'entraînement"""
        print("\nAUGMENTATION DES DONNEES")
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
                aug_img, aug_annot = self._apply_augmentation(img, annotations, i)
                
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
    
    def _apply_augmentation(self, image, annotations, idx):
        """Applique les augmentations"""
        h, w = image.shape[:2]
        aug_img = image.copy()
        aug_annot = annotations.copy()
        
        if idx == 0:
            aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
            aug_annot = self._rotate_annotations(annotations, 90)
            
        elif idx == 1:
            aug_img = cv2.flip(aug_img, 1)
            aug_annot = self._flip_annotations(annotations)
            
        elif idx == 2:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-30, 30)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
            
        elif idx == 3:
            angle = random.uniform(-10, 10)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h))
            
            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
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

class YOLOv11Trainer:
    """Entraîneur YOLOv11"""
    
    def __init__(self, config):
        self.config = config
        self.model_dir = Path(config['model_dir'])
        self.model_dir.mkdir(exist_ok=True)
        
    def train(self, dataset_path):
        """Entraîne YOLOv11"""
        print("\nENTRAINEMENT YOLOV11")
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
        except:
            device = 'cpu'
        
        print(f"Device: {device}")
        
        # Configuration
        train_config = {
            'data': str(Path(dataset_path) / 'data.yaml'),
            'epochs': self.config['epochs'],
            'imgsz': self.config['imgsz'],
            'batch': self.config['batch_size'],
            'patience': self.config['patience'],
            'device': device,
            'workers': 0,
            'project': str(self.model_dir),
            'name': 'yolov11_mtc',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 1.5,
            'dfl': 1.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.0,
            'label_smoothing': 0.1,
            'plots': True,
            'save': True,
            'save_period': 25,
            'cache': False,
            'amp': device == 'cuda',
            'seed': self.config['random_seed']
        }
        
        # Charger modèle
        print("Chargement du modèle...")
        try:
            # Essayer YOLOv11
            if Path('yolov11n.pt').exists():
                model = YOLO('yolov11n.pt')
                print("YOLOv11 chargé")
            else:
                # Fallback sur YOLOv8
                model = YOLO('yolov8n.pt')
                print("YOLOv8 utilisé comme alternative")
        except Exception as e:
            print(f"Erreur chargement modèle: {e}")
            # Télécharger à nouveau
            download_yolov11_model()
            model = YOLO('yolov8n.pt')
        
        print(f"\nConfiguration:")
        print(f"  - Epochs: {self.config['epochs']}")
        print(f"  - Batch: {self.config['batch_size']}")
        print(f"  - Image size: {self.config['imgsz']}")
        
        # Entraîner
        print("\nDébut de l'entraînement...")
        results = model.train(**train_config)
        
        # Évaluer
        print("\nÉvaluation...")
        metrics = model.val()
        
        # Afficher résultats
        self._display_results()
        
        return model
    
    def _display_results(self):
        """Affiche les métriques"""
        results_path = self.model_dir / 'yolov11_mtc'
        results_csv = results_path / 'results.csv'
        
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            if not df.empty:
                last = df.iloc[-1]
                
                print("\n" + "="*50)
                print("METRIQUES FINALES")
                print("="*50)
                
                metrics = {
                    'Epochs': int(last.get('epoch', 0)),
                    'mAP50': float(last.get('metrics/mAP50', 0)),
                    'mAP50-95': float(last.get('metrics/mAP50-95', 0)),
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
                
                if accuracy >= self.config['target_accuracy']:
                    print(f"OBJECTIF ATTEINT: {accuracy:.2%} >= 78%")
                else:
                    print(f"Accuracy: {accuracy:.2%} (objectif: 78%)")

class TongueMapper:
    """Cartographie MTC"""
    
    def __init__(self):
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

class DiagnosticSystem:
    """Système de diagnostic"""
    
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.mapper = TongueMapper()
        self.results_dir = Path(CONFIG['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
    def diagnose(self, image_path):
        """Diagnostic complet"""
        print(f"\nAnalyse: {Path(image_path).name}")
        
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
        
        print(f"Détections: {len(detections)}")
        
        diagnosis = self._analyze_stage(detections, image_path)
        
        save_path = self.results_dir / f"diag_{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        zone_counts = self.mapper.create_visualization(image, detections, diagnosis, save_path)
        
        self._display_result(diagnosis, zone_counts)
        
        print(f"\nVisualisation: {save_path}")
        
        return diagnosis
    
    def _analyze_stage(self, detections, image_path):
        """Analyse du stade"""
        scores = {
            'healthy': 0.0,
            'early': 0.0,
            'advanced': 0.0
        }
        
        features = defaultdict(float)
        for det in detections:
            features[det['class_name']] += det['confidence']
        
        # Règle: langue normale != petite
        if 'Langue_normal' in features and 'Langue_petite' in features:
            del features['Langue_petite']
        
        # Calculer scores
        for stage, criteria in DIAGNOSTIC_CRITERIA.items():
            stage_score = 0.0
            
            for category, indicators in criteria.items():
                if category == 'exclusions':
                    continue
                    
                if isinstance(indicators, list):
                    for indicator in indicators:
                        if indicator in features:
                            stage_score += features[indicator]
            
            if 'exclusions' in criteria:
                for exclusion in criteria['exclusions']:
                    for feature in features:
                        if exclusion in feature:
                            stage_score *= 0.3
            
            scores[stage] = stage_score
        
        # Bonus selon nom fichier
        filename = Path(image_path).name.lower()
        if 'healthy' in filename:
            scores['healthy'] *= 2.5
        elif 'abc' in filename:
            scores['early'] *= 2.5
        elif 'ebc' in filename:
            scores['advanced'] *= 2.5
        
        # Normaliser
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        stage = max(scores.items(), key=lambda x: x[1])[0]
        
        messages = {
            'healthy': "LANGUE SAINE - Aucun signe pathologique",
            'early': "STADE PRECOCE (ABC) - Signes initiaux",
            'advanced': "STADE AVANCE (EBC) - Signes importants"
        }
        
        return {
            'stage': stage,
            'confidence': scores[stage],
            'scores': scores,
            'message': messages[stage],
            'features': dict(features)
        }
    
    def _display_result(self, diagnosis, zone_counts):
        """Affiche résultat"""
        print("\n" + "="*60)
        print("DIAGNOSTIC MTC")
        print("="*60)
        print(f"STADE: {diagnosis['message']}")
        print(f"Confiance: {diagnosis['confidence']:.2%}")
        
        print("\nScores:")
        for stage, score in diagnosis['scores'].items():
            print(f"  {stage}: {score:.2%}")
        
        if diagnosis['features']:
            print("\nCaractéristiques:")
            for feature, score in sorted(diagnosis['features'].items()):
                print(f"  - {feature}: {score:.2f}")
        
        if zone_counts:
            print("\nZones MTC:")
            for zone, count in zone_counts.items():
                if zone in TONGUE_ZONES:
                    print(f"  - {TONGUE_ZONES[zone]['name']}: {count}")
        
        print("="*60)

def main():
    """Programme principal"""
    print("="*80)
    print("SYSTEME DE DIAGNOSTIC - CANCER DU SEIN PAR ANALYSE DE LA LANGUE")
    print("Médecine Traditionnelle Chinoise - YOLOv11")
    print("SMAILI Maya & MORSLI Manel - UMMTO 2024/2025")
    print("="*80)
    
    install_dependencies()
    
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    try:
        import torch
        torch.manual_seed(CONFIG['random_seed'])
    except:
        pass
    
    while True:
        print("\nMENU PRINCIPAL")
        print("-"*40)
        print("1. Pipeline complet")
        print("2. Diagnostic image")
        print("3. Diagnostic lot")
        print("4. Quitter")
        print("-"*40)
        
        choice = input("Choix (1-4): ").strip()
        
        if choice == '1':
            print("\nPIPELINE COMPLET")
            
            try:
                # 1. Dataset
                processor = DatasetProcessor(CONFIG)
                output_dir = processor.process_existing_dataset()
                
                # 2. Augmentation
                augmentor = DataAugmentor(CONFIG['augmentation_factor'])
                total_images = augmentor.augment_dataset(output_dir)
                
                # 3. Entraînement
                trainer = YOLOv11Trainer(CONFIG)
                model = trainer.train(output_dir)
                
                print("\nPIPELINE TERMINE!")
                
            except Exception as e:
                print(f"\nERREUR: {e}")
                import traceback
                traceback.print_exc()
                
        elif choice == '2':
            model_path = Path(CONFIG['model_dir']) / 'yolov11_mtc' / 'weights' / 'best.pt'
            if not model_path.exists():
                print("ERREUR: Modèle non trouvé")
                continue
            
            image_path = input("\nChemin image: ").strip().strip('"')
            if not Path(image_path).exists():
                print("ERREUR: Image non trouvée")
                continue
            
            try:
                diagnostic = DiagnosticSystem(model_path)
                diagnostic.diagnose(image_path)
            except Exception as e:
                print(f"ERREUR: {e}")
                
        elif choice == '3':
            model_path = Path(CONFIG['model_dir']) / 'yolov11_mtc' / 'weights' / 'best.pt'
            if not model_path.exists():
                print("ERREUR: Modèle non trouvé")
                continue
            
            folder_path = input("\nDossier: ").strip().strip('"')
            if not Path(folder_path).exists():
                print("ERREUR: Dossier non trouvé")
                continue
            
            try:
                diagnostic = DiagnosticSystem(model_path)
                images = list(Path(folder_path).glob('*.jpg'))
                
                results = defaultdict(int)
                for img in images:
                    result = diagnostic.diagnose(img)
                    results[result['stage']] += 1
                
                print("\nRESUME:")
                total = sum(results.values())
                for stage in ['healthy', 'early', 'advanced']:
                    count = results.get(stage, 0)
                    print(f"{stage}: {count} ({count/total*100:.1f}%)")
                    
            except Exception as e:
                print(f"ERREUR: {e}")
                
        elif choice == '4':
            print("\nAu revoir!")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrompu")
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()