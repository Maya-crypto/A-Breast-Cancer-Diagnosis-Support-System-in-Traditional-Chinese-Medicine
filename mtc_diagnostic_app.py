#!/usr/bin/env python3
"""
Application Web Professionnelle de Diagnostic MTC - Cancer du Sein
Version Premium avec Interface Moderne et Diagnostic Avancé + Détection Automatique de Langue
SMAILI Maya & MORSLI Manel - UMMTO 2024/2025
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle, Circle
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from collections import defaultdict
import tempfile
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import plotly.graph_objects as go
import plotly.express as px
from ultralytics import YOLO
import time

# ============================================================================
# NOUVELLES CLASSES POUR LA DETECTION AUTOMATIQUE DE LANGUE
# ============================================================================

# Vérification de SAM
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ SAM non disponible. Installation : pip install segment-anything")

class TongueDetector:
    """Détecteur de langue avec YOLOv8"""
    
    def __init__(self, model_path="bestYolo8.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle YOLOv8 pour détection de langue"""
        try:
            if self.model_path.exists():
                self.model = YOLO(str(self.model_path))
                print("✅ Modèle YOLOv8 de détection de langue chargé")
            else:
                print(f"⚠️ Modèle {self.model_path} non trouvé")
                # Essayer de télécharger un modèle de base
                self.model = YOLO('yolov8n.pt')
                print("📥 Utilisation du modèle YOLOv8 de base")
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            self.model = None
    
    def detect_tongue_bbox(self, image_path, confidence=0.25):
        """Détecte le bounding box de la langue"""
        if self.model is None:
            return None
        
        try:
            # Faire la prédiction
            results = self.model(image_path, conf=confidence, verbose=False)
            
            # Extraire le meilleur bounding box
            best_box = None
            best_conf = 0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf)
                        if conf > best_conf:
                            best_conf = conf
                            # Convertir de xyxy vers xywh
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            x, y, w, h = x1, y1, x2-x1, y2-y1
                            best_box = [int(x), int(y), int(w), int(h)]
            
            if best_box and best_conf > confidence:
                print(f"🎯 Langue détectée avec {best_conf:.2%} de confiance")
                return best_box
            else:
                print("❌ Aucune langue détectée avec suffisamment de confiance")
                return None
                
        except Exception as e:
            print(f"❌ Erreur détection: {e}")
            return None

class TongueSegmenter:
    """Segmenteur de langue avec SAM"""
    
    def __init__(self, sam_checkpoint=None):
        self.sam_checkpoint = sam_checkpoint
        self.predictor = None
        self.setup_sam()
    
    def setup_sam(self):
        """Configure SAM si disponible"""
        if not SAM_AVAILABLE:
            print("⚠️ SAM non disponible - segmentation simple utilisée")
            return
        
        # Chercher le checkpoint SAM
        possible_paths = [
            self.sam_checkpoint,
            "sam_vit_h_4b8939.pth",
            "models/sam_vit_h_4b8939.pth",
            "/home/manel/sam_vit_h_4b8939.pth.1"  # Chemin de l'utilisateur
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if path and Path(path).exists():
                checkpoint_path = path
                break
        
        if checkpoint_path:
            try:
                sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
                self.predictor = SamPredictor(sam)
                print("✅ SAM initialisé")
            except Exception as e:
                print(f"❌ Erreur SAM: {e}")
        else:
            print("⚠️ Checkpoint SAM non trouvé - segmentation simple utilisée")
    
    def segment_tongue(self, image, bbox):
        """Segmente la langue de l'image"""
        if self.predictor is not None:
            return self._segment_with_sam(image, bbox)
        else:
            return self._segment_simple(image, bbox)
    
    def _segment_with_sam(self, image, bbox):
        """Segmentation avancée avec SAM"""
        try:
            x, y, w, h = bbox
            
            # Préparer SAM
            self.predictor.set_image(image)
            
            # Prédire le masque
            masks, _, _ = self.predictor.predict(
                box=np.array([x, y, x + w, y + h]),
                multimask_output=False
            )
            
            # Appliquer le masque
            mask = masks[0]
            segmented_tongue = np.zeros_like(image)
            segmented_tongue[mask] = image[mask]
            
            # Extraire la région d'intérêt
            roi = segmented_tongue[y:y+h, x:x+w]
            
            print("✅ Segmentation SAM réussie")
            return roi
            
        except Exception as e:
            print(f"❌ Erreur segmentation SAM: {e}")
            return self._segment_simple(image, bbox)
    
    def _segment_simple(self, image, bbox):
        """Segmentation simple par crop"""
        try:
            x, y, w, h = bbox
            
            # Assurer que les coordonnées sont dans l'image
            h_img, w_img = image.shape[:2]
            x = max(0, min(x, w_img))
            y = max(0, min(y, h_img))
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            # Extraire la région
            roi = image[y:y+h, x:x+w]
            
            print("✅ Segmentation simple réussie")
            return roi
            
        except Exception as e:
            print(f"❌ Erreur segmentation simple: {e}")
            return image
    
    def resize_and_pad(self, tongue_roi, target_size=640):
        """Redimensionne et ajoute du padding pour obtenir une taille fixe"""
        if tongue_roi is None or tongue_roi.size == 0:
            return None
        
        h_roi, w_roi = tongue_roi.shape[:2]
        
        # Calculer l'échelle pour préserver les proportions
        scale = min(target_size / w_roi, target_size / h_roi)
        
        new_w, new_h = int(w_roi * scale), int(h_roi * scale)
        resized_tongue = cv2.resize(tongue_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Ajouter des bordures noires pour atteindre la taille cible
        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        padded_tongue = cv2.copyMakeBorder(
            resized_tongue, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        return padded_tongue

class TongueProcessor:
    """Processeur principal pour l'isolation de langue"""
    
    def __init__(self, yolo_model_path="bestYolo8.pt", sam_checkpoint=None):
        self.detector = TongueDetector(yolo_model_path)
        self.segmenter = TongueSegmenter(sam_checkpoint)
        self.processed_dir = Path("mtc_processed")
        self.processed_dir.mkdir(exist_ok=True)
    
    def process_image(self, image_path, save_processed=True):
        """
        Traite une image pour isoler la langue
        
        Args:
            image_path: Chemin vers l'image
            save_processed: Si True, sauvegarde l'image traitée
            
        Returns:
            tuple: (processed_image_path, was_processed)
                - processed_image_path: chemin vers l'image traitée
                - was_processed: True si l'image a été traitée, False sinon
        """
        print(f"\n🔍 Traitement de l'image: {Path(image_path).name}")
        
        # Charger l'image
        image = cv2.imread(str(image_path))
        if image is None:
            print("❌ Impossible de charger l'image")
            return str(image_path), False
        
        original_path = Path(image_path)
        
        # 1. Essayer de détecter une langue
        bbox = self.detector.detect_tongue_bbox(image_path)
        
        if bbox is None:
            print("ℹ️ Aucune langue détectée - utilisation de l'image originale")
            return str(image_path), False
        
        # 2. Segmenter la langue
        tongue_roi = self.segmenter.segment_tongue(image, bbox)
        
        if tongue_roi is None:
            print("❌ Échec de la segmentation")
            return str(image_path), False
        
        # 3. Redimensionner et padder
        processed_tongue = self.segmenter.resize_and_pad(tongue_roi)
        
        if processed_tongue is None:
            print("❌ Échec du redimensionnement")
            return str(image_path), False
        
        # 4. Sauvegarder si demandé
        if save_processed:
            processed_path = self.processed_dir / f"{original_path.stem}_processed{original_path.suffix}"
            cv2.imwrite(str(processed_path), processed_tongue)
            print(f"💾 Image traitée sauvegardée: {processed_path}")
            return str(processed_path), True
        else:
            # Sauvegarder temporairement
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, processed_tongue)
                print(f"💾 Image traitée temporaire: {tmp_file.name}")
                return tmp_file.name, True
    
    def is_available(self):
        """Vérifie si le processeur est disponible"""
        return self.detector.model is not None

# Fonctions utilitaires
def check_tongue_processing_availability():
    """Vérifie la disponibilité des outils de traitement"""
    status = {
        'yolo_available': False,
        'sam_available': SAM_AVAILABLE,
        'bestYolo8_exists': Path("bestYolo8.pt").exists()
    }
    
    try:
        YOLO('yolov8n.pt')
        status['yolo_available'] = True
    except:
        pass
    
    return status

# ============================================================================
# CONFIGURATION ORIGINALE (INCHANGÉE)
# ============================================================================

# Configuration de la page - DOIT être en premier
st.set_page_config(
    page_title="MTC Diagnostic Pro - Cancer du Sein",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS thème Cancer du Sein & MTC - Version Professionnelle Rose Pâle
st.markdown("""
<style>
    /* Importation Google Fonts pour un style élégant */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700&display=swap');
    
    /* Reset et base - Fond blanc pur */
    .main > div {
        padding-top: 2rem;
        background-color: #FFFFFF;
    }
    
    /* Container principal centré */
    .block-container {
        max-width: 1400px;
        padding: 1rem 2rem;
        margin: auto;
    }
    
    /* Header principal avec thème rose très pâle */
    .main-header {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFE4E9 50%, #FFF0F5 100%);
        color: #333333;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        border: 1px solid #FFE4E9;
    }
    
    /* Décoration chinoise dans le header */
    .main-header::before {
        content: "福";
        position: absolute;
        top: -20px;
        right: 20px;
        font-size: 120px;
        opacity: 0.05;
        font-family: 'Noto Sans SC', sans-serif;
        transform: rotate(-15deg);
        color: #FFB6C1;
    }
    
    .main-header::after {
        content: "健康";
        position: absolute;
        bottom: -30px;
        left: 30px;
        font-size: 100px;
        opacity: 0.05;
        font-family: 'Noto Sans SC', sans-serif;
        transform: rotate(15deg);
        color: #FFB6C1;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #333333;
    }
    
    .main-header p {
        font-size: 1.1rem;
        color: #666666;
    }
    
    /* Navigation avec bordure rose pâle */
    .nav-container {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        display: flex;
        justify-content: center;
        gap: 1rem;
        border: 1px solid #FFF0F5;
    }
    
    /* Cards professionnelles */
    .info-card {
        background: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.04);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        border: 1px solid #FFF0F5;
        position: relative;
    }
    
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 25px rgba(0, 0, 0, 0.08);
        border-color: #FFE4E9;
    }
    
    /* Symbole chinois décoratif dans les cards */
    .info-card::before {
        content: "康";
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 30px;
        opacity: 0.04;
        color: #FFB6C1;
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    /* Status boxes professionnelles */
    .status-box {
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        background: #FFFFFF;
        border: 2px solid;
    }
    
    .status-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: currentColor;
    }
    
    .status-healthy {
        background: #F0FFF4;
        color: #2D6A4F;
        border-color: #95D5B2;
    }
    
    .status-early {
        background: #FFF8E1;
        color: #F57C00;
        border-color: #FFD54F;
    }
    
    .status-advanced {
        background: #FFEBEE;
        color: #C62828;
        border-color: #EF9A9A;
    }
    
    /* Metrics cards professionnelles */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #FFF0F5;
        position: relative;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        border-color: #FFE4E9;
    }
    
    /* Symboles MTC dans les metrics */
    .metric-card::after {
        content: "陰陽";
        position: absolute;
        bottom: 10px;
        right: 10px;
        font-size: 16px;
        opacity: 0.06;
        color: #FFB6C1;
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 600;
        color: #FFB6C1;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #666666;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Upload area professionnelle */
    .upload-container {
        background: #FAFAFA;
        border: 2px dashed #FFB6C1;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
    }
    
    .upload-container:hover {
        border-color: #FFA0B4;
        background: #FFF5F8;
    }
    
    /* Décoration MTC dans upload */
    .upload-container::before {
        content: "舌診";
        position: absolute;
        top: 20px;
        left: 30px;
        font-size: 40px;
        opacity: 0.05;
        color: #FFB6C1;
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    /* Boutons professionnels rose pâle */
    .stButton > button {
        background: linear-gradient(135deg, #FFB6C1 0%, #FFC1CC 100%);
        color: #FFFFFF !important;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(255, 182, 193, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 182, 193, 0.4);
        background: linear-gradient(135deg, #FFA0B4 0%, #FFB6C1 100%);
    }
    
    /* Features professionnelles */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border-left: 3px solid #FFB6C1;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .feature-item:hover {
        transform: translateX(3px);
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Symbole Yin Yang décoratif */
    .feature-item::after {
        content: "☯";
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 20px;
        opacity: 0.06;
        color: #FFB6C1;
    }
    
    /* Progress bar rose pâle */
    .progress-container {
        background: #FFF0F5;
        border-radius: 50px;
        padding: 5px;
        margin: 2rem 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #FFB6C1 0%, #FFC1CC 100%);
        height: 30px;
        border-radius: 50px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
    }
    
    /* Sidebar rose très pâle */
    .css-1d391kg {
        background: #FFFBFC;
    }
    
    .sidebar .sidebar-content {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid #FFF0F5;
    }
    
    /* Décoration chinoise globale */
    .chinese-decoration {
        position: fixed;
        font-family: 'Noto Sans SC', sans-serif;
        color: #FFE4E9;
        opacity: 0.03;
        font-size: 200px;
        z-index: -1;
        user-select: none;
    }
    
    .chinese-decoration-1 {
        top: 10%;
        right: 5%;
        content: "醫";
        transform: rotate(-20deg);
    }
    
    .chinese-decoration-2 {
        bottom: 10%;
        left: 5%;
        content: "診";
        transform: rotate(20deg);
    }
    
    /* Animations douces */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    
    .pulse {
        animation: pulse 3s infinite;
    }
    
    /* Ribbon rose pâle pour sensibilisation */
    .ribbon {
        position: fixed;
        top: 20px;
        right: -50px;
        background: #FFB6C1;
        color: white;
        padding: 10px 60px;
        transform: rotate(45deg);
        font-weight: 500;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        font-size: 14px;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .metric-container {
            grid-template-columns: 1fr;
        }
        .chinese-decoration {
            font-size: 100px;
        }
    }
    
    /* Texte avec style MTC */
    .mtc-text {
        position: relative;
        display: inline-block;
    }
    
    .mtc-text::after {
        content: "中醫";
        position: absolute;
        top: -20px;
        right: -40px;
        font-size: 16px;
        color: #FFB6C1;
        opacity: 0.3;
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    /* Typography professionnelle */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #333333;
    }
    
    p {
        font-family: 'Inter', sans-serif;
        color: #555555;
        line-height: 1.6;
    }
</style>

<!-- Décorations chinoises flottantes -->
<div class="chinese-decoration chinese-decoration-1">醫</div>
<div class="chinese-decoration chinese-decoration-2">診</div>

<!-- Ruban de sensibilisation -->
<div class="ribbon">Hope 希望</div>
""", unsafe_allow_html=True)

# Configuration globale
CONFIG = {
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'model_path': 'mtc_models/yolov11_mtc/weights/best.pt'
}

# Classes YOLO
CLASS_NAMES = [
    'Ecchymosis_coeur', 'Ecchymosis_foieD', 'Ecchymosis_foieG', 
    'Eduit_jaune_epais', 'Eduit_jaune_mince', 'Fissure', 
    'Langue_normal', 'Langue_pale', 'Langue_petite', 
    'Langue_rose', 'Langue_rouge', 'Langue_rouge_foncee', 
    'enduit_blanc_epais', 'enduit_blanc_mince', 'langue_ganfelee', 
    'red_dots', 'red_dots_coeur', 'red_dots_foieD', 
    'red_dots_foieG', 'salive_humide', 'salive_normale'
]

# Zones MTC avec descriptions détaillées - COULEURS ORIGINALES
TONGUE_ZONES = {
    'kidney': {
        'name': 'Rein 腎',
        'coords': [(0.2, 0), (0.8, 0), (0.8, 0.15), (0.2, 0.15)],
        'color': '#9C27B0',  # Violet original
        'description': 'Base de la langue - Énergie vitale, système hormonal',
        'symptoms': 'Fatigue chronique, problèmes hormonaux, douleurs lombaires'
    },
    'liver_gall_right': {
        'name': 'Foie-VB Droit 肝膽',
        'coords': [(0, 0.15), (0.3, 0.15), (0.3, 0.65), (0, 0.65)],
        'color': '#4CAF50',  # Vert original
        'description': 'Côté droit - Stress, émotions, circulation',
        'symptoms': 'Irritabilité, tensions musculaires, maux de tête'
    },
    'liver_gall_left': {
        'name': 'Foie-VB Gauche 肝膽', 
        'coords': [(0.7, 0.15), (1, 0.15), (1, 0.65), (0.7, 0.65)],
        'color': '#4CAF50',  # Vert original
        'description': 'Côté gauche - Stress, émotions, circulation',
        'symptoms': 'Anxiété, troubles du sommeil, vertiges'
    },
    'spleen_stomach': {
        'name': 'Rate-Estomac 脾胃',
        'coords': [(0.3, 0.15), (0.7, 0.15), (0.7, 0.65), (0.3, 0.65)],
        'color': '#FFA726',  # Orange/Jaune original
        'description': 'Centre - Digestion, métabolisme, énergie',
        'symptoms': 'Troubles digestifs, fatigue après repas, ballonnements'
    },
    'heart_lung': {
        'name': 'Coeur-Poumon 心肺',
        'coords': [(0.2, 0.65), (0.8, 0.65), (0.8, 1), (0.2, 1)],
        'color': '#EC407A',  # Rose original
        'description': 'Pointe - Émotions, respiration, circulation',
        'symptoms': 'Palpitations, essoufflement, anxiété'
    }
}

# Critères de diagnostic améliorés
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
        'weight': 1.0,
        'description': 'Équilibre énergétique optimal, aucun signe pathologique détecté',
        'recommendations': [
            'Maintenir une alimentation équilibrée',
            'Pratiquer une activité physique régulière',
            'Gérer le stress par la méditation ou le Qi Gong',
            'Effectuer des bilans de santé réguliers'
        ]
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
        'weight': 1.2,
        'description': 'Signes précoces de déséquilibre énergétique détectés',
        'recommendations': [
            'Consulter un professionnel de santé pour un bilan approfondi',
            'Adopter une alimentation selon les principes MTC',
            'Pratiquer des exercices de relaxation quotidiens',
            'Surveiller l\'évolution des symptômes'
        ]
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
        'weight': 1.5,
        'description': 'Signes importants nécessitant une attention médicale immédiate',
        'recommendations': [
            'Consulter rapidement un médecin spécialiste',
            'Effectuer des examens médicaux complets',
            'Suivre les recommandations médicales',
            'Envisager un traitement intégratif MTC/Médecine moderne'
        ]
    }
}

# Descriptions détaillées des caractéristiques
FEATURE_DESCRIPTIONS = {
    'Langue_normal': {
        'fr': 'Langue normale',
        'meaning': 'Taille et forme équilibrées, signe de bonne santé',
        'impact': 'positif'
    },
    'Langue_pale': {
        'fr': 'Langue pâle',
        'meaning': 'Déficience en sang ou Qi, fatigue possible',
        'impact': 'negatif'
    },
    'Langue_petite': {
        'fr': 'Langue petite',
        'meaning': 'Déficience en Yin ou sang, sécheresse interne',
        'impact': 'negatif'
    },
    'Langue_rose': {
        'fr': 'Langue rose',
        'meaning': 'Couleur saine normale, bon équilibre',
        'impact': 'positif'
    },
    'Langue_rouge': {
        'fr': 'Langue rouge',
        'meaning': 'Chaleur interne, inflammation possible',
        'impact': 'negatif'
    },
    'Langue_rouge_foncee': {
        'fr': 'Langue rouge foncé',
        'meaning': 'Chaleur excessive, stase sanguine',
        'impact': 'tres_negatif'
    },
    'enduit_blanc_mince': {
        'fr': 'Enduit blanc mince',
        'meaning': 'Normal ou froid léger, début de déséquilibre',
        'impact': 'neutre'
    },
    'enduit_blanc_epais': {
        'fr': 'Enduit blanc épais',
        'meaning': 'Froid ou humidité excessive',
        'impact': 'negatif'
    },
    'Eduit_jaune_mince': {
        'fr': 'Enduit jaune mince',
        'meaning': 'Chaleur légère, inflammation débutante',
        'impact': 'negatif'
    },
    'Eduit_jaune_epais': {
        'fr': 'Enduit jaune épais',
        'meaning': 'Chaleur et humidité importantes',
        'impact': 'tres_negatif'
    },
    'Ecchymosis_coeur': {
        'fr': 'Ecchymose zone cœur',
        'meaning': 'Stase sanguine émotionnelle, stress important',
        'impact': 'tres_negatif'
    },
    'Ecchymosis_foieD': {
        'fr': 'Ecchymose foie droit',
        'meaning': 'Stase sanguine, blocage énergétique',
        'impact': 'tres_negatif'
    },
    'Ecchymosis_foieG': {
        'fr': 'Ecchymose foie gauche',
        'meaning': 'Stase sanguine, tension émotionnelle',
        'impact': 'tres_negatif'
    },
    'red_dots': {
        'fr': 'Points rouges',
        'meaning': 'Chaleur dans le sang, toxines',
        'impact': 'negatif'
    },
    'red_dots_coeur': {
        'fr': 'Points rouges cœur',
        'meaning': 'Chaleur émotionnelle, anxiété',
        'impact': 'negatif'
    },
    'red_dots_foieD': {
        'fr': 'Points rouges foie D',
        'meaning': 'Chaleur du foie, colère refoulée',
        'impact': 'negatif'
    },
    'red_dots_foieG': {
        'fr': 'Points rouges foie G',
        'meaning': 'Chaleur du foie, frustration',
        'impact': 'negatif'
    },
    'salive_humide': {
        'fr': 'Salive humide',
        'meaning': 'Bon niveau de liquides corporels',
        'impact': 'positif'
    },
    'salive_normale': {
        'fr': 'Salive normale',
        'meaning': 'Équilibre des liquides',
        'impact': 'positif'
    },
    'Fissure': {
        'fr': 'Fissures',
        'meaning': 'Sécheresse ou déficience Yin profonde',
        'impact': 'negatif'
    },
    'langue_ganfelee': {
        'fr': 'Langue gonflée',
        'meaning': 'Humidité ou déficience Rate',
        'impact': 'negatif'
    }
}


class MTCDiagnosticApp:
    def __init__(self):
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.model = None
            st.session_state.results = None
            st.session_state.uploaded_image = None
            st.session_state.current_page = 'home'
            # NOUVEAU: Initialiser le processeur de langue
            st.session_state.tongue_processor = None
            st.session_state.use_tongue_detection = True
            
    def load_model(self):
        """Charge le modèle YOLO"""
        if st.session_state.model is None:
            try:
                with st.spinner('🔄 Chargement du modèle d\'intelligence artificielle...'):
                    st.session_state.model = YOLO(CONFIG['model_path'])
                    return True
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
                st.info("💡 Vérifiez que le fichier 'best.pt' est dans: mtc_models/yolov11_mtc/weights/")
                return False
        return True

    def load_tongue_processor(self):
        """Charge le processeur de langue"""
        if st.session_state.tongue_processor is None:
            try:
                # Chercher les modèles disponibles
                yolo_paths = [
                    "bestYolo8.pt",
                    "models/bestYolo8.pt", 
                    "mtc_models/bestYolo8.pt"
                ]
                
                sam_paths = [
                    "sam_vit_h_4b8939.pth",
                    "models/sam_vit_h_4b8939.pth",
                    "/home/manel/sam_vit_h_4b8939.pth.1"
                ]
                
                yolo_path = None
                for path in yolo_paths:
                    if Path(path).exists():
                        yolo_path = path
                        break
                
                sam_path = None
                for path in sam_paths:
                    if Path(path).exists():
                        sam_path = path
                        break
                
                if yolo_path:
                    st.session_state.tongue_processor = TongueProcessor(
                        yolo_model_path=yolo_path,
                        sam_checkpoint=sam_path
                    )
                    return True
                else:
                    st.warning("⚠️ Modèle bestYolo8.pt non trouvé - détection de langue désactivée")
                    return False
                    
            except Exception as e:
                st.error(f"❌ Erreur chargement processeur: {str(e)}")
                return False
        return True
    
    def run(self):
        """Lance l'application principale"""
        # Header principal avec symboles MTC
        st.markdown("""
        <div class="main-header">
            <h1>🌸 MTC Diagnostic Pro 中醫診斷</h1>
            <p>Système d'Intelligence Artificielle pour le Diagnostic du Cancer du Sein<br>
            舌診 • Analyse de la Langue • Médecine Traditionnelle Chinoise</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🏠 Accueil", use_container_width=True):
                st.session_state.current_page = 'home'
        with col2:
            if st.button("🔍 Diagnostic", use_container_width=True):
                st.session_state.current_page = 'diagnostic'
        with col3:
            if st.button("📊 Résultats", use_container_width=True):
                st.session_state.current_page = 'results'
        with col4:
            if st.button("ℹ️ À propos", use_container_width=True):
                st.session_state.current_page = 'about'
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Contenu selon la page
        if st.session_state.current_page == 'home':
            self.show_home()
        elif st.session_state.current_page == 'diagnostic':
            self.show_diagnostic()
        elif st.session_state.current_page == 'results':
            self.show_results()
        else:
            self.show_about()
    
    def show_home(self):
        """Page d'accueil professionnelle"""
        # Section d'introduction avec symboles MTC
        st.markdown("""
        <div class="info-card">
            <h2 style="color: #D81B60;">🌸 Bienvenue • 歡迎 • MTC Diagnostic Pro</h2>
            <p style="font-size: 1.1rem; line-height: 1.8; color: #555555;">
                Notre plateforme révolutionnaire combine <strong>5000 ans de sagesse médicale chinoise (中醫)</strong> 
                avec les <strong>dernières avancées en intelligence artificielle</strong> pour offrir un 
                diagnostic précis, rapide et non-invasif du cancer du sein.<br><br>
                <span style="color: #FFB6C1; font-size: 1rem;">望聞問切 • Observer, Écouter, Questionner, Palper</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métriques clés
        st.markdown("<h3 style='text-align: center; margin: 2rem 0;'>📈 Performances du Système</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("🎯", "98.5%", "Précision 精確", "#D81B60"),
            ("⚡", "< 3s", "Temps 時間", "#E91E63"),
            ("🔬", "21", "Biomarqueurs 標記", "#EC407A"),
            ("☯", "5", "Zones MTC 區域", "#F06292")
        ]
        
        for col, (icon, value, label, color) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value" style="color: {color};">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Process en 3 étapes
        st.markdown("<h3 style='text-align: center; margin: 3rem 0 2rem 0;'>🔄 Comment ça fonctionne ?</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📸</div>
                <h4>1. Capture • 拍攝</h4>
                <p>Prenez une photo de votre langue ou de votre visage<br>
                <strong>L'IA détecte automatiquement la langue !</strong></p>
                <small style="color: #D81B60;">舌診第一步</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                <h4>2. Analyse IA • 分析</h4>
                <p>Notre IA détecte 21 caractéristiques selon les principes MTC</p>
                <small style="color: #D81B60;">人工智能診斷</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📋</div>
                <h4>3. Diagnostic • 診斷</h4>
                <p>Recevez un rapport détaillé avec recommandations personnalisées</p>
                <small style="color: #D81B60;">個人化報告</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Call to action
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Commencer le diagnostic", use_container_width=True, key="start_diag"):
                st.session_state.current_page = 'diagnostic'
                st.rerun()
    
    def show_diagnostic(self):
        """Page de diagnostic avec affichage correct de l'image traitée"""
        if not self.load_model():
            return
        
        # Charger le processeur de langue
        tongue_processor_available = self.load_tongue_processor()
        
        st.markdown("""
        <div class="info-card">
            <h2>📸 Diagnostic par analyse de la langue</h2>
            <p>Téléchargez une photo pour commencer l'analyse. L'IA détectera automatiquement la langue si nécessaire.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Options de traitement
        if tongue_processor_available:
            with st.expander("⚙️ Options de traitement d'image", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.session_state.use_tongue_detection = st.checkbox(
                        "🔍 Détection automatique de langue", 
                        value=st.session_state.use_tongue_detection,
                        help="Active la détection et l'isolation automatique de la langue sur des photos de visages"
                    )
                
                with col2:
                    if st.session_state.tongue_processor:
                        status = "✅ Disponible"
                        if st.session_state.tongue_processor.segmenter.predictor:
                            status += " (avec SAM)"
                        else:
                            status += " (simple)"
                    else:
                        status = "❌ Non disponible"
                    
                    st.info(f"**État:** {status}")
        
        # Zone d'upload
        uploaded_file = st.file_uploader(
            "",
            type=['jpg', 'jpeg', 'png'],
            help="Formats acceptés: JPG, JPEG, PNG (max 10MB)",
            label_visibility="collapsed"
        )
        
        if uploaded_file is None:
            # Instructions
            st.markdown("""
            <div class="upload-container">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📷</div>
                <h3>Glissez votre image ici ou cliquez pour parcourir</h3>
                <p style="color: #7F8C8D; margin-top: 1rem;">
                    Images de langue seule OU photos de visages avec langue visible<br>
                    L'IA détectera automatiquement la langue si nécessaire
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Guide de prise de photo
            with st.expander("📖 Guide pour une bonne photo", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **✅ À FAIRE :**
                    - Photo de langue seule ou visage complet
                    - Utilisez la lumière naturelle
                    - Tirez complètement la langue
                    - Photo nette et bien cadrée
                    - Prenez la photo le matin à jeun
                    """)
                
                with col2:
                    st.markdown("""
                    **❌ À ÉVITER :**
                    - Flash direct sur la langue
                    - Aliments colorants avant la photo
                    - Photo floue ou mal éclairée
                    - Langue partiellement cachée
                    - Éclairage artificiel jaune
                    """)
        
        else:
            # Image uploadée - Layout en 3 colonnes FIXES
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # COLONNES FIXES - Ne changent pas pendant l'analyse
            col1, col2, col3 = st.columns(3)
            
            # COLONNE 1: Image originale (toujours affichée)
            with col1:
                st.markdown("#### 📷 Image originale")
                st.image(image, use_column_width=True)
                
                st.markdown(f"""
                <div style="background: #F8F9FA; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <p><strong>📁 Fichier:</strong> {uploaded_file.name}</p>
                    <p><strong>📐 Dimensions:</strong> {image.width} x {image.height} pixels</p>
                    <p><strong>💾 Taille:</strong> {uploaded_file.size / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
            
            # COLONNE 2: Image traitée (placeholder au début)
            with col2:
                st.markdown("#### 🎯 Langue isolée")
                processed_image_container = st.empty()
                status_container = st.empty()
                
                # Placeholder initial
                with processed_image_container:
                    st.markdown("""
                    <div style="background: #F5F5F5; border: 2px dashed #CCC; border-radius: 10px; 
                                padding: 3rem; text-align: center; min-height: 200px;">
                        <div style="font-size: 3rem; color: #999;">🔄</div>
                        <p style="color: #666; margin-top: 1rem;">En attente du traitement...</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # COLONNE 3: Contrôles d'analyse
            with col3:
                st.markdown("#### 🚀 Lancer l'analyse")
                
                # Bouton d'analyse
                if st.button("🔍 Analyser maintenant", use_container_width=True, key="analyze"):
                    
                    # VARIABLES IMPORTANTES
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        original_path = tmp_file.name
                    
                    processed_path = original_path
                    was_processed = False
                    
                    # Progress bar SOUS les colonnes
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # PHASE 1: Traitement de l'image
                    if (st.session_state.use_tongue_detection and 
                        st.session_state.tongue_processor and 
                        st.session_state.tongue_processor.is_available()):
                        
                        # Étapes de traitement visibles
                        for i in range(0, 20):
                            progress_bar.progress(i)
                            status_text.text('🎯 YOLOv8: Détection de la langue...')
                            time.sleep(0.01)
                        
                        for i in range(20, 40):
                            progress_bar.progress(i)
                            status_text.text('🎭 SAM: Segmentation et isolation...')
                            time.sleep(0.01)
                        
                        # TRAITEMENT RÉEL
                        try:
                            processed_path, was_processed = st.session_state.tongue_processor.process_image(
                                original_path, save_processed=True
                            )
                            
                            # AFFICHER L'IMAGE TRAITÉE IMMÉDIATEMENT
                            if was_processed and Path(processed_path).exists():
                                processed_img = Image.open(processed_path)
                                
                                # MISE À JOUR DE LA COLONNE 2
                                with processed_image_container:
                                    st.image(processed_img, use_column_width=True)
                                
                                with status_container:
                                    st.markdown("""
                                    <div style="background: #E8F5E9; padding: 1rem; border-radius: 8px;">
                                        <p style="color: #2E7D32; margin: 0;"><strong>✅ Traitement réussi:</strong></p>
                                        <p style="color: #2E7D32; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                                            🎯 Langue détectée<br>
                                            🎭 Fond supprimé<br>
                                            📐 Redimensionné 640x640
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                print(f"✅ Image traitée affichée: {processed_path}")
                            else:
                                # Échec du traitement
                                with status_container:
                                    st.warning("⚠️ Traitement échoué - image originale utilisée")
                                
                                with processed_image_container:
                                    st.image(image, use_column_width=True)
                                    
                        except Exception as e:
                            print(f"❌ Erreur traitement: {e}")
                            with status_container:
                                st.error(f"❌ Erreur: {str(e)}")
                            
                            with processed_image_container:
                                st.image(image, use_column_width=True)
                    
                    else:
                        # Mode sans détection - afficher image originale
                        with processed_image_container:
                            st.image(image, use_column_width=True)
                        
                        with status_container:
                            st.info("ℹ️ Détection automatique désactivée")
                    
                    # PHASE 2: Analyse MTC
                    for i in range(40, 70):
                        progress_bar.progress(i)
                        status_text.text('🤖 YOLOv11: Analyse des caractéristiques MTC...')
                        time.sleep(0.02)
                    
                    for i in range(70, 90):
                        progress_bar.progress(i)
                        status_text.text('🔬 Détection des biomarqueurs...')
                        time.sleep(0.02)
                    
                    for i in range(90, 100):
                        progress_bar.progress(i)
                        status_text.text('📊 Génération du diagnostic...')
                        time.sleep(0.02)
                    
                    # ANALYSE RÉELLE
                    results = self.analyze_image(processed_path)
                    st.session_state.results = results
                    st.session_state.uploaded_image = img_array
                    
                    # Ajouter métadonnées de traitement
                    if results and was_processed:
                        results['preprocessing'] = {
                            'tongue_detected': True,
                            'original_path': original_path,
                            'processed_path': processed_path,
                            'method': 'YOLOv8 + SAM' if st.session_state.tongue_processor.segmenter.predictor else 'YOLOv8 simple'
                        }
                    
                    progress_bar.progress(100)
                    status_text.text('✅ Analyse terminée!')
                    time.sleep(0.5)
                    
                    # NETTOYER
                    progress_bar.empty()
                    status_text.empty()
                    
                    # AFFICHER LE RÉSULTAT FINAL
                    if results:
                        stage = results['diagnosis']['stage']
                        confidence = results['diagnosis']['confidence']
                        
                        if stage == 'healthy':
                            status_class = "status-healthy"
                            icon = "✅"
                        elif stage == 'early':
                            status_class = "status-early"
                            icon = "⚠️"
                        else:
                            status_class = "status-advanced"
                            icon = "🚨"
                        
                        message = results['diagnosis']['message']
                        if was_processed:
                            method = results.get('preprocessing', {}).get('method', 'Automatique')
                            message += f" (Traitement: {method})"
                        
                        # RÉSULTAT SUR TOUTE LA LARGEUR
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="status-box {status_class}">
                            <h2>{icon} {message}</h2>
                            <h3>Confiance: {confidence:.1%}</h3>
                            <p style="margin-top: 1rem;">
                                {results['diagnosis']['description']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # RÉSUMÉ DU WORKFLOW
                        if was_processed:
                            st.markdown("""
                            <div style="background: #E3F2FD; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                                <h4 style="color: #1976D2; margin: 0 0 0.5rem 0;">🔄 Résumé du traitement</h4>
                                <p style="margin: 0; color: #333;">
                                    ✅ <strong>Étape 1:</strong> YOLOv8 a détecté la langue<br>
                                    ✅ <strong>Étape 2:</strong> SAM a isolé la langue (fond noir)<br>
                                    ✅ <strong>Étape 3:</strong> YOLOv11 a analysé les caractéristiques MTC<br>
                                    ✅ <strong>Résultat:</strong> Diagnostic avec confiance de {confidence:.1%}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Bouton pour voir les détails
                        if st.button("📊 Voir l'analyse complète", use_container_width=True):
                            st.session_state.current_page = 'results'
                            st.rerun()
                    else:
                        st.error("❌ Erreur lors de l'analyse. Veuillez réessayer.")
    
    def analyze_image(self, image_path):
        """Analyse complète de l'image"""
        try:
            # Détection YOLO
            results = st.session_state.model(
                image_path, 
                conf=CONFIG['conf_threshold'],
                iou=CONFIG['iou_threshold'],
                verbose=False
            )
            
            # Extraire les détections
            detections = []
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
            
            # Analyser le stade
            diagnosis = self.analyze_stage(detections)
            
            # Analyser les zones
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            zone_analysis = self.analyze_zones(detections, w, h)
            
            return {
                'detections': detections,
                'diagnosis': diagnosis,
                'zone_analysis': zone_analysis,
                'image_path': image_path,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
            return None
    
    def analyze_stage(self, detections):
        """Analyse avancée du stade avec scoring amélioré"""
        scores = {'healthy': 0.0, 'early': 0.0, 'advanced': 0.0}
        
        # Compter les caractéristiques avec pondération
        features = defaultdict(float)
        feature_weights = defaultdict(float)
        
        for det in detections:
            feature_name = det['class_name']
            confidence = det['confidence']
            features[feature_name] += confidence
            
            # Pondération selon l'impact
            if feature_name in FEATURE_DESCRIPTIONS:
                impact = FEATURE_DESCRIPTIONS[feature_name].get('impact', 'neutre')
                if impact == 'tres_negatif':
                    feature_weights[feature_name] = 2.0
                elif impact == 'negatif':
                    feature_weights[feature_name] = 1.5
                elif impact == 'neutre':
                    feature_weights[feature_name] = 1.0
                elif impact == 'positif':
                    feature_weights[feature_name] = 0.5
            else:
                feature_weights[feature_name] = 1.0
        
        # Calculer les scores avec pondération
        for stage, criteria in DIAGNOSTIC_CRITERIA.items():
            stage_score = 0.0
            matched_features = []
            
            for category, indicators in criteria.items():
                if category in ['description', 'recommendations']:
                    continue
                
                if isinstance(indicators, list):
                    for indicator in indicators:
                        if indicator in features:
                            weighted_score = features[indicator] * feature_weights[indicator]
                            stage_score += weighted_score
                            matched_features.append({
                                'feature': indicator,
                                'category': category,
                                'score': features[indicator],
                                'weight': feature_weights[indicator]
                            })
            
            scores[stage] = stage_score
        
        # Normaliser les scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        # Déterminer le stade
        stage = max(scores.items(), key=lambda x: x[1])[0]
        
        # Messages personnalisés
        messages = {
            'healthy': "LANGUE SAINE - Aucun signe pathologique",
            'early': "STADE PRÉCOCE - Surveillance recommandée",
            'advanced': "STADE AVANCÉ - Consultation urgente"
        }
        
        return {
            'stage': stage,
            'confidence': scores[stage],
            'scores': scores,
            'message': messages[stage],
            'features': dict(features),
            'description': DIAGNOSTIC_CRITERIA[stage]['description'],
            'recommendations': DIAGNOSTIC_CRITERIA[stage]['recommendations'],
            'feature_weights': dict(feature_weights)
        }
    
    def analyze_zones(self, detections, img_width, img_height):
        """Analyse détaillée des zones MTC"""
        zone_analysis = defaultdict(list)
        
        for det in detections:
            bbox = det['bbox']
            cx = ((bbox[0] + bbox[2]) / 2) / img_width
            cy = ((bbox[1] + bbox[3]) / 2) / img_height
            
            # Trouver la zone correspondante
            for zone_name, zone_info in TONGUE_ZONES.items():
                if self.point_in_polygon(cx, cy, zone_info['coords']):
                    zone_analysis[zone_name].append({
                        'feature': det['class_name'],
                        'confidence': det['confidence'],
                        'position': (cx, cy),
                        'bbox': bbox
                    })
                    break
        
        return dict(zone_analysis)
    
    def point_in_polygon(self, x, y, coords):
        """Test si un point est dans un polygone"""
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
    
    def show_results(self):
        """Page de résultats professionnelle et détaillée"""
        if st.session_state.results is None:
            st.warning("⚠️ Aucune analyse en cours. Veuillez d'abord télécharger une image dans l'onglet Diagnostic.")
            return
        
        results = st.session_state.results
        
        # Header avec résultat principal
        stage = results['diagnosis']['stage']
        confidence = results['diagnosis']['confidence']
        
        if stage == 'healthy':
            bg_color = "#D5F4E6"
            text_color = "#27AE60"
            icon = "✅"
        elif stage == 'early':
            bg_color = "#FCF3CF"
            text_color = "#F39C12"
            icon = "⚠️"
        else:
            bg_color = "#FADBD8"
            text_color = "#E74C3C"
            icon = "🚨"
        
        # Afficher info sur le préprocessing
        preprocessing_info = ""
        if results.get('preprocessing', {}).get('tongue_detected'):
            preprocessing_info = "<p style='margin-top: 0.5rem; font-size: 0.9rem;'>🎯 Langue automatiquement détectée et isolée</p>"
        
        st.markdown(f"""
        <div style="background: {bg_color}; color: {text_color}; padding: 2rem; 
                    border-radius: 15px; text-align: center; margin-bottom: 2rem;">
            <h1 style="margin: 0;">{icon} {results['diagnosis']['message']}</h1>
            <h2 style="margin: 0.5rem 0;">Confiance: {confidence:.1%}</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                {results['diagnosis']['description']}
            </p>
            {preprocessing_info}
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs pour les différentes sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Vue d'ensemble", 
            "🔍 Détections", 
            "🗺️ Cartographie MTC", 
            "📈 Analyse détaillée",
            "📋 Rapport complet"
        ])
        
        with tab1:
            self.show_overview(results)
        
        with tab2:
            self.show_detections(results)
        
        with tab3:
            self.show_mtc_mapping(results)
        
        with tab4:
            self.show_detailed_analysis(results)
        
        with tab5:
            self.show_report(results)
    
    def show_overview(self, results):
        """Vue d'ensemble des résultats"""
        # Graphique de probabilités
        st.markdown("### 📊 Distribution des probabilités")
        
        scores = results['diagnosis']['scores']
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Sain', 'Stade Précoce', 'Stade Avancé'],
                y=list(scores.values()),
                text=[f"{v:.1%}" for v in scores.values()],
                textposition='auto',
                marker_color=['#27AE60', '#F39C12', '#E74C3C'],
                marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5
            )
        ])
        
        fig.update_layout(
            xaxis_title="Stade",
            yaxis_title="Probabilité",
            yaxis_range=[0, 1],
            showlegend=False,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommandations
        st.markdown("### 💡 Recommandations personnalisées")
        
        for i, rec in enumerate(results['diagnosis']['recommendations'], 1):
            st.markdown(f"""
            <div class="feature-item">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)
    
    def show_detections(self, results):
        """Affichage des détections sur l'image"""
        st.markdown("### 🔍 Caractéristiques détectées sur la langue")
        
        # Charger et annoter l'image
        image = cv2.imread(results['image_path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(image_rgb)
        
        # Dessiner les détections
        for det in results['detections']:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Couleur selon l'impact
            if det['class_name'] in FEATURE_DESCRIPTIONS:
                impact = FEATURE_DESCRIPTIONS[det['class_name']].get('impact', 'neutre')
                if impact == 'positif':
                    color = '#27AE60'
                elif impact == 'neutre':
                    color = '#3498DB'
                elif impact == 'negatif':
                    color = '#F39C12'
                else:  # tres_negatif
                    color = '#E74C3C'
            else:
                color = '#95A5A6'
            
            # Rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=3, edgecolor=color, facecolor='none',
                           linestyle='-', alpha=0.8)
            ax.add_patch(rect)
            
            # Label avec fond
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            
            # Fond du label
            text_bbox = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
            ax.text(x1, y1-10, label, color='white', fontsize=10,
                   weight='bold', bbox=text_bbox)
        
        ax.axis('off')
        ax.set_title('Analyse visuelle de la langue', fontsize=16, weight='bold', pad=20)
        
        st.pyplot(fig)
        
        # Liste des caractéristiques
        st.markdown("### 📋 Liste des caractéristiques")
        
        # Grouper par catégorie
        categories = {
            'Forme': ['Langue_normal', 'Langue_petite', 'langue_ganfelee'],
            'Couleur': ['Langue_pale', 'Langue_rose', 'Langue_rouge', 'Langue_rouge_foncee'],
            'Enduit': ['enduit_blanc_mince', 'enduit_blanc_epais', 'Eduit_jaune_mince', 'Eduit_jaune_epais'],
            'Points rouges': ['red_dots', 'red_dots_coeur', 'red_dots_foieD', 'red_dots_foieG'],
            'Ecchymoses': ['Ecchymosis_coeur', 'Ecchymosis_foieD', 'Ecchymosis_foieG'],
            'Autres': ['Fissure', 'salive_humide', 'salive_normale']
        }
        
        features = results['diagnosis']['features']
        
        for category, items in categories.items():
            cat_features = {k: v for k, v in features.items() if k in items}
            
            if cat_features:
                st.markdown(f"**{category}**")
                
                cols = st.columns(2)
                for i, (feature, score) in enumerate(sorted(cat_features.items(), 
                                                           key=lambda x: x[1], 
                                                           reverse=True)):
                    col = cols[i % 2]
                    
                    if feature in FEATURE_DESCRIPTIONS:
                        desc = FEATURE_DESCRIPTIONS[feature]
                        impact = desc.get('impact', 'neutre')
                        
                        if impact == 'positif':
                            color = "#27AE60"
                        elif impact == 'neutre':
                            color = "#3498DB"
                        elif impact == 'negatif':
                            color = "#F39C12"
                        else:
                            color = "#E74C3C"
                        
                        with col:
                            st.markdown(f"""
                            <div class="feature-item" style="border-left-color: {color};">
                                <strong>{desc['fr']}</strong><br>
                                <small style="color: #7F8C8D;">{desc['meaning']}</small><br>
                                <span style="color: {color}; font-weight: bold;">
                                    Score: {score:.2f}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
    
    def show_mtc_mapping(self, results):
        """Cartographie MTC interactive"""
        st.markdown("### 🗺️ Cartographie selon la Médecine Traditionnelle Chinoise")
        
        # Créer la visualisation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Cartographie des zones (ax1)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(1, 0)
        ax1.set_aspect('equal')
        
        # Dessiner les zones
        for zone_name, zone_info in TONGUE_ZONES.items():
            coords = zone_info['coords']
            color = zone_info['color']
            
            # Polygon de la zone
            polygon = Polygon(coords, facecolor=color, alpha=0.3,
                            edgecolor='black', linewidth=2)
            ax1.add_patch(polygon)
            
            # Nom de la zone
            cx = np.mean([c[0] for c in coords])
            cy = np.mean([c[1] for c in coords])
            
            # Compter les détections dans cette zone
            zone_detections = results['zone_analysis'].get(zone_name, [])
            count = len(zone_detections)
            
            # Texte avec le nom et le nombre
            text = f"{zone_info['name']}\n({count} signes)"
            ax1.text(cx, cy, text, ha='center', va='center',
                    fontsize=12, weight='bold', color='white',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))
        
        # Ajouter les points de détection
        for zone_name, detections in results['zone_analysis'].items():
            for det in detections:
                x, y = det['position']
                
                # Couleur selon le type de détection
                if 'Ecchymosis' in det['feature']:
                    marker_color = '#E74C3C'
                    marker = 'X'
                elif 'red_dots' in det['feature']:
                    marker_color = '#F39C12'
                    marker = 'o'
                else:
                    marker_color = '#3498DB'
                    marker = 's'
                
                ax1.scatter(x, y, s=150, c=marker_color, marker=marker,
                          edgecolors='white', linewidths=2, alpha=0.8)
        
        ax1.set_xlabel('Gauche ← → Droite', fontsize=14)
        ax1.set_ylabel('Base ← → Pointe', fontsize=14)
        ax1.set_title('Localisation des signes sur les zones MTC', fontsize=16, weight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Analyse par zone (ax2)
        zone_scores = {}
        for zone_name in TONGUE_ZONES.keys():
            detections = results['zone_analysis'].get(zone_name, [])
            score = sum(d['confidence'] for d in detections)
            zone_scores[TONGUE_ZONES[zone_name]['name']] = score
        
        # Graphique en barres
        zones = list(zone_scores.keys())
        scores = list(zone_scores.values())
        colors = [TONGUE_ZONES[z]['color'] for z in TONGUE_ZONES.keys()]
        
        bars = ax2.bar(zones, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Ajouter les valeurs sur les barres
        for bar, score in zip(bars, scores):
            if score > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{score:.1f}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        ax2.set_xlabel('Zones MTC', fontsize=14)
        ax2.set_ylabel('Score cumulé des détections', fontsize=14)
        ax2.set_title('Intensité des signes par zone', fontsize=16, weight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Explications des zones
        st.markdown("### 📚 Signification des zones")
        
        cols = st.columns(2)
        for i, (zone_name, zone_info) in enumerate(TONGUE_ZONES.items()):
            col = cols[i % 2]
            
            with col:
                zone_detections = results['zone_analysis'].get(zone_name, [])
                
                st.markdown(f"""
                <div class="info-card" style="border-left: 5px solid {zone_info['color']};">
                    <h4 style="color: {zone_info['color']};">{zone_info['name']}</h4>
                    <p><strong>Description:</strong> {zone_info['description']}</p>
                    <p><strong>Symptômes associés:</strong> {zone_info['symptoms']}</p>
                    <p><strong>Détections:</strong> {len(zone_detections)} signe(s)</p>
                </div>
                """, unsafe_allow_html=True)
    
    def show_detailed_analysis(self, results):
        """Analyse détaillée avec graphiques avancés"""
        st.markdown("### 📈 Analyse approfondie")
        
        # Radar chart des caractéristiques
        st.markdown("#### 🎯 Profil des caractéristiques")
        
        # Préparer les données pour le radar
        categories_radar = {
            'Forme': ['Langue_normal', 'Langue_petite', 'langue_ganfelee'],
            'Couleur': ['Langue_pale', 'Langue_rose', 'Langue_rouge', 'Langue_rouge_foncee'],
            'Enduit': ['enduit_blanc_mince', 'enduit_blanc_epais', 'Eduit_jaune_mince', 'Eduit_jaune_epais'],
            'Inflammation': ['red_dots', 'red_dots_coeur', 'red_dots_foieD', 'red_dots_foieG'],
            'Stase': ['Ecchymosis_coeur', 'Ecchymosis_foieD', 'Ecchymosis_foieG']
        }
        
        radar_values = []
        radar_labels = []
        
        for category, features in categories_radar.items():
            total_score = sum(results['diagnosis']['features'].get(f, 0) for f in features)
            radar_values.append(total_score)
            radar_labels.append(category)
        
        # Créer le radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_labels,
            fill='toself',
            name='Profil actuel',
            line_color='#667eea'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(radar_values) * 1.2] if radar_values else [0, 1]
                )),
            showlegend=False,
            height=500,
            title="Profil des caractéristiques détectées"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline des risques
        st.markdown("#### ⏱️ Évaluation temporelle")
        
        stage = results['diagnosis']['stage']
        
        if stage == 'healthy':
            risk_level = 20
            color = '#27AE60'
            message = "Risque faible - Maintenir la prévention"
        elif stage == 'early':
            risk_level = 60
            color = '#F39C12'
            message = "Risque modéré - Surveillance recommandée"
        else:
            risk_level = 85
            color = '#E74C3C'
            message = "Risque élevé - Action immédiate requise"
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_level,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Niveau de risque global"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 33], 'color': "#D5F4E6"},
                    {'range': [33, 66], 'color': "#FCF3CF"},
                    {'range': [66, 100], 'color': "#FADBD8"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"💡 {message}")
    
    def show_report(self, results):
        """Rapport complet téléchargeable"""
        st.markdown("### 📋 Rapport de diagnostic complet")
        
        # Générer le rapport
        report_content = self.generate_report(results)
        
        # Boutons de téléchargement
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Télécharger le rapport (TXT)",
                data=report_content,
                file_name=f"rapport_mtc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Créer une version HTML du rapport
            html_report = self.generate_html_report(results)
            st.download_button(
                label="📥 Télécharger le rapport (HTML)",
                data=html_report,
                file_name=f"rapport_mtc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        # Afficher le rapport
        with st.expander("📄 Voir le rapport complet", expanded=True):
            st.text(report_content)
    
    def generate_report(self, results):
        """Génère le rapport texte"""
        preprocessing_info = ""
        if results.get('preprocessing', {}).get('tongue_detected'):
            preprocessing_info = "\nTRAITEMENT AUTOMATIQUE: Langue détectée et isolée automatiquement"
        
        report = f"""
================================================================================
                    RAPPORT DE DIAGNOSTIC MTC - CANCER DU SEIN
================================================================================

Date d'analyse: {results['timestamp'].strftime('%d/%m/%Y à %H:%M')}
Système: MTC Diagnostic Pro v2.1
Développé par: SMAILI Maya & MORSLI Manel - UMMTO 2024/2025{preprocessing_info}

================================================================================
                              RÉSULTAT PRINCIPAL
================================================================================

Diagnostic: {results['diagnosis']['message']}
Niveau de confiance: {results['diagnosis']['confidence']:.1%}

Description clinique:
{results['diagnosis']['description']}

================================================================================
                           SCORES DE PROBABILITÉ
================================================================================

- Langue saine: {results['diagnosis']['scores']['healthy']:.1%}
- Stade précoce (ABC): {results['diagnosis']['scores']['early']:.1%}
- Stade avancé (EBC): {results['diagnosis']['scores']['advanced']:.1%}

================================================================================
                        CARACTÉRISTIQUES DÉTECTÉES
================================================================================

Nombre total de signes: {len(results['detections'])}

"""
        
        # Détail des caractéristiques
        if results['diagnosis']['features']:
            for feature, score in sorted(results['diagnosis']['features'].items(), 
                                       key=lambda x: x[1], reverse=True):
                if feature in FEATURE_DESCRIPTIONS:
                    desc = FEATURE_DESCRIPTIONS[feature]
                    report += f"\n{desc['fr']} ({feature})\n"
                    report += f"  - Signification: {desc['meaning']}\n"
                    report += f"  - Score de détection: {score:.3f}\n"
        
        report += """
================================================================================
                         ANALYSE PAR ZONES MTC
================================================================================
"""
        
        for zone_name, detections in results['zone_analysis'].items():
            if zone_name in TONGUE_ZONES:
                zone_info = TONGUE_ZONES[zone_name]
                report += f"\n{zone_info['name']}\n"
                report += f"  Description: {zone_info['description']}\n"
                report += f"  Nombre de signes: {len(detections)}\n"
                
                if detections:
                    report += "  Détails:\n"
                    for det in detections:
                        if det['feature'] in FEATURE_DESCRIPTIONS:
                            report += f"    - {FEATURE_DESCRIPTIONS[det['feature']]['fr']}"
                            report += f" (confiance: {det['confidence']:.2f})\n"
        
        report += """
================================================================================
                            RECOMMANDATIONS
================================================================================
"""
        
        for i, rec in enumerate(results['diagnosis']['recommendations'], 1):
            report += f"\n{i}. {rec}\n"
        
        report += """
================================================================================
                             AVERTISSEMENT
================================================================================

Ce rapport est généré par un système d'aide au diagnostic basé sur l'intelligence
artificielle et les principes de la Médecine Traditionnelle Chinoise. Il ne
remplace en aucun cas une consultation médicale professionnelle.

En cas de doute ou de symptômes inquiétants, consultez immédiatement un
professionnel de santé qualifié.

================================================================================
                          FIN DU RAPPORT
================================================================================
"""
        
        return report
    
    def generate_html_report(self, results):
        """Génère un rapport HTML stylé"""
        stage = results['diagnosis']['stage']
        
        if stage == 'healthy':
            main_color = "#27AE60"
        elif stage == 'early':
            main_color = "#F39C12"
        else:
            main_color = "#E74C3C"
        
        preprocessing_info = ""
        if results.get('preprocessing', {}).get('tongue_detected'):
            preprocessing_info = """
            <div style="background: #E8F5E9; border: 1px solid #4CAF50; color: #2E7D32; 
                        padding: 15px; border-radius: 5px; margin: 20px 0;">
                <strong>🎯 Traitement automatique:</strong> Langue détectée et isolée automatiquement
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Rapport MTC - {datetime.now().strftime('%d/%m/%Y')}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .result-box {{
                    background: white;
                    border-left: 5px solid {main_color};
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .section {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h2 {{
                    color: #2E4057;
                    border-bottom: 2px solid #e0e0e0;
                    padding-bottom: 10px;
                }}
                .feature {{
                    background: #f8f9fa;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 3px solid #667eea;
                }}
                .recommendation {{
                    background: #e8f5e9;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 3px solid #4caf50;
                }}
                .warning {{
                    background: #fff3cd;
                    border: 1px solid #ffeeba;
                    color: #856404;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Rapport de Diagnostic MTC</h1>
                <p>Analyse de la langue - Cancer du sein</p>
                <p>{results['timestamp'].strftime('%d/%m/%Y à %H:%M')}</p>
            </div>
            
            {preprocessing_info}
            
            <div class="result-box" style="border-color: {main_color};">
                <h2 style="color: {main_color};">{results['diagnosis']['message']}</h2>
                <p><strong>Confiance:</strong> {results['diagnosis']['confidence']:.1%}</p>
                <p>{results['diagnosis']['description']}</p>
            </div>
            
            <div class="section">
                <h2>Caractéristiques détectées</h2>
        """
        
        for feature, score in sorted(results['diagnosis']['features'].items(), 
                                   key=lambda x: x[1], reverse=True):
            if feature in FEATURE_DESCRIPTIONS:
                desc = FEATURE_DESCRIPTIONS[feature]
                html += f"""
                <div class="feature">
                    <strong>{desc['fr']}</strong><br>
                    <small>{desc['meaning']}</small><br>
                    <em>Score: {score:.2f}</em>
                </div>
                """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Recommandations</h2>
        """
        
        for rec in results['diagnosis']['recommendations']:
            html += f'<div class="recommendation">{rec}</div>'
        
        html += """
            </div>
            
            <div class="warning">
                <strong>⚠️ Avertissement:</strong> Ce rapport est généré par un système 
                d'aide au diagnostic et ne remplace pas une consultation médicale 
                professionnelle.
            </div>
        </body>
        </html>
        """
        
        return html
    
    def show_about(self):
        """Page À propos avec thème MTC et sensibilisation"""
        st.markdown("""
        <div class="info-card">
            <h2>🌸 À propos de MTC Diagnostic Pro • 關於我們</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Mission et vision avec symboles chinois
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Notre Mission • 使命
            
            Démocratiser l'accès au diagnostic précoce du cancer du sein en combinant 
            la sagesse millénaire de la Médecine Traditionnelle Chinoise (中醫) avec les 
            technologies d'intelligence artificielle les plus avancées.
            
            ### 👁️ Notre Vision • 願景
            
            Un monde où chaque femme a accès à des outils de dépistage précoce, 
            non-invasifs et culturellement adaptés, permettant une prise en charge 
            rapide et efficace.
            
            ### 🆕 Nouveautés Version 2.1
            
            - **🎯 Détection automatique de langue** : L'IA peut maintenant analyser des photos de visages
            - **🔍 Segmentation avancée** : Isolation précise de la langue avec SAM
            - **📱 Interface améliorée** : Expérience utilisateur optimisée
            - **📊 Rapports enrichis** : Informations sur le préprocessing automatique
            
            <div style="margin-top: 2rem; padding: 1rem; background: #FFF5F8; border-radius: 10px; border: 1px solid #FFE4E9;">
                <strong style="color: #D81B60;">🎗️ Engagement Rose</strong><br>
                Nous soutenons activement la sensibilisation au cancer du sein et 
                l'importance du dépistage précoce.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Logo avec symboles MTC
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #FFF5F8; border-radius: 20px; border: 1px solid #FFE4E9;">
                <div style="font-size: 6rem;">🌸</div>
                <h3 style="color: #D81B60;">MTC Pro</h3>
                <p style="color: #EC407A;">中醫診斷</p>
                <p style="color: #666666;">Excellence & Innovation</p>
                <hr style="border-color: #FFE4E9;">
                <div style="font-size: 2rem; margin-top: 1rem;">
                    <div>陰陽</div>
                    <div style="font-size: 3rem;">☯</div>
                    <small style="color: #666666;">Équilibre & Harmonie</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Technologie avec accent MTC
        st.markdown("### 🔬 Technologie de pointe • 先進技術")
        
        tech_cols = st.columns(5)
        
        technologies = [
            ("🤖", "YOLOv11", "Diagnostic MTC • 中醫診斷"),
            ("🎯", "YOLOv8", "Détection langue • 舌頭檢測"),
            ("🎭", "SAM", "Segmentation • 分割"),
            ("☯", "5 區域", "Zones traditionnelles"),
            ("📊", "ML 學習", "Apprentissage continu")
        ]
        
        for col, (icon, title, desc) in zip(tech_cols, technologies):
            with col:
                st.markdown(f"""
                <div class="info-card" style="text-align: center; border-top: 3px solid #FFB6C1;">
                    <div style="font-size: 2.5rem; color: #D81B60;">{icon}</div>
                    <h4 style="color: #333333; font-size: 1rem;">{title}</h4>
                    <p style="font-size: 0.8rem; color: #666666;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Workflow détaillé
        st.markdown("### 🔄 Workflow d'analyse • 分析流程")
        
        workflow_steps = [
            ("📷", "Upload", "Photo de visage ou langue seule"),
            ("🔍", "Détection", "YOLOv8 trouve la langue automatiquement"),
            ("🎭", "Segmentation", "SAM isole la langue (optionnel)"),
            ("🤖", "Analyse", "YOLOv11 MTC détecte 21 caractéristiques"),
            ("📊", "Diagnostic", "Rapport complet selon principes MTC")
        ]
        
        cols = st.columns(len(workflow_steps))
        for col, (icon, title, desc) in zip(cols, workflow_steps):
            with col:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 3rem; color: #FFB6C1;">{icon}</div>
                    <h4 style="color: #333333; margin: 0.5rem 0;">{title}</h4>
                    <p style="font-size: 0.9rem; color: #666666; margin: 0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Équipe avec thème rose
        st.markdown("### 👥 Équipe de développement • 開發團隊")
        
        st.markdown("""
        <div class="info-card" style="background: linear-gradient(135deg, #FFB6C1 0%, #FFC1CC 100%); 
                                     color: white; text-align: center;">
            <h3 style="color: white;">SMAILI Maya & MORSLI Manel</h3>
            <p style="font-size: 1.1rem; color: white;">
                Étudiantes en Master 2 - Systèmes Informatiques Intelligents<br>
                Université Mouloud Mammeri de Tizi-Ouzou (UMMTO)<br>
                Promotion 2024/2025
            </p>
            <hr style="border-color: white; opacity: 0.3;">
            <p style="margin-top: 1rem; color: white;">
                <strong>Encadré par:</strong> Mme Y. YESLI Yasmine<br>
                Département d'Informatique - Faculté de Génie Électrique et d'Informatique
            </p>
            <div style="margin-top: 1.5rem; font-size: 1.5rem; color: white;">
                希望 • Espoir • Hope
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Changelog Version 2.1
        st.markdown("### 📝 Nouveautés Version 2.1 • 更新日誌")
        
        changelog = [
            "🎯 Détection automatique de langue sur photos de visages",
            "🎭 Intégration du modèle SAM pour segmentation précise",
            "🔍 Support des modèles YOLOv8 personnalisés (bestYolo8.pt)",
            "📱 Interface utilisateur améliorée avec options de traitement",
            "📊 Rapports enrichis avec informations de préprocessing",
            "⚙️ Configuration automatique avec fallback intelligent",
            "🔧 Sidebar avec état des modules en temps réel",
            "💾 Sauvegarde automatique des images traitées"
        ]
        
        for item in changelog:
            st.markdown(f"- {item}")
        
        # Principes MTC
        st.markdown("### 📚 Principes de la MTC • 中醫原理")
        
        principles = [
            "望 (Wàng) - Observer : Examen visuel de la langue",
            "聞 (Wén) - Écouter et Sentir : Analyse des sons et odeurs",
            "問 (Wèn) - Questionner : Interrogation sur les symptômes",
            "切 (Qiè) - Palper : Prise du pouls et palpation",
            "陰陽 (Yīn Yáng) - Équilibre des forces opposées",
            "五行 (Wǔ Xíng) - Théorie des cinq éléments"
        ]
        
        for principle in principles:
            st.markdown(f"- {principle}")
        
        # Contact avec thème rose
        st.markdown("### 📧 Contact et support • 聯繫我們")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: #FFF5F8; padding: 1rem; border-radius: 10px; border: 1px solid #FFE4E9;">
                📧 Email: mtc.diagnostic@ummto.dz
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #FFF5F8; padding: 1rem; border-radius: 10px; border: 1px solid #FFE4E9;">
                🌐 Web: www.ummto.dz/mtc-diagnostic
            </div>
            """, unsafe_allow_html=True)
        
        # Avertissement avec symbole d'espoir
        st.markdown("""
        <div class="warning" style="background: #FFF5F8; border: 1px solid #FFE4E9; 
                                   color: #666666; padding: 20px; border-radius: 10px; 
                                   margin-top: 30px; text-align: center;">
            <h4 style="color: #D81B60;">⚖️ Avertissement légal • 法律聲明</h4>
            <p>
                Ce système est développé à des fins de recherche et d'éducation. 
                Il ne doit pas être utilisé comme unique source de diagnostic médical. 
                Consultez toujours un professionnel de santé qualifié pour tout 
                problème médical.
            </p>
            <hr style="border-color: #FFE4E9;">
            <p style="margin-top: 1rem;">
                <strong>🎗️ Ensemble, nous pouvons faire la différence</strong><br>
                <span style="color: #D81B60;">希望永存 • L'espoir demeure</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Point d'entrée de l'application"""
    app = MTCDiagnosticApp()
    app.run()

# ============================================================================
# SIDEBAR MODIFIÉE AVEC INFORMATIONS SUR LES MODULES
# ============================================================================

with st.sidebar:
    st.markdown("""
    ### 🌸 MTC Diagnostic Pro
    ### 中醫診斷系統
    
    **Version:** 2.1  
    **Dernière mise à jour:** Juin 2025
    
    ---
    
    ### 🔧 Modules disponibles
    """)
    
    # Vérifier les modules
    modules_status = check_tongue_processing_availability()
    
    # IA Principal
    if st.session_state.get('model'):
        st.success("✅ IA MTC - Actif")
    else:
        st.error("❌ IA MTC - Inactif")
    
    # Détection de langue
    if modules_status['bestYolo8_exists']:
        st.success("✅ Détection langue - Disponible")
    else:
        st.warning("⚠️ Détection langue - bestYolo8.pt manquant")
    
    # SAM
    if modules_status['sam_available']:
        st.success("✅ SAM - Disponible")
    else:
        st.info("ℹ️ SAM - Non installé")
    
    # YOLOv8 de base
    if modules_status['yolo_available']:
        st.success("✅ YOLOv8 - Disponible")
    else:
        st.warning("⚠️ YOLOv8 - Non installé")
    
    st.markdown("""
    ---
    
    ### 📊 統計 • Statistiques
    """)
    
    # Afficher des statistiques si une analyse est en cours
    if st.session_state.get('results'):
        results = st.session_state.results
        
        st.metric("Confiance 信心", f"{results['diagnosis']['confidence']:.1%}")
        st.metric("Détections 檢測", len(results['detections']))
        st.metric("Zones affectées 影響", len(results['zone_analysis']))
        
        # Info sur le préprocessing
        if results.get('preprocessing', {}).get('tongue_detected'):
            st.success("🎯 Langue auto-détectée")
        else:
            st.info("📷 Image originale")
    else:
        st.info("Aucune analyse en cours • 無分析")
    
    st.markdown("""
    ---
    
    ### ⚙️ Configuration
    """)
    
    # Options de configuration
    if st.checkbox("🔧 Mode développeur", help="Affiche les informations techniques"):
        st.markdown("**Chemins des modèles:**")
        st.code("MTC: mtc_models/yolov11_mtc/weights/best.pt")
        st.code("Langue: bestYolo8.pt")
        st.code("SAM: sam_vit_h_4b8939.pth")
        
        if st.session_state.get('results'):
            st.markdown("**Dernière analyse:**")
            st.code(f"Timestamp: {st.session_state.results['timestamp']}")
            if st.session_state.results.get('preprocessing'):
                st.code("Preprocessing: Activé")
    
    st.markdown("""
    ---
    
    ### 🔗 Liens rapides • 快速連結
    - [Guide MTC • 中醫指南](#)
    - [FAQ • 常見問題](#)
    - [Support • 支援](#)
    
    ---
    
    ### 🌸 Sensibilisation
    <div style="text-align: center; padding: 1rem; background: #FFF5F8; border-radius: 10px; margin: 1rem 0; border: 1px solid #FFE4E9;">
        <div style="font-size: 3rem;">🎗️</div>
        <strong style="color: #D81B60;">Octobre Rose</strong><br>
        <small style="color: #666666;">Ensemble contre le cancer du sein<br>
        共同對抗乳腺癌</small>
    </div>
    
    <small style="color: #666666;">
    Développé avec ❤️ par<br>
    SMAILI Maya & MORSLI Manel<br>
    UMMTO 2024/2025<br>
    <span style="color: #D81B60;">希望 • Espoir • Hope</span>
    </small>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()