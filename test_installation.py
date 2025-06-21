#!/usr/bin/env python3
"""
Script de vérification rapide pour MTC Diagnostic Pro v2.1
SMAILI Maya & MORSLI Manel - UMMTO 2024/2025
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def print_header():
    print("=" * 70)
    print("🩺 VERIFICATION MTC DIAGNOSTIC PRO v2.1")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_files():
    """Vérifier les fichiers requis"""
    print("📁 VERIFICATION DES FICHIERS")
    print("-" * 40)
    
    required_files = {
        'mtc_diagnostic_app.py': 'Application principale',
        'bestYolo8.pt': 'Modèle YOLOv8 (REQUIS pour détection)',
        'mtc_models/yolov11_mtc/weights/best.pt': 'Modèle MTC principal'
    }
    
    optional_files = {
        'sam_vit_h_4b8939.pth': 'Modèle SAM (optionnel)',
        'quick_start.py': 'Script de démarrage',
        'requirements.txt': 'Liste des dépendances'
    }
    
    all_good = True
    
    print("Fichiers REQUIS:")
    for file_path, description in required_files.items():
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path:<40} - {description}")
        if not exists:
            all_good = False
    
    print("\nFichiers OPTIONNELS:")
    for file_path, description in optional_files.items():
        exists = Path(file_path).exists()
        status = "✅" if exists else "⚠️ "
        print(f"  {status} {file_path:<40} - {description}")
    
    return all_good

def check_dependencies():
    """Vérifier les dépendances Python"""
    print("\n📦 VERIFICATION DES DEPENDANCES")
    print("-" * 40)
    
    dependencies = {
        'streamlit': 'Framework web',
        'ultralytics': 'YOLOv8/v11',
        'cv2': 'OpenCV',
        'numpy': 'Calculs numériques',
        'matplotlib': 'Visualisation',
        'plotly': 'Graphiques interactifs',
        'PIL': 'Traitement d\'images',
        'pandas': 'Manipulation de données'
    }
    
    optional_deps = {
        'segment_anything': 'SAM (segmentation avancée)',
        'torch': 'PyTorch (IA)'
    }
    
    missing_required = []
    missing_optional = []
    
    print("Dépendances REQUISES:")
    for module, description in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                import PIL
            else:
                __import__(module)
            print(f"  ✅ {module:<20} - {description}")
        except ImportError:
            print(f"  ❌ {module:<20} - {description}")
            missing_required.append(module)
    
    print("\nDépendances OPTIONNELLES:")
    for module, description in optional_deps.items():
        try:
            __import__(module)
            print(f"  ✅ {module:<20} - {description}")
        except ImportError:
            print(f"  ⚠️  {module:<20} - {description}")
            missing_optional.append(module)
    
    return missing_required, missing_optional

def check_model_functionality():
    """Tester le chargement des modèles"""
    print("\n🤖 TEST DES MODELES")
    print("-" * 40)
    
    models_status = {}
    
    # Test YOLOv8 de base
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Modèle de base
        print("  ✅ YOLOv8 de base - Fonctionnel")
        models_status['yolo_base'] = True
    except Exception as e:
        print(f"  ❌ YOLOv8 de base - Erreur: {e}")
        models_status['yolo_base'] = False
    
    # Test modèle de détection de langue
    if Path('bestYolo8.pt').exists():
        try:
            from ultralytics import YOLO
            model = YOLO('bestYolo8.pt')
            print("  ✅ bestYolo8.pt - Chargement OK")
            models_status['tongue_detection'] = True
        except Exception as e:
            print(f"  ❌ bestYolo8.pt - Erreur: {e}")
            models_status['tongue_detection'] = False
    else:
        print("  ⚠️  bestYolo8.pt - Fichier manquant")
        models_status['tongue_detection'] = False
    
    # Test modèle MTC principal
    mtc_model_path = Path('mtc_models/yolov11_mtc/weights/best.pt')
    if mtc_model_path.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(mtc_model_path))
            print("  ✅ Modèle MTC - Chargement OK")
            models_status['mtc_main'] = True
        except Exception as e:
            print(f"  ❌ Modèle MTC - Erreur: {e}")
            models_status['mtc_main'] = False
    else:
        print("  ❌ Modèle MTC - Fichier manquant")
        models_status['mtc_main'] = False
    
    # Test SAM (optionnel)
    try:
        from segment_anything import SamPredictor, sam_model_registry
        sam_paths = [
            'sam_vit_h_4b8939.pth',
            'models/sam_vit_h_4b8939.pth'
        ]
        
        sam_found = False
        for path in sam_paths:
            if Path(path).exists():
                try:
                    sam = sam_model_registry["vit_h"](checkpoint=path)
                    print(f"  ✅ SAM - Chargement OK ({path})")
                    models_status['sam'] = True
                    sam_found = True
                    break
                except Exception as e:
                    continue
        
        if not sam_found:
            print("  ⚠️  SAM - Aucun modèle trouvé (optionnel)")
            models_status['sam'] = False
            
    except ImportError:
        print("  ⚠️  SAM - Module non installé (optionnel)")
        models_status['sam'] = False
    
    return models_status

def check_streamlit_compatibility():
    """Vérifier Streamlit"""
    print("\n🌐 TEST STREAMLIT")
    print("-" * 40)
    
    try:
        import streamlit as st
        version = st.__version__
        print(f"  ✅ Streamlit installé - Version {version}")
        
        if version >= "1.31.0":
            print("  ✅ Version compatible")
            return True
        else:
            print("  ⚠️  Version ancienne - Mise à jour recommandée")
            return True
    except ImportError:
        print("  ❌ Streamlit non installé")
        return False

def generate_recommendations(files_ok, missing_required, missing_optional, models_status):
    """Générer des recommandations"""
    print("\n💡 RECOMMANDATIONS")
    print("=" * 70)
    
    if not files_ok:
        print("\n🔴 ACTIONS REQUISES:")
        if not Path('mtc_diagnostic_app.py').exists():
            print("  ❌ Remplacez le contenu de mtc_diagnostic_app.py par le nouveau code")
        
        if not Path('bestYolo8.pt').exists():
            print("  ❌ Placez votre modèle bestYolo8.pt dans la racine du projet")
        
        if not Path('mtc_models/yolov11_mtc/weights/best.pt').exists():
            print("  ❌ Vérifiez l'emplacement de votre modèle MTC principal")
    
    if missing_required:
        print(f"\n🔴 INSTALLER LES DEPENDANCES MANQUANTES:")
        for dep in missing_required:
            if dep == 'cv2':
                print("  pip install opencv-python")
            elif dep == 'PIL':
                print("  pip install pillow")
            else:
                print(f"  pip install {dep}")
    
    if missing_optional:
        print(f"\n🟡 DEPENDANCES OPTIONNELLES (pour de meilleures performances):")
        for dep in missing_optional:
            print(f"  pip install {dep}")
    
    if not models_status.get('sam', False):
        print(f"\n🟡 POUR UNE SEGMENTATION PLUS PRECISE:")
        print("  1. pip install segment-anything")
        print("  2. wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    
    # Score global
    total_checks = 4  # files, required deps, models, streamlit
    passed_checks = 0
    
    if files_ok:
        passed_checks += 1
    if not missing_required:
        passed_checks += 1
    if models_status.get('mtc_main', False):
        passed_checks += 1
    if check_streamlit_compatibility():
        passed_checks += 1
    
    score = (passed_checks / total_checks) * 100
    
    print(f"\n📊 SCORE GLOBAL: {score:.0f}%")
    
    if score >= 75:
        print("🟢 PRET ! Vous pouvez lancer l'application")
        print("   Commande: streamlit run mtc_diagnostic_app.py")
    elif score >= 50:
        print("🟡 PRESQUE PRET - Corrigez les points ci-dessus")
    else:
        print("🔴 CONFIGURATION INCOMPLETE - Suivez les recommandations")

def main():
    """Fonction principale"""
    print_header()
    
    # Vérifications
    files_ok = check_files()
    missing_required, missing_optional = check_dependencies()
    models_status = check_model_functionality()
    streamlit_ok = check_streamlit_compatibility()
    
    # Recommandations
    generate_recommendations(files_ok, missing_required, missing_optional, models_status)
    
    print("\n" + "=" * 70)
    print("✅ Vérification terminée !")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Vérification interrompue")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()