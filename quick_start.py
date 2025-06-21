#!/usr/bin/env python3
"""
Script de démarrage rapide pour l'application MTC
Lance automatiquement l'installation et l'application
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Affiche la bannière"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     SYSTÈME DE DIAGNOSTIC MTC - CANCER DU SEIN             ║
    ║     Médecine Traditionnelle Chinoise + IA                  ║
    ║     SMAILI Maya & MORSLI Manel - UMMTO 2024/2025          ║
    ╔════════════════════════════════════════════════════════════╗
    """)

def check_python():
    """Vérifie la version Python"""
    print("🔍 Vérification de Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ requis. Version actuelle:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor} détecté")
    return True

def create_venv():
    """Crée l'environnement virtuel"""
    venv_path = Path("mtc_env")
    
    if venv_path.exists():
        print("✅ Environnement virtuel déjà existant")
        return venv_path
    
    print("📦 Création de l'environnement virtuel...")
    subprocess.run([sys.executable, "-m", "venv", "mtc_env"])
    print("✅ Environnement créé")
    return venv_path

def get_pip_command(venv_path):
    """Retourne la commande pip selon l'OS"""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "pip.exe")
    else:
        return str(venv_path / "bin" / "pip")

def install_dependencies(venv_path):
    """Installe les dépendances"""
    pip_cmd = get_pip_command(venv_path)
    
    print("\n📚 Installation des dépendances...")
    
    packages = [
        "streamlit==1.31.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "plotly>=5.18.0",
        "pandas>=2.0.0"
    ]
    
    # Mise à jour pip
    print("   Mise à jour de pip...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Installation des packages
    for i, package in enumerate(packages, 1):
        print(f"   [{i}/{len(packages)}] Installation de {package.split('=')[0]}...")
        result = subprocess.run([pip_cmd, "install", package], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            print(f"   ⚠️  Erreur avec {package}, tentative sans version...")
            subprocess.run([pip_cmd, "install", package.split('=')[0]], 
                          stdout=subprocess.DEVNULL)
    
    print("✅ Dépendances installées")

def check_model():
    """Vérifie la présence du modèle"""
    model_path = Path("mtc_models/yolov11_mtc/weights/best.pt")
    
    if not model_path.exists():
        print("\n⚠️  ATTENTION: Modèle non trouvé!")
        print(f"   Placez votre modèle 'best.pt' dans:")
        print(f"   {model_path.absolute()}")
        
        # Créer les dossiers
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print("\n   Dossiers créés. Placez-y votre modèle et relancez.")
        return False
    
    print("✅ Modèle trouvé")
    return True

def check_app_file():
    """Vérifie la présence de l'application"""
    app_path = Path("mtc_diagnostic_app.py")
    
    if not app_path.exists():
        print("\n❌ Fichier 'mtc_diagnostic_app.py' non trouvé!")
        print("   Assurez-vous qu'il est dans le même dossier que ce script.")
        return False
    
    print("✅ Application trouvée")
    return True

def get_streamlit_command(venv_path):
    """Retourne la commande streamlit selon l'OS"""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "streamlit.exe")
    else:
        return str(venv_path / "bin" / "streamlit")

def launch_app(venv_path):
    """Lance l'application Streamlit"""
    streamlit_cmd = get_streamlit_command(venv_path)
    
    print("\n🚀 Lancement de l'application...")
    print("="*60)
    print("L'application va s'ouvrir dans votre navigateur")
    print("URL: http://localhost:8501")
    print("Pour arrêter: Ctrl+C dans ce terminal")
    print("="*60)
    
    try:
        subprocess.run([streamlit_cmd, "run", "mtc_diagnostic_app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Application fermée. À bientôt!")

def main():
    """Fonction principale"""
    print_banner()
    
    # Vérifications
    if not check_python():
        input("\nAppuyez sur Entrée pour quitter...")
        return
    
    if not check_app_file():
        input("\nAppuyez sur Entrée pour quitter...")
        return
    
    # Setup
    venv_path = create_venv()
    
    # Demander l'installation
    if not (venv_path / "Lib" / "site-packages" / "streamlit").exists() and \
       not (venv_path / "lib" / "python*" / "site-packages" / "streamlit").exists():
        response = input("\n📦 Installer les dépendances? (O/n): ").lower()
        if response != 'n':
            install_dependencies(venv_path)
    else:
        print("✅ Dépendances déjà installées")
    
    # Vérifier le modèle
    if not check_model():
        input("\nAppuyez sur Entrée pour quitter...")
        return
    
    # Lancer
    print("\n" + "="*60)
    response = input("🎯 Lancer l'application maintenant? (O/n): ").lower()
    if response != 'n':
        launch_app(venv_path)
    else:
        print("\nPour lancer plus tard, utilisez:")
        if platform.system() == "Windows":
            print(f"   {venv_path}\\Scripts\\activate")
        else:
            print(f"   source {venv_path}/bin/activate")
        print("   streamlit run mtc_diagnostic_app.py")
    
    print("\n✅ Terminé!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        input("\nAppuyez sur Entrée pour quitter...")