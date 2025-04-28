@echo off
REM === Script pour configurer automatiquement l'environnement venv_p9 ===

echo 🚀 Activation de l'environnement virtuel venv_p9...
call venv_p9\Scripts\activate

echo 🛠️ Vérification installation de torch...
python -c "import torch" 2>NUL
if errorlevel 1 (
    echo ❗ Torch non installé. Installation en cours...
    pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
) else (
    echo ✅ Torch est déjà installé.
)

echo 🛠️ Vérification installation de TensorFlow...
python -c "import tensorflow" 2>NUL
if errorlevel 1 (
    echo ❗ TensorFlow non installé. Installation en cours...
    pip install tensorflow==2.10.0
) else (
    echo ✅ TensorFlow est déjà installé.
)

echo 🎯 Environnement prêt ! Bon travail !
pause
