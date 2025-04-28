# src/utils.py

"""
Module utils.py
Contient les fonctions utilitaires génériques pour le projet :
- Vérification GPU
- Affichage arborescence dossier
- Gestion démarrage/arrêt MLflow Server
- Décorateurs pour MLflow et timing
"""

import os
import sys
import subprocess
import psutil
import time
import mlflow
from functools import wraps
import torch
import tensorflow as tf
from pathlib import Path

# Décorateur de mesure et affichage du temps d'exécution d'une fonction.
def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"⏱️ Fonction {func.__name__} exécutée en {duration:.2f} secondes.")
        return result
    return wrapper

# Décorateur de sécurisation de l'enregistrement MLflow ===
def mlflow_run_safety(experiment_name="Default"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# === Vérification disponibilité GPU Torch et Tensorflow ===
@timing
def check_gpu():
    print(f"🚀 Torch CUDA Available: {torch.cuda.is_available()}")
    print(f"🚀 TensorFlow GPUs: {tf.config.list_physical_devices('GPU')}")

# === Affichage de l'arborescence d'un dossier ===
@timing
def show_tree(directory):
    directory = Path(directory)
    for path in directory.rglob('*'):
        depth = len(path.relative_to(directory).parts)
        spacer = ' ' * 4 * depth
        print(f"{spacer}{path.name}")

# Tuer tout processus qui occupe un port réseau spécifique
@timing
def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        for conn in proc.info['connections'] or []:
            if conn.laddr.port == port:
                print(f"⛔️ Killing process {proc.info['name']} (PID {proc.info['pid']}) on port {port}")
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                p.wait()

# Démarre le serveur MLflow UI sur le port spécifié.
@timing
def start_mlflow_server(port=5000):
    kill_process_on_port(port)
    mlflow_server = subprocess.Popen(["mlflow", "ui", "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"✅ MLflow Server démarré sur http://127.0.0.1:{port}")
    return mlflow_server

# === Arrêter proprement un serveur MLflow ===
@timing
def stop_mlflow_server(server_process):
    if server_process:
        server_process.terminate()
        server_process.wait()
        print("✅ MLflow Server arrêté proprement.")

# === Affichage de l'utilisation actuelle du GPU ===
@timing
def print_gpu_utilization():
    """
    Affiche l'état actuel de l'utilisation GPU :
    - ID GPU
    - Charge (%)
    - Mémoire utilisée, libre et totale
    """
    import GPUtil
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID {gpu.id}: {gpu.name}")
        print(f"  Load: {gpu.load*100:.1f}%")
        print(f"  Free memory: {gpu.memoryFree}MB")
        print(f"  Used memory: {gpu.memoryUsed}MB")
        print(f"  Total memory: {gpu.memoryTotal}MB")