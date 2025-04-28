# src/data_preprocessing.py

"""
Module data_preprocessing.py
Contient les fonctions pour nettoyer et préparer le dataset :
- Nettoyage de texte
- Lemmatisation
- Prétraitement complet du dataset tweets.csv
- Export du dataset nettoyé
"""

import os
import re
import pandas as pd
import emoji
import spacy
from pathlib import Path
from utils import mlflow_run_safety, timing

# Initialisation SpaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# === Nettoyage basique du texte ===
def clean_text(text):
    """
    Nettoyage du texte brut :
    - passage en minuscules,
    - suppression des urls, mentions, hashtags, emojis,
    - suppression des caractères spéciaux.
    """
    text = text.lower()
    text = re.sub(r'http\\S+|www\\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = ' '.join([word for word in text.split() if len(word) > 1])
    return text

# === Lemmatisation et suppression stopwords ===
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

# === Pipeline complet de nettoyage Dataset ===
@mlflow_run_safety(experiment_name="P9_sentiment_analysis_preprocessing")
@timing
def clean_and_save_dataset(input_path="../data/tweets.csv", output_path="../data/tweets_cleaned.csv"):
    """
    Charge, nettoie et sauvegarde le dataset tweets.csv.:
    - conserve uniquement les colonnes utiles,
    - convertit les labels 0 et 4 en 0 et 1,
    - supprime les doublons sur 'text',
    - applique un nettoyage avancé sur le texte,
    - filtre les tweets trop courts,
    - sauvegarde un fichier nettoyé.
    """
    print(f"📂 Chargement du dataset brut depuis : {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8')

    # Conserver colonnes utiles
    df = df[['text', 'label']]

    # Conversion labels (4 ➔ 1)
    df['label'] = df['label'].replace(4, 1)

    # Suppression des doublons sur text
    n_doublons = df.duplicated(subset=['text']).sum()
    print(f"🔍 Nombre de doublons : {n_doublons}")
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)

    # Nettoyage avancé du texte
    print("🚀 Nettoyage avancé des textes...")
    df['text'] = df['text'].astype(str).apply(clean_text)
    df['text'] = df['text'].apply(lemmatize_text)

    # Suppression des tweets trop courts (<5 mots)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= 5].drop(columns=['word_count']).reset_index(drop=True)

    # Sauvegarde finale
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"✅ Dataset nettoyé sauvegardé sous : {output_path}")
