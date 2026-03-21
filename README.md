# InvoiceAI — Extracteur Intelligent de Factures

> Pipeline OCR + NLP + Detection d'anomalies pour l'automatisation du traitement de factures

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-green)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Portfolio%20PoC-orange)

---

## Le probleme metier

Le traitement manuel de factures coute en moyenne **10 a 15 minutes par document** avec un taux d'erreur humain de 3 a 5%. Les fraudes sur factures (doublons, surfacturation, fournisseurs fantomes) representent **1 a 5% du chiffre d'affaires** en industrie.

## La solution

Un pipeline end-to-end qui :
1. **Extrait le texte** de n'importe quel scan de facture (PDF, photo, image) via OCR
2. **Structure les donnees** (fournisseur, date, montants, lignes) via extraction intelligente
3. **Detecte les anomalies** via un moteur de regles metier (coherence arithmetique, loi de Benford, doublons, valeurs aberrantes)

## Architecture

```
Image/PDF  -->  Pretraitement (OpenCV)  -->  OCR (EasyOCR)
                                                  |
                                                  v
                                        Texte brut extrait
                                                  |
                                                  v
                                    Extraction structuree (Regex/LLM)
                                                  |
                                                  v
                                         JSON structure (Pydantic)
                                                  |
                                                  v
                                    Moteur de regles (anomalies/fraudes)
                                                  |
                                                  v
                                      Interface Streamlit (dashboard)
```

## Resultats

| Metrique | Valeur |
|---|---|
| Taux de reconnaissance OCR moyen | ~85% (tickets reels SROIE) |
| Extraction des montants | Detection du total TTC sur la majorite des tickets |
| Regles d'anomalies implementees | 4 (arithmetique, doublons, outliers, Benford) |
| Temps de traitement par facture | < 5 secondes (CPU) |

## Dataset

**SROIE (Scanned Receipts OCR and Information Extraction)** — Dataset de reference pour l'OCR sur tickets de caisse reels, avec images et annotations ground truth.

Source : [Kaggle - SROIE Dataset v2](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2)

## Stack technique

| Composant | Technologie | Justification |
|---|---|---|
| Pretraitement image | OpenCV | Deskew, debruitage, binarisation adaptative |
| OCR | EasyOCR | Installation simple, support FR/EN, confiance par bloc |
| Extraction structuree | Regex + Pydantic | Robuste sans dependance LLM, extensible |
| Detection d'anomalies | Python pur + SciPy | Loi de Benford (chi2), Z-score, matching |
| Validation donnees | Pydantic | Typage strict, validation automatique |
| Interface | Streamlit | Prototypage rapide, upload interactif |

## Structure du projet

```
InvoiceAI/
├── README.md
├── requirements.txt
├── invoice_ai.py                    # Notebook Colab complet
├── cours_projet1.md                 # Cours pedagogique detaille
└── app_streamlit.py                 # Interface Streamlit (generee)
```

## Installation et execution

```bash
# Cloner le repo
git clone https://github.com/<votre-username>/InvoiceAI.git
cd InvoiceAI

# Installer les dependances
pip install -r requirements.txt

# Lancer le notebook (Colab recommande)
# Ouvrir invoice_ai.py dans Google Colab

# Ou lancer l'interface Streamlit
streamlit run app_streamlit.py
```

## Regles de detection implementees

### 1. Coherence arithmetique
Verifie que `somme(lignes) == total_HT` et `HT + TVA == TTC` avec tolerance d'arrondi.

### 2. Detection de doublons
Compare chaque facture a l'historique : meme fournisseur + meme montant + dates proches (< 7 jours).

### 3. Valeurs aberrantes (outliers)
Pour un fournisseur donne, alerte si le montant depasse 3 ecarts-types de la moyenne historique (Z-score).

### 4. Loi de Benford
Analyse statistique de la distribution des premiers chiffres significatifs des montants. Un ecart a la loi de Benford (test du chi-deux, p < 0.05) signale une potentielle manipulation. Necessaire un minimum de 50 montants.

## Extensions possibles

- **LLM local (Ollama/Mistral)** pour l'extraction structuree a la place du regex
- **Base de donnees SQLite** pour l'historique des factures
- **API REST (FastAPI)** pour l'integration dans un SI existant
- **Support multi-pages PDF** avec concatenation du texte OCR

## Auteur

Projet realise dans le cadre d'un portfolio Data Science / IA industrielle.

---

*Ce projet demontre la capacite a chainer OCR + NLP + logique metier dans un pipeline robuste, avec une gestion explicite des cas d'erreur et une detection proactive de fraudes.*
