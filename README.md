# InvoiceAI -- Extracteur Intelligent de Factures

> Pipeline VLM multimodal + detection d'anomalies statistiques pour l'automatisation du traitement de factures

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![VLM](https://img.shields.io/badge/Extraction-LLaVA%20Multimodal-purple)
![Ollama](https://img.shields.io/badge/Inference-Ollama%20Local-green)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Portfolio%20PoC-orange)

---

## Le probleme metier

Le traitement manuel de factures coute en moyenne 10 a 15 minutes par document avec un taux d'erreur humain de 3 a 5%. Les fraudes (doublons, surfacturation, fournisseurs fantomes) representent 1 a 5% du chiffre d'affaires en industrie.

## La solution

Un pipeline end-to-end qui extrait automatiquement les donnees structurees d'une facture et applique une batterie de controles statistiques pour detecter les anomalies et fraudes potentielles.

## Ce qui distingue ce projet

L'approche est en 3 niveaux de degradation gracieuse. Le pipeline fonctionne TOUJOURS, meme si le modele le plus avance est indisponible.

```
Niveau 1 (VLM)      : Image brute -> LLaVA multimodal -> JSON -> Pydantic
Niveau 2 (OCR+LLM)  : Image -> Preprocess -> EasyOCR -> Mistral -> JSON -> Pydantic
Niveau 3 (Regex)    : Image -> Preprocess -> EasyOCR -> Regex -> Pydantic
                                    |
                                    v
                     Moteur de regles (agnostique de la source)
                     Benford | Z-score | Doublons | Arithmetique
```

**Niveau 1 (etat de l'art 2026)** : L'image brute est envoyee directement a un modele Vision-Language (LLaVA via Ollama). Le VLM "regarde" la facture, comprend la structure spatiale (colonnes, tableaux, alignements) et retourne le JSON. Zero preprocessing. Zero OCR. Zero regex.

**Niveau 2 (fallback robuste)** : Si le VLM n'est pas installe, EasyOCR extrait le texte et Mistral (LLM textuel) le structure en JSON.

**Niveau 3 (dernier recours)** : Si aucun LLM n'est disponible, un parsing regex minimal assure que le pipeline ne plante pas.

Le moteur de regles en aval est totalement agnostique de la methode d'extraction : les memes controles statistiques s'appliquent quel que soit le niveau utilise.

## Resultats

| Metrique | Valeur |
|---|---|
| Extraction VLM (LLaVA) | Comprehension du layout 2D natif |
| Regles d'anomalies | 4 (arithmetique, doublons, outliers Z-score, Benford) |
| Traitement par facture | < 10s (VLM local), < 5s (OCR+LLM) |
| Confidentialite | 100% local (Ollama), zero donnee envoyee |

## Dataset

**SROIE (Scanned Receipts OCR and Information Extraction)** -- Dataset de reference pour l'OCR sur tickets de caisse reels avec images et annotations ground truth.

Source : [Kaggle - SROIE Dataset v2](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2)

## Stack technique

| Composant | Technologie | Justification |
|---|---|---|
| Extraction niveau 1 | LLaVA (Ollama) | VLM multimodal, comprend le layout 2D sans OCR |
| Extraction niveau 2 | EasyOCR + Mistral (Ollama) | Fallback textuel si pas de VLM |
| Extraction niveau 3 | EasyOCR + Regex | Dernier recours, zero dependance LLM |
| Preprocessing (niv 2-3) | OpenCV | Deskew, debruitage, binarisation adaptative |
| Validation donnees | Pydantic v2 | Contrat strict entre extraction et regles |
| Detection d'anomalies | Python + SciPy | Benford (chi2), Z-score, doublons, arithmetique |
| Inference locale | Ollama | Gratuit, offline, RGPD |
| Interface | Streamlit | Upload + resultats + anomalies |

## Structure du projet

```
InvoiceAI/
├── README.md
├── requirements.txt
├── invoice_ai.py           # Pipeline complet (3 niveaux + regles + visualisations)
└── cours_projet1.md        # Cours pedagogique detaille
```

## Installation et execution

```bash
# Cloner
git clone https://github.com/<votre-username>/InvoiceAI.git
cd InvoiceAI

# Dependances Python
pip install -r requirements.txt

# Installer Ollama (https://ollama.ai)
# Puis telecharger les modeles :
ollama pull llava       # Niveau 1 : VLM multimodal
ollama pull mistral     # Niveau 2 : LLM textuel

# Lancer le pipeline
python invoice_ai.py

# Ou l'interface Streamlit
streamlit run app.py
```

Le pipeline fonctionne meme sans Ollama (niveau 3 regex), mais les niveaux 1 et 2 necessitent qu'Ollama soit lance (`ollama serve`).

## Regles de detection implementees

### 1. Coherence arithmetique
Verifie que la somme des lignes correspond au total HT et que HT + TVA == TTC.

### 2. Detection de doublons
Compare chaque facture a l'historique : meme fournisseur + meme montant.

### 3. Valeurs aberrantes (Z-score)
Pour un fournisseur donne, alerte si le montant depasse 3 ecarts-types de la moyenne historique.

### 4. Loi de Benford
Analyse la distribution des premiers chiffres significatifs des montants via un test du chi-deux. Un ecart significatif (p < 0.05) signale une potentielle manipulation. Necessite 50+ montants.

## Pourquoi VLM plutot que OCR + Regex ?

| Approche | Avantage | Inconvenient |
|---|---|---|
| Regex | Zero dependance | Casse a chaque nouveau format |
| OCR + LLM textuel | Comprend la semantique | Perd le layout 2D (colonnes, tableaux) |
| VLM multimodal | Comprend layout + semantique | Modele plus lourd (~4 Go RAM) |

L'approche VLM est l'etat de l'art en Document AI en 2026. Le modele voit l'image comme un humain : il comprend que "50.00" est aligne sous la colonne "TVA" sans avoir besoin de regex pour le deviner.

## Extensions possibles

- **Qwen2-VL ou LLaVA-NeXT** pour une meilleure precision multimodale
- **Base SQLite** pour l'historique des factures
- **API REST (FastAPI)** pour l'integration SI
- **LayoutLMv3** comme alternative locale sans Ollama
- **Export PDF** du rapport d'anomalies

## Auteur

Projet realise dans le cadre d'un portfolio Data Science / IA industrielle.

---

*Ce projet demontre la maitrise de l'architecture de degradation gracieuse (VLM -> OCR+LLM -> Regex), la separation extraction / validation, et l'application de statistiques avancees (Benford, Z-score) a la detection de fraude.*
