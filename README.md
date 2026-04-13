# InvoiceAI -- Extracteur Intelligent de Factures

Pipeline d'extraction automatique de factures avec detection de fraudes en 4 niveaux.

---

## A quoi ca sert

On donne une image de facture. Le systeme extrait les donnees (fournisseur, date, montants, lignes de detail), les stocke dans une base vectorielle, et les passe dans 4 niveaux de controle pour detecter les anomalies et fraudes potentielles.

## Comment ca marche

### Extraction

Trois methodes, de la plus performante a la plus basique. Le systeme essaie la meilleure disponible et descend automatiquement si besoin :

- **VLM multimodal (LLaVA)** : le modele de vision regarde directement l'image, comprend le layout, retourne le JSON. Si le JSON est invalide, il retente une fois en injectant l'erreur dans le prompt.
- **OCR + LLM (Mistral)** : EasyOCR extrait le texte, Mistral le structure. Meme logique de retry.
- **OCR + Regex** : parsing basique. Fragile mais fonctionne sans modele IA.

### Detection d'anomalies (4 niveaux)

**Niveau 1 -- Regles deterministes** : coherence arithmetique (HT+TVA=TTC, somme des lignes = total), doublons exacts (hash SHA256), doublons semantiques (cosine similarity via ChromaDB, seuil > 0.95).

**Niveau 2 -- Rapprochement ERP** : verification que la facture correspond a un bon de commande reel. Hors scope PoC mais prevu dans l'architecture.

**Niveau 3 -- Scoring ML** : Isolation Forest en multivarie (montant + nb lignes + ratio TTC/HT + ecart lignes), Z-score monovarie par fournisseur, loi de Benford sur la distribution des premiers chiffres (test du chi-deux).

**Niveau 4 -- LLM reasoning** : quand les niveaux precedents declenchent des alertes, un LLM analyse l'ensemble des indices en Chain-of-Thought et produit une recommandation (APPROVE / REVIEW / BLOCK).

### Stockage

ChromaDB (base vectorielle locale) stocke chaque facture avec ses metadonnees. La recherche de doublons semantiques se fait par requete vectorielle filtree par fournisseur.

## Dataset

**High-Quality Invoice Images for OCR** -- Factures B2B reelles.

[Kaggle](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr)

## Technologies

| Quoi | Avec quoi |
|---|---|
| Extraction VLM | LLaVA via Ollama |
| Extraction fallback | EasyOCR + Mistral |
| Preprocessing | OpenCV + Sauvola (scikit-image) |
| Validation | Pydantic v2 (avec field_validator croise) |
| Stockage vectoriel | ChromaDB |
| Anomalies ML | Isolation Forest (scikit-learn) |
| Anomalies stats | Z-score, Benford (SciPy) |
| LLM reasoning | Mistral via Ollama |

## Fichiers

```
InvoiceAI/
├── README.md
├── requirements.txt
├── invoice_ai.py           # Pipeline complet
└── cours_projet1.md        # Cours pedagogique
```

## Lancer en local

```bash
pip install -r requirements.txt

# Recommande : installer Ollama + modeles
# https://ollama.com/download
ollama serve &
ollama pull llava       # VLM multimodal
ollama pull mistral     # LLM textuel + raisonnement fraude

python invoice_ai.py
```

Fonctionne aussi sans Ollama (mode regex + regles deterministes uniquement), mais les performances sont limitees.

## Lancer sur Google Colab

Sur Colab, Ollama necessite d'etre installe manuellement. Active le GPU (Runtime > Change runtime type > T4 GPU) pour des performances acceptables.

### Cellule 1 : Installation d'Ollama

```bash
# Installer zstd (prerequis manquant sur Colab)
!apt-get install -y zstd

# Installer Ollama
!curl -fsSL https://ollama.com/install.sh | sh

# Lancer le serveur en background
!nohup ollama serve > ollama.log 2>&1 &
!sleep 5

# Verifier que le serveur repond
!curl http://localhost:11434/api/tags

# Telecharger les modeles (5-10 min chacun)
!ollama pull mistral
!ollama pull llava

# Verifier
!ollama list
```

### Cellule 2 : Dependances Python

```bash
!pip install -r requirements.txt
```

### Cellule 3 : Lancer le pipeline

```python
!python invoice_ai.py
```

Au demarrage, le script detecte Ollama et affiche :
```
[OK] Ollama detecte. Modeles : ['mistral', 'llava']
```

### Limitations Colab

- Session qui expire apres 90 min d'inactivite (modeles a re-telecharger)
- GPU T4 limite a ~3-4h par jour en version gratuite
- Sans GPU : LLaVA met 30-60 secondes par image

**Alternative recommandee** : installer Ollama en local sur ta machine pour un workflow persistant. Les modeles restent telecharges entre sessions.

