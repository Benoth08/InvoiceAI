# InvoiceAI -- Extracteur Intelligent de Factures

Pipeline d'extraction automatique de donnees de factures avec detection de fraudes et d'anomalies.

---

## A quoi ca sert

On donne une image de facture au systeme. Il extrait automatiquement les informations cles (fournisseur, date, montants, lignes de detail) et verifie si quelque chose est suspect (erreur de calcul, doublon, montant anormal, manipulation statistique).

## Comment ca marche

Le pipeline fonctionne en deux etapes independantes :

**Etape 1 -- Extraction des donnees**

Le systeme lit la facture et en extrait un JSON structure. Trois methodes sont disponibles, de la plus performante a la plus basique :

- **VLM multimodal (LLaVA)** : un modele de vision analyse directement l'image. Il comprend le layout (colonnes, tableaux) sans avoir besoin d'OCR. C'est la methode la plus fiable.
- **OCR + LLM textuel (Mistral)** : EasyOCR extrait le texte, puis un LLM le structure en JSON. Moins precis car on perd l'information spatiale.
- **OCR + Regex** : parsing basique du texte OCR. Fragile mais fonctionne sans aucun modele IA.

Le systeme essaie automatiquement la meilleure methode disponible. Si le VLM n'est pas installe, il passe au LLM textuel. Si aucun LLM n'est disponible, il utilise le regex. Le pipeline ne plante jamais.

**Etape 2 -- Detection d'anomalies**

Quelle que soit la methode d'extraction, les memes controles s'appliquent :

- **Coherence arithmetique** : est-ce que HT + TVA = TTC ? Est-ce que la somme des lignes correspond au total ?
- **Doublons** : meme fournisseur + meme montant = alerte.
- **Montants aberrants** : si un fournisseur facture habituellement 500 euros et qu'une facture arrive a 5 000 euros, le Z-score le detecte.
- **Loi de Benford** : test statistique (chi-deux) sur la distribution des premiers chiffres des montants. Un ecart a la loi theorique signale une possible manipulation comptable. Necessite 50+ montants pour etre fiable.

## Dataset

**High-Quality Invoice Images for OCR** -- Factures B2B reelles (formats europeens et internationaux), avec colonnes, tableaux, TVA et lignes de detail.

[Kaggle - High-Quality Invoice Images](https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr)

## Technologies utilisees

| Quoi | Avec quoi | Pourquoi |
|---|---|---|
| Extraction (methode principale) | LLaVA via Ollama | Comprend l'image directement, pas besoin d'OCR |
| Extraction (fallback) | EasyOCR + Mistral | Si le VLM n'est pas installe |
| Validation des donnees | Pydantic v2 | Schema strict, rejet automatique si le JSON est invalide |
| Detection d'anomalies | SciPy + NumPy | Benford (chi-deux), Z-score, comparaison arithmetique |
| Inference locale | Ollama | Gratuit, offline, les donnees restent sur la machine |
| Interface | Streamlit | Upload de facture + resultats + anomalies |

## Fichiers

```
InvoiceAI/
├── README.md
├── requirements.txt
├── invoice_ai.py           # Pipeline complet
└── cours_projet1.md        # Cours pedagogique detaille
```

## Lancer le projet

```bash
pip install -r requirements.txt

# Pour la methode VLM (recommande) :
# Installer Ollama (https://ollama.ai) puis :
ollama pull llava

# Pour la methode OCR + LLM :
ollama pull mistral

# Lancer
python invoice_ai.py
```

Le projet tourne aussi sans Ollama (methode regex), mais les resultats sont moins bons.

## Ce que ca montre en entretien

- Savoir chainer extraction IA + validation metier dans un pipeline robuste
- Appliquer des statistiques avancees (Benford, Z-score) a un probleme concret de fraude
- Coder pour la production (Pydantic, gestion d'erreurs, tracabilite de la methode utilisee)
- Garder les donnees en local (Ollama) pour la confidentialite
