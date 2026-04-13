# -*- coding: utf-8 -*-
"""
Extracteur Intelligent de Factures (VLM + Detection de Fraude Multi-Niveaux)
Dataset : High-Quality Invoice Images for OCR (Kaggle)
Extraction : VLM multimodal (LLaVA) > OCR+LLM (Mistral) > Regex
Detection : Regles deterministes + Isolation Forest + Cosine Similarity + LLM Reasoning
Stockage : ChromaDB (base vectorielle locale)
"""

import os
import re
import math
import json
import base64
import hashlib
import warnings
from pathlib import Path
from collections import Counter
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pydantic import BaseModel, field_validator
from scipy import stats
from skimage.filters import threshold_sauvola
import requests
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

print("[OK] Imports termines.")


# ---------------------------------------------------------------------------
# MODELES DE DONNEES (PYDANTIC) -- Avec validation croisee integree
# ---------------------------------------------------------------------------
# Le field_validator verifie HT + TVA == TTC directement au parsing.
# Si le LLM retourne des montants incoherents, Pydantic le detecte
# AVANT que la donnee n'entre dans le pipeline.

class InvoiceLine(BaseModel):
    designation: str = ""
    quantite: float = 1.0
    prix_unitaire: float = 0.0
    total_ligne: float = 0.0


class InvoiceData(BaseModel):
    numero_facture: Optional[str] = None
    fournisseur: Optional[str] = None
    date_facture: Optional[str] = None
    lignes: list[InvoiceLine] = []
    total_ht: Optional[float] = None
    tva_taux: Optional[float] = None
    tva_montant: Optional[float] = None
    total_ttc: Optional[float] = None
    raw_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    extraction_method: Optional[str] = None
    pydantic_ttc_valid: Optional[bool] = None  # Resultat de la validation croisee

    @field_validator("total_ttc")
    @classmethod
    def check_ttc_consistency(cls, v, info):
        """Validation croisee : TTC doit etre coherent avec HT + TVA.
        Ne bloque PAS le parsing (on veut quand meme la donnee),
        mais flagge l'incoherence via pydantic_ttc_valid.
        """
        ht = info.data.get("total_ht")
        tva = info.data.get("tva_montant")
        if v is not None and ht is not None and tva is not None:
            expected = round(ht + tva, 2)
            if abs(v - expected) > 0.05:
                # On ne raise PAS, on laisse passer mais on flagge
                pass
        return v


class AnomalyReport(BaseModel):
    invoice_id: str = "unknown"
    anomalies: list[dict] = []
    overall_level: str = "ok"
    benford_pvalue: Optional[float] = None
    isolation_forest_score: Optional[float] = None
    llm_decision: Optional[str] = None  # APPROVE / REVIEW / BLOCK


# ---------------------------------------------------------------------------
# TELECHARGEMENT DU DATASET
# ---------------------------------------------------------------------------

import kagglehub

print("Telechargement du dataset 'High-Quality Invoice Images for OCR'...")

try:
    dataset_path = kagglehub.dataset_download(
        "osamahosamabdellatif/high-quality-invoice-images-for-ocr"
    )
    print(f"[OK] Dataset telecharge dans : {dataset_path}")
except Exception as e:
    print(f"[ERREUR] {e}")
    print("Alternative : https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr")
    dataset_path = "./invoice_data"


def find_invoice_images(base_path: str) -> list:
    """Collecte toutes les images de factures du dataset."""
    base = Path(base_path)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
    images = []
    for ext in extensions:
        images.extend([str(p) for p in base.rglob(ext)])
    images = sorted(images)
    print(f"[OK] {len(images)} images de factures trouvees.")
    return images


dataset_images = find_invoice_images(dataset_path)

n_preview = min(4, len(dataset_images))
if n_preview > 0:
    fig, axes = plt.subplots(1, n_preview, figsize=(5 * n_preview, 6))
    if n_preview == 1:
        axes = [axes]
    for i in range(n_preview):
        img = cv2.imread(dataset_images[i])
        if img is not None:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(Path(dataset_images[i]).name, fontsize=8)
        axes[i].axis("off")
    plt.suptitle("Exemples de factures du dataset", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# PREPROCESSING AMELIORE (Sauvola + Blur Check + Deskew)
# ---------------------------------------------------------------------------
# Tire du cours DocumentAI 2026 Module 1 :
# - Sauvola au lieu de la binarisation adaptative OpenCV (meilleure sur
#   les fonds jaunis, ombres de reliure, scans anciens)
# - Blur check pour rejeter les images trop floues AVANT traitement
# - Resize intelligent (ne jamais upscaler)

def blur_check(image: np.ndarray, threshold: float = 80.0) -> bool:
    """Verifie si l'image est assez nette pour l'OCR.

    Utilise la variance du Laplacien : une image floue a une faible
    variance (peu de hautes frequences spatiales).
    Seuil typique : 80-100. En dessous = rejet.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= threshold


def resize_document(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Redimensionne sans jamais upscaler. Cible : 1200x1600 px."""
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    return image


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Correction d'inclinaison via detection de l'angle dominant."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=50, maxLineGap=10)
    if lines is None:
        return image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 30:
            angles.append(a)
    if not angles:
        return image
    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def binarize_sauvola(image: np.ndarray, window_size: int = 51) -> np.ndarray:
    """Binarisation Sauvola : superieure a un seuil global car elle gere
    les variations de luminosite locales (fond jauni, ombres de reliure).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = threshold_sauvola(gray, window_size=window_size, k=0.2)
    binary = (gray > thresh).astype(np.uint8) * 255
    return binary


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Pipeline complet : resize -> deskew -> Sauvola."""
    image = resize_document(image)
    image = deskew_image(image)
    binary = binarize_sauvola(image)
    return binary


if len(dataset_images) > 0:
    sample_img = cv2.imread(dataset_images[0])
    if sample_img is not None:
        is_sharp = blur_check(sample_img)
        preprocessed = preprocess_for_ocr(sample_img)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Originale (nette={is_sharp})")
        ax1.axis("off")
        ax2.imshow(preprocessed, cmap="gray")
        ax2.set_title("Apres preprocessing (Sauvola)")
        ax2.axis("off")
        plt.tight_layout()
        plt.show()

print("[OK] Preprocessing pret.")


# ---------------------------------------------------------------------------
# OCR (EASYOCR) -- Conserve pour le fallback et le monitoring
# ---------------------------------------------------------------------------

import logging
logging.getLogger("easyocr").setLevel(logging.ERROR)

import easyocr

print("Initialisation d'EasyOCR...")
reader = easyocr.Reader(["en", "fr"], gpu=False)
print("[OK] EasyOCR initialise.")


def extract_text_ocr(image: np.ndarray, confidence_threshold: float = 0.3) -> dict:
    """Extrait le texte + confiance via EasyOCR."""
    results = reader.readtext(image)
    blocks, texts, confidences = [], [], []
    for (bbox, text, confidence) in results:
        if confidence >= confidence_threshold:
            blocks.append({"text": text, "confidence": round(confidence, 3)})
            texts.append(text)
            confidences.append(confidence)
    return {
        "full_text": "\n".join(texts),
        "blocks": blocks,
        "avg_confidence": round(np.mean(confidences), 3) if confidences else 0.0
    }


# ---------------------------------------------------------------------------
# DETECTION DES CAPABILITIES OLLAMA (une seule fois au demarrage)
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
VLM_MODEL = "llava"
LLM_MODEL = "mistral"


def detect_ollama_capabilities() -> dict:
    """Detecte les modeles disponibles au demarrage."""
    caps = {"ollama_running": False, "vlm_available": False, "llm_available": False}
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if resp.status_code == 200:
            caps["ollama_running"] = True
            models = [m.get("name", "").split(":")[0] for m in resp.json().get("models", [])]
            caps["vlm_available"] = VLM_MODEL in models
            caps["llm_available"] = LLM_MODEL in models
            print(f"[OK] Ollama detecte. Modeles : {models}")
    except requests.ConnectionError:
        print("[INFO] Ollama non detecte. Pipeline en mode regex.")
        print("       Pour activer : ollama serve && ollama pull llava")
    return caps


OLLAMA_CAPS = detect_ollama_capabilities()


# ---------------------------------------------------------------------------
# EXTRACTION VLM MULTIMODALE (NIVEAU 1) -- Avec retry sur erreur JSON
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Tu es un systeme d'extraction de donnees de factures.
Analyse cette image de facture et retourne UNIQUEMENT un JSON valide.
Si un champ est illisible ou absent, utilise null.
Les montants doivent etre des nombres decimaux sans symboles.

SCHEMA JSON :
{{"numero_facture": "string ou null", "fournisseur": "string ou null",
"date_facture": "string ou null",
"lignes": [{{"designation": "string", "quantite": number, "prix_unitaire": number, "total_ligne": number}}],
"total_ht": number ou null, "tva_taux": number ou null,
"tva_montant": number ou null, "total_ttc": number ou null}}

JSON :"""

RETRY_PROMPT = """Le JSON que tu as produit a echoue la validation :
ERREUR : {error}

Corrige le JSON et retourne UNIQUEMENT le JSON corrige, sans texte avant ni apres.
JSON corrige :"""


def _call_ollama(model: str, prompt: str, images: list = None, timeout: int = 120) -> Optional[str]:
    """Appel bas niveau a l'API Ollama."""
    payload = {"model": model, "prompt": prompt, "temperature": 0,
               "stream": False, "format": "json"}
    if images:
        payload["images"] = images
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()["response"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return raw
    except Exception:
        return None


def _parse_invoice_json(raw_json: str, method: str, raw_text: str = None) -> Optional[InvoiceData]:
    """Parse le JSON brut en InvoiceData avec gestion d'erreur."""
    try:
        data = json.loads(raw_json)
        invoice = InvoiceData(**data, extraction_method=method, raw_text=raw_text)

        # Validation croisee TTC
        if all(v is not None for v in [invoice.total_ht, invoice.tva_montant, invoice.total_ttc]):
            expected = round(invoice.total_ht + invoice.tva_montant, 2)
            invoice.pydantic_ttc_valid = abs(invoice.total_ttc - expected) <= 0.05
        return invoice
    except Exception:
        return None


def extract_with_vlm(image_path: str) -> Optional[InvoiceData]:
    """Niveau 1 : VLM multimodal avec retry sur erreur JSON."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    raw = _call_ollama(VLM_MODEL, EXTRACTION_PROMPT, images=[img_b64])
    if raw is None:
        return None

    invoice = _parse_invoice_json(raw, method="vlm")
    if invoice is not None:
        return invoice

    # RETRY : reinjecter l'erreur dans le prompt
    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        error_msg = str(e)
    else:
        error_msg = "Structure JSON non conforme au schema attendu"

    raw_retry = _call_ollama(VLM_MODEL, RETRY_PROMPT.format(error=error_msg), images=[img_b64])
    if raw_retry:
        return _parse_invoice_json(raw_retry, method="vlm_retry")
    return None


# ---------------------------------------------------------------------------
# EXTRACTION OCR + LLM TEXTUEL (NIVEAU 2) -- Avec retry
# ---------------------------------------------------------------------------

OCR_PROMPT = """Tu es un systeme d'extraction de donnees de factures.
Retourne UNIQUEMENT un JSON valide. Si un champ est absent, utilise null.

SCHEMA : {{"numero_facture": str, "fournisseur": str, "date_facture": str,
"lignes": [{{"designation": str, "quantite": float, "prix_unitaire": float, "total_ligne": float}}],
"total_ht": float, "tva_taux": float, "tva_montant": float, "total_ttc": float}}

TEXTE OCR :
{ocr_text}

JSON :"""


def extract_with_ocr_llm(image_path: str) -> Optional[InvoiceData]:
    """Niveau 2 : OCR textuel + LLM textuel avec retry."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    preprocessed = preprocess_for_ocr(img)
    ocr_out = extract_text_ocr(preprocessed)
    if not ocr_out["full_text"].strip():
        return None

    raw = _call_ollama(LLM_MODEL, OCR_PROMPT.format(ocr_text=ocr_out["full_text"]))
    if raw is None:
        return None

    invoice = _parse_invoice_json(raw, method="ocr_llm", raw_text=ocr_out["full_text"])
    if invoice is not None:
        invoice.ocr_confidence = ocr_out["avg_confidence"]
        return invoice

    # Retry
    raw_retry = _call_ollama(LLM_MODEL, RETRY_PROMPT.format(error="JSON invalide") +
                             "\nTexte original:\n" + ocr_out["full_text"])
    if raw_retry:
        inv = _parse_invoice_json(raw_retry, method="ocr_llm_retry", raw_text=ocr_out["full_text"])
        if inv:
            inv.ocr_confidence = ocr_out["avg_confidence"]
        return inv
    return None


# ---------------------------------------------------------------------------
# EXTRACTION REGEX (NIVEAU 3) -- Dernier recours
# ---------------------------------------------------------------------------

def extract_with_regex(image_path: str) -> InvoiceData:
    """Niveau 3 : Regex minimal, toujours disponible."""
    img = cv2.imread(image_path)
    if img is None:
        return InvoiceData(extraction_method="regex_fallback")

    preprocessed = preprocess_for_ocr(img)
    ocr_out = extract_text_ocr(preprocessed)
    text = ocr_out["full_text"]
    data = InvoiceData(raw_text=text, extraction_method="regex_fallback",
                       ocr_confidence=ocr_out["avg_confidence"])

    for p in [r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", r"(\d{4}[/\-\.]\d{2}[/\-\.]\d{2})"]:
        m = re.search(p, text)
        if m:
            data.date_facture = m.group(1)
            break

    montants = [float(m.replace(",", ".")) for m in re.findall(r"(\d+[.,]\d{2})", text)]
    if montants:
        data.total_ttc = max(montants)

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        data.fournisseur = lines[0]

    max_val = data.total_ttc or 0
    for match in re.finditer(r"(.{2,40}?)\s+(\d+[.,]\d{2})\s*$", text, re.MULTILINE):
        designation = match.group(1).strip()
        if re.search(r"(?i)(total|subtotal|tax|change|cash|tendered|amount|gst|round)", designation):
            continue
        montant = float(match.group(2).replace(",", "."))
        if 0 < montant < max_val:
            data.lignes.append(InvoiceLine(designation=designation,
                                           total_ligne=montant, prix_unitaire=montant))
    return data


# ---------------------------------------------------------------------------
# ORCHESTRATEUR
# ---------------------------------------------------------------------------

def extract_structured_data(image_path: str) -> InvoiceData:
    """Essaie VLM > OCR+LLM > Regex. Avec retry a chaque niveau."""
    if OLLAMA_CAPS["vlm_available"]:
        invoice = extract_with_vlm(image_path)
        if invoice is not None:
            return invoice

    if OLLAMA_CAPS["llm_available"]:
        invoice = extract_with_ocr_llm(image_path)
        if invoice is not None:
            return invoice

    return extract_with_regex(image_path)


# Test d'extraction sur une image
if len(dataset_images) > 0:
    test_invoice = extract_structured_data(dataset_images[0])
    print(f"\n--- Donnees extraites (methode : {test_invoice.extraction_method}) ---")
    print(f"  Fournisseur : {test_invoice.fournisseur}")
    print(f"  Date        : {test_invoice.date_facture}")
    print(f"  Total TTC   : {test_invoice.total_ttc}")
    print(f"  Nb lignes   : {len(test_invoice.lignes)}")
    if test_invoice.pydantic_ttc_valid is not None:
        print(f"  TTC valide  : {test_invoice.pydantic_ttc_valid}")


# ---------------------------------------------------------------------------
# STOCKAGE VECTORIEL (CHROMADB) -- Remplace la liste Python
# ---------------------------------------------------------------------------
# Chaque facture est vectorisee et stockee avec ses metadonnees.
# La recherche de doublons se fait par similarite cosinus filtree
# par fournisseur -- plus rapide et scalable qu'une boucle for.

import chromadb

chroma_client = chromadb.Client()  # In-memory (local, zero config)
invoice_collection = chroma_client.get_or_create_collection(
    name="invoices",
    metadata={"hnsw:space": "cosine"}
)

print("[OK] ChromaDB initialise (collection 'invoices').")


def store_invoice(invoice: InvoiceData, invoice_idx: int):
    """Stocke une facture dans ChromaDB avec ses metadonnees."""
    text = invoice.raw_text or ""
    doc_id = f"inv_{invoice_idx}"

    metadata = {
        "fournisseur": (invoice.fournisseur or "unknown").lower().strip(),
        "total_ttc": invoice.total_ttc or 0.0,
        "date": invoice.date_facture or "",
        "extraction_method": invoice.extraction_method or "",
        "nb_lignes": len(invoice.lignes),
    }

    invoice_collection.upsert(
        ids=[doc_id],
        documents=[text],
        metadatas=[metadata]
    )


def find_similar_invoices(invoice: InvoiceData, exclude_id: str = None,
                           n_results: int = 5) -> list:
    """Recherche les factures similaires par cosine similarity.
    exclude_id : ID a exclure (pour eviter que la facture se trouve elle-meme).
    """
    text = invoice.raw_text or ""
    if not text.strip():
        return []

    try:
        results = invoice_collection.query(
            query_texts=[text],
            n_results=n_results + 1  # +1 pour compenser l'auto-match
        )
        similar = []
        if results and results["distances"]:
            ids = results["ids"][0]
            for i, (doc_id, dist, meta) in enumerate(zip(ids,
                                                          results["distances"][0],
                                                          results["metadatas"][0])):
                if doc_id == exclude_id:
                    continue
                similarity = 1 - dist
                similar.append({"id": doc_id, "similarity": round(similarity, 3), **meta})
                if len(similar) >= n_results:
                    break
        return similar
    except Exception:
        return []


# ---------------------------------------------------------------------------
# MOTEUR DE REGLES DETERMINISTES (NIVEAU 1)
# ---------------------------------------------------------------------------

def check_arithmetic(invoice: InvoiceData) -> list:
    """Coherence arithmetique : somme lignes, HT+TVA=TTC."""
    issues = []
    if invoice.lignes:
        s = sum(l.total_ligne for l in invoice.lignes)
        ref = invoice.total_ht or invoice.total_ttc
        if ref and abs(s - ref) > 0.05 * ref + 0.02:
            issues.append({"rule": "arithmetic_lines", "level": "warning",
                           "detail": f"Somme lignes ({s:.2f}) vs total ({ref:.2f})"})

    if all(v is not None for v in [invoice.total_ht, invoice.tva_montant, invoice.total_ttc]):
        exp = round(invoice.total_ht + invoice.tva_montant, 2)
        if abs(exp - invoice.total_ttc) > 0.05:
            issues.append({"rule": "arithmetic_ttc", "level": "warning",
                           "detail": f"HT+TVA ({exp:.2f}) != TTC ({invoice.total_ttc:.2f})"})
    return issues


def check_hash_duplicate(invoice: InvoiceData, seen_hashes: set) -> list:
    """Detection de doublons exacts par hash SHA256."""
    issues = []
    text = invoice.raw_text or ""
    if not text.strip():
        return issues
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest in seen_hashes:
        issues.append({"rule": "hash_duplicate", "level": "critical",
                       "detail": f"Doublon exact detecte (hash={digest[:12]}...)"})
    seen_hashes.add(digest)
    return issues


def check_semantic_duplicate(invoice: InvoiceData, invoice_idx: int = None) -> list:
    """Detection de doublons semantiques via cosine similarity (ChromaDB)."""
    issues = []
    exclude_id = f"inv_{invoice_idx}" if invoice_idx is not None else None
    similar = find_similar_invoices(invoice, exclude_id=exclude_id, n_results=3)
    for s in similar:
        if s["similarity"] > 0.95:
            issues.append({"rule": "semantic_duplicate", "level": "critical",
                           "detail": f"Doublon semantique (sim={s['similarity']:.3f}, "
                                     f"fournisseur={s['fournisseur']}, ttc={s['total_ttc']})"})
        elif s["similarity"] > 0.85 and s["fournisseur"] == (invoice.fournisseur or "").lower().strip():
            issues.append({"rule": "near_duplicate", "level": "warning",
                           "detail": f"Facture tres similaire (sim={s['similarity']:.3f})"})
    return issues


# ---------------------------------------------------------------------------
# DETECTION ML -- ISOLATION FOREST + Z-SCORE (NIVEAU 3)
# ---------------------------------------------------------------------------
# Isolation Forest travaille en multidimensionnel :
# (montant, nb_lignes, ratio_ttc_ht) au lieu du Z-score monovarie.

from sklearn.ensemble import IsolationForest


def build_feature_vector(invoice: InvoiceData) -> Optional[np.ndarray]:
    """Construit le vecteur de features pour Isolation Forest."""
    if invoice.total_ttc is None or invoice.total_ttc <= 0:
        return None

    montant = invoice.total_ttc
    nb_lignes = len(invoice.lignes)
    ratio_ttc_ht = 1.0
    if invoice.total_ht and invoice.total_ht > 0:
        ratio_ttc_ht = invoice.total_ttc / invoice.total_ht

    sum_lignes = sum(l.total_ligne for l in invoice.lignes) if invoice.lignes else 0
    ecart_lignes = abs(montant - sum_lignes) / montant if montant > 0 else 0

    return np.array([montant, nb_lignes, ratio_ttc_ht, ecart_lignes])


def train_isolation_forest(invoices: list) -> Optional[IsolationForest]:
    """Entraine un Isolation Forest sur l'historique des factures."""
    vectors = []
    for inv in invoices:
        v = build_feature_vector(inv)
        if v is not None:
            vectors.append(v)

    if len(vectors) < 10:
        return None

    X = np.array(vectors)
    clf = IsolationForest(contamination=0.05, n_estimators=200, random_state=42)
    clf.fit(X)
    return clf


def check_isolation_forest(invoice: InvoiceData, model: IsolationForest) -> list:
    """Score une facture avec Isolation Forest."""
    issues = []
    v = build_feature_vector(invoice)
    if v is None:
        return issues

    score = model.decision_function(v.reshape(1, -1))[0]
    if score < -0.1:
        issues.append({"rule": "isolation_forest", "level": "warning",
                       "detail": f"Anomalie multivariee (IF score={score:.3f})"})
    return issues


def check_zscore(invoice: InvoiceData, history: list) -> list:
    """Z-score monovarie sur le montant par fournisseur."""
    issues = []
    if not invoice.fournisseur or invoice.total_ttc is None:
        return issues
    amounts = [h.total_ttc for h in history
               if h.fournisseur and h.total_ttc is not None
               and h.fournisseur.lower().strip() == invoice.fournisseur.lower().strip()]
    if len(amounts) < 5:
        return issues
    mu, sigma = np.mean(amounts), np.std(amounts)
    if sigma > 0:
        z = (invoice.total_ttc - mu) / sigma
        if abs(z) > 3:
            issues.append({"rule": "zscore", "level": "warning",
                           "detail": f"Z-score={z:.1f} (moy={mu:.2f}, sigma={sigma:.2f})"})
    return issues


def check_benford(amounts: list) -> dict:
    """Loi de Benford : P(d) = log10(1 + 1/d). Test du chi-deux."""
    if len(amounts) < 50:
        return {"anomaly": False, "pvalue": None}
    first_digits = []
    for a in amounts:
        if a > 0:
            s = str(a).lstrip("0").replace(".", "").replace("-", "")
            if s and s[0].isdigit() and int(s[0]) >= 1:
                first_digits.append(int(s[0]))
    if len(first_digits) < 50:
        return {"anomaly": False, "pvalue": None}
    counts = Counter(first_digits)
    observed = [counts.get(d, 0) for d in range(1, 10)]
    n = len(first_digits)
    expected = [n * math.log10(1 + 1 / d) for d in range(1, 10)]
    chi2, pvalue = stats.chisquare(observed, expected)
    return {"anomaly": pvalue < 0.05, "pvalue": round(pvalue, 6), "chi2": round(chi2, 2)}


# ---------------------------------------------------------------------------
# LLM REASONING (NIVEAU 4) -- Chain-of-Thought pour decisions ambigues
# ---------------------------------------------------------------------------
# Quand les niveaux precedents declenchent des alertes, le LLM analyse
# l'ensemble des indices et produit une recommandation.

FRAUD_COT_PROMPT = """Tu es un expert en fraude documentaire. Analyse cette facture :

DONNEES EXTRAITES :
- Fournisseur : {fournisseur}
- Date : {date}
- Total HT : {total_ht}
- TVA : {tva_montant}
- Total TTC : {total_ttc}
- Nombre de lignes : {nb_lignes}
- Methode d'extraction : {method}

ALERTES DECLENCHEES :
{alerts}

Raisonne etape par etape :
1. Les montants sont-ils coherents ?
2. Y a-t-il des indices de doublon ou de manipulation ?
3. Quelle est la probabilite de fraude (0-100%) ?

Conclus avec exactement : DECISION: [APPROVE|REVIEW|BLOCK] - RAISON: [explication courte]"""


def llm_fraud_analysis(invoice: InvoiceData, anomalies: list) -> Optional[str]:
    """Niveau 4 : analyse LLM Chain-of-Thought sur les alertes."""
    if not OLLAMA_CAPS["llm_available"] or not anomalies:
        return None

    alerts_text = "\n".join([f"- [{a['level'].upper()}] {a['rule']}: {a['detail']}" for a in anomalies])

    prompt = FRAUD_COT_PROMPT.format(
        fournisseur=invoice.fournisseur or "?",
        date=invoice.date_facture or "?",
        total_ht=invoice.total_ht,
        tva_montant=invoice.tva_montant,
        total_ttc=invoice.total_ttc,
        nb_lignes=len(invoice.lignes),
        method=invoice.extraction_method,
        alerts=alerts_text
    )

    raw = _call_ollama(LLM_MODEL, prompt, timeout=60)
    if raw:
        # Extraire la decision
        for line in raw.split("\n"):
            if "DECISION:" in line.upper():
                return line.strip()
        return raw[-200:]  # Derniers 200 chars si pas de format standard
    return None


# ---------------------------------------------------------------------------
# PIPELINE D'ANALYSE COMPLET (4 niveaux en cascade)
# ---------------------------------------------------------------------------

def analyze_invoice(invoice: InvoiceData, history: list, seen_hashes: set,
                    if_model: IsolationForest = None,
                    invoice_idx: int = None) -> AnomalyReport:
    """Pipeline de detection en 4 niveaux :
    N1 : Regles deterministes (arithmetique, doublons hash, doublons semantiques)
    N2 : (ERP matching - hors scope PoC)
    N3 : ML scoring (Isolation Forest, Z-score, Benford)
    N4 : LLM reasoning (Chain-of-Thought)
    """
    anomalies = []

    # Le regex_fallback extrait des donnees peu fiables (fournisseur = premiere
    # ligne, lignes = toutes les lignes avec un montant). On ne lance PAS les
    # regles qui dependent de la qualite d'extraction sur ce mode.
    is_reliable = invoice.extraction_method not in ("regex_fallback",)

    # --- Niveau 1 : Regles deterministes ---
    if is_reliable:
        anomalies.extend(check_arithmetic(invoice))
    anomalies.extend(check_hash_duplicate(invoice, seen_hashes))
    if is_reliable:
        anomalies.extend(check_semantic_duplicate(invoice, invoice_idx=invoice_idx))

    # --- Niveau 3 : ML scoring ---
    anomalies.extend(check_zscore(invoice, history))
    if if_model is not None:
        anomalies.extend(check_isolation_forest(invoice, if_model))

    # Benford sur l'historique complet
    all_amounts = []
    for h in history + [invoice]:
        if h.total_ttc and h.total_ttc > 0:
            all_amounts.append(h.total_ttc)
        for line in h.lignes:
            if line.total_ligne > 0:
                all_amounts.append(line.total_ligne)
    benford = check_benford(all_amounts)
    if benford["anomaly"]:
        anomalies.append({"rule": "benford", "level": "warning",
                          "detail": f"chi2={benford['chi2']}, p={benford['pvalue']}"})

    # --- Niveau 4 : LLM reasoning (si des alertes existent) ---
    llm_decision = None
    if anomalies and OLLAMA_CAPS["llm_available"]:
        llm_decision = llm_fraud_analysis(invoice, anomalies)

    # IF score pour le rapport
    if_score = None
    if if_model is not None:
        v = build_feature_vector(invoice)
        if v is not None:
            if_score = round(float(if_model.decision_function(v.reshape(1, -1))[0]), 4)

    levels = [a["level"] for a in anomalies]
    overall = "critical" if "critical" in levels else ("warning" if "warning" in levels else "ok")

    return AnomalyReport(
        invoice_id=invoice.numero_facture or "unknown",
        anomalies=anomalies,
        overall_level=overall,
        benford_pvalue=benford["pvalue"],
        isolation_forest_score=if_score,
        llm_decision=llm_decision
    )


print("[OK] Pipeline d'analyse pret (4 niveaux).")


# ---------------------------------------------------------------------------
# EXECUTION SUR LE DATASET
# ---------------------------------------------------------------------------

print("\n--- Phase 1 : Extraction des factures ---")
all_invoices = []
methods_count = {}

n_images = min(50, len(dataset_images))
for i in tqdm(range(n_images), desc="Extraction", unit="facture"):
    img_path = dataset_images[i]

    # Blur check
    img = cv2.imread(img_path)
    if img is None:
        continue
    if not blur_check(img, threshold=50.0):
        continue

    invoice = extract_structured_data(img_path)
    if invoice is not None:
        all_invoices.append(invoice)
        m = invoice.extraction_method or "unknown"
        methods_count[m] = methods_count.get(m, 0) + 1

print(f"[OK] {len(all_invoices)} factures extraites.")
print(f"  Methodes : {methods_count}")


# --- Phase 2 : Stockage vectoriel ---
print("\n--- Phase 2 : Indexation dans ChromaDB ---")
for i, inv in enumerate(tqdm(all_invoices, desc="Indexation", unit="facture")):
    store_invoice(inv, i)
print(f"[OK] {len(all_invoices)} factures indexees.")


# --- Phase 3 : Entrainement Isolation Forest ---
print("\n--- Phase 3 : Entrainement Isolation Forest ---")
if_model = train_isolation_forest(all_invoices)
if if_model:
    print(f"[OK] Isolation Forest entraine sur {len(all_invoices)} factures.")
else:
    print("[INFO] Pas assez de donnees pour Isolation Forest (min 10).")


# --- Phase 4 : Analyse d'anomalies ---
print("\n--- Phase 4 : Detection d'anomalies (4 niveaux) ---")
reports = []
seen_hashes = set()

for i, inv in enumerate(tqdm(all_invoices, desc="Analyse", unit="facture")):
    history = all_invoices[:i]
    report = analyze_invoice(inv, history, seen_hashes, if_model, invoice_idx=i)
    reports.append(report)

n_ok = sum(1 for r in reports if r.overall_level == "ok")
n_warn = sum(1 for r in reports if r.overall_level == "warning")
n_crit = sum(1 for r in reports if r.overall_level == "critical")
print(f"\n  OK: {n_ok}  |  Warnings: {n_warn}  |  Critical: {n_crit}")


# ---------------------------------------------------------------------------
# VISUALISATIONS
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution des montants
all_ttc = [inv.total_ttc for inv in all_invoices if inv.total_ttc and inv.total_ttc > 0]
if all_ttc:
    axes[0, 0].hist(all_ttc, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0, 0].set_title("Distribution des montants TTC")
    axes[0, 0].set_xlabel("Montant")

# 2. Loi de Benford
all_benford = []
for inv in all_invoices:
    if inv.total_ttc and inv.total_ttc > 0:
        all_benford.append(inv.total_ttc)
    for line in inv.lignes:
        if line.total_ligne > 0:
            all_benford.append(line.total_ligne)

if len(all_benford) >= 10:
    fd = []
    for a in all_benford:
        s = str(a).lstrip("0").replace(".", "").replace("-", "")
        if s and s[0].isdigit() and int(s[0]) >= 1:
            fd.append(int(s[0]))
    if fd:
        c = Counter(fd)
        obs = [c.get(d, 0) / len(fd) for d in range(1, 10)]
        theo = [math.log10(1 + 1 / d) for d in range(1, 10)]
        x = range(1, 10)
        w = 0.35
        axes[0, 1].bar([i - w/2 for i in x], obs, w, label="Observe", color="steelblue")
        axes[0, 1].bar([i + w/2 for i in x], theo, w, label="Benford", color="coral")
        axes[0, 1].set_title("Loi de Benford")
        axes[0, 1].legend()

# 3. Anomalies
labels_p = ["OK", "Warning", "Critical"]
sizes_p = [n_ok, n_warn, n_crit]
colors_p = ["#4CAF50", "#FF9800", "#F44336"]
nz = [(l, s, c) for l, s, c in zip(labels_p, sizes_p, colors_p) if s > 0]
if nz:
    l, s, c = zip(*nz)
    axes[1, 0].pie(s, labels=l, colors=c, autopct="%1.0f%%", startangle=90)
    axes[1, 0].set_title("Anomalies detectees")

# 4. Isolation Forest scores
if if_model:
    if_scores = []
    for inv in all_invoices:
        v = build_feature_vector(inv)
        if v is not None:
            score = if_model.decision_function(v.reshape(1, -1))[0]
            if_scores.append(score)
    if if_scores:
        axes[1, 1].hist(if_scores, bins=20, color="teal", edgecolor="white", alpha=0.8)
        axes[1, 1].axvline(-0.1, color="red", linestyle="--", label="Seuil anomalie")
        axes[1, 1].set_title("Scores Isolation Forest")
        axes[1, 1].set_xlabel("Score (< -0.1 = anomalie)")
        axes[1, 1].legend()

plt.suptitle("Analyse du portefeuille de factures", fontsize=14)
plt.tight_layout()
plt.show()

# Methodes d'extraction
if methods_count:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors_m = {"vlm": "#2196F3", "vlm_retry": "#64B5F6",
                "ocr_llm": "#FF9800", "ocr_llm_retry": "#FFB74D",
                "regex_fallback": "#F44336"}
    labels_m = list(methods_count.keys())
    sizes_m = list(methods_count.values())
    cols_m = [colors_m.get(m, "#999") for m in labels_m]
    ax.pie(sizes_m, labels=labels_m, colors=cols_m, autopct="%1.0f%%", startangle=90)
    ax.set_title("Methodes d'extraction utilisees")
    plt.tight_layout()
    plt.show()

# Exemple detaille
for i, r in enumerate(reports):
    if r.anomalies:
        inv = all_invoices[i]
        print(f"\n=== Facture {i} : {inv.fournisseur or '?'} (via {inv.extraction_method}) ===")
        print(f"  TTC : {inv.total_ttc}")
        if r.isolation_forest_score is not None:
            print(f"  IF score : {r.isolation_forest_score}")
        for a in r.anomalies:
            print(f"  [{a['level'].upper()}] {a['rule']} : {a['detail']}")
        if r.llm_decision:
            print(f"  LLM : {r.llm_decision}")
        break

print("\n[OK] Pipeline termine.")
print("\nArchitecture de detection en 4 niveaux :")
print("  N1 : Regles deterministes (arithmetique, doublons hash, doublons cosinus)")
print("  N2 : Rapprochement ERP (hors scope PoC)")
print("  N3 : ML scoring (Isolation Forest multivarie + Z-score + Benford)")
print("  N4 : LLM reasoning Chain-of-Thought (si alertes et Ollama dispo)")
print(f"\n  Resultats : {n_ok} OK / {n_warn} warnings / {n_crit} critiques")
