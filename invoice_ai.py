# -*- coding: utf-8 -*-
"""
Extracteur Intelligent de Factures (VLM Multimodal + Regles Metier)
Dataset : SROIE (Scanned Receipts OCR and Information Extraction)
Extraction : VLM multimodal (LLaVA via Ollama) -- on "regarde" l'image, on ne parse plus
Fallback : OCR (EasyOCR) + LLM textuel (Mistral) + regex en dernier recours
"""

import os
import re
import math
import json
import base64
import warnings
from pathlib import Path
from collections import Counter
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pydantic import BaseModel
from scipy import stats
import requests

warnings.filterwarnings("ignore")

print("[OK] Imports termines.")


# ---------------------------------------------------------------------------
# MODELES DE DONNEES (PYDANTIC)
# ---------------------------------------------------------------------------
# C'est le contrat strict entre l'extraction (quel que soit le backend)
# et le moteur de regles. Pydantic valide, type, et rejette si non conforme.

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
    extraction_method: Optional[str] = None  # "vlm", "ocr_llm", "regex_fallback"


class AnomalyReport(BaseModel):
    invoice_id: str = "unknown"
    anomalies: list[dict] = []
    overall_level: str = "ok"
    benford_pvalue: Optional[float] = None


# ---------------------------------------------------------------------------
# TELECHARGEMENT DU DATASET SROIE
# ---------------------------------------------------------------------------

import kagglehub

print("Telechargement du dataset SROIE depuis Kaggle...")

try:
    dataset_path = kagglehub.dataset_download("urbikn/sroie-datasetv2")
    print(f"[OK] Dataset telecharge dans : {dataset_path}")
except Exception as e:
    print(f"[ERREUR] {e}")
    print("Alternative : https://www.kaggle.com/datasets/urbikn/sroie-datasetv2")
    dataset_path = "./sroie_data"

for root, dirs, files in os.walk(dataset_path):
    level = root.replace(str(dataset_path), "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 2:
        sub_indent = " " * 2 * (level + 1)
        for f in files[:5]:
            print(f"{sub_indent}{f}")
        if len(files) > 5:
            print(f"{sub_indent}... ({len(files)} fichiers)")


# ---------------------------------------------------------------------------
# CHARGEMENT DES IMAGES
# ---------------------------------------------------------------------------

def find_images_and_labels(base_path: str) -> dict:
    """Parcourt le dataset SROIE et associe images et annotations."""
    base = Path(base_path)
    data = {"images": [], "texts": []}
    img_files = list(base.rglob("*.jpg")) + list(base.rglob("*.png"))
    txt_files = list(base.rglob("*.txt"))
    img_map = {p.stem: p for p in img_files}
    txt_map = {p.stem: p for p in txt_files}

    for stem, img_path in sorted(img_map.items()):
        data["images"].append(str(img_path))
        if stem in txt_map:
            with open(txt_map[stem], "r", encoding="utf-8", errors="ignore") as f:
                data["texts"].append(f.read())
        else:
            data["texts"].append("")

    print(f"[OK] {len(data['images'])} images trouvees.")
    return data


dataset = find_images_and_labels(dataset_path)

n_preview = min(4, len(dataset["images"]))
if n_preview > 0:
    fig, axes = plt.subplots(1, n_preview, figsize=(5 * n_preview, 6))
    if n_preview == 1:
        axes = [axes]
    for i in range(n_preview):
        img = cv2.imread(dataset["images"][i])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Receipt {i+1}", fontsize=10)
        axes[i].axis("off")
    plt.suptitle("Exemples de tickets du dataset SROIE", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# NIVEAU 1 : EXTRACTION VLM MULTIMODALE (ETAT DE L'ART 2026)
# ---------------------------------------------------------------------------
# C'est la revolution Document AI : on envoie l'IMAGE BRUTE directement
# a un modele Vision-Language (LLaVA, Qwen-VL, Llava-Phi3...) via Ollama.
#
# Le modele "regarde" la facture, comprend la structure spatiale (colonnes,
# tableaux, alignements), et retourne le JSON structure.
#
# Avantages :
# - Zero preprocessing (pas de deskew, pas de binarisation)
# - Zero OCR intermediaire (pas d'EasyOCR, pas de perte d'info spatiale)
# - Comprend le layout 2D natalement
#
# Prerequis : ollama pull llava (ou llava-phi3, ou qwen2-vl)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
VLM_MODEL = "llava"       # Modele multimodal (vision + langage)
LLM_MODEL = "mistral"     # Modele textuel (fallback niveau 2)


def detect_ollama_capabilities() -> dict:
    """Detecte une seule fois au demarrage quels backends sont disponibles.

    Evite de retenter Ollama a chaque image si le serveur n'est pas lance.
    Verifie aussi quels modeles sont installes (llava, mistral, etc.).
    """
    capabilities = {"ollama_running": False, "vlm_available": False, "llm_available": False}

    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if resp.status_code == 200:
            capabilities["ollama_running"] = True
            models = [m.get("name", "").split(":")[0] for m in resp.json().get("models", [])]
            capabilities["vlm_available"] = VLM_MODEL in models
            capabilities["llm_available"] = LLM_MODEL in models
            print(f"[OK] Ollama detecte. Modeles installes : {models}")
            if not capabilities["vlm_available"]:
                print(f"  [INFO] VLM '{VLM_MODEL}' non installe (ollama pull {VLM_MODEL})")
            if not capabilities["llm_available"]:
                print(f"  [INFO] LLM '{LLM_MODEL}' non installe (ollama pull {LLM_MODEL})")
    except requests.ConnectionError:
        print("[INFO] Ollama non detecte. Le pipeline utilisera le fallback regex.")
        print("       Pour activer les niveaux 1-2 : ollama serve && ollama pull llava")

    return capabilities


OLLAMA_CAPS = detect_ollama_capabilities()

EXTRACTION_PROMPT = """Tu es un systeme d'extraction de donnees de factures et tickets de caisse.
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


def image_to_base64(image_path: str) -> str:
    """Encode une image en base64 pour l'API multimodale d'Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_with_vlm(image_path: str) -> Optional[InvoiceData]:
    """Niveau 1 : Extraction multimodale (VLM).

    L'image brute est envoyee directement au modele de vision.
    Pas d'OCR, pas de preprocessing. Le VLM "regarde" et structure.
    """
    try:
        img_b64 = image_to_base64(image_path)

        response = requests.post(OLLAMA_URL, json={
            "model": VLM_MODEL,
            "prompt": EXTRACTION_PROMPT,
            "images": [img_b64],
            "temperature": 0,
            "stream": False,
            "format": "json"
        }, timeout=120)
        response.raise_for_status()

        raw = response.json()["response"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        data = json.loads(raw)
        return InvoiceData(**data, extraction_method="vlm")

    except requests.ConnectionError:
        print("[VLM] Ollama non disponible.")
        return None
    except json.JSONDecodeError as e:
        print(f"[VLM] JSON invalide : {e}")
        return None
    except Exception as e:
        print(f"[VLM] Erreur : {e}")
        return None


# ---------------------------------------------------------------------------
# NIVEAU 2 : OCR + LLM TEXTUEL (FALLBACK ROBUSTE)
# ---------------------------------------------------------------------------
# Si le VLM n'est pas disponible (pas de modele llava installe),
# on revient a l'approche OCR texte + LLM textuel.
# EasyOCR extrait le texte brut, Mistral le structure.

import logging
logging.getLogger("easyocr").setLevel(logging.ERROR)

import easyocr

print("Initialisation d'EasyOCR (fallback)...")
reader = easyocr.Reader(["en", "fr"], gpu=False)
print("[OK] EasyOCR initialise.")


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocessing classique pour l'OCR textuel.

    Conserve pour le fallback niveau 2. En niveau 1 (VLM), ce bloc
    n'est jamais appele -- le modele multimodal gere le bruit lui-meme.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=15, C=4
    )
    return binary


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


OCR_PROMPT = """Tu es un systeme d'extraction de donnees de factures.
Retourne UNIQUEMENT un JSON valide. Si un champ est absent, utilise null.
Les montants en nombres decimaux.

SCHEMA : {{"numero_facture": str, "fournisseur": str, "date_facture": str,
"lignes": [{{"designation": str, "quantite": float, "prix_unitaire": float, "total_ligne": float}}],
"total_ht": float, "tva_taux": float, "tva_montant": float, "total_ttc": float}}

TEXTE OCR :
{ocr_text}

JSON :"""


def extract_with_ocr_llm(image_path: str) -> Optional[InvoiceData]:
    """Niveau 2 : OCR textuel + LLM textuel."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        preprocessed = preprocess_for_ocr(img)
        ocr_out = extract_text_ocr(preprocessed)

        if not ocr_out["full_text"].strip():
            return None

        response = requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": OCR_PROMPT.format(ocr_text=ocr_out["full_text"]),
            "temperature": 0,
            "stream": False,
            "format": "json"
        }, timeout=60)
        response.raise_for_status()

        raw = response.json()["response"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        data = json.loads(raw)
        invoice = InvoiceData(**data, extraction_method="ocr_llm",
                              raw_text=ocr_out["full_text"],
                              ocr_confidence=ocr_out["avg_confidence"])
        return invoice

    except Exception as e:
        print(f"[OCR+LLM] Erreur : {e}")
        return None


# ---------------------------------------------------------------------------
# NIVEAU 3 : REGEX FALLBACK (DERNIER RECOURS)
# ---------------------------------------------------------------------------

def extract_with_regex(image_path: str) -> InvoiceData:
    """Niveau 3 : Regex minimal si aucun LLM n'est disponible."""
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

    return data


# ---------------------------------------------------------------------------
# ORCHESTRATEUR : DEGRADATION GRACIEUSE
# ---------------------------------------------------------------------------

def extract_structured_data(image_path: str) -> InvoiceData:
    """Point d'entree unique. Tente les niveaux disponibles en cascade.

    La detection des capabilities est faite UNE SEULE FOIS au demarrage
    (via detect_ollama_capabilities). On ne retente pas Ollama a chaque image
    si le serveur n'est pas lance.
    """
    # Niveau 1 : VLM multimodal (si disponible)
    if OLLAMA_CAPS["vlm_available"]:
        invoice = extract_with_vlm(image_path)
        if invoice is not None:
            return invoice

    # Niveau 2 : OCR + LLM textuel (si disponible)
    if OLLAMA_CAPS["llm_available"]:
        invoice = extract_with_ocr_llm(image_path)
        if invoice is not None:
            return invoice

    # Niveau 3 : Regex (toujours disponible)
    return extract_with_regex(image_path)


# Test d'extraction
if len(dataset["images"]) > 0:
    test_invoice = extract_structured_data(dataset["images"][0])
    print(f"\n--- Donnees extraites (methode : {test_invoice.extraction_method}) ---")
    print(f"  Fournisseur : {test_invoice.fournisseur}")
    print(f"  Date        : {test_invoice.date_facture}")
    print(f"  Total TTC   : {test_invoice.total_ttc}")
    print(f"  Nb lignes   : {len(test_invoice.lignes)}")


# ---------------------------------------------------------------------------
# MOTEUR DE REGLES (DETECTION D'ANOMALIES)
# ---------------------------------------------------------------------------
# Independant de la methode d'extraction. Que les donnees viennent du VLM,
# de l'OCR+LLM ou du regex, les regles s'appliquent identiquement.

def check_arithmetic(invoice: InvoiceData) -> list:
    issues = []
    if invoice.lignes:
        s = sum(l.total_ligne for l in invoice.lignes)
        ref = invoice.total_ht or invoice.total_ttc
        if ref and abs(s - ref) > 0.05 * ref + 0.02:
            issues.append({"rule": "arithmetic_lines", "level": "warning",
                           "detail": f"Somme lignes ({s:.2f}) vs total ({ref:.2f})"})
    if all(v is not None for v in [invoice.total_ht, invoice.tva_montant, invoice.total_ttc]):
        exp = invoice.total_ht + invoice.tva_montant
        if abs(exp - invoice.total_ttc) > 0.05:
            issues.append({"rule": "arithmetic_ttc", "level": "warning",
                           "detail": f"HT+TVA ({exp:.2f}) != TTC ({invoice.total_ttc:.2f})"})
    return issues


def check_duplicates(invoice: InvoiceData, history: list) -> list:
    issues = []
    for hist in history:
        same = (invoice.fournisseur and hist.fournisseur
                and invoice.fournisseur.lower().strip() == hist.fournisseur.lower().strip())
        same_amt = (invoice.total_ttc is not None and hist.total_ttc is not None
                    and abs(invoice.total_ttc - hist.total_ttc) < 0.01)
        if same and same_amt:
            issues.append({"rule": "duplicate", "level": "critical",
                           "detail": f"Doublon : {invoice.fournisseur} / {invoice.total_ttc}"})
    return issues


def check_outlier(invoice: InvoiceData, history: list) -> list:
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
            issues.append({"rule": "outlier", "level": "warning",
                           "detail": f"Z-score={z:.1f} (moy={mu:.2f})"})
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


def analyze_invoice(invoice: InvoiceData, history: list = None) -> AnomalyReport:
    """Pipeline de detection d'anomalies (arithmetique, doublons, outliers, Benford)."""
    history = history or []
    anomalies = []
    anomalies.extend(check_arithmetic(invoice))
    anomalies.extend(check_duplicates(invoice, history))
    anomalies.extend(check_outlier(invoice, history))

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

    levels = [a["level"] for a in anomalies]
    overall = "critical" if "critical" in levels else ("warning" if "warning" in levels else "ok")
    return AnomalyReport(invoice_id=invoice.numero_facture or "unknown",
                         anomalies=anomalies, overall_level=overall,
                         benford_pvalue=benford["pvalue"])


print("[OK] Moteur de regles pret.")


# ---------------------------------------------------------------------------
# PIPELINE COMPLET SUR LE DATASET
# ---------------------------------------------------------------------------

print("Execution du pipeline sur le dataset...")
all_invoices = []
methods_count = {}

n_images = min(30, len(dataset["images"]))
for i in range(n_images):
    invoice = extract_structured_data(dataset["images"][i])
    if invoice is not None:
        all_invoices.append(invoice)
        m = invoice.extraction_method or "unknown"
        methods_count[m] = methods_count.get(m, 0) + 1
    if (i + 1) % 5 == 0:
        print(f"  Traite {i+1}/{n_images}...")

print(f"[OK] {len(all_invoices)} factures extraites.")
print(f"  Methodes utilisees : {methods_count}")

print("\nAnalyse des anomalies...")
reports = []
for i, inv in enumerate(all_invoices):
    reports.append(analyze_invoice(inv, all_invoices[:i]))

n_ok = sum(1 for r in reports if r.overall_level == "ok")
n_warn = sum(1 for r in reports if r.overall_level == "warning")
n_crit = sum(1 for r in reports if r.overall_level == "critical")
print(f"  OK: {n_ok}  |  Warnings: {n_warn}  |  Critical: {n_crit}")


# ---------------------------------------------------------------------------
# VISUALISATIONS
# ---------------------------------------------------------------------------

all_ttc = [inv.total_ttc for inv in all_invoices if inv.total_ttc and inv.total_ttc > 0]
if all_ttc:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(all_ttc, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title("Distribution des montants TTC")

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
            axes[1].bar([i - w/2 for i in x], obs, w, label="Observe", color="steelblue")
            axes[1].bar([i + w/2 for i in x], theo, w, label="Benford", color="coral")
            axes[1].set_title("Loi de Benford")
            axes[1].legend()

    labels_p = ["OK", "Warning", "Critical"]
    sizes_p = [n_ok, n_warn, n_crit]
    colors_p = ["#4CAF50", "#FF9800", "#F44336"]
    nz = [(l, s, c) for l, s, c in zip(labels_p, sizes_p, colors_p) if s > 0]
    if nz:
        l, s, c = zip(*nz)
        axes[2].pie(s, labels=l, colors=c, autopct="%1.0f%%", startangle=90)
        axes[2].set_title("Anomalies")

    plt.tight_layout()
    plt.show()

# Methodes d'extraction utilisees (pie chart)
if methods_count:
    fig, ax = plt.subplots(figsize=(6, 4))
    method_colors = {"vlm": "#2196F3", "ocr_llm": "#FF9800", "regex_fallback": "#F44336"}
    labels_m = list(methods_count.keys())
    sizes_m = list(methods_count.values())
    colors_m = [method_colors.get(m, "#999") for m in labels_m]
    ax.pie(sizes_m, labels=labels_m, colors=colors_m, autopct="%1.0f%%", startangle=90)
    ax.set_title("Methodes d'extraction utilisees")
    plt.tight_layout()
    plt.show()

# Exemple detaille
for i, r in enumerate(reports):
    if r.anomalies:
        inv = all_invoices[i]
        print(f"\n=== Facture {i} : {inv.fournisseur or '?'} (via {inv.extraction_method}) ===")
        print(f"  TTC : {inv.total_ttc}")
        for a in r.anomalies:
            print(f"  [{a['level'].upper()}] {a['rule']} : {a['detail']}")
        break

print("\n[OK] Pipeline termine.")
print("\nArchitecture de degradation gracieuse :")
print("  Niveau 1 (VLM)     : Image brute -> LLaVA multimodal -> JSON -> Pydantic")
print("  Niveau 2 (OCR+LLM) : Image -> preprocess -> EasyOCR -> Mistral -> JSON -> Pydantic")
print("  Niveau 3 (Regex)   : Image -> preprocess -> EasyOCR -> regex -> Pydantic")
print(f"  Methodes utilisees sur ce run : {methods_count}")
