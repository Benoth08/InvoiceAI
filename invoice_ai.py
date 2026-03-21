# -*- coding: utf-8 -*-
"""
Extracteur Intelligent de Factures (OCR + Regles Metier)
Dataset : SROIE (Scanned Receipts OCR and Information Extraction)
"""

# =============================================================================
# CELLULE 1 : INSTALLATION DES DEPENDANCES
# =============================================================================
# !pip install -q easyocr opencv-python-headless pydantic pillow kagglehub scipy

# =============================================================================
# CELLULE 2 : IMPORTS
# =============================================================================
import os
import re
import math
import json
import warnings
from pathlib import Path
from datetime import date, timedelta
from collections import Counter
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from scipy import stats

warnings.filterwarnings("ignore")

print("[OK] Imports termines.")

# =============================================================================
# CELLULE 3 : TELECHARGEMENT DU DATASET SROIE
# =============================================================================
# Le dataset SROIE contient des images de tickets de caisse reels
# avec les annotations ground truth (texte, entites).

import kagglehub

print("Telechargement du dataset SROIE depuis Kaggle...")
print("(Necessite un compte Kaggle. En Colab : Settings > Secrets > KAGGLE_USERNAME / KAGGLE_KEY)")

try:
    dataset_path = kagglehub.dataset_download("urbikn/sroie-datasetv2")
    print(f"[OK] Dataset telecharge dans : {dataset_path}")
except Exception as e:
    print(f"[ERREUR] Impossible de telecharger via kagglehub : {e}")
    print("Alternative : telechargez manuellement depuis https://www.kaggle.com/datasets/urbikn/sroie-datasetv2")
    print("et placez les fichiers dans ./sroie_data/")
    dataset_path = "./sroie_data"

# Explorer la structure du dataset
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

# =============================================================================
# CELLULE 4 : CHARGEMENT DES IMAGES ET ANNOTATIONS
# =============================================================================

def find_images_and_labels(base_path: str) -> dict:
    """Parcourt le dataset SROIE et associe images et annotations."""
    base = Path(base_path)
    data = {"images": [], "texts": [], "entities": []}

    # SROIE structure : img/ pour les images, box/ ou annot/ pour les annotations
    img_dirs = list(base.rglob("*.jpg")) + list(base.rglob("*.png"))
    txt_dirs = list(base.rglob("*.txt"))

    # Associer par nom de fichier (sans extension)
    img_map = {p.stem: p for p in img_dirs}
    txt_map = {p.stem: p for p in txt_dirs}

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

# Afficher quelques exemples
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

# =============================================================================
# CELLULE 5 : PREPROCESSING DES IMAGES
# =============================================================================

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Pipeline de pretraitement optimise pour l'OCR sur tickets de caisse.
    
    Ordre : niveaux de gris -> deskew -> debruitage -> binarisation adaptative.
    """
    # Conversion en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Correction d'inclinaison (deskew)
    angle = _detect_skew(gray)
    if abs(angle) > 0.5:
        gray = _rotate(gray, angle)

    # Debruitage (filtre bilateral : preserve les aretes des caracteres)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Binarisation adaptative (gere l'eclairage non uniforme)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=4
    )
    return binary


def _detect_skew(image: np.ndarray) -> float:
    """Detecte l'angle d'inclinaison via la transformee de Hough."""
    edges = cv2.Canny(image, 50, 200, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle_deg) < 30:
            angles.append(angle_deg)
    return float(np.median(angles)) if angles else 0.0


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotation autour du centre."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# Visualiser le pretraitement sur un exemple
if len(dataset["images"]) > 0:
    sample_img = cv2.imread(dataset["images"][0])
    preprocessed = preprocess_for_ocr(sample_img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Image originale")
    ax1.axis("off")
    ax2.imshow(preprocessed, cmap="gray")
    ax2.set_title("Apres pretraitement (binarisation adaptative)")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()

print("[OK] Module de pretraitement pret.")

# =============================================================================
# CELLULE 6 : EXTRACTION OCR AVEC EASYOCR
# =============================================================================
import easyocr

print("Initialisation d'EasyOCR (premier lancement : telechargement des modeles)...")
reader = easyocr.Reader(["en", "fr"], gpu=False)
print("[OK] EasyOCR initialise.")


def extract_text_ocr(image: np.ndarray, confidence_threshold: float = 0.3) -> dict:
    """Extrait le texte d'une image via EasyOCR.
    
    Returns:
        dict avec full_text, blocks (detail par zone), avg_confidence.
    """
    results = reader.readtext(image)

    blocks = []
    texts = []
    confidences = []

    for (bbox, text, confidence) in results:
        if confidence >= confidence_threshold:
            blocks.append({
                "text": text,
                "confidence": round(confidence, 3),
                "bbox": bbox
            })
            texts.append(text)
            confidences.append(confidence)

    return {
        "full_text": "\n".join(texts),
        "blocks": blocks,
        "avg_confidence": round(np.mean(confidences), 3) if confidences else 0.0
    }


# Test OCR sur un exemple
if len(dataset["images"]) > 0:
    test_img = cv2.imread(dataset["images"][0])
    test_preprocessed = preprocess_for_ocr(test_img)
    ocr_result = extract_text_ocr(test_preprocessed)

    print(f"\n--- Texte OCR extrait (confiance moyenne : {ocr_result['avg_confidence']:.1%}) ---")
    print(ocr_result["full_text"][:500])
    print("---")

# =============================================================================
# CELLULE 7 : MODELES DE DONNEES (PYDANTIC)
# =============================================================================

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


class AnomalyReport(BaseModel):
    invoice_id: str = "unknown"
    anomalies: list[dict] = []
    overall_level: str = "ok"  # ok, warning, critical
    benford_pvalue: Optional[float] = None


print("[OK] Modeles de donnees definis.")

# =============================================================================
# CELLULE 8 : EXTRACTION STRUCTUREE (REGEX)
# =============================================================================

def extract_structured_data(ocr_text: str) -> InvoiceData:
    """Extrait les donnees structurees d'un texte OCR par expressions regulieres.
    
    Cette approche fonctionne sans LLM et couvre les patterns les plus courants
    sur les tickets de caisse (dates, montants, totaux).
    """
    data = InvoiceData(raw_text=ocr_text)

    # --- Date ---
    date_patterns = [
        r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})",   # DD/MM/YYYY
        r"(\d{4}[/\-\.]\d{2}[/\-\.]\d{2})",    # YYYY/MM/DD
        r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{2})",    # DD/MM/YY
    ]
    for pattern in date_patterns:
        match = re.search(pattern, ocr_text)
        if match:
            data.date_facture = match.group(1)
            break

    # --- Numero de facture / ticket ---
    invoice_patterns = [
        r"(?:invoice|facture|receipt|ticket|no?\.?\s*)[:#]?\s*(\w[\w\-]+)",
        r"(?:INV|FAC|REC)[:\-#]?\s*(\w+)",
    ]
    for pattern in invoice_patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            data.numero_facture = match.group(1)
            break

    # --- Montants (tous les nombres avec decimales trouves dans le texte) ---
    montants = re.findall(r"(\d+[.,]\d{2})", ocr_text)
    montants_float = [float(m.replace(",", ".")) for m in montants]

    # Le plus grand montant est probablement le total TTC
    if montants_float:
        data.total_ttc = max(montants_float)

    # --- Total HT et TVA ---
    ht_patterns = [
        r"(?:subtotal|sous.?total|total\s*ht|net|hors\s*taxe)[:\s]*(\d+[.,]\d{2})",
    ]
    for pattern in ht_patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            data.total_ht = float(match.group(1).replace(",", "."))
            break

    tva_patterns = [
        r"(?:tax|tva|vat|gst)[:\s]*(\d+[.,]\d{2})",
    ]
    for pattern in tva_patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            data.tva_montant = float(match.group(1).replace(",", "."))
            break

    # --- Fournisseur (premiere ligne non vide du texte) ---
    lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
    if lines:
        data.fournisseur = lines[0]

    # --- Lignes de detail (pattern : texte + montant) ---
    line_pattern = r"([A-Za-z][\w\s]{2,30})\s+(\d+[.,]\d{2})"
    for match in re.finditer(line_pattern, ocr_text):
        designation = match.group(1).strip()
        montant = float(match.group(2).replace(",", "."))
        if montant != data.total_ttc:  # Exclure le total
            data.lignes.append(InvoiceLine(
                designation=designation,
                total_ligne=montant,
                prix_unitaire=montant
            ))

    return data


# Test sur l'exemple OCR
if len(dataset["images"]) > 0:
    test_data = extract_structured_data(ocr_result["full_text"])
    print("\n--- Donnees extraites ---")
    print(f"  Fournisseur  : {test_data.fournisseur}")
    print(f"  Date         : {test_data.date_facture}")
    print(f"  No. facture  : {test_data.numero_facture}")
    print(f"  Total TTC    : {test_data.total_ttc}")
    print(f"  Total HT     : {test_data.total_ht}")
    print(f"  TVA          : {test_data.tva_montant}")
    print(f"  Nb lignes    : {len(test_data.lignes)}")

# =============================================================================
# CELLULE 9 : MOTEUR DE REGLES (DETECTION D'ANOMALIES)
# =============================================================================

def check_arithmetic(invoice: InvoiceData) -> list:
    """Verifie la coherence arithmetique interne de la facture."""
    issues = []
    tolerance = 0.05

    # Somme des lignes vs Total HT ou TTC
    if invoice.lignes:
        sum_lines = sum(l.total_ligne for l in invoice.lignes)
        ref = invoice.total_ht or invoice.total_ttc
        if ref and abs(sum_lines - ref) > tolerance * ref + 0.02:
            issues.append({
                "rule": "arithmetic_lines_total",
                "level": "warning",
                "detail": f"Somme lignes ({sum_lines:.2f}) vs total ({ref:.2f}) : ecart significatif"
            })

    # HT + TVA = TTC
    if all(v is not None for v in [invoice.total_ht, invoice.tva_montant, invoice.total_ttc]):
        expected = invoice.total_ht + invoice.tva_montant
        if abs(expected - invoice.total_ttc) > 0.05:
            issues.append({
                "rule": "arithmetic_ht_tva_ttc",
                "level": "warning",
                "detail": f"HT ({invoice.total_ht:.2f}) + TVA ({invoice.tva_montant:.2f}) != TTC ({invoice.total_ttc:.2f})"
            })

    return issues


def check_duplicates(invoice: InvoiceData, history: list) -> list:
    """Detecte les doublons potentiels (meme fournisseur + montant + date proche)."""
    issues = []
    for hist in history:
        same_supplier = (
            invoice.fournisseur and hist.fournisseur
            and invoice.fournisseur.lower().strip() == hist.fournisseur.lower().strip()
        )
        same_amount = (
            invoice.total_ttc is not None and hist.total_ttc is not None
            and abs(invoice.total_ttc - hist.total_ttc) < 0.01
        )
        if same_supplier and same_amount:
            issues.append({
                "rule": "duplicate_suspected",
                "level": "critical",
                "detail": f"Doublon potentiel : meme fournisseur ({invoice.fournisseur}) et montant ({invoice.total_ttc})"
            })
    return issues


def check_outlier(invoice: InvoiceData, history: list) -> list:
    """Detecte les montants aberrants par fournisseur (Z-score > 3)."""
    issues = []
    if not invoice.fournisseur or invoice.total_ttc is None:
        return issues

    amounts = [
        h.total_ttc for h in history
        if h.fournisseur and h.total_ttc is not None
        and h.fournisseur.lower().strip() == invoice.fournisseur.lower().strip()
    ]
    if len(amounts) < 5:
        return issues

    mu = np.mean(amounts)
    sigma = np.std(amounts)
    if sigma > 0:
        z = (invoice.total_ttc - mu) / sigma
        if abs(z) > 3:
            issues.append({
                "rule": "amount_outlier",
                "level": "warning",
                "detail": f"Montant {invoice.total_ttc:.2f} anormal (Z-score={z:.1f}, moyenne={mu:.2f})"
            })
    return issues


def check_benford(amounts: list) -> dict:
    """Applique la loi de Benford sur un ensemble de montants.
    
    Loi de Benford : P(d) = log10(1 + 1/d) pour d in {1,...,9}
    Test du chi-deux pour comparer distribution empirique vs theorique.
    """
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
    """Pipeline complet de detection d'anomalies."""
    history = history or []
    anomalies = []

    anomalies.extend(check_arithmetic(invoice))
    anomalies.extend(check_duplicates(invoice, history))
    anomalies.extend(check_outlier(invoice, history))

    # Benford sur l'historique complet
    all_amounts = []
    for h in history + [invoice]:
        if h.total_ttc and h.total_ttc > 0:
            all_amounts.append(h.total_ttc)
        for line in h.lignes:
            if line.total_ligne > 0:
                all_amounts.append(line.total_ligne)

    benford_result = check_benford(all_amounts)
    benford_pvalue = benford_result["pvalue"]
    if benford_result["anomaly"]:
        anomalies.append({
            "rule": "benford_violation",
            "level": "warning",
            "detail": f"Distribution non conforme a Benford (chi2={benford_result['chi2']}, p={benford_pvalue})"
        })

    levels = [a["level"] for a in anomalies]
    if "critical" in levels:
        overall = "critical"
    elif "warning" in levels:
        overall = "warning"
    else:
        overall = "ok"

    return AnomalyReport(
        invoice_id=invoice.numero_facture or "unknown",
        anomalies=anomalies,
        overall_level=overall,
        benford_pvalue=benford_pvalue
    )


print("[OK] Moteur de regles pret.")

# =============================================================================
# CELLULE 10 : PIPELINE COMPLET SUR LE DATASET
# =============================================================================

def run_full_pipeline(image_path: str) -> tuple:
    """Execute le pipeline complet sur une image : pretraitement -> OCR -> extraction -> regles."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    preprocessed = preprocess_for_ocr(img)
    ocr_out = extract_text_ocr(preprocessed)
    invoice = extract_structured_data(ocr_out["full_text"])
    invoice.ocr_confidence = ocr_out["avg_confidence"]

    return img, ocr_out, invoice


# Traiter toutes les images du dataset
print("Execution du pipeline sur l'ensemble du dataset...")
all_invoices = []
results = []

n_images = min(30, len(dataset["images"]))  # Limiter pour le temps d'execution
for i in range(n_images):
    img_path = dataset["images"][i]
    img, ocr_out, invoice = run_full_pipeline(img_path)
    if invoice is not None:
        all_invoices.append(invoice)
        results.append({"index": i, "path": img_path, "ocr": ocr_out, "invoice": invoice})
    if (i + 1) % 5 == 0:
        print(f"  Traite {i+1}/{n_images} images...")

print(f"[OK] {len(all_invoices)} factures extraites.")

# Analyse d'anomalies sur chaque facture avec historique
print("\nAnalyse des anomalies...")
reports = []
for i, invoice in enumerate(all_invoices):
    history = all_invoices[:i]  # Historique = factures precedentes
    report = analyze_invoice(invoice, history)
    reports.append(report)

n_ok = sum(1 for r in reports if r.overall_level == "ok")
n_warn = sum(1 for r in reports if r.overall_level == "warning")
n_crit = sum(1 for r in reports if r.overall_level == "critical")
print(f"  OK: {n_ok}  |  Warnings: {n_warn}  |  Critical: {n_crit}")

# =============================================================================
# CELLULE 11 : VISUALISATION DES RESULTATS
# =============================================================================

# --- Distribution des montants extraits ---
all_ttc = [inv.total_ttc for inv in all_invoices if inv.total_ttc and inv.total_ttc > 0]
if all_ttc:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogramme des montants
    axes[0].hist(all_ttc, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title("Distribution des montants TTC extraits")
    axes[0].set_xlabel("Montant (devise)")
    axes[0].set_ylabel("Frequence")

    # Loi de Benford
    all_amounts_benford = []
    for inv in all_invoices:
        if inv.total_ttc and inv.total_ttc > 0:
            all_amounts_benford.append(inv.total_ttc)
        for line in inv.lignes:
            if line.total_ligne > 0:
                all_amounts_benford.append(line.total_ligne)

    if len(all_amounts_benford) >= 10:
        first_digits = []
        for a in all_amounts_benford:
            s = str(a).lstrip("0").replace(".", "").replace("-", "")
            if s and s[0].isdigit() and int(s[0]) >= 1:
                first_digits.append(int(s[0]))

        if first_digits:
            counts = Counter(first_digits)
            observed_freq = [counts.get(d, 0) / len(first_digits) for d in range(1, 10)]
            benford_freq = [math.log10(1 + 1 / d) for d in range(1, 10)]

            x = range(1, 10)
            width = 0.35
            axes[1].bar([i - width/2 for i in x], observed_freq, width, label="Observe", color="steelblue")
            axes[1].bar([i + width/2 for i in x], benford_freq, width, label="Benford theorique", color="coral")
            axes[1].set_title("Loi de Benford : premiers chiffres")
            axes[1].set_xlabel("Premier chiffre")
            axes[1].set_ylabel("Frequence relative")
            axes[1].legend()

    # Repartition des anomalies
    labels_pie = ["OK", "Warning", "Critical"]
    sizes_pie = [n_ok, n_warn, n_crit]
    colors_pie = ["#4CAF50", "#FF9800", "#F44336"]
    non_zero = [(l, s, c) for l, s, c in zip(labels_pie, sizes_pie, colors_pie) if s > 0]
    if non_zero:
        labels_f, sizes_f, colors_f = zip(*non_zero)
        axes[2].pie(sizes_f, labels=labels_f, colors=colors_f, autopct="%1.0f%%", startangle=90)
        axes[2].set_title("Repartition des niveaux d'anomalie")

    plt.tight_layout()
    plt.show()

# --- Confiance OCR ---
confs = [inv.ocr_confidence for inv in all_invoices if inv.ocr_confidence]
if confs:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confs, bins=15, color="teal", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(confs), color="red", linestyle="--", label=f"Moyenne = {np.mean(confs):.2f}")
    ax.set_title("Distribution de la confiance OCR")
    ax.set_xlabel("Confiance moyenne par facture")
    ax.set_ylabel("Frequence")
    ax.legend()
    plt.tight_layout()
    plt.show()

# --- Affichage detaille d'un exemple avec anomalies ---
for i, report in enumerate(reports):
    if report.anomalies:
        inv = all_invoices[i]
        print(f"\n=== Facture {i} : {inv.fournisseur or 'Inconnu'} ===")
        print(f"  Montant TTC : {inv.total_ttc}")
        print(f"  Niveau : {report.overall_level}")
        for a in report.anomalies:
            print(f"  [{a['level'].upper()}] {a['rule']} : {a['detail']}")
        break  # Afficher un seul exemple

print("\n[OK] Pipeline complet termine.")
print("Pour l'interface Streamlit, voir la cellule suivante.")

# =============================================================================
# CELLULE 12 : GENERATION DU FICHIER STREAMLIT (app.py)
# =============================================================================

streamlit_code = '''
# -*- coding: utf-8 -*-
"""
Extracteur de Factures -- Interface Streamlit
Lancer avec : streamlit run app_projet1.py
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Importer les modules du pipeline (copier les fonctions ci-dessus dans src/)
# Pour la demo, on integre les fonctions directement.

st.set_page_config(page_title="Extracteur de Factures IA", layout="wide")
st.title("Extracteur Intelligent de Factures")
st.caption("OCR + Detection d'anomalies")

uploaded = st.file_uploader("Deposez une facture (image)", type=["png", "jpg", "jpeg"])

if uploaded:
    image = np.array(Image.open(uploaded))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Document original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Resultats")
        st.info("Pipeline OCR + Extraction en cours...")
        # Integrer ici le pipeline complet
        st.write("Connectez les modules du pipeline pour voir les resultats.")
'''

with open("app_projet1.py", "w") as f:
    f.write(streamlit_code)

print("[OK] Fichier app_projet1.py genere.")
print("Pour lancer : streamlit run app_projet1.py")
