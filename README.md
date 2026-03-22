# Portfolio IA Industrielle

> 5 projets de Data Science / IA appliquee a l'industrie, demontrant des competences en OCR, Computer Vision, NLP, Time Series, Detection d'objets, Tracking et Metrologie.

---

## Projets

### 1. [InvoiceAI](./InvoiceAI/) -- Extracteur Intelligent de Factures
Pipeline VLM multimodal + detection d'anomalies (loi de Benford, doublons, coherence arithmetique).

**Stack** : LLaVA (Ollama), EasyOCR, Pydantic, SciPy, Streamlit

### 2. [DefectVision](./DefectVision/) -- Controle Qualite Industriel
Classification de defauts sur pieces moulees par transfer learning (ResNet-18) avec explicabilite Grad-CAM.

**Stack** : PyTorch, Albumentations, OpenCV (CLAHE), scikit-learn, Streamlit

### 3. [PredictivePulse](./PredictivePulse/) -- Maintenance Predictive
Prediction de la duree de vie residuelle (RUL) de moteurs turbofan via XGBoost et LSTM sur donnees capteurs.

**Stack** : PyTorch LSTM, XGBoost, pandas, scikit-learn, Streamlit

### 4. [CrackVision](./CrackVision/) -- Detection et Metrologie de Fissures
Detection de fissures dans le beton par YOLO11 avec estimation dimensionnelle et calibration metrologique.

**Stack** : YOLO11 (Ultralytics), PyTorch, Roboflow, Pandas, Streamlit

### 5. [TrafficPulse](./TrafficPulse/) -- Analyse de Trafic et Estimation de Vitesse
Detection, tracking et estimation de vitesse de vehicules par YOLO11 + homographie (transformation de perspective).

**Stack** : YOLO11, ByteTrack, OpenCV (homographie), NumPy, Streamlit

---

## Profil

Docteur-Ingenieur avec 10 ans d'experience en physique, analyse de signaux et capteurs (ex-CEA).
Reconversion vers la Data Science et l'IA industrielle.

## Competences demonstrees

| Competence | Projet(s) |
|---|---|
| Detection d'objets (YOLO11) | CrackVision, TrafficPulse |
| Tracking multi-objets (ByteTrack) | TrafficPulse |
| Computer Vision (CNN, Transfer Learning) | DefectVision |
| Geometrie projective (Homographie) | TrafficPulse |
| OCR et extraction de donnees | InvoiceAI |
| Series temporelles et signal processing | PredictivePulse |
| Deep Learning (PyTorch, LSTM, CNN) | DefectVision, PredictivePulse |
| Machine Learning classique (XGBoost) | PredictivePulse |
| Metrologie et calibration | CrackVision, TrafficPulse |
| Explicabilite IA (Grad-CAM) | DefectVision |
| Detection d'anomalies | InvoiceAI |
| Pipeline end-to-end + Streamlit | Les 5 projets |

## Execution

Chaque projet contient un fichier .py executable et un requirements.txt pour l'installation locale.
