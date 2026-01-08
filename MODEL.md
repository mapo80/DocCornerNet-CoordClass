# DocCornerNet-CoordClass: Documentazione Completa del Modello

## Panoramica

**DocCornerNetV3** è una rete neurale leggera per la rilevazione degli angoli di documenti basata su **Marginal Coordinate Classification (SimCC)** - un metodo che tratta la rilevazione delle coordinate come problemi di classificazione 1D invece della tradizionale regressione.

**Obiettivo:** <1M parametri, IoU ≥ 0.99 a 224×224

---

## 1. Architettura del Modello

### 1.1 Design Generale

```
Input Image [B, H, W, 3]
        │
        ▼
┌─────────────────────┐
│   Backbone          │  MobileNetV2 (alpha=0.35) o MobileNetV3
│   (Feature Extract) │  Output: C2, C3, C4, C5 multi-scala
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Mini-FPN (Neck)   │  Feature Pyramid Network
│                     │  Merge C2, C3, C4 con top-down pathway
│                     │  Output: P2 [B, H/4, W/4, fpn_ch]
└─────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│   SimCC Head        │    │   Score Head        │
│   (Coordinate Pred) │    │   (Document Detect) │
└─────────────────────┘    └─────────────────────┘
        │                          │
        ▼                          ▼
   simcc_x [B, 4, bins]      score_logit [B, 1]
   simcc_y [B, 4, bins]
        │
        ▼
   coords [B, 8] (soft-argmax decode)
```

### 1.2 Backbone

| Variante | Alpha | Parametri | Note |
|----------|-------|-----------|------|
| MobileNetV2 | 0.35 | 495,353 | **Raccomandato per produzione** |
| MobileNetV2 | 0.50 | ~600K | Compromesso velocità/accuratezza |
| MobileNetV3-Small | 0.75 | 742,417 | Teacher per distillazione |
| MobileNetV3-Small | 0.35 | 669,761 | Student distillato |

**Feature Scales estratte:**
- C2: H/4 × W/4 (alta risoluzione)
- C3: H/8 × W/8
- C4: H/16 × W/16
- C5: H/32 × W/32 (contesto globale)

### 1.3 Feature Pyramid Network (FPN/Neck)

- **Input:** Features multi-scala C2, C3, C4 dal backbone
- **Processo:** Top-down pathway con upsampling 2× nearest neighbor
- **Output:** P2 a risoluzione 56×56 (per input 224×224) con `fpn_ch` canali
- **Default:** fpn_ch = 32 (student/MobileNetV2), 48 (teacher)

**Ottimizzazione XNNPACK:**
- Upsampling 2× implementato come RESHAPE+MUL (evita RESIZE_NEAREST_NEIGHBOR)
- Completamente delegabile a XNNPACK per inferenza WASM

### 1.4 SimCC Head (Innovazione Core)

A differenza delle tradizionali heatmap 2D o regressione diretta, SimCC produce distribuzioni di probabilità 1D:

**Elaborazione Marginale Spaziale:**
```
P2 [B, 56, 56, fpn_ch]
        │
        ├─── reduce lungo Y ──→ X marginal [B, 56, fpn_ch]
        │
        └─── reduce lungo X ──→ Y marginal [B, 56, fpn_ch]
```

**Raffinamento Conv1D:**
- X-axis: Conv1D(simcc_ch, k=5) → Conv1D(simcc_ch//2, k=3)
- Y-axis: Conv1D(simcc_ch, k=5) → Conv1D(simcc_ch//2, k=3)
- Default: simcc_ch = 96 (student), 128 (teacher)

**Fusione Contesto Globale:**
- GlobalAveragePool2D sulle features
- Dense layer → broadcast a tutte le posizioni dei bin
- Concatena features locali + globali

**Output Coordinate:**
- 4 teste di predizione (una per angolo)
- Output: simcc_x [B, 4, num_bins], simcc_y [B, 4, num_bins]
- Soft-argmax decoding → coordinate normalizzate [B, 8] ∈ [0,1]

### 1.5 Score Head

- GlobalAveragePool2D dalle features C5
- Single Dense layer → logit presenza documento
- Bias inizializzato a 1.75 (default)
- Output: score_logit [B, 1]

---

## 2. Specifiche Input/Output

### 2.1 Input

| Parametro | Valore |
|-----------|--------|
| Shape | [B, H, W, 3] |
| Dimensioni default | 224×224 o 256×256 |
| Data type | float32 |
| Range valori | [0, 1] (dopo normalizzazione) |

**Normalizzazione ImageNet:**
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalized = (image - mean) / std
```

### 2.2 Output Training Model

| Output | Shape | Descrizione |
|--------|-------|-------------|
| simcc_x | [B, 4, num_bins] | Logits coordinata X per 4 angoli |
| simcc_y | [B, 4, num_bins] | Logits coordinata Y per 4 angoli |
| score_logit | [B, 1] | Logit presenza documento |
| coords | [B, 8] | Coordinate decodificate [0, 1] |

### 2.3 Output Inference Model

| Output | Shape | Descrizione |
|--------|-------|-------------|
| coords | [B, 8] | [x₀, y₀, x₁, y₁, x₂, y₂, x₃, y₃] |
| score_logit | [B, 1] | Applicare sigmoid per probabilità |

**Ordine angoli:**
- Angolo 0: Top-Left (TL)
- Angolo 1: Top-Right (TR)
- Angolo 2: Bottom-Right (BR)
- Angolo 3: Bottom-Left (BL)

### 2.4 Formati Export TFLite

**coords9 (standard):**
- Output: [B, 9] = [x₀, y₀, x₁, y₁, x₂, y₂, x₃, y₃, score]
- Supporta: float32, float16, dynamic quantization

**simcc_logits (avanzato, per INT8):**
- Output 1: simcc_xy logits [B, num_bins, 8] (bins_first) o [B, 8, num_bins]
- Output 2: score_logit [B, 1]
- Decode esterno: softmax → expectation over bin centers

---

## 3. Configurazione Training

### 3.1 Funzioni di Loss

**SimCCLoss (primaria):**
```python
# Cross-entropy con target Gaussiani soft
# sigma_px = 3.0 (larghezza Gaussiana in pixel)
# tau = 1.0 (temperatura softmax)
loss = -Σ(gaussian_target · log_softmax(logits / tau))
```

**CoordLoss (ausiliaria):**
```python
# Supervisione diretta coordinate
# Tipo: L1 o SmoothL1
loss = |predicted_coords - target_coords|
```

**ScoreLoss:**
```python
# Binary cross-entropy per presenza documento
# Solo campioni positivi contribuiscono
loss = BCE(sigmoid(score_logit), has_document)
```

### 3.2 Pesi Loss Default

| Loss | Peso |
|------|------|
| w_simcc | 1.0 |
| w_coord | 0.5 |
| w_score | 0.5 |

### 3.3 Ottimizzatore e Learning Rate

| Parametro | Valore Default |
|-----------|----------------|
| Optimizer | Adam |
| Learning Rate iniziale | 0.0002 (teacher), 0.00075 (large batch) |
| Warmup | Linear, 5 epochs |
| LR Schedule | ReduceOnPlateau |
| Patience (LR) | 7 epochs |
| Factor (LR) | 0.5 |
| Min LR | 1e-6 |
| Weight Decay | 0.0001 |

### 3.4 Pipeline di Augmentation

```python
augmentation_config = {
    "rotation_degrees": 5,      # ±5° rotazione
    "scale_range": (0.9, 1.0),  # 90-100% zoom
    "brightness": 0.2,          # ±20% luminosità
    "contrast": 0.2,            # ±20% contrasto
    "saturation": 0.1,          # ±10% saturazione
    "blur_prob": 0.1,           # 10% probabilità blur
    "blur_kernel": 3,           # Kernel Gaussiano
    "translate": 0.0,           # Nessuna traslazione
    "perspective": (0.0, 0.03), # Max ±3% prospettiva
}
```

### 3.5 Parametri Training

| Parametro | Valore Standard | Large-Scale |
|-----------|-----------------|-------------|
| Batch Size | 64 | 256-512 |
| Epochs | 100 | 200 |
| Early Stopping Patience | 20 | 30 |

### 3.6 Gestione Outlier

```python
# File con nomi immagini problematiche
outlier_list = "outliers.txt"
# Peso di campionamento per questi esempi
outlier_weight = 3.0  # 3× più probabilità di essere campionati
```

---

## 4. Dataset

### 4.1 Formato Dati

**Struttura directory:**
```
dataset/
├── images/              # Immagini positive (con documento)
├── images-negative/     # Immagini negative (senza documento)
└── labels/              # Label in formato YOLO OBB
    └── {image_name}.txt
```

**Formato Label (YOLO OBB):**
```
class_id x₀ y₀ x₁ y₁ x₂ y₂ x₃ y₃
```
- `class_id`: 0 (documento)
- Coordinate: normalizzate [0, 1]

### 4.2 Split Dataset

| Split | Descrizione |
|-------|-------------|
| train.txt | Training set principale |
| val.txt | Validation set |
| test.txt | Test set |
| train_with_negative_v2.txt | Training con negativi |
| train_clean.txt | Solo IoU ≥ 0.99 |
| train_clean_iter3.txt | Iterazione 3 raffinamento |
| train_clean_iter4_mix.txt | Mix finale ottimizzato |

### 4.3 Statistiche Dataset (doc-scanner-dataset-labeled)

| Metrica | Valore |
|---------|--------|
| Train images | 88,143 |
| Val images | 23,288 |
| Totale | 111,431 |

**Distribuzione qualità IoU:**
| Soglia | Percentuale |
|--------|-------------|
| IoU ≥ 0.99 | 15% |
| IoU ≥ 0.98 | 48% |
| IoU ≥ 0.97 | 68% |
| IoU ≥ 0.95 | 86% |

---

## 5. Metriche di Valutazione

### 5.1 Metriche Geometriche

**Polygon IoU (primaria):**
- Intersezione su unione polygon-to-polygon reale
- Usa Shapely per geometria esatta
- Range: [0, 1]

**Corner Error (pixel):**
- Distanza Euclidea per angolo
- Riportato: mean, max, p95

### 5.2 Metriche di Classificazione

- Document presence accuracy
- Precision, Recall, F1-score
- Soglia: 0.5 (dopo sigmoid)

### 5.3 Recall a Soglie

| Metrica | Descrizione |
|---------|-------------|
| Recall@IoU50 | % campioni con IoU > 0.50 |
| Recall@IoU75 | % campioni con IoU > 0.75 |
| Recall@IoU90 | % campioni con IoU > 0.90 |
| Recall@IoU95 | % campioni con IoU > 0.95 |
| Recall@IoU99 | % campioni con IoU > 0.99 |

---

## 6. Risultati e Modelli Migliori

### 6.1 Modelli Raccomandati per Produzione

**Per velocità/accuratezza (raccomandato):**
```
checkpoints/mobilenetv2_224_best
├── Parametri: 495K
├── Input: 224×224
├── Val mIoU: 0.9894
└── Robustezza outlier: 0.7075
```

**Per massima accuratezza:**
```
checkpoints/mobilenetv2_256_best
├── Parametri: 495K
├── Input: 256×256
├── Val mIoU: 0.9902 (best su clean)
└── Robustezza outlier: 0.6281
```

**Per generalizzazione cross-dataset:**
```
mobilenetv2_224_from256_clean_iter3
├── Fine-tuned da 256→224
├── Worst-case mIoU: 0.9047
└── Miglior performance bilanciata
```

### 6.2 Benchmark TFLite (CPU, batch=1)

**Modelli Float16 (coords9 output):**

| Modello | Input | Latenza (p50) | Size | mIoU |
|---------|-------|---------------|------|------|
| mnv2_224_best | 224 | 4.24 ms | 0.98 MB | 0.9894 |
| mnv2_256_best | 256 | 8.18 ms | 0.98 MB | 0.9902 |
| mnv3_224 | 224 | 3.96 ms | 1.47 MB | 0.9842 |

**Modelli INT8 Full-Quant (simcc_logits output):**

| Modello | Input | Latenza (p50) | Size | mIoU |
|---------|-------|---------------|------|------|
| mnv2_224_best | 224 | **2.53 ms** | 0.82 MB | 0.9888 |
| mnv2_256_best | 256 | 2.92 ms | 0.84 MB | 0.9893 |
| mnv3_224 | 224 | 3.15 ms | 1.04 MB | 0.3519 |

**Vincitore:** mnv2_224_best INT8 - più veloce (2.53ms) con eccellente accuratezza (0.9888)

### 6.3 Speedup Float16 vs INT8

- MobileNetV2: 1.6-2.8× più veloce con perdita minima (<0.001 mIoU)
- MobileNetV3: PTQ INT8 **non raccomandato** (collasso accuratezza)

---

## 7. Formati di Export

### 7.1 Export TFLite Float32 (base)

```bash
python export_tflite.py \
  --model_path ./checkpoints/mobilenetv2_224_best \
  --output ./exported_tflite/model_float32.tflite
```
- Output: [B, 9] coords9
- Nessuna quantizzazione
- File più grande

### 7.2 Export TFLite Float16 (raccomandato per WASM)

```bash
python export_tflite.py \
  --model_path ./checkpoints/mobilenetv2_224_best \
  --output ./exported_tflite/model_float16.tflite \
  --float16
```
- Output: [B, 9] coords9
- ~50% più piccolo di float32
- Perdita accuratezza minima

### 7.3 Export TFLite INT8 (più veloce, full XNNPACK)

```bash
python export_tflite_int8.py \
  --checkpoint ./checkpoints/mobilenetv2_224_best \
  --data_root $DATA_ROOT \
  --split val_cleaned \
  --quantization int8 \
  --io_dtype int8 \
  --output_dtype int8 \
  --output_format simcc_logits \
  --simcc_packed_layout bins_first \
  --axis_mean_impl dwconv_full \
  --global_pool_impl dwconv_strided
```
- Output: packed SimCC logits + score (int8)
- Decode coordinate esterno al modello
- 1.6-2.8× più veloce di float16
- 100% delegazione XNNPACK su WASM

### 7.4 Opzioni Export

**Schemi di quantizzazione:**
| Schema | Descrizione |
|--------|-------------|
| int8 | Full int8 dove possibile |
| int16x8 | Attivazioni int16 + pesi int8 (maggior accuratezza) |
| dynamic | Solo pesi quantizzati (attivazioni float) |

**I/O dtypes:**
| Dtype | Note |
|-------|------|
| float32 | Standard, più grande |
| int8 | Richiede calibrazione |
| uint8 | Alternativa int8 |

**Implementazioni pooling (per XNNPACK):**
| Impl | Descrizione |
|------|-------------|
| mean | Standard (potrebbe non delegare completamente) |
| avgpool | Basato su pool |
| dwconv_full | Singola depthwise conv |
| dwconv_strided | Riduzione con stride |
| dwconv_pyramid | Riduzione gerarchica |

---

## 8. Varianti e Configurazioni del Modello

### 8.1 Configurazione MobileNetV2 Small (Produzione)

```python
config = {
    "backbone": "mobilenetv2",
    "alpha": 0.35,
    "fpn_ch": 32,
    "simcc_ch": 96,
    "img_size": 224,
    "num_bins": 224,
    "tau": 1.0,
    "sigma_px": 3.0,
}
# Parametri: 495,353
# Latenza: 3.78 ms (TFLite float16)
```

### 8.2 Configurazione MobileNetV2 Medium

```python
config = {
    "backbone": "mobilenetv2",
    "alpha": 0.35,
    "fpn_ch": 32,
    "simcc_ch": 96,
    "img_size": 256,
    "num_bins": 256,
}
# Parametri: 495,353 (stessa arch, input più grande)
# Latenza: 4.81 ms (TFLite float16)
```

### 8.3 Configurazione Teacher (MobileNetV3-Small)

```python
config = {
    "backbone": "mobilenetv3_small",
    "alpha": 0.75,
    "fpn_ch": 48,
    "simcc_ch": 128,
    "img_size": 224,
    "num_bins": 224,
}
# Parametri: 742,417
# Latenza: 4.93 ms (TFLite float32)
```

### 8.4 Sensibilità Parametri

**Alpha (moltiplicatore larghezza):**
| Alpha | Parametri | Uso |
|-------|-----------|-----|
| 0.35 | ~495K | Veloce, produzione |
| 0.50 | ~600K | Compromesso |
| 0.75 | ~742K | Più capacità |
| 1.0 | Full width | Più lento |

**FPN channels:**
| Canali | Effetto |
|--------|---------|
| 24-32 | Meno parametri, più veloce |
| 48-64 | Più capacità, più lento |

**SimCC channels:**
| Canali | Effetto |
|--------|---------|
| 64-96 | Teste leggere |
| 128-256 | Teste più pesanti |

**Image size:**
| Size | Caratteristiche |
|------|-----------------|
| 224 | Veloce, bassa memoria |
| 256 | Bilanciato |
| 320 | Accurato, lento |
| 384+ | Non raccomandato |

---

## 9. Script e Pipeline

### 9.1 Training

**Training standard:**
```bash
python train.py \
  --data_root ./data \
  --img_size 256 \
  --backbone mobilenetv2 \
  --alpha 0.35 \
  --fpn_ch 32 \
  --simcc_ch 96 \
  --batch_size 256 \
  --epochs 100 \
  --lr 0.00075 \
  --output_dir ./checkpoints
```

**Training ultra-ottimizzato (GPU):**
```bash
python train_ultra.py \
  --data_root ./data \
  --img_size 224 \
  --backbone mobilenetv2 \
  --alpha 0.35 \
  --batch_size 512 \
  --epochs 200 \
  --cache_images \
  --output_dir ./checkpoints
```

**Knowledge distillation:**
```bash
python train_student.py \
  --teacher_path ./checkpoints/teacher_model \
  --data_root ./data \
  --student_alpha 0.35 \
  --output_dir ./checkpoints_student
```

### 9.2 Valutazione

```bash
python evaluate.py \
  --model_path ./checkpoints/mobilenetv2_256_best \
  --data_root ./data \
  --split val_cleaned
```

**Valutazione TFLite:**
```bash
python eval_tflite.py \
  --model_path ./exported_tflite/model_float16.tflite \
  --data_root ./data \
  --split val_cleaned
```

**Benchmark latenza:**
```bash
python benchmark_tflite.py \
  --model_path ./exported_tflite/model_int8.tflite \
  --iterations 1000
```

### 9.3 Export

Vedi Sezione 7 per comandi dettagliati di export.

---

## 10. Innovazioni Tecniche

### 10.1 Perché SimCC > Regressione/Heatmap

| Aspetto | SimCC | Regressione | Heatmap 2D |
|---------|-------|-------------|------------|
| Supervisione | 224 bin per asse | 1 scalare | 56×56 pixel |
| Gradienti | Cross-entropy (stabili) | L1/L2 (instabili) | CE (stabili) |
| Consapevolezza spaziale | FPN + Conv1D | GAP collassa | Mantiene |
| Precisione sub-pixel | Soft-argmax | Limitata | Limitata |
| Efficienza | 1D processing | Minima | 2D processing |

### 10.2 Ottimizzazioni XNNPACK (WASM)

Operazioni standard sostituite con alternative XNNPACK-friendly:

| Operazione Standard | Problema | Soluzione XNNPACK |
|---------------------|----------|-------------------|
| STRIDED_SLICE | Non delegabile | Reshape away singleton dim |
| TILE | Non delegabile | MUL broadcasting |
| SUM/PACK | Non delegabile | Matmul per expectation |
| RESIZE_NEAREST_NEIGHBOR | Non delegabile | RESHAPE+MUL |
| REDUCE ops | Parzialmente delegabile | dwconv alternatives |

**Risultato:** 100% delegazione XNNPACK su browser WASM

### 10.3 Pipeline Self-Training

```
1. Train su dati IoU ≥ 0.99
        │
        ▼
2. Ri-valuta dataset completo
        │
        ▼
3. Trova immagini dove modello predice meglio di GT originale (0.95-0.99)
        │
        ▼
4. Verifica manuale e aggiungi a training set
        │
        ▼
5. Itera fino a convergenza
```

**Beneficio:** Correzione automatica errori GT, scalabile a grandi dataset

---

## 11. File di Configurazione

### 11.1 Struttura config.json

```json
{
  "backbone": "mobilenetv2",
  "alpha": 0.35,
  "fpn_ch": 32,
  "simcc_ch": 96,
  "img_size": 256,
  "num_bins": 256,
  "tau": 1.0,
  "sigma_px": 3.0,
  "w_simcc": 1.0,
  "w_coord": 0.5,
  "w_score": 0.5,
  "batch_size": 256,
  "epochs": 200,
  "lr": 0.00075,
  "weight_decay": 0.0001,
  "warmup_epochs": 5,
  "patience": 20,
  "lr_patience": 7,
  "lr_factor": 0.5,
  "min_lr": 1e-06,
  "augment": true
}
```

### 11.2 Struttura Checkpoint

```
checkpoints/mobilenetv2_256_best/
├── config.json              # Configurazione completa training
├── best_model.weights.h5    # Pesi migliore modello
├── best_model.keras         # Formato Keras (legacy)
├── training_log.txt         # Storico training
└── metrics.json             # Metriche validazione
```

---

## 12. Quick Reference

### Training Minimo

```bash
python train.py \
  --data_root ./data \
  --img_size 256 \
  --backbone mobilenetv2 \
  --alpha 0.35 \
  --batch_size 64 \
  --epochs 100 \
  --output_dir ./checkpoints
```

### Valutazione Minima

```bash
python evaluate.py \
  --model_path ./checkpoints/mobilenetv2_256_best \
  --data_root ./data \
  --split val
```

### Export Minimo

```bash
python export_tflite.py \
  --model_path ./checkpoints/mobilenetv2_256_best \
  --output ./model.tflite \
  --float16
```

### Export INT8 Minimo (più veloce)

```bash
python export_tflite_int8.py \
  --checkpoint ./checkpoints/mobilenetv2_224_best \
  --data_root ./data \
  --split val \
  --quantization int8 \
  --output_format simcc_logits
```

---

## 13. Riassunto

DocCornerNetV3 è un modello state-of-the-art leggero per la rilevazione degli angoli di documenti con:

| Caratteristica | Valore |
|----------------|--------|
| **Efficienza** | 495K parametri, 2-4ms inferenza (TFLite) |
| **Accuratezza** | 98.9%+ mIoU su dati puliti |
| **Robustezza** | 70.75% su outlier |
| **Deployment** | Float32/Float16/INT8 con full XNNPACK delegation |
| **Flessibilità** | Multipli backbone, dimensioni input, schemi quantizzazione |
| **Production-ready** | Pesi pre-trainati, dati benchmark, valutazione cross-dataset |

**Casi d'uso ideali:**
- Scansione documenti
- Elaborazione form
- Applicazioni mobile/web
- Edge computing

---

## 14. Riferimenti

- **Dataset:** `mapo80/DocCornerDataset` (HuggingFace)
- **Framework:** TensorFlow/Keras 2.x
- **Backbone:** MobileNetV2/V3 (tf.keras.applications)
- **Metodo:** SimCC (Marginal Coordinate Classification)
- **Export:** TensorFlow Lite con XNNPACK delegation
