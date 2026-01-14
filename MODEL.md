# DocCornerNet-CoordClass: Documentazione Completa del Modello

## Panoramica

**DocCornerNetV3** è una rete neurale leggera per la rilevazione degli angoli di documenti basata su **Marginal Coordinate Classification (SimCC)** - un metodo che tratta la rilevazione delle coordinate come problemi di classificazione 1D invece della tradizionale regressione.

**Obiettivo:** <1M parametri, IoU ≥ 0.99 a 224×224

---

## 🏆 Quick Summary: Best Model (Gennaio 2026)

| Metrica | Float16 | INT8 SimCC |
|---------|---------|------------|
| **Modello** | `geom_aug_plateau_ohem` | `geom_aug_plateau_ohem` |
| **Backbone** | MobileNetV2 α=0.35 | MobileNetV2 α=0.35 |
| **Input** | 320×320 | 320×320 |
| **Parametri** | ~500K | ~500K |
| **Test mIoU** | **0.9219** | **0.9183** |
| **Test R@95** | **68.9%** | ~68% |
| **TFLite Size** | 1.11 MB | **0.89 MB** |
| **Latenza CPU** | 4.64 ms | **4.41 ms** |
| **XNNPACK** | ✅ Full | ✅ Full |
| **Decode** | Interno | **Esterno** |

**Caratteristiche chiave:**
- GAU attention + fc_expansion=256
- OHEM (Online Hard Example Mining)
- Geometric augmentation (rotation ±10°, perspective 0.03)
- LR schedule: plateau (non cosine)

**File Float16:** `checkpoints_remote/geom_aug_plateau_ohem/model_float16.tflite`
**File INT8 (BEST):** `checkpoints_remote/geom_aug_plateau_ohem/model_int8_simcc_static.tflite`

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

### 6.1 🏆 Modello Vincitore: geom_aug_plateau_ohem (320×320)

```
checkpoints_remote/geom_aug_plateau_ohem
├── Backbone: MobileNetV2 alpha=0.35
├── Input: 320×320
├── Parametri: ~500K
├── GAU attention + fc_expansion=256
├── OHEM (threshold=12px, weight=2.5)
├── Geometric augmentation (rotation=10°, perspective=0.03)
├── Val mIoU: 0.9879
└── Test mIoU: 0.9221 ← BEST GENERALIZATION
```

**Perché è il vincitore:**
- Miglior generalizzazione sul test set (+8.6% vs modelli 224/256)
- Geometric augmentation migliora robustezza a rotazioni e prospettive
- GAU attention cattura dipendenze spaziali globali
- OHEM focalizza training su casi difficili

### 6.2 Confronto Modelli su Test Set

| Modello | Input | GAU | GeomAug | Test mIoU | Test R@95 | Val mIoU |
|---------|-------|-----|---------|-----------|-----------|----------|
| mobilenetv2_224_best | 224 | No | No | 0.8572 | 48.5% | 0.9894 |
| mobilenetv2_256_best | 256 | No | No | 0.8572 | 48.2% | 0.9902 |
| mobilenetv2_320_gau_ohem | 320 | Yes | No | 0.9014 | 63.2% | ~0.988 |
| **geom_aug_plateau_ohem** | **320** | **Yes** | **Yes** | **0.9221** | **68.9%** | **0.9879** |

### 6.3 Benchmark TFLite su Test Set (CPU, batch=1)

**Float16 (raccomandato per WASM/browser):**

| Modello | Input | Size | Test mIoU | Test R@95 | Latenza (p50) |
|---------|-------|------|-----------|-----------|---------------|
| mnv2_224_revlast_f16 | 224 | 0.98 MB | 0.8361 | 52.1% | 2.60 ms |
| mnv2_256_revlast_f16 | 256 | 0.98 MB | 0.8344 | 52.0% | 3.05 ms |
| **geom_aug_320_f16** | **320** | **1.11 MB** | **0.9219** | **68.9%** | **4.64 ms** |

**INT8 PTQ (XNNPACK delegated):**

| Modello | Input | Size | Test mIoU | Latenza (p50) | XNNPACK | Decode |
|---------|-------|------|-----------|---------------|---------|--------|
| geom_aug_320_int8 | 320 | 1.16 MB | 0.9108 | 5.78 ms | ⚠️ Parziale | Interno |
| geom_aug_320_int8_static | 320 | 0.94 MB | 0.9050 | 4.43 ms | ✅ Full | Interno |
| **geom_aug_320_int8_simcc_static** | **320** | **0.89 MB** | **0.9183** | **4.41 ms** | ✅ Full | **Esterno** |

**Trade-off Float16 vs INT8:**
- Float16: migliore accuratezza (0.9219 IoU), decode interno
- INT8 coords9 static: veloce (4.43 ms), ma accuratezza ridotta (0.9050 IoU)
- **INT8 simcc_logits static:** **BEST** - veloce (4.41 ms) E alta accuratezza (0.9183 IoU)
- **Raccomandato:** `model_int8_simcc_static.tflite` con decode esterno per massime performance

**Trade-off generale:**
- geom_aug_320 è 1.8× più lento dei modelli 224px ma +10% IoU sul test set
- Per applicazioni real-time con requisiti di generalizzazione, geom_aug_320 è raccomandato

### 6.4 Modelli Legacy per Produzione

**Per velocità estrema (se generalizzazione non è critica):**
```
checkpoints/mobilenetv2_224_best
├── Parametri: 495K
├── Input: 224×224
├── Val mIoU: 0.9894
├── Test mIoU: 0.8572
└── Latenza: 2.60 ms (float16)
```

### 6.5 Speedup Float16 vs INT8

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

### 7.3 Export TFLite INT8 (PTQ Quantization)

#### Comando Base (con dataset legacy)

```bash
python export_tflite_int8.py \
    --checkpoint ./checkpoints/mobilenetv2_224_best \
    --data_root /path/to/doc-scanner-dataset-labeled \
    --split val_cleaned \
    --quantization int8 \
    --io_dtype int8 \
    --output_dtype int8 \
    --output_format simcc_logits \
    --simcc_packed_layout bins_first \
    --axis_mean_impl dwconv_full \
    --global_pool_impl dwconv_strided \
    --output exported_tflite/model_int8.tflite
```

#### Comando con HuggingFace Parquet Dataset

```bash
python export_tflite_int8.py \
    --checkpoint checkpoints_remote/geom_aug_plateau_ohem \
    --hf_dataset ./hf_dataset \
    --split val \
    --quantization int8 \
    --io_dtype int8 \
    --output_dtype int8 \
    --output_format simcc_logits \
    --simcc_packed_layout bins_first \
    --axis_mean_impl dwconv_full \
    --global_pool_impl dwconv_strided \
    --output exported_tflite/geom_aug_320_int8_simcc.tflite
```

#### 🚀 INT8 Full XNNPACK Delegate con SimCC Logits (BEST - Raccomandato)

Per ottenere **massima accuratezza (mIoU ≥ 0.91) E velocità**, usa `simcc_logits` con decode esterno:

```bash
python export_tflite_int8.py \
    --checkpoint checkpoints_remote/geom_aug_plateau_ohem \
    --hf_dataset ./hf_dataset \
    --split val \
    --quantization int8 \
    --io_dtype int8 \
    --output_dtype int8 \
    --output_format simcc_logits \
    --simcc_packed_layout 8_first \
    --static_batch \
    --axis_mean_impl dwconv_full \
    --global_pool_impl dwconv_strided \
    --output checkpoints_remote/geom_aug_plateau_ohem/model_int8_simcc_static.tflite
```

**Risultati:**
- ✅ mIoU: **0.9183** (target ≥ 0.91)
- ✅ Latenza: **4.41 ms** (target ≤ 4.43 ms)
- ✅ Full XNNPACK delegation (nessun warning)
- ✅ Size: **0.89 MB** (più piccolo di float16)

**Decode esterno in Python/NumPy:**
```python
def decode_simcc(simcc_xy, tau=1.0):
    """
    Decode SimCC logits to normalized coordinates.

    Args:
        simcc_xy: [1, 8, num_bins] - packed logits (8_first layout)
                  First 4 channels are X coords, next 4 are Y coords
        tau: softmax temperature (default 1.0)

    Returns:
        coords: [1, 8] - (x0,y0,x1,y1,x2,y2,x3,y3) in [0,1]
    """
    simcc_xy = simcc_xy.astype(np.float32)
    simcc_x = simcc_xy[:, :4, :]  # [1, 4, num_bins]
    simcc_y = simcc_xy[:, 4:, :]  # [1, 4, num_bins]

    num_bins = simcc_x.shape[2]
    centers = np.linspace(0, 1, num_bins, dtype=np.float32)

    # Stable softmax
    sx = simcc_x / tau
    sy = simcc_y / tau
    sx = sx - np.max(sx, axis=-1, keepdims=True)
    sy = sy - np.max(sy, axis=-1, keepdims=True)
    px = np.exp(sx) / (np.sum(np.exp(sx), axis=-1, keepdims=True) + 1e-8)
    py = np.exp(sy) / (np.sum(np.exp(sy), axis=-1, keepdims=True) + 1e-8)

    # Expectation (soft-argmax)
    ex = np.sum(px * centers, axis=-1)  # [1, 4]
    ey = np.sum(py * centers, axis=-1)  # [1, 4]

    # Interleave: [x0,y0,x1,y1,x2,y2,x3,y3]
    coords = np.stack([ex, ey], axis=-1).reshape(-1, 8)
    return np.clip(coords, 0, 1)
```

#### INT8 coords9 con Static Batch (alternativa più semplice)

Se preferisci decode interno (meno codice, ma accuratezza inferiore):

```bash
python export_tflite_int8.py \
    --checkpoint checkpoints_remote/geom_aug_plateau_ohem \
    --hf_dataset ./hf_dataset \
    --split val \
    --quantization int8 \
    --io_dtype int8 \
    --output_dtype int8 \
    --output_format coords9 \
    --static_batch \
    --axis_mean_impl dwconv_full \
    --global_pool_impl dwconv_strided \
    --output checkpoints_remote/geom_aug_plateau_ohem/model_int8_static.tflite
```

**Vantaggi di `--static_batch`:**
- ✅ Nessun warning "dynamic-sized tensors"
- ✅ Full XNNPACK delegation per INT8
- ✅ Output coords9 con decode interno (più semplice da usare)
- ⚠️ Accuratezza ridotta: mIoU 0.9050 (vs 0.9183 con simcc_logits)
- ⚠️ Solo batch_size=1 (standard per inference mobile)

### 7.4 Parametri Export INT8

| Parametro | Valori | Default | Descrizione |
|-----------|--------|---------|-------------|
| `--checkpoint` | path | (required) | Directory checkpoint o file .h5 |
| `--data_root` | path | None | Dataset legacy per calibrazione |
| `--hf_dataset` | path | None | Dataset HuggingFace parquet |
| `--split` | train/val/test | val_cleaned | Split per calibrazione |
| `--num_calib` | int | 500 | Numero campioni calibrazione |
| `--quantization` | int8/int16x8/dynamic | int8 | Schema quantizzazione |
| `--io_dtype` | float32/int8/uint8 | float32 | Tipo I/O modello |
| `--output_dtype` | float32/int8/uint8 | float32 | Tipo output modello |
| `--output_format` | coords9/simcc_logits | coords9 | Formato output |
| `--simcc_packed_layout` | 8_first/bins_first | 8_first | Layout packed logits |
| `--axis_mean_impl` | mean/avgpool/dwconv_* | dwconv_full | Impl riduzione assi |
| `--global_pool_impl` | mean/avgpool/dwconv_* | dwconv_full | Impl global pooling |
| `--allow_float_fallback` | flag | False | Permetti fallback float |
| `--static_batch` | flag | False | Batch statico=1 per full XNNPACK |

### 7.5 Output Formats INT8

| Format | Output Shape | Decode | XNNPACK | mIoU | Note |
|--------|--------------|--------|---------|------|------|
| `coords9` | [B, 9] | Interno | ⚠️ Parziale | 0.9108 | Dynamic tensors senza --static_batch |
| `coords9 + --static_batch` | [1, 9] | Interno | ✅ Full | 0.9050 | Semplice ma meno accurato |
| `simcc_logits` | [B, 8, num_bins] | Esterno | ⚠️ Parziale | ~0.92 | Richiede decode esterno |
| **`simcc_logits + --static_batch`** | **[1, 8, num_bins]** | **Esterno** | ✅ **Full** | **0.9183** | **BEST: veloce + accurato** |

**simcc_logits con `8_first` layout (raccomandato):**
- Output shape: [1, 8, 320] (con --static_batch)
- Layout: primi 4 canali sono X coords, successivi 4 sono Y coords
- Decoding esterno: dequantize → softmax → expectation (vedi funzione decode_simcc sopra)

**simcc_logits con `bins_first` layout (alternativa):**
- Output shape: [1, 320, 8]
- Layout: [x0, x1, x2, x3, y0, y1, y2, y3] per ogni bin
- Decoding esterno: transpose → softmax → expectation

### 7.6 Schemi Quantizzazione

| Schema | Descrizione | Velocità | Accuratezza |
|--------|-------------|----------|-------------|
| `int8` | Full INT8 | ★★★ | ★★ |
| `int16x8` | Attivazioni INT16 + pesi INT8 | ★★ | ★★★ |
| `dynamic` | Solo pesi quantizzati | ★ | ★★★ |

### 7.7 Implementazioni Pooling (XNNPACK)

| Impl | Descrizione | XNNPACK |
|------|-------------|---------|
| `mean` | tf.reduce_mean | ❌ Parziale |
| `avgpool` | tf.nn.avg_pool2d | ❌ Parziale |
| `dwconv_full` | Singola depthwise conv | ✅ Full |
| `dwconv_strided` | Multi-step con stride | ✅ Full |
| `dwconv_pyramid` | Riduzione gerarchica 2x2 | ✅ Full |

**Raccomandato per full XNNPACK:** `--axis_mean_impl dwconv_full --global_pool_impl dwconv_strided`

### 7.8 Risultati INT8 su Test Set

| Modello | Format | Size | Test mIoU | Latency (p50) | XNNPACK | Decode |
|---------|--------|------|-----------|---------------|---------|--------|
| geom_aug_320_float16 | coords9 | 1.11 MB | **0.9219** | 4.64 ms | ✅ | Interno |
| geom_aug_320_int8 | coords9 | 1.16 MB | 0.9108 | 5.78 ms | ⚠️ | Interno |
| geom_aug_320_int8_static | coords9 | 0.94 MB | 0.9050 | 4.43 ms | ✅ | Interno |
| **geom_aug_320_int8_simcc_static** | **simcc_logits** | **0.89 MB** | **0.9183** | **4.41 ms** | ✅ | **Esterno** |

**Raccomandazione:**
- **Massima accuratezza assoluta:** float16 (mIoU 0.9219) - decode interno
- **Miglior rapporto accuratezza/velocità:** INT8 simcc_logits + `--static_batch` (mIoU 0.9183, 4.41 ms) - **BEST**
- **Più semplice:** INT8 coords9 + `--static_batch` (4.43 ms, full XNNPACK) - ma mIoU solo 0.9050
- **Evitare:** INT8 senza `--static_batch` (warning tensori dinamici, più lento)

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
