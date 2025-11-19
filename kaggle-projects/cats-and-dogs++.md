# Cats & Dogs++ — Full Project Report

The Cats & Dogs++ competition extended the classic binary “cats vs dogs” task into a **multi-label** problem with extremely **rare subclasses** (Moris, Motya, Biatrix).
Since the official metric is **Macro F1**, rare labels became the main bottleneck and shaped the entire solution.

This project taught me how to build full deep-learning CV pipelines: augmentation strategies, transfer learning, rare-class sampling, threshold tuning, and ensemble methods.

---

## 1. Data Insights

The dataset revealed three properties that determined the whole strategy:

1. **Multi-label setup:**
   A single image can have several labels (e.g., Cat + Moris).

2. **Label hierarchy:**

   * Moris and Motya are types of Cat
   * Biatrix is a type of Dog

   The model does not know this by default → needs post-processing.

3. **Severe imbalance:**
   Rare labels appear dozens of times; common labels appear thousands.
   Since Macro F1 treats all classes equally → rare classes dominate the metric.

---

## 2. Pre-processing & Augmentation

### 2.1 Pre-processing

To preserve animal proportions:

* **pad_to_square** instead of destructive stretching
* Resize to model’s native resolution (EffNetV2-S uses **384×384**)
* Normalize with ImageNet mean & std

### 2.2 Differentiated Augmentation Strategy

Rare animals (“Moris”, “Motya”, “Biatrix”) needed aggressive augmentation to avoid overfitting.

I used two separate pipelines inside the dataset:

* **light augmentations** → common classes

  * RandomResizedCrop (low scale)
  * HorizontalFlip

* **heavy augmentations** → rare classes

  * stronger RandomResizedCrop
  * ColorJitter (brightness/contrast/saturation shifts)

This produced synthetic variation for rare labels and helped the model generalize their shapes.

---

## 3. Models & Training Strategy

### 3.1 Baseline CNN (failed)

A small ConvNet was trained from scratch to test the data pipeline.

**Result:**

* Macro F1 ≈ 0.35
* Completely ignored rare labels → useless for the final solution.

---

### 3.2 Transfer Learning Models (final approach)

I used two strong ImageNet models:

1. **EfficientNet_V2_S** (torchvision)
2. **EfficientNet_B3-NS** (timm)

Both models were fine-tuned using a **two-stage schedule**:

#### Stage A — train only the head

* freeze backbone
* high LR
* quick convergence

#### Stage B — fine-tune everything

* unfreeze backbone
* very low LR for feature extractor (`1e-5`)
* medium LR for classifier
* important to avoid destroying pretrained weights

### Key training parameters

* `IMG_SIZE = 384`
* `BATCH_SIZE = 24`
* `FOLDS = 3` (MultilabelStratifiedKFold)
* `EPOCHS_A = 2`, `EPOCHS_B = 5`
* Optimizer: AdamW
* Loss: BCEWithLogitsLoss
* `pos_weight` per class (handles imbalance)
* WeightedRandomSampler (`K_SAMPLER = 2.0`)

---

## 4. Handling Class Imbalance

A three-level strategy was applied:

### 1) **Data level** — WeightedRandomSampler

Samples with rare labels were drawn more frequently.

### 2) **Loss level** — BCEWithLogits + pos_weight

Rare classes received higher penalties.

### 3) **Augmentation level** — heavy transforms for rare labels

Gave the model visually diverse examples of Moris/Motya/Biatrix.

Together, these three layers dramatically improved Macro F1.

---

## 5. Post-processing (Most Important Part)

Raw model outputs were not used directly.
The biggest improvements came after inference.

### 5.1 Blending (model ensembling)

I blended predictions from EfficientNet-V2-S and EfficientNet-B3 using **per-class α**, optimized on OOF:

```
final_prob[class] = α[class] * prob_v2s + (1 − α[class]) * prob_b3
```

Rare labels preferred different backbones:

* V2-S: strongest for Moris/Motya
* B3: important for Dog/Biatrix

### 5.2 Label hierarchy enforcement

To avoid logically impossible predictions:

```
if Moris == 1 → Cat = 1
if Motya == 1 → Cat = 1
if Biatrix == 1 → Dog = 1
```

This fixed many inconsistent outputs.

### 5.3 Threshold Optimization

Using coordinate descent, I optimized 5 independent thresholds, one per class:

* common labels (Cat, Dog): ~0.50
* rare labels: ~0.92–0.97

This was crucial for Macro F1.

---

## 6. Results

### Final scores

* **OOF Macro F1:** **0.9630**
* **Kaggle Public Macro F1:** **0.9490**

The gap is small → strong generalization, no leakage, stable ensemble.

### Breakdown

| Component         | Impact                         |
| ----------------- | ------------------------------ |
| EfficientNet-V2-S | Base accuracy, strong for cats |
| EfficientNet-B3   | Complementary patterns         |
| Weighted sampling | Huge boost for rare labels     |
| α-blending        | Smoothed fold variance         |
| Hierarchy         | Removed impossible predictions |
| Threshold tuning  | Final major boost              |

---

## 7. Lessons Learned

* Rare classes can dominate metrics → need special treatment at every stage.
* Transfer learning + careful LR scheduling outperform training from scratch.
* Per-class thresholds are often more important than deeper architectures.
* Hierarchical constraints dramatically reduce logical errors.
* Ensembling two strong models provides better stability than trying to overfit one.


