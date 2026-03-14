# Bounded By LLMs

### Priyanka S - PES1UG23AM219

### Padarthi Neha Sai - PES1UG23AM200
---

## 🔍 Overview

This project fine-tunes a **ViT-GPT2 VisionEncoderDecoder** model on the [RICO Screen2Words](https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words) dataset to generate descriptive natural language captions for mobile app UI screenshots.

Given a screenshot of any Android app screen — a gallery page, a login form, a social profile, a settings menu — the model outputs a short human-readable description of what is displayed.

**Real-world applications:**
- Accessibility tools that describe app screens for visually impaired users
- UI/UX testing pipelines that generate readable descriptions of app states
- Semantic search over mobile app screenshot datasets
- Automated annotation for large-scale UI research datasets

The full pipeline — data loading, preprocessing, fine-tuning, evaluation, and inference — is designed to run end-to-end on a **free T4 GPU** (Google Colab or Kaggle).

---

## 🤗 Model on HuggingFace

The fine-tuned model is publicly available on HuggingFace Hub:

**[pnehasai/rico-screen-caption-model](https://huggingface.co/pnehasai/rico-screen-caption-model)**

The repository contains all three required components:
- Model weights (`pytorch_model.bin`)
- Feature extractor config (`preprocessor_config.json`)
- Tokenizer config (`tokenizer_config.json`, `vocab.json`, `merges.txt`)

---

## 🎬 Demo

| Input Screenshot | Predicted Caption | Reference Caption |
|---|---|---|
| Gallery page with space images | `display of screen shows images on gallery page of app` | `display of screen shows images on gallery` |
| Drawing of a girl | `page displaying drawing image of a girl with long hair` | `page displaying drawing image of a girl with long hair` |
| App settings / options list | `display of list of options in an app with app settings page` | `different kinds of rewards and membership details in app` |

---

## 📦 Dataset

**[RICO Screen2Words](https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words)**

RICO (Rico Is a Collection of UIs) is a large-scale dataset of Android UI screenshots. The Screen2Words extension pairs each screenshot with **5 human-written captions**, making it well-suited for image captioning tasks on mobile interfaces.

| Property | Detail |
|---|---|
| Total images | ~22,000 |
| Captions per image | 5 (human-written) |
| Caption style | Short functional descriptions (avg. ~8 words) |
| Subset used for training | 500 samples (400 train / 100 val) |

**Why 500 samples?** Training on the full dataset would exceed practical time limits on free-tier T4 compute. 500 samples keeps training under 15 minutes while still producing a meaningful fine-tuned checkpoint. Scaling to the full dataset is the most direct path to higher BLEU scores.

---

## 🧠 Model Architecture

We use `nlpconnect/vit-gpt2-image-captioning` as the base model — a **VisionEncoderDecoder** combining:

| Component | Role |
|---|---|
| **Encoder**: Vision Transformer (ViT) | Extracts visual features from the 224×224 screenshot |
| **Decoder**: GPT-2 | Generates natural language captions autoregressively from visual features |

**Why ViT-GPT2?**
- It is a Small Language Model (SLM) that fits comfortably within T4 GPU memory (~3.5GB)
- It is pretrained on general image captioning (COCO), giving it a strong visual-language foundation before domain-specific fine-tuning
- The VisionEncoderDecoder architecture is natively supported by HuggingFace `Trainer`, keeping the training loop simple and well-tested

**Note on padding:** GPT-2 has no dedicated padding token. We set `tokenizer.pad_token = tokenizer.eos_token` as the standard workaround, which is why the following warning appears during inference:

```
The attention mask is not set and cannot be inferred from input because
pad token is same as eos token.
```

To suppress this and ensure reliable outputs, always pass `attention_mask` explicitly when calling `model.generate()` (see [Inference](#inference)).

---

## ⚙️ Training Setup

All training was done on a **T4 GPU (16GB VRAM)** using HuggingFace `Trainer`.

| Hyperparameter | Value | Reasoning |
|---|---|---|
| `per_device_train_batch_size` | 4 | Keeps GPU memory under ~8GB, well within T4 limits |
| `num_train_epochs` | 5 | 2,000 total gradient steps — enough to adapt to the UI domain without overfitting on 400 samples |
| `learning_rate` | 3e-5 | Lower than default (5e-5) to avoid catastrophic forgetting of pretrained COCO captioning knowledge |
| `warmup_steps` | 20 | Short linear warmup prevents large initial gradient updates from destabilizing pretrained weights |
| `fp16` | True | Mixed precision halves memory usage and speeds up training ~1.5–2× on T4 Tensor Cores |
| `max_length` (tokenizer) | 32 | Screen2Words captions average ~8 words; 32 tokens provides comfortable headroom |
| `save_strategy` | none | Avoids writing large checkpoint files to Colab's limited disk during training |

**Label masking:** Padding positions in the labels tensor are set to `-100`. PyTorch's `CrossEntropyLoss` ignores index `-100` by default, so the model is penalized only on real caption tokens — not on padding.

**Fine-tuning library:** HuggingFace `Trainer` (from the `transformers` library)

---

## 📊 Results

### BLEU-4 Score: **0.1722**

BLEU-4 measures 4-gram precision overlap between predicted and reference captions. For domain-specific captioning systems trained on small subsets, scores in the 0.10–0.25 range are typical.

| BLEU-4 Range | Interpretation |
|---|---|
| < 0.10 | Very poor — little structural overlap |
| 0.10 – 0.20 | Below average — some caption structure learned |
| 0.20 – 0.30 | Moderate — reasonable semantic accuracy |
| 0.30 – 0.50 | Good — strong match to references |
| > 0.50 | Excellent — near human-level |

A score of **0.1722** given only 500 training samples is a reasonable baseline. The model has learned the language style and structure of Screen2Words captions and produces fluent output, but needs more training data to reliably capture screen-specific semantics.

---

### Sample Predictions

| # | Predicted | Reference | Assessment |
|---|---|---|---|
| 1 | page displaying drawing image of a girl with long hair | page displaying drawing image of a girl with long hair | ✅ Exact match |
| 2 | display of list of options in an app with app settings page | different kinds of rewards and membership details in app | ❌ Wrong semantic focus |
| 3 | display of a page showing of workout app with workout app settings page displayed | details of a person in a social app | ❌ Wrong app type identified |
| 4 | page asking to enter login details for verification of account details | notification displaying an information | ⚠️ Partial — detected input screen, missed notification context |
| 5 | display of a page showing of payment options for app | page asking to continue with the app | ⚠️ Partial — detected app page, wrong functional context |

### Analysis

**What the model gets right:**
- Visually distinctive screens (e.g., a drawing) are captioned with high accuracy — Image 1 is an exact match
- All predictions are grammatically fluent and follow the Screen2Words caption style correctly
- The model reliably identifies the medium ("display of a page", "app") even when specific content is wrong

**Where the model struggles:**
- Context-dependent screens (notifications, reward pages, social profiles) are frequently misidentified because they share visual structure with higher-frequency screen types (settings pages, login forms, option lists)
- With only 500 training samples, the model defaults to common patterns rather than learning fine-grained visual differences between similar screen types

**Why Image 1 succeeded:** A drawing of a girl is visually unlike any typical app UI element. The ViT encoder produces a distinctive feature representation for it, making caption generation unambiguous.

**Why Images 2–5 struggled:** Reward pages, profile cards, notification banners, and onboarding CTAs all share generic list/card/form visual structures. Limited training data means the model cannot reliably distinguish these at inference time.

---

## 🚀 Inference

### Pull from HuggingFace and run inference

This is the primary way to use the model. No local training required.

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load all three components directly from HuggingFace Hub
hf_model_name = "foreseeitwithme/rico-vit-gpt2-finetuned"

model = VisionEncoderDecoderModel.from_pretrained(hf_model_name)
feature_extractor = ViTImageProcessor.from_pretrained(hf_model_name)
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

model = model.to("cuda")
model.eval()  # disable dropout for deterministic inference

# Load your screenshot
image = Image.open("your_screenshot.png").convert("RGB")

# Preprocess — resizes and normalizes to ViT's expected input (224x224, ImageNet stats)
inputs = feature_extractor(images=image, return_tensors="pt").to("cuda")

# Generate caption
# torch.no_grad() disables gradient tracking — not needed at inference, saves memory
with torch.no_grad():
    output_ids = model.generate(
        inputs.pixel_values,
        max_length=32   # matches the max_length used during training
    )

caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Caption:", caption)
```

### Run on multiple images from the dataset

```python
from datasets import load_dataset

dataset = load_dataset("rootsautomation/RICO-Screen2Words")

for i in [0, 5, 10]:
    sample = dataset["train"][i]
    image = sample["image"].convert("RGB")

    inputs = feature_extractor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(inputs.pixel_values, max_length=32)

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Predicted : {caption}")
    print(f"Reference : {sample['captions'][0]}")
    print()
```

> **No adapter merging required.** This model was trained with full fine-tuning via HuggingFace `Trainer` — not LoRA or PEFT. The weights loaded from the Hub are the complete fine-tuned model, ready to use directly without any merging step.

---


## ⚠️ Limitations

- **Small training set:** 500 of ~22,000 available samples. The model has not seen enough screen diversity to reliably distinguish visually similar screen types (e.g. notification vs. login, reward page vs. settings).
- **Single reference during training:** Only the first of 5 available captions per image is used for training. A multi-reference training setup could improve caption diversity.
- **Attention mask warning:** Because GPT-2 shares the pad and EOS token IDs, passing `attention_mask` explicitly during generation is recommended for reliable results.
- **Domain specificity:** The model is tuned for Android mobile UI screenshots and will not generalize well to web interfaces, desktop UIs, or general photographs.
- **BLEU as a sole metric:** BLEU-4 underestimates true caption quality when captions have high wording variance. CIDEr or METEOR would give a more complete picture.

---

## 🔭 Future Work

- [ ] Scale training to the full ~22,000 sample dataset for substantially higher BLEU scores
- [ ] Add CIDEr and METEOR metrics for a more complete evaluation
- [ ] Fine-tune a larger vision-language model (e.g., BLIP-2, LLaVA) for comparison
- [ ] Use all 5 reference captions during training instead of just the first
- [ ] Add attention heatmap visualization to inspect which UI regions drive the caption
- [ ] Evaluate on held-out screen type categories to measure generalization across app types

---
