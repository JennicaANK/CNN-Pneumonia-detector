# Pneumonia Detection — RAG Clinical Report Generator

---

## Which model to use

Use **ResNet18 only** (`finetuned_resnet.pth`).

The custom CNN had a critical flaw: it correctly identified only 40% of normal patients,
meaning it flagged 140 out of 234 healthy X-rays as pneumonia. The ResNet gets 80.8%
of normals right and 96.7% of pneumonia cases, giving 90.7% overall test accuracy.

---

## Folder structure

Set up your project exactly like this before running anything:

```
your-project/
├── models/
│   └── finetuned_resnet.pth     ← move your checkpoint here
├── guidelines/                   ← optional: add PDF guidelines here
├── chroma_db/                    ← auto-created by setup script
├── .env                          ← your Groq API key
├── requirements.txt
├── setup_knowledge_base.py
├── pipeline.py
└── app.py
```

Create the `models/` folder and move your checkpoint:
```bash
mkdir models
mv finetuned_resnet.pth models/
```

---

## What you need from me

1. **Your `finetuned_resnet.pth` checkpoint** — the ResNet18 one, not the CNN.
2. **A Groq API key** — completely free at https://console.groq.com/ (no credit card needed)

---

## Setup (one time)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your API key
Create a file called `.env` in your project folder:
```
GROQ_API_KEY=gsk_your-key-here
```
Replace the value with your actual key. Never commit this file to GitHub —
add `.env` to your `.gitignore`.

### 3. Build the knowledge base
```bash
python setup_knowledge_base.py
```
This runs once. It embeds 10 built-in guideline excerpts into ChromaDB.
If you want better coverage, drop any pneumonia PDF guidelines into the
`guidelines/` folder and re-run.

---

## Running the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. Upload any chest X-ray JPEG and
the app will show you:

- The original image and Grad-CAM attention heatmap side by side
- A colour-coded prediction badge with confidence
- A structured clinical report from Claude citing the retrieved guidelines

---

## Adding real guidelines (optional but recommended)

Download any of these free PDFs and place them in `guidelines/`:

| Source | URL |
|---|---|
| WHO Pocket Book of Hospital Care for Children | apps.who.int |
| CDC Community-Acquired Pneumonia | cdc.gov |
| BTS Guidelines for Pneumonia | brit-thoracic.org.uk |

Then re-run `python setup_knowledge_base.py` to rebuild the index.

---

## File descriptions

| File | What it does |
|---|---|
| `pipeline.py` | Core logic: model loading, preprocessing, inference, Grad-CAM, RAG retrieval, Claude API call |
| `app.py` | Streamlit UI that calls `pipeline.run_pipeline()` and displays results |
| `setup_knowledge_base.py` | One-time script to embed guidelines into ChromaDB |
| `requirements.txt` | Python dependencies |
| `.env` | Your API key (never commit this) |

---

## Troubleshooting

**`Model file not found`** — Check that `models/finetuned_resnet.pth` exists.

**`GROQ_API_KEY` error** — Make sure `.env` is in the project root and contains the key.

**Report skipped / ChromaDB empty** — Run `python setup_knowledge_base.py` first.

**Slow on CPU** — The model runs fine on CPU, just slower. MPS (Apple Silicon) and CUDA are auto-detected.

**`weights_only` warning from torch.load** — Safe to ignore; it's a PyTorch deprecation notice about pickle security that doesn't affect functionality.

---

## Disclaimer

This project is for educational use only. The model output does not constitute
medical advice and must not be used for clinical decision-making.
