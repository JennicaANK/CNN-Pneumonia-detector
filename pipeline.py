"""
pipeline.py
===========
Core logic for the pneumonia detection + RAG report pipeline.
Uses a pure PyTorch Grad-CAM implementation — no cv2 or opencv dependency.
"""

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────
GROQ_MODEL           = "llama-3.1-8b-instant"
MODEL_PATH           = Path("models/finetuned_resnet.pth")
CHROMA_DB_PATH       = Path("chroma_db")
CLASS_NAMES          = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE           = 224
CONFIDENCE_THRESHOLD = 0.70


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: Path = MODEL_PATH) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(
        str(model_path),
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess(image: Image.Image) -> torch.Tensor:
    return _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model: nn.Module, tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx   = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs


# ── Pure-PyTorch Grad-CAM (no cv2) ───────────────────────────────────────────

def _apply_colormap(cam: np.ndarray) -> np.ndarray:
    """Convert a [0,1] grayscale array to an RGB heatmap using matplotlib jet."""
    cmap    = plt.get_cmap("jet")
    heatmap = cmap(cam)[:, :, :3]   # RGBA → RGB
    return heatmap.astype(np.float32)


def generate_gradcam(model: nn.Module, tensor: torch.Tensor, pred_idx: int):
    """
    Compute Grad-CAM using plain PyTorch hooks — no cv2, no external library.

    Returns:
      overlay       uint8 RGB numpy array (224×224×3)
      grayscale_cam float32 array shaped (1, 224, 224), values in [0, 1]
    """
    activations = []
    gradients   = []

    def forward_hook(module, input, output):
        activations.append(output.detach().cpu())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach().cpu())

    target_layer = model.layer4[-1]
    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward then backward for the predicted class
    model.zero_grad()
    output = model(tensor)
    output[0, pred_idx].backward()

    fwd.remove()
    bwd.remove()

    grad = gradients[0].numpy()[0]    # (C, H, W)
    act  = activations[0].numpy()[0]  # (C, H, W)

    # Global-average-pool the gradients to get per-channel weights
    weights = grad.mean(axis=(1, 2))  # (C,)

    # Weighted sum of activation maps
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    # ReLU: keep only positive influences
    cam = np.maximum(cam, 0)

    # Resize to input image size using PIL (no cv2 needed)
    cam_pil = Image.fromarray(cam).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    cam     = np.array(cam_pil, dtype=np.float32)

    # Normalise to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Reconstruct display image from normalised tensor
    img_np = tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    img_np = (img_np * 0.5 + 0.5).clip(0, 1).astype(np.float32)

    # Blend original image with heatmap
    heatmap = _apply_colormap(cam)
    overlay = (0.5 * img_np + 0.5 * heatmap)
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)

    return overlay, cam[np.newaxis, :]   # cam shaped (1, H, W) for describe_gradcam


def describe_gradcam(grayscale_cam: np.ndarray) -> str:
    cam  = grayscale_cam[0]
    h, w = cam.shape
    hot  = cam > 0.65

    if hot.sum() == 0:
        return "no strongly activated region"

    ys, xs   = np.where(hot)
    cy, cx   = ys.mean() / h, xs.mean() / w
    coverage = hot.sum() / (h * w)

    v_zone = "upper"  if cy < 0.38 else "lower"    if cy > 0.62 else "mid"
    h_zone = "left"   if cx < 0.38 else "right"    if cx > 0.62 else "bilateral"
    extent = "focal"  if coverage < 0.10 else "patchy" if coverage < 0.25 else "diffuse"

    return f"{extent} attention in the {v_zone} {h_zone} lung field"


# ── RAG retrieval ─────────────────────────────────────────────────────────────

_EMBEDDER  = None
_COLLECTION = None

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER

def _get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        client      = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        _COLLECTION = client.get_or_create_collection("pneumonia_guidelines")
    return _COLLECTION

def retrieve_guidelines(prediction: str, confidence: float, n: int = 3) -> list:
    collection = _get_collection()
    if collection.count() == 0:
        return []
    query     = f"{prediction} chest X-ray pneumonia diagnosis management confidence {confidence:.0%}"
    embedding = _get_embedder().encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=min(n, collection.count()),
        include=["documents", "metadatas"],
    )
    return [
        {"text": doc, "source": meta.get("source", "Unknown"), "section": meta.get("section", "")}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


# ── Groq report ───────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are an AI assistant reviewing an automated chest X-ray model output for \
educational purposes. This is NOT a clinical diagnosis.

MODEL OUTPUT
  Prediction  : {prediction}
  Confidence  : {confidence:.1%}
  P(NORMAL)   : {p_normal:.1%}
  P(PNEUMONIA): {p_pneumonia:.1%}

GRAD-CAM ATTENTION
  {gradcam_desc}

RETRIEVED GUIDELINE CONTEXT
{guideline_text}

Return ONLY valid JSON — no markdown fences, no prose — with this exact schema:
{{
  "assessment": "one-sentence summary",
  "urgency_level": "ROUTINE | ELEVATED | URGENT",
  "key_observations": ["obs1", "obs2", "obs3"],
  "recommended_next_steps": ["step1", "step2"],
  "relevant_guideline_points": ["point from context"],
  "disclaimer": "This output is generated by an AI model for educational use only and must not replace qualified clinical judgment."
}}"""


def generate_report(prediction, confidence, probs, guideline_chunks, gradcam_desc) -> dict:
    guideline_text = "\n".join(
        f"[{c['source']} — {c['section']}]\n{c['text']}" for c in guideline_chunks
    ) or "No guidelines retrieved."

    prompt = _PROMPT_TEMPLATE.format(
        prediction    = prediction,
        confidence    = confidence,
        p_normal      = float(probs[0]),
        p_pneumonia   = float(probs[1]),
        gradcam_desc  = gradcam_desc,
        guideline_text= guideline_text,
    )

    client   = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model     = GROQ_MODEL,
        messages  = [{"role": "user", "content": prompt}],
        max_tokens= 1024,
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(image: Image.Image, model: nn.Module) -> dict:
    tensor                       = preprocess(image)
    prediction, confidence, probs = predict(model, tensor)
    pred_idx                     = CLASS_NAMES.index(prediction)
    overlay, gcam                = generate_gradcam(model, tensor, pred_idx)
    gradcam_desc                 = describe_gradcam(gcam)
    inconclusive                 = confidence < CONFIDENCE_THRESHOLD

    report           = None
    guideline_chunks = []

    if not inconclusive:
        guideline_chunks = retrieve_guidelines(prediction, confidence)
        try:
            report = generate_report(prediction, confidence, probs, guideline_chunks, gradcam_desc)
        except json.JSONDecodeError as e:
            report = {"error": f"Malformed JSON from model: {e}"}
        except Exception as e:
            report = {"error": str(e)}

    return {
        "prediction":       prediction,
        "confidence":       confidence,
        "probs":            probs,
        "overlay":          overlay,
        "gradcam_desc":     gradcam_desc,
        "inconclusive":     inconclusive,
        "report":           report,
        "guideline_chunks": guideline_chunks,
    }
