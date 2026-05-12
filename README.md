# CNN-Pneumonia-detector

A deep learning project that detects pneumonia from chest X-ray images using a fine-tuned ResNet18 model, Grad-CAM attention visualization, and a RAG-powered clinical report generator backed by medical guidelines.

**Authors:** Aye Nyein Kyaw and Isiah Ketton  

🔗 **[Live Demo →](https://aye-pneumonia-detector-cazzdtx94laxcycqyhra7m.streamlit.app/)** 

---

## What was built

This project went beyond the original plan to include a full end-to-end clinical AI pipeline:

| Component | Description |
|---|---|
| Data preprocessing | PyTorch `ImageFolder` + augmentation (flip, rotate, colour jitter) |
| Baseline CNN | 4-block custom CNN trained from scratch |
| Transfer learning | Fine-tuned ResNet18 from `torchvision.models` |
| Grad-CAM | Attention heatmaps showing where the model looks in the X-ray |
| RAG pipeline | ChromaDB vector store + sentence-transformers retrieval from medical guidelines |
| Clinical report | Groq LLM synthesises prediction + retrieved guidelines into a structured report |
| Deployment | Streamlit app deployed publicly on Streamlit Community Cloud |

---

## Results

| Model | Test Accuracy | Normal Recall | Pneumonia Recall |
|---|---|---|---|
| Custom CNN (scratch) | 77.4% | 40.2% | 99.7% |
| ResNet18 (fine-tuned) | **90.7%** | **80.8%** | **96.7%** |

The ResNet18 model is used in the deployed app. The CNN's low normal recall (40%) made it unsuitable for clinical screening — flagging 60% of healthy patients as sick. The ResNet was the better choice for balanced performance across both classes.

---

## Project Outline

**1. Data Preprocessing**
- Load the dataset using PyTorch's `ImageFolder` and `DataLoader`
- Apply image augmentation (rotation, flipping, colour jitter, normalization) to reduce overfitting
- Split into training, validation, and test sets
- Identified and corrected a critically small validation set (16 images) — combined with training set and re-split 80/20

**2. Model Construction**
- Built a baseline CNN from scratch with 4 convolutional blocks and ReLU activations
- Fine-tuned a pretrained ResNet18 for comparison
- Trained with weighted cross-entropy loss (to address class imbalance) and Adam optimizer
- Evaluated with confusion matrix, precision/recall, and F1 score

**3. Analysis & Validation**
- Plotted training/validation loss and accuracy curves
- Generated Grad-CAM heatmaps for both models
- ResNet attention was more anatomically focused; CNN attention was diffuse and scattered
- Compared per-class accuracy — ResNet significantly outperformed CNN on the NORMAL class

**4. Extended: RAG Clinical Report Generator**
- Chunked medical guidelines (WHO, CDC, pediatric) and embedded with `sentence-transformers`
- Stored in ChromaDB vector database for semantic retrieval
- On each prediction, top-3 relevant guideline chunks are retrieved
- Groq LLM generates a structured JSON clinical summary including urgency level, observations, and next steps
- Confidence gate: reports are only generated above 70% model confidence

**5. Ethical Reflection**
- The dataset has a 2.89:1 PNEUMONIA:NORMAL imbalance, which biased the CNN toward over-predicting pneumonia
- Class weights were used in the loss function to partially correct this
- The deployed app includes a mandatory disclaimer on every report: AI output must not replace clinical judgment
- A confidence threshold prevents the model from generating reports when uncertain

---

## Running the app locally

See [SETUP.md](SETUP.md) for full installation and usage instructions.

Quick start:
```bash
pip install -r requirements.txt
python setup_knowledge_base.py
streamlit run app.py
```

---

## Project Timeline

| Week | Task | Status |
|---|---|---|
| Week 9 (10/14–10/20) | GitHub repo, project idea, role assignment | 
| Week 10–11 | Data preprocessing and EDA | 
| Week 12–13 | CNN + ResNet18 training | 
| Week 14–15 | Evaluation, Grad-CAM, ethical discussion | 
| Week 16 (12/2–12/4) | Final presentation | 
| 12/11 | Submit notebooks and project files | 
| Extended | RAG pipeline + Streamlit deployment | 

---

## 📚 Data Acknowledgment

This project uses the **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney,
originally published on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
and sourced from the **Guangzhou Women and Children's Medical Center**.

Dataset License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Source: https://data.mendeley.com/datasets/rscbjbr9sj/2  
Citation: Kermany, D.S. et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell (2018).

---

> ⚠️ **Disclaimer:** This tool is for educational use only and does not constitute medical advice. Model output must not replace qualified clinical or radiological judgment.