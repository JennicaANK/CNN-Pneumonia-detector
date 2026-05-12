"""
setup_knowledge_base.py
=======================
Run this ONCE before launching the app to build the ChromaDB vector store.

    python setup_knowledge_base.py

It will:
  1. Look for PDF files in guidelines/
  2. If none found, use a built-in set of pneumonia guideline excerpts
  3. Chunk the text, embed it with sentence-transformers, and persist to chroma_db/

You can add real PDFs at any time and re-run to rebuild.
Suggested free PDFs:
  - WHO Pneumonia fact sheet (who.int)
  - CDC CAP guidelines (cdc.gov)
  - BTS lower respiratory tract guidelines
"""

import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

GUIDELINES_DIR = Path("guidelines")
CHROMA_DB_PATH  = Path("chroma_db")
CHUNK_SIZE      = 500   # words per chunk
CHUNK_OVERLAP   = 50    # word overlap between consecutive chunks


# ── Built-in fallback knowledge base ─────────────────────────────────────────
# These are representative guideline excerpts so the app works out of the box.
# Replace or supplement with real PDFs in guidelines/ for better coverage.

FALLBACK_CHUNKS = [
    {
        "source": "WHO Pneumonia Guidelines 2022",
        "section": "Diagnosis",
        "text": (
            "Pneumonia is diagnosed clinically based on the presence of fever, cough, "
            "and rapid breathing. Chest radiograph is used to confirm and classify pneumonia. "
            "Radiographic findings include consolidation, interstitial infiltrates, and pleural "
            "effusion. Lobar consolidation suggests bacterial etiology whereas bilateral "
            "interstitial infiltrates are more consistent with viral or atypical pneumonia."
        ),
    },
    {
        "source": "WHO Pneumonia Guidelines 2022",
        "section": "Severity Classification",
        "text": (
            "Severe pneumonia in children is characterised by chest in-drawing, inability to "
            "drink, and stridor in a calm child. Very severe disease includes central cyanosis, "
            "convulsions, or abnormal drowsiness. Children with any sign of very severe disease "
            "or severe pneumonia should be urgently referred to hospital for inpatient treatment "
            "and oxygen therapy when SpO2 falls below 90%."
        ),
    },
    {
        "source": "WHO Pneumonia Guidelines 2022",
        "section": "Treatment",
        "text": (
            "First-line antibiotic treatment for non-severe pneumonia is oral amoxicillin for "
            "5 days. For severe pneumonia, ampicillin plus gentamicin is recommended for at least "
            "5 days followed by oral amoxicillin for a further 5 days. Oxygen therapy is indicated "
            "when SpO2 is below 90%. Supportive care includes ensuring adequate hydration and "
            "nutrition, and antipyretics for fever."
        ),
    },
    {
        "source": "CDC Community-Acquired Pneumonia 2019",
        "section": "Imaging",
        "text": (
            "Chest radiography is recommended for patients with suspected community-acquired "
            "pneumonia to confirm the diagnosis, assess severity, and identify complications. "
            "Radiographic findings may include alveolar infiltrates, air bronchograms, and "
            "consolidation. CT scan provides more detail but is not routinely indicated for "
            "uncomplicated pneumonia. Follow-up radiograph at 6–8 weeks is indicated for patients "
            "aged over 50 years or those who smoke to exclude underlying malignancy."
        ),
    },
    {
        "source": "CDC Community-Acquired Pneumonia 2019",
        "section": "Severity Assessment",
        "text": (
            "The CURB-65 score assesses pneumonia severity: Confusion, Urea > 7 mmol/L, "
            "Respiratory rate >= 30/min, Blood pressure < 90 mmHg systolic, Age >= 65. "
            "Score 0–1: low severity, suitable for outpatient treatment. "
            "Score 2: moderate severity, consider short-stay inpatient or supervised outpatient. "
            "Score 3+: severe pneumonia, hospital admission required. "
            "PSI/PORT score provides an alternative validated risk stratification tool."
        ),
    },
    {
        "source": "Pediatric Infectious Disease Society Guidelines",
        "section": "Pediatric Pneumonia Diagnosis",
        "text": (
            "Bacterial pneumonia in children typically presents with fever, tachypnea, and cough. "
            "The most common bacterial pathogen is Streptococcus pneumoniae. Viral pneumonia, "
            "most often caused by respiratory syncytial virus (RSV), is more common in children "
            "under 2 years. Chest X-ray alone cannot reliably distinguish bacterial from viral "
            "pneumonia. Procalcitonin and CRP may aid in distinguishing bacterial from viral "
            "aetiology but are not routinely required for outpatient management."
        ),
    },
    {
        "source": "Pediatric Infectious Disease Society Guidelines",
        "section": "Pediatric Management",
        "text": (
            "Children with mild to moderate community-acquired pneumonia can be treated as "
            "outpatients with oral antibiotics if they tolerate oral medications, have reliable "
            "follow-up, and show no signs of severity. Amoxicillin is the preferred first-line "
            "agent for children 3 months to 5 years. Azithromycin is used for atypical organisms "
            "in school-age children. Children with severe respiratory distress, oxygen requirement, "
            "or inability to maintain oral intake require hospitalisation."
        ),
    },
    {
        "source": "BTS Lower Respiratory Tract Guidelines",
        "section": "Normal Chest Radiograph",
        "text": (
            "A normal chest radiograph shows clear lung fields without consolidation or infiltrates, "
            "a normal cardiac silhouette within 50% of thoracic diameter, intact hemidiaphragms, "
            "and sharp costophrenic angles. Normal lung markings are visible as branching "
            "bronchovascular shadows extending to the lung periphery. Absence of these findings "
            "does not exclude early or mild pneumonia, particularly in immunocompromised patients."
        ),
    },
    {
        "source": "BTS Lower Respiratory Tract Guidelines",
        "section": "Radiographic Patterns",
        "text": (
            "Lobar or segmental consolidation on chest X-ray is the classical radiographic pattern "
            "of bacterial pneumonia. Patchy bilateral infiltrates suggest atypical or viral "
            "pneumonia. Cavitation may indicate Staphylococcus aureus, Klebsiella, or anaerobic "
            "infection. Pleural effusion is present in about 40% of bacterial pneumonias and "
            "requires assessment for empyema if large or non-resolving."
        ),
    },
    {
        "source": "WHO Pneumonia Guidelines 2022",
        "section": "Prevention",
        "text": (
            "Vaccination is the most effective intervention for preventing pneumonia. "
            "Pneumococcal conjugate vaccine (PCV) and Haemophilus influenzae type b (Hib) "
            "vaccine are recommended for all children. Annual influenza vaccination is recommended "
            "for high-risk groups. Exclusive breastfeeding for the first 6 months of life reduces "
            "the risk of pneumonia. Reducing household air pollution from indoor cooking fires "
            "substantially reduces the burden of childhood pneumonia in low-income settings."
        ),
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def load_pdfs(directory: Path) -> list[dict]:
    """Try to load and chunk PDF files from a directory.
    
    Returns a list of {source, section, text} dicts.
    Returns [] if pypdf is not installed or no PDFs found.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  pypdf not installed — skipping PDF loading. "
              "Run: pip install pypdf")
        return []

    chunks = []
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        return []

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        try:
            reader = PdfReader(str(pdf_path))
            full_text = " ".join(
                page.extract_text() or "" for page in reader.pages
            )
            # Collapse whitespace
            full_text = re.sub(r"\s+", " ", full_text).strip()

            for i, chunk in enumerate(chunk_text(full_text)):
                chunks.append({
                    "source": pdf_path.stem,
                    "section": f"chunk_{i}",
                    "text": chunk,
                })
        except Exception as exc:
            print(f"  Warning: could not parse {pdf_path.name}: {exc}")

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def build_knowledge_base():
    print("\n── Setting up ChromaDB knowledge base ────────────────────────────")

    GUIDELINES_DIR.mkdir(exist_ok=True)
    CHROMA_DB_PATH.mkdir(exist_ok=True)

    # Decide which chunks to use
    pdf_chunks = load_pdfs(GUIDELINES_DIR)
    if pdf_chunks:
        print(f"  Loaded {len(pdf_chunks)} chunks from {GUIDELINES_DIR}/")
        all_chunks = pdf_chunks
    else:
        print("  No PDFs found — using built-in fallback knowledge base.")
        all_chunks = FALLBACK_CHUNKS

    texts     = [c["text"]    for c in all_chunks]
    metadatas = [{"source": c["source"], "section": c["section"]} for c in all_chunks]
    ids       = [f"chunk_{i}" for i in range(len(all_chunks))]

    # Embed
    print("  Embedding with sentence-transformers/all-MiniLM-L6-v2 …")
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    # Persist to ChromaDB
    print("  Writing to ChromaDB …")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Delete old version to avoid duplicate IDs on re-run
    try:
        client.delete_collection("pneumonia_guidelines")
    except Exception:
        pass

    collection = client.create_collection("pneumonia_guidelines")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"  Done. {collection.count()} chunks indexed in {CHROMA_DB_PATH}/")
    print("─" * 60)
    print("Knowledge base ready. You can now run: streamlit run app.py\n")


if __name__ == "__main__":
    build_knowledge_base()
