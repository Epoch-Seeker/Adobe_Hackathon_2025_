import argparse
import json
import time
import re
import os
import fitz  # PyMuPDF
import torch
import numpy as np
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# --- MODEL CONFIGURATION ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
SUMMARIZER_MODEL_NAME = 'google/flan-t5-small'

def load_models():
    """Loads the sentence embedding and summarization models into memory."""
    print("➡️ Loading embedding model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    print("➡️ Loading summarization model...")
    summarizer = pipeline(
        "summarization",
        model=SUMMARIZER_MODEL_NAME,
        tokenizer=SUMMARIZER_MODEL_NAME,
        device=-1 if device == 'cpu' else 0
    )
    print("✅ Models loaded successfully.")
    return embedding_model, summarizer

def parse_documents_structurally(doc_paths: list) -> list:
    """Parses PDFs to extract text chunks based on structural heuristics (headings)."""
    print(f"📄 Parsing {len(doc_paths)} documents...")
    all_chunks = []
    heading_font_size_factor = 1.15
    
    for doc_path in doc_paths:
        doc_name = os.path.basename(doc_path)
        try:
            doc = fitz.open(doc_path)
        except Exception as e:
            print(f"⚠️ Could not open or read {doc_path}. Skipping. Error: {e}")
            continue
            
        current_section = {"title": "Introduction / Abstract", "content": "", "page_num": 1}

        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            spans = [span for b in blocks if 'lines' in b for l in b['lines'] for span in l['spans']]
            
            if not spans:
                continue

            font_sizes = [span['size'] for span in spans]
            median_font_size = np.median(font_sizes) if font_sizes else 12.0

            for span in spans:
                text = span['text'].strip()
                font_size = span['size']

                is_heading = (
                    font_size > median_font_size * heading_font_size_factor and
                    len(text.split()) < 10 and 
                    not text.endswith('.')
                )

                if is_heading and text:
                    if current_section["content"].strip():
                        all_chunks.append({
                            "doc_name": doc_name,
                            "page_num": current_section["page_num"],
                            "title": current_section["title"],
                            "content": re.sub(r'\s+', ' ', current_section["content"]).strip()
                        })
                    
                    current_section = { "title": text, "content": "", "page_num": page_num }
                else:
                    current_section["content"] += text + " "
        
        if current_section["content"].strip():
            all_chunks.append({
                "doc_name": doc_name,
                "page_num": current_section["page_num"],
                "title": current_section["title"],
                "content": re.sub(r'\s+', ' ', current_section["content"]).strip()
            })

    print(f"✅ Found {len(all_chunks)} semantic sections.")
    return all_chunks

def rank_sections(query: str, chunks: list, model: SentenceTransformer) -> list:
    """Ranks text chunks against a query using cosine similarity."""
    print("🔎 Ranking sections for relevance...")
    if not chunks:
        return []

    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_contents = [chunk['content'] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_contents, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
    for i, chunk in enumerate(chunks):
        chunk['score'] = cosine_scores[i].item()
    
    ranked_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
    
    for i, chunk in enumerate(ranked_chunks):
        chunk['importance_rank'] = i + 1
        
    print("✅ Ranking complete.")
    return ranked_chunks

def generate_refined_text(chunks: list, summarizer, persona: str, job: str, max_sections=5) -> list:
    """Generates a persona-aware summary for the top-ranked sections."""
    print(f"✍️ Generating analysis for top {max_sections} sections...")
    analyzed_chunks = []
    
    for chunk in chunks[:max_sections]:
        content = chunk['content']
        prompt = f"Summarize the following text for a '{persona}' who needs to '{job}':\n\n{content}"
        
        summary = summarizer(prompt, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        
        chunk['refined_text'] = summary
        analyzed_chunks.append(chunk)

    print("✅ Analysis complete.")
    return analyzed_chunks

def main():
    """Main function to run the document intelligence pipeline from a JSON input."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Persona-Driven Document Intelligence System.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    args = parser.parse_args()

    # Get the directory where the JSON file is located to build correct file paths
    json_dir = os.path.dirname(os.path.abspath(args.input_json))

    with open(args.input_json, 'r') as f:
        config = json.load(f)
    
    # --- MODIFIED SECTION TO PARSE NEW JSON STRUCTURE ---
    # Extract document paths by iterating through the 'documents' list
    # and prepending the directory path.
    doc_paths = [os.path.join(json_dir, doc['filename']) for doc in config['documents']]
    
    # Extract persona from the nested object
    persona = config["persona"]["role"]
    
    # Extract job from the nested object
    job = config["job_to_be_done"]["task"]
    # --- END OF MODIFIED SECTION ---

    # 1. Load Models
    embedding_model, summarizer = load_models()
    
    # 2. Parse Documents
    chunks = parse_documents_structurally(doc_paths)
    
    # 3. Rank Sections
    query = f"As a {persona}, my goal is to {job}."
    ranked_chunks = rank_sections(query, chunks, embedding_model)
    
    # 4. Generate Sub-section Analysis
    analyzed_chunks = generate_refined_text(ranked_chunks, summarizer, persona, job)
    
    # 5. Format Output
    output = {
        "metadata": {
            "input_documents": [os.path.basename(doc) for doc in doc_paths],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now(timezone.utc).isoformat()
        },
        "extracted_sections": [
            {
                "document": chunk["doc_name"],
                "page_number": chunk["page_num"],
                "section_title": chunk["title"],
                "importance_rank": chunk["importance_rank"]
            } for chunk in ranked_chunks
        ],
        "sub_section_analysis": [
            {
                "document": chunk["doc_name"],
                "section_title": chunk["title"],
                "page_number": chunk["page_num"],
                "refined_text": chunk["refined_text"]
            } for chunk in analyzed_chunks
        ]
    }
    
    # --- MODIFIED SECTION: SAVE OUTPUT TO FILE ---
    # output_filename = "output.json"
    output_filename = os.path.join("/app/output", "output.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Output successfully saved to {output_filename}")
    # --- END OF MODIFIED SECTION ---
    
    end_time = time.time()
    print(f"\n--- Total processing time: {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()