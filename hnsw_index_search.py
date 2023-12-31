import re
import hnswlib
import heapq
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pdfminer.high_level import extract_text
import argparse
from fuzzywuzzy import process
from fuzzywuzzy import utils
from fuzzywuzzy import fuzz

def chunk_text(text, chunk_size, overlap):
    """
    Splits the text into chunks of specified size with overlap, using a for loop.
    """
    chunks = []
    for start in range(0, len(text), chunk_size - overlap):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
    return chunks

def embed_text(chunks, model):
    """
    Embeds the text chunks using the provided model.
    """
    return np.array(model.encode(chunks, show_progress_bar=True))

def build_index(embedded_chunks):
    """
    Builds an HNSW index from the embedded text chunks.
    """
    num_elements, dim = embedded_chunks.shape

    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    p.add_items(embedded_chunks, np.arange(num_elements))
    p.set_ef(50)
    
    return p

def search_index(query, index, model, top_k=3):
    """
    Searches the index with the query and returns top_k results.
    """
    query_vector = model.encode([query])
    labels, distances = index.knn_query(query_vector, k=top_k)
    return labels.flatten()

def fuzzy_keyword_search(query, chunks, top_k=3):
    scorer = fuzz.WRatio

    choices = []
    for index, choice in enumerate(chunks):
        score = scorer(query, choice)
        if score >= 0:
            choices.append((choice, score, index))

    final = heapq.nlargest(top_k, choices, key=lambda i: i[1])
    return [x[2] for x in final]

def clean_text(text):
    """
    Cleans the text by removing lines that don't meet certain criteria.
    """
    cleaned_lines = []
    regex_pattern = r"^(?:\s*\w\s*|\s*\d+\s*|\s*\b(File|Edit|View|Help)\b\s*)$"
    
    for line in text.split('\n'):
        trimmed_line = line.strip()
        if trimmed_line and not re.search(regex_pattern, trimmed_line):
            cleaned_lines.append(trimmed_line)
    
    return '\n'.join(cleaned_lines)

def segment_text(text):
    """
    Segments the text into lines.
    """
    return text.split('\n')

def merge_texts(texts):
    """
    Merges multiple texts into one, ensuring that each line is unique.
    """
    merged_text = ""
    lines_seen = set()

    for text in texts:
        cleaned_text = clean_text(text)
        for line in segment_text(cleaned_text):
            if line not in lines_seen:
                merged_text += f"{line}\n"
                lines_seen.add(line)

    return merged_text

def main():
    parser = argparse.ArgumentParser(description="Text search using HNSW and BM25.")
    parser.add_argument("query", type=str, help="Search query.")
    parser.add_argument("-f", "--file", type=str, help="Path to a text file to use as input.")
    parser.add_argument("-t", "--text", type=str, help="Direct text input.")
    args = parser.parse_args()

    if args.file:
        if args.file.lower().endswith('.pdf'):
            text = extract_text(args.file)
        else:
            with open(args.file, 'r') as file:
                text = file.read()
    elif args.text:
        text = args.text
    else:
        parser.error("No text input provided. Use -t for direct text or -f to specify a file.")

    query = args.query
    chunk_size = 500
    overlap = 100

    model = SentenceTransformer('thenlper/gte-small', device='mps')

    text = merge_texts([text])

    chunks = chunk_text(text, chunk_size, overlap)
    embedded_chunks = embed_text(chunks, model)
    index = build_index(embedded_chunks)

    hnsw_results = search_index(query, index, model, top_k=5)
    fuzzy_matches = fuzzy_keyword_search(query, chunks, top_k=5)

    # Combine and deduplicate results
    combined_results = np.unique(np.concatenate((hnsw_results, fuzzy_matches)))

    for result in combined_results[:10]:  # Return up to 10 unique results
        print(chunks[int(result)])

if __name__ == "__main__":
    main()

