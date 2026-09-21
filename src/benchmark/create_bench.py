import os
import json
import random
import logging
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# Repository-relative data directory (…/src/benchmark/ -> …/data/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")

def load_json_file(filepath):
    """Load JSON data from a given file path."""
    with open(filepath, "r") as f:
        return json.load(f)

def save_json_file(data, filepath):
    """Save JSON data to a given file path."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def compute_embeddings(texts, model):
    """
    Compute embeddings for a list of texts using the provided SentenceTransformer model.
    Returns a dictionary mapping each text to its embedding (as a numpy array).
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return {text: emb for text, emb in zip(texts, embeddings)}

def cosine_sim(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return cosine_similarity([emb1], [emb2])[0][0]

def build_descriptor_pool(pf16_types):
    """
    Build a dictionary mapping PF16 key -> set of its descriptors (from high and low).
    Also return a global set (union) of all descriptors.
    """
    pool_by_key = {}
    global_pool = set()
    for pf in pf16_types:
        key = pf["key"]
        descriptors = set(pf.get("high", [])) | set(pf.get("low", []))
        pool_by_key[key.lower()] = descriptors
        global_pool.update(descriptors)
    return pool_by_key, global_pool

def generate_personality_task(context, context_emb, pf16_types, descriptor_embeddings, pool_by_key, global_pool, model):
    """
    Generate Task 1 (Personality Identification):
      - Correct option: randomly choose one from context["high"].
      - Distractors: from global_pool excluding descriptors of the current PF16 type,
        pick 3 candidates that are most dissimilar to the context (using cosine similarity).
    """
    # Correct option from the context's high descriptors.
    correct_option = random.choice(context["high"])
    
    # Build candidate pool: all descriptors that do NOT belong to the current PF16 key.
    current_key = context["key"].lower()
    excluded = pool_by_key.get(current_key, set())
    candidates = [desc for desc in global_pool if desc not in excluded]
    
    # Compute similarity between the context embedding and each candidate descriptor.
    scores = []
    for cand in candidates:
        cand_emb = np.array(descriptor_embeddings[cand])
        sim = cosine_sim(context_emb, cand_emb)
        scores.append((cand, sim))
    
    # Sort by similarity ascending (i.e. lowest similarity = most dissimilar).
    scores.sort(key=lambda x: x[1])
    distractors = [cand for cand, _ in scores[:3]]
    
    options = distractors + [correct_option]
    random.shuffle(options)
    labels = ['A', 'B', 'C', 'D']
    correct_label = labels[options.index(correct_option)]
    
    return options, correct_option, correct_label

def generate_chord_task(context, chord_map):
    """
    Generate Task 2 (Chord Progression Matching):
      - Correct option: random chord progression from chord_map for context["alike_genre"].
      - Distractors: randomly choose 3 chord progressions from genres other than context["alike_genre"].
    """
    target_genre = context["alike_genre"]
    if target_genre not in chord_map:
        logging.error("Genre '%s' not found in chord map.", target_genre)
        return [], "", ""
    
    correct_list = chord_map[target_genre]["chord_progressions"]
    correct_option = random.choice(correct_list)
    
    # Build pool of chord progressions from genres other than the target.
    pool = []
    for genre, data in chord_map.items():
        if genre != target_genre:
            pool.extend(data.get("chord_progressions", []))
    
    # Choose 3 unique distractors randomly.
    distractors = random.sample(pool, 3) if len(pool) >= 3 else pool
    options = distractors + [correct_option]
    random.shuffle(options)
    labels = ['A', 'B', 'C', 'D']
    correct_label = labels[options.index(correct_option)]
    
    return options, correct_option, correct_label

def generate_benchmark(contexts_file, pf16_types_file, chord_map_file, output_file, cache_dir=None):
    """
    Generate the easy benchmark.
      1. Load contexts, PF16 types, and chord progression mapping.
      2. Compute and store embeddings for all context texts and for all unique personality descriptors.
      3. For each context, generate Task 1 and Task 2.
      4. Output a flat JSON file (no nested metadata) with the following keys:
         index, id, context, question_1, label_1, label_text_1, ans_text_1, answer_1,
         question_2, label_2, label_text_2, ans_text_2, answer_2, domain, subdomain,
         Difficulty, genre, pf16_type.
    """
    # Load data from JSON files.
    contexts = load_json_file(contexts_file)
    pf16_types = load_json_file(pf16_types_file)
    chord_map = load_json_file(chord_map_file)
    
    # Initialize the transformer model. When cache_dir is None the default
    # Hugging Face cache location (or $HF_HOME) is used.
    model = SentenceTransformer('all-mpnet-base-v2', cache_folder=cache_dir)
    
    # Precompute embeddings for all contexts.
    context_texts = [c["context"] for c in contexts]
    context_embeddings = model.encode(context_texts, show_progress_bar=True)
    context_emb_dict = {c["context"]: emb for c, emb in zip(contexts, context_embeddings)}
    
    # Build descriptor pool (global and by PF16 key) from PF16 types.
    pool_by_key, global_pool = build_descriptor_pool(pf16_types)
    
    # Precompute embeddings for all unique personality descriptors.
    candidate_descriptors = list(global_pool)
    descriptor_embeddings = compute_embeddings(candidate_descriptors, model)
    
    benchmark = []
    index_val = 1
    
    # Define fixed questions for both tasks.
    question_1 = "Based on the context, which of the following best describes the individual's personality traits?"
    question_2 = "Which chord progression best matches the user's music preference?"
    labels = ["A", "B", "C", "D"]
    Difficulty = "Easy"
    
    for context in contexts:
        ctx_text = context["context"]
        # Get precomputed embedding for the context.
        ctx_emb = context_emb_dict[ctx_text]
        
        # Task 1: Personality Identification.
        label_text_1, correct_personality, answer_1 = generate_personality_task(
            context, ctx_emb, pf16_types, descriptor_embeddings, pool_by_key, global_pool, model
        )
        ans_text_1 = correct_personality  # Correct personality option.
        
        # Task 2: Chord Progression Matching.
        label_text_2, correct_chord, answer_2 = generate_chord_task(context, chord_map)
        ans_text_2 = correct_chord
        
        # Create benchmark item with flat structure.
        benchmark_item = {
            "index": index_val,
            "id": f"pma_{str(index_val).zfill(3)}",
            "context": ctx_text,
            "question_1": question_1,
            "label_1": labels,
            "label_text_1": label_text_1,
            "ans_text_1": ans_text_1,
            "answer_1": answer_1,
            "question_2": question_2,
            "label_2": labels,
            "label_text_2": label_text_2,
            "ans_text_2": ans_text_2,
            "answer_2": answer_2,
            "domain": context["domain"],
            "subdomain": context["subdomain"],
            "Difficulty": Difficulty,
            "genre": context["alike_genre"],
            "pf16_type": context["key"]
        }
        benchmark.append(benchmark_item)
        index_val += 1
    
    # Optionally shuffle benchmark items.
    random.shuffle(benchmark)
    save_json_file(benchmark, output_file)
    logging.info("Benchmark generated with %d entries and saved to %s", len(benchmark), output_file)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate easy benchmark with transformer embeddings for PF16 tasks.")
    parser.add_argument("--contexts", type=str, required=True, help="Path to JSON file with generated contexts.")
    parser.add_argument("--pf16_types", type=str, default=os.path.join(DATA_DIR, "pf16_map.json"), help="Path to PF16 types JSON file.")
    parser.add_argument("--chord_map", type=str, default=os.path.join(DATA_DIR, "genre_chord_map.json"), help="Path to chord progression mapping JSON file.")
    parser.add_argument("--output", type=str, default="benchmark_easy.json", help="Output JSON file for the benchmark.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional cache directory for the sentence-transformers model.")
    parser.add_argument("-l", "--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return parser.parse_args()

def main():
    args = parse_args()
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(level=numeric_level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.info("Starting easy benchmark generation...")
    generate_benchmark(args.contexts, args.pf16_types, args.chord_map, args.output, cache_dir=args.cache_dir)
    logging.info("Benchmark generation complete.")

if __name__ == "__main__":
    main()
