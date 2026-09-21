import os
import json
import random
import openai
import logging
import argparse
import time
from tqdm.auto import tqdm

# Repository-relative data directory (…/src/benchmark/ -> …/data/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")

def load_json_file(filepath):
    """Load JSON data from a given file path."""
    with open(filepath, "r") as f:
        return json.load(f)

def call_openai(prompt, model="gpt-4o"):
    """
    Call the OpenAI API with the given prompt.
    
    Parameters:
      - prompt (str): The prompt to send.
      - model (str): The model to use (default "gpt-4o").
    
    Returns:
      - str: The generated context.
    
    Raises:
      - Exception: If the API call fails.
    """
    logger = logging.getLogger("call_openai")
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a creative assistant with deep knowledge in personality traits and musical styles."},
                {"role": "user", "content": prompt}
            ]
        )
        logger.debug("Received response from OpenAI.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Error calling OpenAI API: %s", e)
        raise

def generate_pf16_dataset(target_key, pf16_types_path, domains_path, api_key):
    """
    Generates a synthetic dataset for a specific PF16 key by iterating over all domains,
    subdomains, alike genres, and two random high-option lists (each with 3 descriptors).
    
    The total number of generated contexts will be:
    domains (10) * subdomains (3) * alike genres (3) * high_options (2) = 180.
    
    The generated context is produced by calling the OpenAI API.
    The final dataset is saved in a JSON file.
    """
    logger = logging.getLogger("generate_pf16_dataset")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("No OpenAI API key provided. Pass --api_key or set OPENAI_API_KEY.")
    openai.api_key = api_key

    # Load PF16 mapping and domain data from JSON files
    pf16_types = load_json_file(pf16_types_path)
    domains_data = load_json_file(domains_path)

    # Find the PF16 entry for the target key (case-insensitive)
    target_pf16 = next((item for item in pf16_types if item["key"].lower() == target_key.lower()), None)
    if not target_pf16:
        logger.error("PF16 type with key '%s' not found.", target_key)
        return

    high_values = target_pf16["high"]
    alike_genres = target_pf16["alike_genre"]

    # Create 2 distinct lists (each with 3 descriptors) from the 'high' attribute
    if len(high_values) >= 6:
        high_list1 = random.sample(high_values, 3)
        remaining = [x for x in high_values if x not in high_list1]
        high_list2 = random.sample(remaining, 3)
    else:
        high_list1 = random.sample(high_values, 3)
        high_list2 = random.sample(high_values, 3)
    high_options = [high_list1, high_list2]

    dataset = []
    total_count = 0

    # Iterate over domain, subdomain, alike genres, and high option lists
    for domain_entry in tqdm(domains_data, desc="Generating contexts"):
        domain = domain_entry["domain"]
        for subdomain in domain_entry["subdomains"]:
            for genre in alike_genres:
                for high_option in high_options:
                    # Build the prompt
                    prompt = f"""
                                You specialize in crafting realistic, engaging scenarios that subtly reflect individuals' personality traits and musical preferences without explicitly mentioning them.
                                Generate a short, structured context (50-70 words) that describes an individual's character and preferred music style in a natural, real-world scenario.
                                Do not explicitly mention the personality trait or the music genre.

                                Instructions:
                                1. The individual exhibits qualities such as: {', '.join(high_option)}.
                                2. Their musical preference subtly aligns with a style reminiscent of {genre}.
                                3. Incorporate contextual details from the domain "{domain}" and subdomain "{subdomain}".

                                Please produce a plain text description.
                            """
                    logger.info("Generating context for Domain: '%s', Subdomain: '%s', Genre: '%s', High descriptors: %s",
                                domain, subdomain, genre, high_option)
                    try:
                        context_text = call_openai(prompt, model="gpt-4o")
                        time.sleep(1)
                    except Exception as e:
                        logger.error("Failed to generate context: %s", e)
                        context_text = ""
                    
                    entry = {
                        "key": target_pf16["key"],
                        "domain": domain,
                        "subdomain": subdomain,
                        "alike_genre": genre,
                        "high": high_option,
                        "context": context_text
                    }
                    dataset.append(entry)
                    total_count += 1
        #             break
        #         break
        #     break
        # break

    logger.info("Total contexts generated: %d", total_count)

    output_filename = f"{target_key}_contexts.json"
    with open(output_filename, "w") as outfile:
        json.dump(dataset, outfile, indent=2)
    logger.info("Dataset saved to %s", output_filename)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark contexts for a PF16 personality type using OpenAI.")
    parser.add_argument("--pf16_key", type=str, required=True, help="The PF16 key to target (e.g., 'Warmth').")
    parser.add_argument("--pf16_types", type=str, default=os.path.join(DATA_DIR, "pf16_map.json"), help="Path to the PF16 types JSON file.")
    parser.add_argument("--domains", type=str, default=os.path.join(DATA_DIR, "domain.json"), help="Path to the domain JSON file.")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.")
    parser.add_argument("-l", "--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return parser.parse_args()

def main():
    args = parse_args()
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(level=numeric_level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("main")
    logger.info("Starting PF16 dataset generation for key: %s", args.pf16_key)
    
    generate_pf16_dataset(args.pf16_key, args.pf16_types, args.domains, args.api_key)
    logger.info("PF16 dataset generation complete.")

if __name__ == "__main__":
    main()
