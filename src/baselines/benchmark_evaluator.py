import os
import time
import json
import argparse
import random
import logging
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, set_seed
import torch
import ast
from peft import PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, f1_score
from string import Template
from typing import Optional
import numpy as np

# Repository-relative data directory (…/src/baselines/ -> …/data/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")

class BenchmarkEvaluator:
    def __init__(
            self, 
            model_type, 
            model_name, 
            model_path, 
            benchmark_path, 
            predictions_output, 
            adapter_path=None, 
            cache_dir=None, 
            is_cot=False,
            is_cot_multihop=False,
            is_verbalize=False,
            pf16_map_path=None,
            music_theory_path=None,
        ):
        # Deterministic Outputs
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        # torch.use_deterministic_algorithms(True)
        # if you’re on CUDA:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        # 🡒 Transformers‑specific helper (also seeds Python/NumPy/CUDA)
        set_seed(SEED)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = torch.Generator(device_str).manual_seed(SEED)

        self.model_type = model_type  # "api", "chatmusician", or "local"
        self.model_name = model_name
        self.model_path = model_path
        self.benchmark_path = benchmark_path  # now a JSON file
        self.predictions_output = predictions_output  # intermediate predictions JSON file
        self.adapter_path = adapter_path
        self.cache_dir = cache_dir
        self.is_cot = is_cot
        self.is_cot_multihop = is_cot_multihop
        self.is_verbalize = is_verbalize
        self.prompt_template = Template("User: ${inst} </s> Assistant: ")

        # Resource files used by the verbalization pipeline. Defaults resolve to the
        # `data/` directory of this repository, and can be overridden by the caller.
        self.pf16_map_path = pf16_map_path or os.path.join(DATA_DIR, "pf16_map.json")
        self.music_theory_path = music_theory_path or os.path.join(DATA_DIR, "music_theory.json")

        # Set Hugging Face cache environment variables if cache_dir is provided
        if self.cache_dir:
            os.environ['HF_HOME'] = self.cache_dir
            os.environ['TRANSFORMERS_CACHE'] = self.cache_dir

        if self.model_type == "api":
            self._initialize_api_client()
        elif self.model_type == "chatmusician":
            self._initialize_chatmusician()
        elif self.model_type == "local":
            self._initialize_local_model()
        else:
            raise ValueError("Invalid model type. Use 'api', 'chatmusician', or 'local'.")

    def _initialize_api_client(self):
        # Initialize API client (e.g., OpenAI). Credentials are read from the
        # environment so that no key ever needs to live in the repository.
        if self.model_name.startswith("gpt-") or self.model_name.startswith("davinci") or "openai" in self.model_name:
            from openai import OpenAI
            openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set. Export it before running the evaluation.")
            # OPENAI_BASE_URL is optional and lets you point at an OpenAI-compatible endpoint.
            base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
            self.client = OpenAI(api_key=openai_api_key, base_url=base_url)
        elif "groq" in self.model_name:
            from groq import Groq
            self.model_name = self.model_name[5:]
            groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is not set. Export it before running the evaluation.")
            self.client = Groq(api_key=groq_api_key)
        else:
            raise ValueError("Unsupported API model name.")

    def _initialize_chatmusician(self):
        """Initialize ChatMusician model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16, device_map="cuda", cache_dir=self.cache_dir
        ).eval()
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.1,
            min_new_tokens=10,
            max_new_tokens=100
        )

    def _initialize_local_model(self):
        print(f"Loading tokenizer and base model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            cache_dir=self.cache_dir
        )
        # ensure there's a pad token - 
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            resume_download=True,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if self.adapter_path:
            print(f"Loading PEFT adapter from {self.adapter_path}")
            # LOAD Custom tokenizer for Chord_expert model
            if self.model_name == "chord_expert":
                # --- Load tokenizer from the adapter folder so you get the added chord tokens ---
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.adapter_path,      # points at chord_language_model_10ksteps/
                    cache_dir=self.cache_dir,
                    use_fast=True
                )
                self.tokenizer.model_max_length = min(self.tokenizer.model_max_length, 2048)
                
                # ensure there's a pad token
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                base_model.resize_token_embeddings(len(self.tokenizer))

            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.model.eval()
            self.use_pipeline = False
            print("Adapter loaded. Using direct generate().")

        else:
            self.model = base_model
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="cuda"
            )
            self.use_pipeline = True
            print("No adapter. Using Hugging Face pipeline().")

    def load_benchmark(self):
        """Load the benchmark dataset from a JSON file."""
        with open(self.benchmark_path, "r") as f:
            data = json.load(f)
        return data

    def generate_response_api(self, prompt):
        """Generate response using the API."""

        seq_length = 50 

        if self.is_cot:
            # If CoT - seq length is 512
            seq_length = 512
        messages = [
            {"role": "system", "content": "You are an expert assistant. Respond concisely."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=seq_length,
            # max_completion_tokens=seq_length,
            top_p=1,
        )
        return response.choices[0].message.content.strip()

    def generate_response_chatmusician(self, instruction):
        """Generate response using ChatMusician."""
        prompt = self.prompt_template.safe_substitute({"inst": instruction})
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        response = self.model.generate(
            input_ids=inputs["input_ids"].to(self.model.device),
            attention_mask=inputs["attention_mask"].to(self.model.device),
            eos_token_id=self.tokenizer.eos_token_id,
            generation_config=self.generation_config,
        )
        return self.tokenizer.decode(response[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def generate_response_local(self, messages):
        """Generate a response using pipeline() or generate(), depending on adapter presence."""

        # Default Params - 
        # seq length = 50
        seq_length = 50 

        if self.is_cot or self.is_verbalize:
            # If CoT - seq length is 512
            seq_length = 512
        
        if "deepseek" in self.model_name:
            seq_length = 1024

        if self.use_pipeline:
            outputs = self.pipeline(
                messages,
                max_new_tokens=seq_length,
                eos_token_id=self.pipeline.tokenizer.eos_token_id,
                temperature=1e-5,
                top_p=1,
            )
            response = outputs[0]["generated_text"][-1]["content"].strip()
            return response

        else:
            # Use generate() after formatting the messages properly
            # Apply chat template to convert messages to raw prompt
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # for chat models expecting the "Assistant:" part
            )

            # Tokenize and send to device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=seq_length,
                    temperature=1e-5,
                    top_p=1.0,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            # Get only the generated tokens
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            return response

    def set_seed_for_generation(self, seed=0):
        torch.random.manual_seed(seed)
        set_seed(seed)

    def evaluate(self):
        """
        Process the benchmark dataset (JSON format), generate predictions for both tasks,
        and store the predictions to the intermediate output file.
        The benchmark JSON is expected to be a list of flat dictionaries.
        """
        benchmark_data = self.load_benchmark()
        results = []
        
        # Task specific instruction - Regular, CoT
        t1_ins = "Based on the context below, which of the following best describes the individual's personality traits?"
        t1_cot_ins = "Based on the context below, which of the following best describes the individual's personality traits? Analyze step by step about the actions and behaviors described in the context, how they reflect specific personality traits, and which option best matches these traits."

        t2_ins = "Which chord progression best matches the user's music preference?"
        t2_cot_ins = "Which chord progression best matches the user's music preference? Please analyze step by step how the context reveals the individual's genre preference—discuss their musical taste, the energy or vibe expressed, and any subtle cues from the context. Finally, state the chosen chord progression."

        # Output Indicator - Same for all
        footer = "Respond with a single letter (A, B, C, or D). No explanation needed."
        cot_footer = "Please limit your thinking within 100 words. Provide your answer enclosed in <answer></answer> tags. For example: <answer>A</answer>"

        # Verbalization
        # T1 - Identifying Personality
        # will add later
        # T2 - Identifying Chords
        trait_to_genres = {}
        genres_to_chords = {}
        if self.is_verbalize:
            with open(self.pf16_map_path, 'r') as file:
                genre_data = json.load(file)
            for item in genre_data:
                trait_to_genres[item['key']] = item['alike_genre']

            with open(self.music_theory_path, 'r') as file:
                genres_to_chords = json.load(file)

        def _build_verbalize_prompt_task2_genre(context: str) -> str:
            # only include the trait name and its aligned genres

            lines = [
                "You are a music expert and songwriter who knows how personality drives musical taste.",
                "Below is a mapping of personality traits to genres:"
            ]
            for trait, genres in trait_to_genres.items():
                lines.append(f"- {trait}: {', '.join(genres)}")
            lines.append("")  # blank line
            lines.append(f"Context: {context}")
            lines.append(
                "Task: In no more than 30 words, identify the dominant personality traits from the context "
                "and recommend 2-3 genres they are most likely to enjoy, based on the mapping above."
            )
            return "\n".join(lines)

        def _build_verbalize_prompt_task2_chord(context: str, personality_genre_knowledge, label_text_2) -> str:
            # only include the trait name and its aligned genres

            lines = [
                "You are a music expert and songwriter who knows how personality influences musical preferences and harmonic structure.",
                "Below is a mapping of genres to music theory knowledge:"
            ]
            for genre, theory in genres_to_chords.items():
                lines.append(f"- {genre}: {theory}")
            lines.append("")  # blank line
            lines.append(f"Context: {context}")
            lines.append(f"Personality and Genre Knowledge: {personality_genre_knowledge}")
            lines.append(
                "Question: Which chord progression best matches the user's music preference? "
            )
            lines.append("Options:")
            for idx, prog in enumerate(label_text_2):
                # A, B, C, D …
                letter = chr(ord("A") + idx)
                lines.append(f"{letter}: {prog}")
            lines.append(
                "Please analyze how the context, predicted genre, and music theory knowledge align to justify your selection."
            )
            lines.append(
                "Limit your answer to 100 words. Conclude with ONLY the letter of your choice inside <answer> tags. "
                "Example: <answer>C</answer>."
            )
            return "\n".join(lines)


        if self.is_cot:
            t1_ins = t1_cot_ins
            t2_ins = t2_cot_ins
            footer = cot_footer

        for idx, row in tqdm(enumerate(benchmark_data), total=len(benchmark_data), desc="Processing benchmark dataset"):
            # Task 1: Personality Prediction
            # (Assume row["label_text_1"] is already a list of 4 options)
            task_1_prompt = f"""
                                Question: {t1_ins}
                                Context: {row['context']}

                                Options:
                                A: {row['label_text_1'][0]}
                                B: {row['label_text_1'][1]}
                                C: {row['label_text_1'][2]}
                                D: {row['label_text_1'][3]}

                                {footer}
                            """
            # Task 2: Chord Progression Prediction
            task_2_prompt = f"""
                                Question: {t2_ins}
                                Context: {row['context']}
                                
                                Options:
                                A: {row['label_text_2'][0]}
                                B: {row['label_text_2'][1]}
                                C: {row['label_text_2'][2]}
                                D: {row['label_text_2'][3]}

                                {footer}
                            """
            # Generate responses based on the model type
            # API inference
            if self.model_type == "api":
                def safe_generate_response(prompt, max_retry_time=100, increment=5, base_sleep=3):
                    total_sleep_time = 0
                    while total_sleep_time <= max_retry_time:
                        try:
                            return self.generate_response_api(prompt)
                        except Exception as e:
                            print(f"API call failed: {e}. Retrying in {base_sleep + increment} seconds...")
                            time.sleep(base_sleep + increment)
                            total_sleep_time += base_sleep + increment
                    print(f"Max retry time exceeded. Returning empty response.")
                    return ""
                task_1_answer = safe_generate_response(task_1_prompt)
                time.sleep(0.25)
                task_2_answer = safe_generate_response(task_2_prompt)
                time.sleep(0.25)
            elif self.model_type == "chatmusician":
                task_1_answer = self.generate_response_chatmusician(task_1_prompt)
                task_2_answer = self.generate_response_chatmusician(task_2_prompt)

            # LOCAL inference
            else:  
                
                if "deepseek" in self.model_name:
                    output_indicator = "\n<think>\n"
                    task_1_answer = self.generate_response_local([{"role": "user", "content": task_1_prompt+output_indicator}])
                    task_2_answer = self.generate_response_local([{"role": "user", "content": task_2_prompt+output_indicator}])

                else:
                    # TASK 1 - Personality prediction
                    task_1_answer = self.generate_response_local([{"role": "user", "content": task_1_prompt}])
                    # TASK 2 - Chord Prediction
                    if self.is_verbalize:
                        task_2_prompt_indermediate = _build_verbalize_prompt_task2_genre(row['context'])
                        task_2_indermediate = self.generate_response_local([{"role": "user", "content": task_2_prompt_indermediate}])
                        # make task 2 prompt with the intermediate genre knowledge
                        task_2_prompt = _build_verbalize_prompt_task2_chord(row['context'],task_2_indermediate, row['label_text_2'])
                        task_2_answer = self.generate_response_local([{"role": "user", "content": task_2_prompt}])

                    else:
                        task_2_answer = self.generate_response_local([{"role": "user", "content": task_2_prompt}])
            
            result_entry = {
                "true_label_1": row["answer_1"],
                "predicted_label_1": task_1_answer,
                "true_label_2": row["answer_2"],
                "predicted_label_2": task_2_answer
            }
            results.append(result_entry)
        # Save the intermediate predictions JSON file
        with open(self.predictions_output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Intermediate predictions saved to '{self.predictions_output}'")
        return results
