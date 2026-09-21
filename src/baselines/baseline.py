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
from scores import Evaluation
from benchmark_evaluator import BenchmarkEvaluator

# -------------------- Main --------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions and evaluate benchmark (JSON based).")
    parser.add_argument("--model_type", type=str, choices=["api", "chatmusician", "local"], required=True, help="Model type.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to local model (if applicable).")
    parser.add_argument("--benchmark_path", type=str, required=True, help="Path to the benchmark JSON file.")
    parser.add_argument("--predictions_output", type=str, required=True, help="Directory (including filename) to store intermediate predictions JSON.")
    parser.add_argument("--evaluation_output", type=str, required=True, help="Directory (including filename) to store final evaluation text output.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path for adapter (if applicable).")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to the cache directory (if applicable).")
    parser.add_argument("--is_cot", type=str, default=False, help="Apply CoT")
    parser.add_argument("--is_cot_multihop", type=str, default=False, help="Apply Multi-hop CoT")
    parser.add_argument("--is_verbalize", type=str, default=False, help="Apply Verbalization")
    parser.add_argument("--pf16_map", type=str, default=None, help="Path to the PF16 trait-to-genre map (defaults to data/pf16_map.json).")
    parser.add_argument("--music_theory", type=str, default=None, help="Path to the genre-to-music-theory map (defaults to data/music_theory.json).")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.model_type.upper(), "INFO"),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    evaluator = BenchmarkEvaluator(
        model_type=args.model_type,
        model_name=args.model_name,
        model_path=args.model_path,
        benchmark_path=args.benchmark_path,
        predictions_output=args.predictions_output,
        adapter_path=args.adapter_path,
        cache_dir=args.cache_dir,
        is_cot=args.is_cot,
        is_cot_multihop=args.is_cot_multihop,
        is_verbalize=args.is_verbalize,
        pf16_map_path=args.pf16_map,
        music_theory_path=args.music_theory
    )
    # Generate predictions and store in intermediate file.
    evaluator.evaluate()
    # Now evaluate predictions.
    eval_obj = Evaluation(
        model_name=args.model_name,
        is_cot=args.is_cot,
        is_cot_multihop=args.is_cot_multihop,
        is_verbalize=args.is_verbalize,
        results_filename=args.predictions_output,
        evaluation_output=args.evaluation_output
    )
    eval_obj.evaluate()

if __name__ == "__main__":
    main()
