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
import re
from typing import Optional

class Evaluation:
    def __init__(
        self, 
        model_name, 
        is_cot, 
        is_cot_multihop, 
        is_verbalize, 
        results_filename, 
        evaluation_output
    ):
        self.model_name = model_name
        self.is_cot = is_cot
        self.is_cot_multihop = is_cot_multihop
        self.is_verbalize = is_verbalize
        self.results_filename = results_filename
        self.evaluation_output = evaluation_output
        self.results = []
        self.valid_options = {"A", "B", "C", "D"}

    def load_results(self):
        with open(self.results_filename, "r") as f:
            self.results = json.load(f)
        print(f"Loaded results from '{self.results_filename}'")

    def extract_answer_from_text(self, text: str) -> Optional[str]:
        """
        Try multiple patterns to pull out A–D:
         1) <answer>A</answer>
         2) “recommended chord progression is A”
         3) any letter before a colon (A:, B:, …) — *take the last match*
         4) a lone letter at the very end
        """
        # 1) explicit tags
        m = re.search(r"<answer>\s*([A-D])\s*</answer>", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 2) prose mention “... is A”
        m = re.search(r"recommended chord progression is\s*([A-D])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 3) any “A:” / “B:” pattern — collect all and pick the last
        all_label_colons = re.findall(r"\b([A-D])\s*:", text, re.IGNORECASE)
        if all_label_colons:
            return all_label_colons[-1].upper()

        # 4) lone letter at very end
        m = re.search(r"([A-D])\s*$", text.strip(), re.IGNORECASE)
        if m:
            return m.group(1).upper()

        return None
    
    def extract_answer_deepseek(self, text):
        pass

    def post_process_predictions(self):
        """
        Post-process predictions to:
        - Extract the first letter from assistant's response
        - Ensure it's one of {A, B, C, D}, otherwise label as NEI
        """
        if self.is_verbalize:
            print("Verbalization Evaluation")
        elif self.is_cot:
            print("CoT Evaluation")
        else:
            print("Baseline Evaluation")

        for entry in self.results:
            if "deepseek" in self.model_name:
                # Expecting think tag ends in the answer - \n</think>\n\n{answer}
                entry["predicted_label_1"] = (
                    entry["predicted_label_1"].strip()[-1]
                    if entry["predicted_label_1"].strip() and entry["predicted_label_1"].strip()[-1] in self.valid_options
                    else "NEI"
                )
                entry["predicted_label_2"] = (
                    entry["predicted_label_2"].strip()[-1]
                    if entry["predicted_label_2"].strip() and entry["predicted_label_2"].strip()[-1] in self.valid_options
                    else "NEI"
                )
            elif self.is_cot:
                # Expecting <answer> tags
                try:
                    pred_label_1_msg = entry["predicted_label_1"]
                    if type(pred_label_1_msg) == list:
                        pred_label_1_msg = pred_label_1_msg[1]["content"].strip()
                    else:
                        pred_label_1_msg = pred_label_1_msg.strip()

                    extracted = self.extract_answer_from_text(pred_label_1_msg)

                    if extracted is not None:
                        pred_label_1_final = extracted.strip()
                    else:
                        pred_label_1_final = pred_label_1_msg[0] if pred_label_1_msg else "NEI"
                except (IndexError, KeyError, TypeError):
                    pred_label_1_final = "NEI"

                if pred_label_1_final not in self.valid_options:
                    pred_label_1_final = "NEI"

                entry["predicted_label_1"] = pred_label_1_final


                # Process Task 2 predictions
                try:
                    pred_label_2_msg = entry["predicted_label_2"]
                    if isinstance(pred_label_2_msg, list):
                        pred_label_2_msg = pred_label_2_msg["content"].strip()
                    else:
                        pred_label_2_msg = pred_label_2_msg.strip()

                    # Try to extract answer from <answer> tags
                    extracted = self.extract_answer_from_text(pred_label_2_msg)

                    if extracted is not None:
                        pred_label_2_final = extracted
                    else:
                        pred_label_2_final = pred_label_2_msg[0] if pred_label_2_msg else "NEI"
                except (IndexError, KeyError, TypeError):
                    pred_label_2_final = "NEI"

                if pred_label_2_final not in self.valid_options:
                    pred_label_2_final = "NEI"

                entry["predicted_label_2"] = pred_label_2_final

            elif self.is_verbalize:
                # Task 1 - single answer response
                entry["predicted_label_1"] = (
                    entry["predicted_label_1"].strip()[0]
                    if entry["predicted_label_1"].strip() and entry["predicted_label_1"].strip()[0] in self.valid_options
                    else "NEI"
                )
                # Process Task 2 predictions
                try:
                    pred_label_2_msg = entry["predicted_label_2"]
                    if isinstance(pred_label_2_msg, list):
                        pred_label_2_msg = pred_label_2_msg["content"].strip()
                    else:
                        pred_label_2_msg = pred_label_2_msg.strip()

                    # Try to extract answer from <answer> tags
                    extracted = self.extract_answer_from_text(pred_label_2_msg)

                    if extracted is not None:
                        pred_label_2_final = extracted
                    else:
                        pred_label_2_final = pred_label_2_msg[0] if pred_label_2_msg else "NEI"
                except (IndexError, KeyError, TypeError):
                    pred_label_2_final = "NEI"

                if pred_label_2_final not in self.valid_options:
                    pred_label_2_final = "NEI"

                entry["predicted_label_2"] = pred_label_2_final


            else:
                # Expecting single answer response
                entry["predicted_label_1"] = (
                    entry["predicted_label_1"].strip()[0]
                    if entry["predicted_label_1"].strip() and entry["predicted_label_1"].strip()[0] in self.valid_options
                    else "NEI"
                )
                entry["predicted_label_2"] = (
                    entry["predicted_label_2"].strip()[0]
                    if entry["predicted_label_2"].strip() and entry["predicted_label_2"].strip()[0] in self.valid_options
                    else "NEI"
                )

    def count_nei_instances(self):
        nei_count_1 = sum(1 for entry in self.results if entry["predicted_label_1"] == "NEI")
        nei_count_2 = sum(1 for entry in self.results if entry["predicted_label_2"] == "NEI")
        return nei_count_1, nei_count_2

    def calculate_metrics(self):
        true_labels_1 = [entry["true_label_1"] for entry in self.results]
        predicted_labels_1 = [entry["predicted_label_1"] for entry in self.results]
        true_labels_2 = [entry["true_label_2"] for entry in self.results]
        predicted_labels_2 = [entry["predicted_label_2"] for entry in self.results]
        task_1_accuracy = accuracy_score(true_labels_1, predicted_labels_1)
        task_1_f1 = f1_score(true_labels_1, predicted_labels_1, average="weighted", zero_division=0)
        task_2_accuracy = accuracy_score(true_labels_2, predicted_labels_2)
        task_2_f1 = f1_score(true_labels_2, predicted_labels_2, average="weighted", zero_division=0)
        return task_1_accuracy, task_1_f1, task_2_accuracy, task_2_f1

    def save_metrics(self, metrics, nei_counts):
        task_1_accuracy, task_1_f1, task_2_accuracy, task_2_f1 = metrics
        nei_count_1, nei_count_2 = nei_counts
        with open(self.evaluation_output, "w") as f:
            f.write(f"Evaluation Results for Model: {self.model_name}\n")
            f.write(f"Task 1 Accuracy: {task_1_accuracy:.4f}\n")
            f.write(f"Task 1 F1 Score: {task_1_f1:.4f}\n")
            f.write(f"Task 1 NEI Count: {nei_count_1}\n")
            f.write(f"Task 2 Accuracy: {task_2_accuracy:.4f}\n")
            f.write(f"Task 2 F1 Score: {task_2_f1:.4f}\n")
            f.write(f"Task 2 NEI Count: {nei_count_2}\n")
        print(f"Evaluation metrics saved to '{self.evaluation_output}'")

    def evaluate(self):
        self.load_results()
        self.post_process_predictions()
        nei_counts = self.count_nei_instances()
        metrics = self.calculate_metrics()
        self.save_metrics(metrics, nei_counts)