# Harmonious Minds

**Benchmarking Intertwined Reasoning of Human Personality and Musical Preference**

Code and data for the Findings of IJCNLP-AACL 2025 paper *Harmonious Minds: Benchmarking
Intertwined Reasoning of Human Personality and Musical Preference*.

This repository releases the **Personality-Music Alignment Bench (PMAB)**, a benchmark that
tests whether large language models can reason across two semantically distant domains:
inferring a person's personality traits from a narrative context, and then selecting the chord
progression that matches the musical taste implied by those traits.

Each benchmark item poses two multiple-choice questions over the same context:

1. **Task 1 — Personality Trait Identification.** Which descriptor best characterises the
   individual? Distractors are the descriptors *least* similar to the context in embedding
   space, drawn from other PF16 factors.
2. **Task 2 — Chord Progression Matching.** Which chord progression matches the individual's
   musical preference? Distractors are progressions sampled from unaligned genres.

## Benchmark statistics

| Attribute | Value |
| --- | --- |
| Context–question pairs | 2,880 |
| Personality traits (Cattell PF16) | 16 (180 contexts each) |
| Musical genres | 19 |
| Domains / subdomains | 10 / 3 per domain |
| Difficulty tier released | Easy |

## Repository layout

```
data/
  bench.json                    # the released PMAB benchmark (2,880 items)
src/
  benchmark/
    context_gen.py              # Stage 1: GPT-4o synthetic context generation
    create_bench.py             # Stage 2: turn contexts into the two-task benchmark
  baselines/
    baseline.py                 # entry point: run predictions + evaluation
    benchmark_evaluator.py      # prompting / inference for API, local and ChatMusician models
    scores.py                   # answer extraction, Accuracy / F1 / NEI metrics
```

### Benchmark schema

Each entry in `data/bench.json` is a flat object:

| Field | Description |
| --- | --- |
| `index`, `id` | item identifiers |
| `context` | the synthetic narrative scenario |
| `question_1`, `label_1`, `label_text_1` | Task 1 question, option letters, option texts |
| `ans_text_1`, `answer_1` | Task 1 gold option text and letter |
| `question_2`, `label_2`, `label_text_2` | Task 2 question, option letters, option texts |
| `ans_text_2`, `answer_2` | Task 2 gold option text and letter |
| `domain`, `subdomain` | topical grounding of the context |
| `genre`, `pf16_type` | aligned musical genre and PF16 trait |
| `Difficulty` | difficulty tier |

## Setup

```bash
git clone https://github.com/<your-org>/Harmonious-Minds.git
cd Harmonious-Minds

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

API credentials are read from the environment — never commit them:

```bash
export OPENAI_API_KEY="..."      # for GPT-4o generation and API-based evaluation
export GROQ_API_KEY="..."        # only if evaluating Groq-hosted models
export HF_HOME="/path/to/hf_cache"   # optional: shared Hugging Face cache
```

## Usage

### Evaluate a model on the benchmark

```bash
python src/baselines/baseline.py \
  --model_type api \
  --model_name gpt-4o \
  --benchmark_path data/bench.json \
  --predictions_output outputs/gpt4o_preds.json \
  --evaluation_output outputs/gpt4o_eval.txt
```

`--model_type` selects the inference backend:

- `api` — OpenAI or Groq hosted models (`--model_name gpt-4o`, `groq-llama-3.1-8b-instant`, …)
- `local` — any Hugging Face causal LM via `--model_path`, optionally with a LoRA
  adapter via `--adapter_path`
- `chatmusician` — the ChatMusician checkpoint, which uses its own prompt template

Prompting strategy flags (mutually exclusive in practice):

| Flag | Effect |
| --- | --- |
| *(none)* | zero-shot baseline |
| `--is_cot True` | chain-of-thought prompting, answer read from `<answer>` tags |
| `--is_verbalize True` | two-step verbalization pipeline (trait → genre, then genre + music theory → chord) |

Verbalization additionally needs the trait-to-genre and genre-to-music-theory maps. They
default to `data/pf16_map.json` and `data/music_theory.json`, and can be pointed elsewhere:

```bash
python src/baselines/baseline.py \
  --model_type local \
  --model_name gemma2-9b-it \
  --model_path google/gemma-2-9b-it \
  --benchmark_path data/bench.json \
  --predictions_output outputs/gemma2_verb_preds.json \
  --evaluation_output outputs/gemma2_verb_eval.txt \
  --is_verbalize True \
  --pf16_map data/pf16_map.json \
  --music_theory data/music_theory.json
```

Results are written as Accuracy, weighted F1, and NEI (*not enough information* — responses
from which no valid option letter could be extracted) for both tasks.

### Regenerate the benchmark

Stage 1 generates synthetic contexts for one PF16 trait at a time:

```bash
python src/benchmark/context_gen.py \
  --pf16_key Warmth \
  --pf16_types data/pf16_types.json \
  --domains data/domain.json
```

Stage 2 assembles the contexts into the two-task benchmark, using
`all-mpnet-base-v2` embeddings to mine the Task 1 distractors:

```bash
python src/benchmark/create_bench.py \
  --contexts Warmth_contexts.json \
  --pf16_types data/pf16_types.json \
  --chord_map data/genre_chord_map.json \
  --output data/bench.json
```

Generation is stochastic, so regenerating will not reproduce `data/bench.json` byte for byte.
Use the released file for comparable numbers.

## Prompts

The prompts in this repository are the ones documented in the paper's appendix:

- **Context generation** (`src/benchmark/context_gen.py`) — Appendix, *Context Generation Prompt*
- **Verbalization step 1, personality → genre** (`benchmark_evaluator.py`) — Appendix,
  *Step 1: Personality-to-Genre Prompt*
- **Verbalization step 2, genre → chord** (`benchmark_evaluator.py`) — Appendix,
  *Step 2: Genre-to-Chord Prompt*

## Citation

```bibtex
@inproceedings{harmonious-minds-2025,
  title     = {Harmonious Minds: Benchmarking Intertwined Reasoning of Human Personality and Musical Preference},
  booktitle = {Findings of the Association for Computational Linguistics: IJCNLP-AACL 2025},
  year      = {2025}
}
```

Chord progressions are sourced from the
[Chordonomicon](https://huggingface.co/datasets/ailsntua/Chordonomicon) dataset; personality
traits follow Cattell's 16 Personality Factor model.

## License

Released for research use. See `LICENSE` if present, otherwise contact the authors.
