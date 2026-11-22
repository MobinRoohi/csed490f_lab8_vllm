import argparse
import gc
import json
import time

import torch
from vllm import LLM, SamplingParams


# ============================================
# Experiments Configure
# ============================================
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
# MODEL_NAME = "NousResearch/Meta-Llama-3-8B"            # Llama-3 open model
DRAFT_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
PROMPT = "Explain the concept of attention mechanism in transformers."
TEMPERATURE = 0.7
MAX_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.8
DEFAULT_BATCH_SIZES = [8]

SAMPLING_PARAMS = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)


def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_experiment(
    model_name: str,
    use_paged_attention: bool,
    batch_sizes,
    use_speculative_decoding: bool,
    max_model_len=None,
):
    """Load one vLLM engine and benchmark every batch size for this setting."""
    # This is the exact simulation described in Lab 8, page 12.
    block_size = 16 if use_paged_attention else 2048

    llm_kwargs = {
        "model": model_name,
        "dtype": torch.float16,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "block_size": block_size,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    if use_speculative_decoding:
        # Lab 8, page 16: TinyLlama draft model proposes 5 tokens.
        llm_kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": 5,
        }

    paged_label = "ON" if use_paged_attention else "OFF (simulated)"
    spec_label = "ON" if use_speculative_decoding else "OFF"
    print(
        f"\nLoading model={model_name}, PagedAttention={paged_label}, "
        f"speculative_decoding={spec_label}, block_size={block_size}"
    )

    load_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_time = time.perf_counter() - load_start
    print(f"Model/engine load time: {load_time:.2f}s")

    results = []
    try:
        # Exclude one-time engine/CUDA initialisation from measured timings.
        _ = llm.generate(
            [PROMPT],
            SamplingParams(temperature=TEMPERATURE, max_tokens=8),
        )

        for batch_size in batch_sizes:
            if batch_size <= 0:
                raise ValueError("Batch sizes must be positive integers.")

            prompts = [PROMPT] * batch_size
            synchronize_gpu()
            start = time.perf_counter()
            outputs = llm.generate(prompts, SAMPLING_PARAMS)
            synchronize_gpu()
            inference_time = time.perf_counter() - start

            generated_tokens = sum(
                len(request.outputs[0].token_ids)
                for request in outputs
                if request.outputs
            )
            throughput = generated_tokens / inference_time if inference_time > 0 else 0.0

            result = {
                "model": model_name,
                "paged_attention": use_paged_attention,
                "paged_attention_mode": "on" if use_paged_attention else "off_simulated",
                "block_size": block_size,
                "speculative_decoding": use_speculative_decoding,
                "batch_size": batch_size,
                "model_load_seconds": load_time,
                "inference_seconds": inference_time,
                "generated_tokens": generated_tokens,
                "throughput_tokens_per_second": throughput,
            }
            results.append(result)

            print(
                f"[paged={paged_label} spec={spec_label} bs={batch_size}] "
                f"time={inference_time:.2f}s, "
                f"generated={generated_tokens} tokens, "
                f"throughput={throughput:.1f} tok/s"
            )

    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PagedAttention settings and speculative decoding in vLLM."
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Target model (default: Llama-2-7B from the Lab instructions).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark (default: 8).",
    )
    parser.add_argument(
        "--paged",
        choices=["on", "off", "both"],
        default="both",
        help="Compare block_size=16, block_size=2048, or both (default: both).",
    )
    parser.add_argument(
        "--spec",
        choices=["off", "on", "both"],
        default="off",
        help="Disable, enable, or compare speculative decoding (default: off).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional context limit; the Lab suggests 2048 only if a max-length error occurs.",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="JSON file in which to save the measured results (default: results.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled GPU is required for this Lab 8 experiment.")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("All batch sizes must be positive integers.")
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("--max-model-len must be a positive integer.")

    paged_modes = {
        "on": [True],
        "off": [False],
        "both": [True, False],
    }[args.paged]
    speculative_modes = {
        "off": [False],
        "on": [True],
        "both": [False, True],
    }[args.spec]

    results = []
    for use_speculative_decoding in speculative_modes:
        for use_paged_attention in paged_modes:
            results.extend(
                run_experiment(
                    model_name=args.model,
                    use_paged_attention=use_paged_attention,
                    batch_sizes=args.batch_sizes,
                    use_speculative_decoding=use_speculative_decoding,
                    max_model_len=args.max_model_len,
                )
            )

    print("\n=== Summary ===")
    print(
        f"{'PagedAttn':<19}{'Spec':<8}{'Batch':<8}"
        f"{'Latency(s)':<13}{'Generated':<13}{'Throughput(tok/s)':<20}"
    )
    for result in results:
        paged_label = "ON" if result["paged_attention"] else "OFF*"
        spec_label = "ON" if result["speculative_decoding"] else "OFF"
        print(
            f"{paged_label:<19}"
            f"{spec_label:<8}"
            f"{result['batch_size']:<8}"
            f"{result['inference_seconds']:<13.2f}"
            f"{result['generated_tokens']:<13}"
            f"{result['throughput_tokens_per_second']:<20.1f}"
        )

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    print(f"\nSaved results to {args.output}")
    print("* OFF is a block_size=2048 simulation, not a true PagedAttention disable switch.")


if __name__ == "__main__":
    main()