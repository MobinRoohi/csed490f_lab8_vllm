import time
import argparse
import gc
import torch
# ============================================
import vllm
# ============================================
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================
# Experiments Configure
# ============================================
MODEL_NAME = "NousResearch/Llama-2-7b-hf"  # Llama-2 open model
PROMPTS = ["What is the difference between supervised and unsupervised learning?",
            "Explain how gradient descent works in simple terms.",
            "What does overfitting mean in machine learning?",
            "Can you describe how convolutional neural networks process images?",
            "What is the role of activation functions in neural networks?",
            "How do word embeddings represent meaning in text?",
            "Explain what reinforcement learning is with an example.",
            "What does fine-tuning mean for large language models?",
            "How does tokenization work in natural language processing?",
            "Explain the difference between precision and recall.",
            "What is the purpose of the softmax function in classification tasks?",
            "How do transformers handle long sequences compared to RNNs?",
            "What is the idea behind self-supervised learning?",
            "Explain what a loss function does in training a model.",
            "What are the benefits of using quantization for large models?",
            "How does attention help models focus on important parts of input data?"]
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.8
DEVICE = "cuda"
DEFAULT_BATCH_SIZES = [16]

VLLM_SAMPLING_PARAMS = vllm.SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_NEW_TOKENS,
)


def synchronize_gpu():
    """Wait for queued CUDA work so wall-clock timings are meaningful."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_batch_prompts(batch_size: int):
    """Return the same deterministic prompt batch for both backends."""
    if batch_size <= 0:
        raise ValueError("Batch sizes must be positive integers.")
    return [PROMPTS[i % len(PROMPTS)] for i in range(batch_size)]


def count_transformers_generated_tokens(output_ids, prompt_width, eos_token_id):
    """Count generated tokens only, including EOS but excluding batch padding."""
    if eos_token_id is None:
        eos_ids = set()
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = set(eos_token_id)
    else:
        eos_ids = {eos_token_id}

    total = 0
    for sequence in output_ids:
        generated_ids = sequence[prompt_width:].tolist()
        for token_id in generated_ids:
            total += 1
            if token_id in eos_ids:
                break
    return total


def run_transformers_inference(batch_sizes):
    """Load the HF model once and benchmark all requested batch sizes."""
    print("\nLoading Hugging Face Transformers model...")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Decoder-only models should use left padding for batched generation.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    load_time = time.perf_counter() - load_start
    print(f"Transformers model load time: {load_time:.2f}s")

    results = []
    try:
        # Warm-up is deliberately excluded from the measured inference times.
        warmup_inputs = tokenizer([PROMPTS[0]], return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            _ = model.generate(
                **warmup_inputs,
                max_new_tokens=8,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.pad_token_id,
            )
        synchronize_gpu()
        del warmup_inputs

        for batch_size in batch_sizes:
            prompts = make_batch_prompts(batch_size)
            synchronize_gpu()
            start = time.perf_counter()

            # Tokenization is included in the timed request path. vLLM also
            # tokenizes its input prompts inside llm.generate().
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(DEVICE)
            prompt_width = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.pad_token_id,
                )

            synchronize_gpu()
            inference_time = time.perf_counter() - start

            generated_tokens = count_transformers_generated_tokens(
                output_ids,
                prompt_width,
                tokenizer.eos_token_id,
            )
            throughput = generated_tokens / inference_time if inference_time > 0 else 0.0
            sample_text = tokenizer.decode(
                output_ids[0, prompt_width:],
                skip_special_tokens=True,
            )

            result = {
                "backend": "Transformers",
                "batch_size": batch_size,
                "model_load_seconds": load_time,
                "inference_seconds": inference_time,
                "generated_tokens": generated_tokens,
                "throughput_tokens_per_second": throughput,
            }
            results.append(result)

            print(
                f"[Transformers bs={batch_size}] "
                f"time={inference_time:.2f}s "
                f"generated={generated_tokens} tokens "
                f"throughput={throughput:.1f} tok/s"
            )
            print("Sample output:", sample_text[:150].replace("\n", " "), "...")

            del inputs, output_ids

    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_vllm_inference(batch_sizes):
    """Load the vLLM engine once and benchmark all requested batch sizes."""
    print("\nLoading vLLM engine...")
    load_start = time.perf_counter()
    llm = vllm.LLM(
        model=MODEL_NAME,
        dtype=torch.float16,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    load_time = time.perf_counter() - load_start
    print(f"vLLM model load time: {load_time:.2f}s")

    results = []
    try:
        # Warm-up is excluded from the measured inference times.
        _ = llm.generate(
            [PROMPTS[0]],
            vllm.SamplingParams(temperature=TEMPERATURE, max_tokens=8),
        )

        for batch_size in batch_sizes:
            prompts = make_batch_prompts(batch_size)
            start = time.perf_counter()
            outputs = llm.generate(prompts, VLLM_SAMPLING_PARAMS)
            inference_time = time.perf_counter() - start

            generated_tokens = sum(
                len(request.outputs[0].token_ids)
                for request in outputs
                if request.outputs
            )
            throughput = generated_tokens / inference_time if inference_time > 0 else 0.0

            result = {
                "backend": "vLLM",
                "batch_size": batch_size,
                "model_load_seconds": load_time,
                "inference_seconds": inference_time,
                "generated_tokens": generated_tokens,
                "throughput_tokens_per_second": throughput,
            }
            results.append(result)

            print(
                f"[vLLM bs={batch_size}] "
                f"time={inference_time:.2f}s "
                f"generated={generated_tokens} tokens "
                f"throughput={throughput:.1f} tok/s"
            )
            print("Sample output:", outputs[0].outputs[0].text[:150].replace("\n", " "), "...")

    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare Hugging Face Transformers and vLLM inference performance."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark (default: 16).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled GPU is required for this Lab 8 experiment.")

    for batch_size in args.batch_sizes:
        if batch_size <= 0:
            parser.error("All batch sizes must be positive integers.")

    transformers_results = run_transformers_inference(args.batch_sizes)
    # The HF model is deleted and the CUDA cache is cleared before vLLM loads.
    vllm_results = run_vllm_inference(args.batch_sizes)

    print("\n=== Transformers vs. vLLM Summary ===")
    print(
        f"{'Backend':<16}{'Batch':<8}{'Latency(s)':<14}"
        f"{'Generated tokens':<20}{'Throughput(tok/s)':<20}"
    )
    for result in transformers_results + vllm_results:
        print(
            f"{result['backend']:<16}"
            f"{result['batch_size']:<8}"
            f"{result['inference_seconds']:<14.2f}"
            f"{result['generated_tokens']:<20}"
            f"{result['throughput_tokens_per_second']:<20.1f}"
        )


if __name__ == "__main__":
    main()