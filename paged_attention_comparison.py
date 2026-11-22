import argparse
import time
import torch
from vllm import LLM, SamplingParams

# ============================================
# Experiments Configure
# ============================================
MODEL_NAME = "NousResearch/Llama-2-7b-hf"       # Llama-2 open model
# MODEL_NAME = "NousResearch/Meta-Llama-3-8B"            # Llama-3 open model

PROMPT = "Explain the concept of attention mechanism in transformers."

SAMPLING_PARAMS = SamplingParams(temperature=0.7, max_tokens=256)
GPU_MEMORY_UTILIZATION = 0.8

SPEC_CONFIG = {
    "method": "draft_model",
    "model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "num_speculative_tokens": 5,
}

def run_inference(use_paged_attention: bool, batch_size: int, use_spec: bool = False, model_name: str = MODEL_NAME):
    """
        PagedAttention ON: block size 16
        PagedAttention OFF simulated : block size 2048
    """

    block_size = 16 if use_paged_attention else 2048

    # Model load
    start = time.time()
    llm_kwargs = dict(model=model_name, dtype=torch.float16,
                      gpu_memory_utilization=GPU_MEMORY_UTILIZATION, block_size=block_size)
    if use_spec:
        llm_kwargs["speculative_config"] = SPEC_CONFIG
        # llm_kwargs["max_model_len"] = 2048
    llm = LLM(**llm_kwargs)
    load_time = time.time() - start
    print(f"Model load time: {load_time:.2f}s")

    # offline batched inference with multiple prompts
    prompts = [PROMPT] * batch_size

    start = time.time()
    outputs = llm.generate(prompts, SAMPLING_PARAMS)
    infer_time = time.time() - start

    # throughput calculation
    tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = tokens_generated / infer_time

    print(f"Inference time: {infer_time:.2f}s")
    print(f"Throughput: {throughput:.1f} tokens/s")

    print("\nSample output:")
    print(outputs[0].outputs[0].text[:150], "...")
    return infer_time, throughput


if __name__ == '__main__':
    # ============================================
    # Run
    # ============================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--spec", action="store_true", help="Step 2-4: speculative decoding experiment")
    args = parser.parse_args()

    if args.spec:
        spec_results = []
        for batch_size in args.batch_sizes or [1, 8, 32]:
            for spec in [False, True]:
                res = run_inference(use_paged_attention=True, batch_size=batch_size,
                                    use_spec=spec, model_name=args.model)
                spec_results.append((spec, batch_size, *res))

        print("\n=== Speculative Decoding Summary ===")
        print(f"{'SpecDec':<12}{'Batch':<8}{'Latency(s)':<12}{'Throughput(tok/s)':<20}")
        for spec, bsz, lat, thpt in spec_results:
            print(f"{'ON' if spec else 'OFF':<12}{bsz:<8}{lat:<12.2f}{thpt:<20.1f}")
    else:
        results = []
        for batch_size in args.batch_sizes or [8]:
            for paged in [True, False]:
                res = run_inference(use_paged_attention=paged, batch_size=batch_size,
                                    model_name=args.model)
                results.append((paged, batch_size, *res))

        # ============================================
        # Results
        # ============================================
        print("\n=== Summary ===")
        print(f"{'PagedAttn':<12}{'Batch':<8}{'Latency(s)':<12}{'Throughput(tok/s)':<20}")
        for paged, bsz, lat, thpt in results:
            print(f"{'ON' if paged else 'OFF':<12}{bsz:<8}{lat:<12.2f}{thpt:<20.1f}")
