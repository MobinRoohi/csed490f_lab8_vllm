# Lab 8 - Using vLLM (PagedAttention)

## 1. Introduction

The purpose of this lab is to understand the use of vLLM as an inference and serving engine for large language models, and to examine the effect of its PagedAttention-related memory configuration. The experiments have two parts. First, Hugging Face Transformers and vLLM are compared in terms of inference latency and generated-token throughput. Second, different vLLM configurations are evaluated by changing the KV-cache block size, batch size, target model, and speculative decoding setting.

## 2. Implementation

### 2.1 Transformers versus vLLM

The first experiment uses the `NousResearch/Llama-2-7b-hf` model with a list of 16 prompts for both inference backends. The initial batch size is 16, and additional batch sizes can be supplied through the command line. Both implementations use a sampling temperature of 0.7 and generate at most 256 new tokens per request.

For the transformers implementation, the tokenizer prepares a padded batch and the model generates tokens using `do_sample=True`. Left padding is used for batched generation with this decoder-only model. For the vLLM implementation, the prompts are passed to `LLM.generate()` with equivalent sampling parameters. Both models are loaded before the measured generation loop, and a warm-up generation is performed so that one-time initialization is excluded from the reported inference time.

The number of generated tokens excludes the input prompt tokens. For Transformers, generated tokens are counted after the padded input width and any padding after an end-of-sequence token is excluded. For vLLM, the number of generated tokens is obtained from each request's output token IDs.

### 2.2 PagedAttention block-size comparison

The lab instructions state that vLLM does not provide a switch that truly disables PagedAttention. Therefore, the experiment follows the lab's approximation: `block_size=16` represents the normal PagedAttention configuration, while `block_size=2048` simulates a configuration with much larger blocks and less efficient memory allocation. The latter must not be described as a genuine implementation with PagedAttention removed.

The default Step 2-1 experiment uses `NousResearch/Llama-2-7b-hf` with a batch size of 8. The model is loaded separately for each block-size configuration. A warm-up request is executed before measuring the batch inference time.

### 2.3 Batch-size experiment

To study the effect of the batch size, the same target model and prompt are tested with smaller and larger batches. The tested batch sizes are `[INSERT THE BATCH SIZES ACTUALLY RUN, e.g. 1, 4, 8, 32, 64]`. For each batch size, both block-size configurations are measured. Large batches may increase GPU memory pressure and may lead to slower execution, so the observed result should be interpreted together with the hardware and runtime conditions.

### 2.4 Model comparison: Llama 2 and Llama 3

The default target model is `NousResearch/Llama-2-7b-hf`. For Step 2-3, it is replaced with `NousResearch/Meta-Llama-3-8B`, while keeping the default batch size of 8 and testing both block-size configurations. The lab material identifies the attention architecture of Llama 2-7B as Multi-Head Attention (MHA) and that of Llama 3-8B as Grouped Query Attention (GQA). This architectural difference is relevant when interpreting KV-cache memory use and the measured impact of the block-size configuration.

### 2.5 Speculative decoding

For Step 2-4, the target model is Llama 2-7B and PagedAttention is kept ON (`block_size=16`). The experiment compares speculative decoding OFF and ON for batch sizes 1, 8, and 32. When speculative decoding is enabled, the draft model is `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`, and `num_speculative_tokens` is set to 5, following the lab instructions.

The draft model proposes candidate tokens, which are then verified by the target model. The expected benefit is a reduction in target-model decoding work when the proposed tokens are accepted. However, the draft model introduces additional computation and memory use, so the measured benefit can depend on the batch size and the acceptance of its proposed tokens.

## 3. Metrics

For every run, inference latency is the wall-clock time spent in the measured batch-generation call. It is therefore the latency of the complete batch, not the average latency of one individual request. Throughput is computed as:

\[
\text{Throughput (tokens/s)} = \frac{\text{Total number of generated tokens}}{\text{Inference time (s)}}.
\]

Model loading time is printed separately and is not included in inference latency. The warm-up request is also excluded from the measured inference time.

## 4. Results

### 4.1 Transformers versus vLLM

| Backend | Batch size | Inference latency (s) | Generated tokens | Throughput (tokens/s) |
|---|---:|---:|---:|---:|
| Transformers | 16 | [INSERT] | [INSERT] | [INSERT] |
| vLLM | 16 | [INSERT] | [INSERT] | [INSERT] |

**Analysis.** In our experiment, [DESCRIBE WHICH BACKEND HAD LOWER BATCH LATENCY AND WHICH HAD HIGHER TOKEN THROUGHPUT]. The difference can be discussed in relation to the execution and batching strategies of the two inference frameworks. The conclusion should be based on the measured values from the same model, prompt batch, and generation settings.

### 4.2 PagedAttention block-size comparison (Llama 2-7B, batch size 8)

| Configuration | Block size | Inference latency (s) | Generated tokens | Throughput (tokens/s) |
|---|---:|---:|---:|---:|
| PagedAttention ON | 16 | [INSERT] | [INSERT] | [INSERT] |
| PagedAttention OFF (simulated) | 2048 | [INSERT] | [INSERT] | [INSERT] |

**Analysis.** The measured difference between block sizes 16 and 2048 was [INSERT OBSERVATION]. PagedAttention divides the KV cache into blocks that can be allocated as needed. The lab explains that conventional serving approaches can suffer from internal fragmentation, reservation, and external fragmentation. The large-block experiment is intended to illustrate the memory-allocation consequences of using much larger blocks. Since block size 2048 is only the lab's simulation, this result is not a direct comparison against a truly non-PagedAttention implementation.

### 4.3 Effect of batch size

| Batch size | Block size 16 latency (s) | Block size 16 throughput (tokens/s) | Block size 2048 latency (s) | Block size 2048 throughput (tokens/s) |
|---:|---:|---:|---:|---:|
| 1 | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| 4 | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| 8 | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| 32 | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| 64 | [INSERT] | [INSERT] | [INSERT] | [INSERT] |

**Analysis.** As the batch size increased, [DESCRIBE THE OBSERVED LATENCY AND THROUGHPUT TRENDS]. A larger batch can improve hardware utilization, but it also increases the number of concurrent sequences and the memory required for their KV caches. Explain whether the block-size difference became more or less visible at larger batch sizes. Mention any run that was unusually slow, failed, or triggered memory offloading rather than treating it as a normal measurement.

### 4.4 Llama 2-7B versus Llama 3-8B (batch size 8)

| Model | Attention type (lab hint) | Block size | Inference latency (s) | Throughput (tokens/s) |
|---|---|---:|---:|---:|
| Llama 2-7B | MHA | 16 | [INSERT] | [INSERT] |
| Llama 2-7B | MHA | 2048 | [INSERT] | [INSERT] |
| Llama 3-8B | GQA | 16 | [INSERT] | [INSERT] |
| Llama 3-8B | GQA | 2048 | [INSERT] | [INSERT] |

**Analysis.** Compared with Llama 2-7B, Llama 3-8B produced [INSERT OBSERVATION]. The lab notes that Llama 2-7B uses MHA, whereas Llama 3-8B uses GQA. This difference should be considered when discussing the KV-cache memory footprint and the observed sensitivity to block size. Avoid attributing every performance difference to the attention architecture alone, because the models also differ in size and other model characteristics.

### 4.5 Speculative decoding (Llama 2-7B, PagedAttention ON)

| Batch size | Speculative decoding OFF latency (s) | OFF throughput (tokens/s) | Speculative decoding ON latency (s) | ON throughput (tokens/s) |
|---:|---:|---:|---:|---:|
| 1 | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| 8 | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| 32 | [INSERT] | [INSERT] | [INSERT] | [INSERT] |

**Analysis.** Speculative decoding [IMPROVED / DID NOT IMPROVE] throughput at batch size [INSERT], while at batch size [INSERT] it [INSERT]. This can be explained by the balance between the time spent generating candidates with the draft model and the time saved when the target model verifies and accepts multiple candidate tokens. State explicitly at which tested batch sizes the method helped and where its extra overhead outweighed its benefit.

## 5. Conclusion

The experiments show that [SUMMARIZE THE MAIN OBSERVATIONS FROM STEP 1]. For the PagedAttention-related experiment, [SUMMARIZE THE EFFECT OF BLOCK SIZE AND BATCH SIZE]. The Llama 2/Llama 3 comparison suggests that [SUMMARIZE THE OBSERVATION, TAKING MHA/GQA INTO ACCOUNT]. Finally, speculative decoding [SUMMARIZE THE BATCH SIZES WHERE IT HELPED OR DID NOT HELP].

Overall, the measurements indicate that inference performance depends not only on the inference framework, but also on KV-cache allocation settings, batch size, model architecture, and the overhead introduced by speculative decoding.

## 6. Code changes used to obtain the results

- **`vllm_vs_transformers.py`:** implemented the missing Hugging Face imports and model/tokenizer loading; used the same prompt batch and sampling settings for both backends; applied the requested batch size; enabled sampling in Transformers with `do_sample=True`; used `max_new_tokens=256` to match vLLM's `max_tokens=256`; counted generated tokens without including prompt tokens; added warm-up and CUDA synchronization around timing; and released the Transformers model before loading vLLM.
- **`paged_attention_comparison.py`:** made model name and batch sizes configurable; tested block sizes 16 and 2048 as specified by the lab; added the TinyLlama speculative-decoding configuration with five speculative tokens; allowed speculative decoding OFF, ON, or both; allowed an optional `max_model_len` parameter for the lab's max-length hint; added warm-up, timing, throughput calculation, and JSON export.

## 7. Reproducibility commands

```bash
# Step 1: Transformers versus vLLM (default batch size 16)
python vllm_vs_transformers.py

# Step 1: optionally compare additional batch sizes
python vllm_vs_transformers.py --batch-sizes 8 16

# Step 2-1: Llama 2-7B, batch size 8, both block-size settings
python paged_attention_comparison.py

# Step 2-2: smaller and larger batch sizes
python paged_attention_comparison.py --batch-sizes 1 4 8 32 64 --paged both

# Step 2-3: Llama 3-8B, batch size 8, both block-size settings
python paged_attention_comparison.py --model NousResearch/Meta-Llama-3-8B --batch-sizes 8 --paged both

# Step 2-4: compare speculative decoding OFF and ON, with PagedAttention ON
python paged_attention_comparison.py --model NousResearch/Llama-2-7b-hf --paged on --spec both --batch-sizes 1 8 32

# If the max-length error mentioned in the lab occurs, retry with:
python paged_attention_comparison.py --paged on --spec both --batch-sizes 1 8 32 --max-model-len 2048
```

The generated `results.json` file records the measured settings and metrics for the vLLM experiments. Record the Transformers results from the console output in the table above.
