# vllm_lab
Lab materials for Deep Learning Implementation class

## Execution guide and report-completion checklist

This README is intended for the teammate who will run the experiments, collect the measurements, complete the report draft, and prepare the final PDF.

The work follows the supplied **Lab 8: Using vLLM (PagedAttention)** handout. The handout defines two main tasks:

1. Compare Hugging Face Transformers with vLLM.
2. Study PagedAttention-related block-size settings in vLLM, including batch-size changes, a second model, and speculative decoding.

The handout asks for measured latency and throughput results, an analysis of those results, and an explanation of which code was changed to obtain them. The report must be no more than **6 pages excluding the title page**, use **11 pt font**, and be submitted as `Lab8_team{team_number}.pdf`.

---

# 1. Project files

The project should contain these two Python scripts:

```text
vllm_lab/
├── vllm_vs_transformers.py
├── paged_attention_comparison.py
└── README.md
```

The report draft is a separate working document:

```text
Lab8_report_draft.md
```

- `vllm_vs_transformers.py`: implements Step 1, the Transformers versus vLLM comparison.
- `paged_attention_comparison.py`: implements Steps 2-1 through 2-4.
- `Lab8_report_draft.md`: contains the report structure, methodology, result tables, and placeholders to replace with actual measurements.

Do not submit the README instead of the report. It is an execution guide for the team.

# 2. Before running anything

## 2.1 Confirm the GPU instance

The lab handout recommends a Vast.ai instance with:

- 1 × NVIDIA RTX 4090
- 24 GB GPU memory
- 64 GB container storage

Check that the instance is running and that the GPU is visible:

```bash
nvidia-smi
```

The command should show the GPU model, memory usage, and running processes. If CUDA is not available in Python, do not start the benchmark.

The handout warns that GPU rental credits are limited. Stop or destroy the instance when it is no longer needed.

## 2.2 Connect to the instance

Follow the SSH connection procedure shown in the lab handout. The handout's environment uses `uv` and gives the following vLLM installation command:

```bash
uv pip install vllm --torch-backend=auto
```

The Step 1 script also imports Hugging Face Transformers. Make sure the environment has the `transformers` package installed. If it is missing, install it in the same environment:

```bash
uv pip install transformers
```

The scripts also use PyTorch. The vLLM installation command above is the installation method specified by the handout; do not independently replace the CUDA/PyTorch environment unless there is an installation problem that needs to be diagnosed.

Check the imports:

```bash
python -c "import torch, vllm, transformers; print('torch:', torch.__version__); print('vLLM:', vllm.__version__); print('transformers:', transformers.__version__); print('CUDA:', torch.cuda.is_available())"
```

If the import check fails, resolve the environment issue before launching the experiments.

## 2.3 Obtain the lab code

The handout gives this repository:

```bash
git clone https://github.com/xvyaward/vllm_lab
cd vllm_lab
```

Use the corrected versions of the two Python files prepared for this project. If the repository already contains files with those names, make sure the corrected files are the ones actually being executed.

Check the files:

```bash
ls
```

Optionally check Python syntax before spending GPU time:

```bash
python -m py_compile vllm_vs_transformers.py paged_attention_comparison.py
```

A successful command produces no output. This only checks syntax; it does not prove that the models can load or that the GPU has enough memory.

---

# 3. Important experimental rules

Apply these rules to every run and when writing the report.

## 3.1 Use the same comparison conditions

For a meaningful comparison, record the model, prompt batch, batch size, generation settings, GPU, and any relevant runtime configuration.

The Step 1 script uses:

- Target model: `NousResearch/Llama-2-7b-hf`
- 16 predefined prompts
- Default batch size: 16
- Temperature: 0.7
- Maximum generated tokens: 256
- FP16 model weights

Both frameworks receive the same prompt batch and use matching sampling settings.

## 3.2 Understand what “PagedAttention OFF” means in this lab

The handout explicitly says that vLLM does not provide a setting to truly disable PagedAttention. Instead, it asks students to simulate disabling it by changing the block size:

| Experiment label | `block_size` |
|---|---:|
| PagedAttention ON / normal configuration | 16 |
| PagedAttention OFF / simulated configuration | 2048 |

Therefore, in the report, call the second condition **“PagedAttention OFF (simulated, block size 2048)”** or **“large-block simulation.”**

Do not claim that the experiment compares vLLM's actual PagedAttention implementation against a separate non-PagedAttention implementation. It does not.

## 3.3 Understand the reported metrics

The scripts report:

**Inference latency (seconds)**

The wall-clock duration of the measured generation call for the entire batch. It is not the average latency of one request.

**Generated tokens**

The total number of newly generated tokens counted across the requests in the batch. Prompt tokens are excluded.

**Throughput (tokens/second)**

```text
throughput = total generated tokens / inference latency
```

Throughput is an aggregate batch throughput, not per-user throughput.

**Model/engine load time**

Printed separately. It is not included in inference latency.

A warm-up generation is performed before timing so that one-time initialization costs are not included in the measured request.

Because generation is sampled (`temperature=0.7`), output lengths can differ between runs. Record the generated-token count alongside latency and throughput; do not compare throughput numbers without considering how many tokens were generated.

---

# 4. Run Step 1 — Transformers versus vLLM

## 4.1 Objective

Compare Hugging Face Transformers and vLLM using the same Llama 2-7B model and prompts. The handout asks for a comparison of throughput and latency.

## 4.2 Default execution

From the directory containing the script, run:

```bash
python vllm_vs_transformers.py
```

The default batch size is 16.

The script performs these operations:

1. Loads the tokenizer and Transformers model.
2. Performs a short warm-up generation.
3. Measures Transformers inference for the requested batch sizes.
4. Releases the Transformers model and clears unused memory.
5. Loads the vLLM engine.
6. Performs a short warm-up generation.
7. Measures vLLM inference for the same batch sizes.
8. Prints a summary containing latency, generated-token count, and throughput.

The Transformers and vLLM models are not loaded simultaneously by design. This reduces GPU memory pressure.

## 4.3 Optional batch-size experiment

The handout's Step 1 does not prescribe a batch-size sweep, but the script supports one. For example:

```bash
python vllm_vs_transformers.py --batch-sizes 8 16
```

Only report these additional measurements if you actually run them. The default handout-oriented comparison is batch size 16.

## 4.4 Record the results

Copy the final summary from the terminal into a results table. Record at least:

| Backend | Batch size | Inference latency (s) | Generated tokens | Throughput (tokens/s) |
|---|---:|---:|---:|---:|
| Transformers | 16 | | | |
| vLLM | 16 | | | |

Also record the GPU model and any relevant software versions for the experimental setup section.

## 4.5 What to discuss

Answer these questions using your measurements:

- Which framework had the lower batch inference latency?
- Which framework achieved the higher aggregate token throughput?
- How large was the difference?
- Were the generated-token counts comparable?
- What features of vLLM's inference/serving design could explain the observed difference?

Do not state that vLLM is always faster based on one measurement. State what happened in this particular experiment and mention the hardware and settings.

---

# 5. Run Step 2-1 — Default PagedAttention block-size comparison

## 5.1 Objective

Use the default model and batch size from the handout:

- Model: `NousResearch/Llama-2-7b-hf`
- Batch size: 8
- Normal block size: 16
- Large-block simulation: 2048

## 5.2 Execution

```bash
python paged_attention_comparison.py
```

By default, the script tests both block-size configurations with speculative decoding disabled.

The script loads the model for one configuration, performs a warm-up, benchmarks the batch, releases the engine, and then proceeds to the next configuration.

## 5.3 Record the results

The script prints a summary and writes a JSON file named `results.json` by default.

Record:

| Configuration | Block size | Batch size | Inference latency (s) | Generated tokens | Throughput (tokens/s) |
|---|---:|---:|---:|---:|---:|
| PagedAttention ON | 16 | 8 | | | |
| PagedAttention OFF (simulated) | 2048 | 8 | | | |

Keep the JSON file. It contains the settings associated with each measurement and is useful when completing the report.

## 5.4 What to discuss

Explain the observed latency and throughput difference between the two block sizes.

Relate the discussion to the handout's explanation of KV-cache memory management:

- Internal fragmentation
- Reservation of unused capacity
- External fragmentation
- Allocation of KV-cache blocks

The handout explains that PagedAttention manages key/value memory in blocks. Use the diagram and explanation on pages 4–5 as the conceptual basis for this section.

Be precise: the result is a comparison between block-size settings, with block size 2048 used as the handout's simulation. It is not a true PagedAttention-off implementation.

---

# 6. Run Step 2-2 — Smaller and larger batch sizes

## 6.1 Objective

The handout asks students to test smaller and larger batch sizes and analyze how the results differ from Step 2-1. Its examples include 1, 4, 32, and 64.

## 6.2 Execution

Run:

```bash
python paged_attention_comparison.py --batch-sizes 1 4 8 32 64 --paged both
```

This tests all five batch sizes under both block-size configurations.

If the GPU runs out of memory, an experiment fails, or a configuration takes an unusually long time, record that fact. Do not silently omit the failed configuration or invent a result. You may retry with a smaller batch size, but label the retry clearly.

## 6.3 Record the results

Use a table such as:

| Batch size | Block 16 latency (s) | Block 16 throughput (tokens/s) | Block 2048 latency (s) | Block 2048 throughput (tokens/s) |
|---:|---:|---:|---:|---:|
| 1 | | | | |
| 4 | | | | |
| 8 | | | | |
| 32 | | | | |
| 64 | | | | |

## 6.4 What to discuss

For each block-size configuration, describe:

- How latency changes as batch size increases.
- How throughput changes as batch size increases.
- Whether larger batches improve aggregate throughput.
- Whether the difference between block sizes becomes more or less visible at larger batch sizes.
- Whether any run shows signs of memory pressure, slowdown, or failure.

Do not assume that latency must decrease with larger batches. Total batch latency and aggregate throughput measure different things: a larger batch can take longer overall while processing more tokens per second.

---

# 7. Run Step 2-3 — Change the model to Llama 3-8B

## 7.1 Objective

The handout asks students to replace Llama 2-7B with Llama 3-8B and, using the default batch size 8, compare the two block-size settings.

The handout's model identifier is:

```text
NousResearch/Meta-Llama-3-8B
```

## 7.2 Execution

```bash
python paged_attention_comparison.py \
  --model NousResearch/Meta-Llama-3-8B \
  --batch-sizes 8 \
  --paged both
```

If the model cannot be downloaded or accessed, record the exact error. Do not silently substitute a different model, because that would no longer be the experiment specified in the handout.

## 7.3 Record the results

Combine the Llama 2 and Llama 3 measurements:

| Model | Attention type (as stated in lab) | Block size | Batch size | Latency (s) | Throughput (tokens/s) |
|---|---|---:|---:|---:|---:|
| Llama 2-7B | MHA | 16 | 8 | | |
| Llama 2-7B | MHA | 2048 | 8 | | |
| Llama 3-8B | GQA | 16 | 8 | | |
| Llama 3-8B | GQA | 2048 | 8 | | |

## 7.4 What to discuss

The handout provides this hint:

- Llama 2-7B uses Multi-Head Attention (MHA).
- Llama 3-8B uses Grouped Query Attention (GQA).

Explain how this architectural difference is relevant to KV-cache memory requirements and interpret the measured results in that context.

Do not attribute every difference to MHA versus GQA alone. The models also differ in parameter count and other architectural details. State what the measurements show, then discuss GQA as one relevant factor.

---

# 8. Run Step 2-4 — Speculative decoding

## 8.1 Objective

The handout asks for a comparison with and without speculative decoding while keeping:

- Target model: Llama 2-7B
- PagedAttention: ON (`block_size=16`)
- Batch sizes: 1, 8, and 32
- Draft model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Number of speculative tokens: 5

## 8.2 Execution

Run:

```bash
python paged_attention_comparison.py \
  --model NousResearch/Llama-2-7b-hf \
  --paged on \
  --spec both \
  --batch-sizes 1 8 32
```

The `--spec both` option runs the experiment once with speculative decoding disabled and once enabled.

If the maximum-length error mentioned in the handout occurs, retry with:

```bash
python paged_attention_comparison.py \
  --model NousResearch/Llama-2-7b-hf \
  --paged on \
  --spec both \
  --batch-sizes 1 8 32 \
  --max-model-len 2048
```

Use `--max-model-len 2048` as the handout suggests for this error case; do not add it by default without a reason.

## 8.3 Record the results

| Batch size | Speculative decoding | Block size | Latency (s) | Generated tokens | Throughput (tokens/s) |
|---:|---|---:|---:|---:|---:|
| 1 | OFF | 16 | | | |
| 1 | ON | 16 | | | |
| 8 | OFF | 16 | | | |
| 8 | ON | 16 | | | |
| 32 | OFF | 16 | | | |
| 32 | ON | 16 | | | |

## 8.4 What to discuss

For each batch size, compare speculative decoding ON with OFF.

Explain:

1. The draft model proposes several candidate tokens.
2. The target model verifies the proposed tokens.
3. Accepted proposals can reduce the number of sequential target-model decoding steps.
4. The draft model also introduces computation and memory overhead.

Identify the batch sizes where speculative decoding improved throughput and where it did not. Base the conclusion on measured values, not on the expectation that speculative decoding must always help.

---

# 9. Complete the report draft

Open `Lab8_report_draft.md`. It already contains the proposed report sections and result tables.

Replace every placeholder such as `[INSERT]`, `[DESCRIBE]`, and `[SUMMARIZE]` with the actual results and interpretation.

## 9.1 Required report content

The final report should contain:

1. **Introduction** — briefly state the purpose of the lab.
2. **Experimental setup** — GPU, model names, precision, generation settings, and batch sizes.
3. **Transformers versus vLLM** — measurements and interpretation.
4. **PagedAttention block-size comparison** — default Llama 2-7B experiment.
5. **Batch-size experiment** — small and large batches.
6. **Model comparison** — Llama 2-7B versus Llama 3-8B, including the MHA/GQA hint.
7. **Speculative decoding** — ON/OFF comparison for batch sizes 1, 8, and 32.
8. **Conclusion** — summarize the main observations.
9. **Code changes** — identify which code was changed to produce the experiments.

## 9.2 How to write the analysis

For every table:

- State the most important numerical observation.
- Compare the relevant configurations directly.
- Explain a plausible reason using the concepts from the handout.
- Mention limitations or unexpected behavior.
- Avoid claims that are not supported by the measurements.

For example, do not write:

> PagedAttention always improves throughput.

Instead, write a result-specific statement:

> In our experiment with Llama 2-7B and batch size 8, the block-size-16 configuration achieved [X] tokens/s, compared with [Y] tokens/s for the block-size-2048 simulation. This corresponds to [describe the measured difference]. Since the second configuration is a large-block simulation rather than a true PagedAttention-off implementation, the result should be interpreted as a block-size comparison.

Do not invent measurements, hardware details, or successful runs. If an experiment could not be completed, state that limitation.

## 9.3 Check report length and submission format

Before exporting:

- Maximum 6 pages excluding the title page.
- Font size 11 pt.
- Save as `Lab8_team{team_number}.pdf`.
- Submit the PDF through PLMS, as specified by the handout.

---

# 10. Suggested run checklist

Use this checklist to track progress.

- [ ] Confirm the GPU instance and CUDA availability.
- [ ] Install/check vLLM, PyTorch, and Transformers.
- [ ] Replace the repository scripts with the corrected project scripts.
- [ ] Run the Python syntax check.
- [ ] Run Step 1 and save the terminal output.
- [ ] Run Step 2-1 and preserve `results.json`.
- [ ] Run Step 2-2 and preserve the results.
- [ ] Run Step 2-3 and preserve the results.
- [ ] Run Step 2-4 and preserve the results.
- [ ] Record the GPU model and software versions.
- [ ] Fill every report placeholder with actual measurements.
- [ ] Write the analysis for each experiment.
- [ ] Check the report's page count and font size.
- [ ] Export the final PDF using the required filename.
- [ ] Submit the PDF on PLMS.
- [ ] Stop/destroy the GPU instance when finished.

---

# 11. Code comparison: original structure versus refactored structure

This section explains what changed relative to the code supplied at the beginning of the project. The changes are mostly organizational and experimental-quality improvements. They do not add a new experiment beyond the lab objectives.

## 11.1 `vllm_vs_transformers.py`

### Original structure

The original script had:

- A fixed model name.
- A list of 16 prompts.
- Separate functions named `run_transformers_inference(batch_size)` and `run_vllm_inference(batch_size)`.
- A loop that used `[16]` as the batch-size list.
- A Transformers generation call using `max_length`.
- A throughput calculation based on the lengths of complete output sequences.

### Refactored structure

The current script:

- Adds `argparse` so batch sizes can be supplied from the terminal.
- Uses `make_batch_prompts()` to construct the same prompt batch for both frameworks.
- Uses `max_new_tokens=256` in Transformers to match vLLM's `max_tokens=256`.
- Enables sampling explicitly in Transformers with `do_sample=True`.
- Counts generated tokens separately from prompt tokens.
- Adds a warm-up generation before timing.
- Synchronizes CUDA around the Transformers timing measurement.
- Loads and releases the Transformers model before loading the vLLM engine.
- Prints generated-token counts as well as latency and throughput.

### Important fairness detail

Transformers tokenization is included in its timed request path. vLLM tokenizes prompts inside `llm.generate()`. This is intended to compare the end-to-end request path, but it should be described accurately in the report. The two frameworks also have different internal execution implementations, so the experiment is a practical framework comparison, not a perfectly isolated kernel benchmark.

## 11.2 `paged_attention_comparison.py`

### Original structure

The original uploaded file:

- Used a global `MODEL_NAME`.
- Defined `run_inference(use_paged_attention, batch_size)`.
- Loaded a new vLLM engine for every individual batch-size/configuration run.
- Tested only batch size 8 in the main loop.
- Compared block sizes 16 and 2048.
- Printed latency and throughput.
- Did not yet implement speculative decoding.

### Refactored structure

The current script:

- Uses `run_experiment()` to load one engine for a selected model/block-size/speculative-decoding configuration and test all requested batch sizes with it.
- Adds command-line arguments for model, batch sizes, block-size mode, speculative-decoding mode, optional maximum model length, and output JSON filename.
- Adds the TinyLlama draft model and five speculative tokens, following the handout.
- Supports speculative decoding OFF, ON, or both.
- Supports PagedAttention normal mode, simulated OFF mode, or both.
- Adds a warm-up generation before measuring each engine configuration.
- Uses CUDA synchronization around timed vLLM generation.
- Records model load time, inference time, generated-token count, throughput, model name, block size, and experiment flags in JSON.
- Releases the vLLM engine and clears unused memory after each configuration.

### Important design choice

The current `run_experiment()` reuses one engine across the batch sizes for a given configuration. This avoids repeatedly loading the same model for every batch size. It still loads a separate engine when the block-size or speculative-decoding configuration changes, because those settings are applied when constructing `LLM`.

## 11.3 Variable and function name changes

The following table lists the main renames between the original code and the current refactored scripts. If the team wants to restore the original names after the README/report work, use this as a guide. Renaming is optional and does not change the experiment as long as every reference is updated consistently.

### `paged_attention_comparison.py`

| Original name | Current name | Notes |
|---|---|---|
| `MODEL_NAME` | `DEFAULT_MODEL_NAME` | The default value for the new CLI argument. The actual run-time model is passed as `model_name`. |
| `run_inference(...)` | `run_experiment(...)` | The function now supports multiple batch sizes and speculative decoding. |
| `use_paged_attention` | `use_paged_attention` | Kept unchanged as the function parameter. |
| `batch_size` | `batch_size` | Kept unchanged inside the loop. |
| `infer_time` | `inference_time` | More explicit metric name. |
| `tokens_generated` | `generated_tokens` | More explicit metric name. |
| `throughput` | `throughput` locally; `throughput_tokens_per_second` in JSON | The JSON key is more descriptive. |
| `SAMPLING_PARAMS` | `SAMPLING_PARAMS` | Replaced by constants `TEMPERATURE`, `MAX_TOKENS`, and the same sampling-parameter object. |
| Not present | `use_speculative_decoding` | New function argument for Step 2-4. |
| Not present | `max_model_len` | New optional argument for the handout's maximum-length hint. |
| Not present | `parse_args()` | New CLI configuration function. |
| Not present | `synchronize_gpu()` | New timing helper. |

### `vllm_vs_transformers.py`

| Original name | Current name | Notes |
|---|---|---|
| `MODEL_NAME` | `MODEL_NAME` | Unchanged. |
| `PROMPTS` | `PROMPTS` | Unchanged. |
| `TEMPERATURE` | `TEMPERATURE` | Unchanged. |
| `MAX_LENGTH` | `MAX_NEW_TOKENS` | Important semantic correction: this now means newly generated tokens, not prompt-plus-generated sequence length. |
| `SAMPLING_PARAMS` | `VLLM_SAMPLING_PARAMS` | Makes clear that this object is specifically the vLLM sampling configuration. |
| `run_transformers_inference(batch_size)` | `run_transformers_inference(batch_sizes)` | Now accepts a list of batch sizes and tests them using one loaded model. |
| `run_vllm_inference(batch_size)` | `run_vllm_inference(batch_sizes)` | Same list-based refactoring for vLLM. |
| `infer_time` | `inference_time` | More explicit metric name. |
| `tokens_generated` | `generated_tokens` | More explicit metric name. |
| `throughput` | `throughput` locally; `throughput_tokens_per_second` in result dictionaries | More descriptive stored result key. |
| Not present | `make_batch_prompts()` | Ensures both backends receive the same prompt batch. |
| Not present | `count_transformers_generated_tokens()` | Excludes prompt tokens from the Transformers throughput calculation. |
| Not present | `parse_args()` / CLI arguments | Allows batch sizes to be set from the terminal. |
| Not present | `synchronize_gpu()` | Synchronizes CUDA around Transformers timing. |

### Should the original names be restored?

No restoration is required for the lab. The current names are more explicit and help distinguish:

- Maximum newly generated tokens from maximum total sequence length.
- The default model constant from the model selected for a particular run.
- vLLM-specific sampling parameters from Transformers generation arguments.
- A single batch size from a list of batch sizes.

If names are restored for consistency with an earlier draft, update every use of the name and keep the corrected behavior. In particular, do **not** change `MAX_NEW_TOKENS` back to `MAX_LENGTH` unless the code and report make clear that it still represents newly generated tokens. The name `MAX_LENGTH` can otherwise be misleading.

---

# 12. Final reminder

The code implementation is only one part of the assignment. The remaining work is to **run every required experiment, preserve the actual outputs, complete the report tables, and write an evidence-based analysis**.

The final report should explain both what was measured and which code settings were changed to obtain each measurement. Do not submit placeholder values or conclusions that have not been verified by an actual run.
