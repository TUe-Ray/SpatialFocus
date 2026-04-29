# SpatialFocus 🔭

SpatialFocus is an ongoing research codebase for spatial reasoning from monocular video, built on top of [VLM-3R](https://github.com/VITA-Group/VLM-3R).

The project explores query-guided visual representations that emphasize spatially relevant information for vision-language reasoning. It extends the VLM-3R framework with components for training, evaluation, offline deployment, and spatial feature extraction in GPU cluster environments.

> This repository is under active development. Some components may change as the research evolves.

## Prerequisites ✅

- Linux environment with GPU support
- Conda (Miniconda or Mambaforge recommended)
- CUDA-compatible GPUs
- Fast local or scratch storage for model caches and datasets

## Project Structure 🗂️

```text
SpatialFocus/
|-- llava/                    # Core LLaVA-NeXT / VLM-3R model code
|-- third_party/
|   |-- CUT3R/                # Geometry encoder submodule
|   |-- EoMT/                 # EoMT submodule
|   `-- Pi3/                  # Pi3 submodule
|-- thinking-in-space/        # VSiBench / VSTiBench evaluation framework
|-- scripts/                  # Training, inference, and utility scripts
|-- vlm_3r_data_process/      # Data processing pipeline
|-- playground/demo/          # Demo videos and images
|-- eval_vsi_snellius.sh      # Example cluster evaluation job
`-- README.md
```

External dependencies are included as git submodules where needed.

## Installation 🛠️

### 1. Clone the repository

```bash
git clone <your-repo-url> SpatialFocus
cd SpatialFocus
git submodule update --init --recursive
```

### 2. Create the training and inference environment

```bash
conda create -n vlm3r python=3.10 -y
conda activate vlm3r

pip install --upgrade pip
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -e ".[train]"
pip install flash-attn==2.7.1.post1 --no-build-isolation
pip install decord openai accelerate==0.29.1
```

Tested package versions in this environment include:

- PyTorch 2.1.1 (CUDA 12.1)
- FlashAttention 2.7.1.post1
- DeepSpeed 0.14.4
- Transformers 4.40.0.dev0
- PEFT 0.4.0
- Accelerate 0.29.1

### 3. Create the evaluation environment

```bash
conda create -n vsibench python=3.10 -y
conda activate vsibench

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.vsibench.txt

cd thinking-in-space
pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
cd ..
```

Tested package versions in this environment include:

- PyTorch 2.1.1 (CUDA 12.1)
- FlashAttention 2.7.3
- Transformers 4.40.0
- PEFT 0.10.0
- s2wrapper 0.1

### 4. Build CUT3R

Run the following from the repository root with the vlm3r environment activated:

```bash
conda activate vlm3r

cd third_party/CUT3R
pip install -r requirements.txt

cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../..
```

To download the CUT3R checkpoint:

```bash
cd third_party/CUT3R/src
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ../../..
```

## Offline Cluster Setup 🧊

This repository supports offline GPU cluster workflows, where compute nodes do not have internet access.

Before submitting jobs to offline nodes, all required models and datasets should be cached in advance.

### Step 1. Pre-cache assets on a node with internet access

If you use the legacy helper script:

```bash
bash prepare_offline_cache.sh
```

Expected cached assets include:

- Journey9ni/vlm-3r-llava-qwen2-lora (LoRA weights)
- lmms-lab/LLaVA-NeXT-Video-7B-Qwen2 (base model)
- google/siglip-so400m-patch14-384 (vision encoder)
- nyu-visionx/VSI-Bench (evaluation dataset)

If any repository requires authentication, set HF_TOKEN in your environment before downloading. 🔐

### Step 2. Submit an evaluation job

```bash
sbatch eval_vsi_snellius.sh
```

The script is configured for offline execution with HF_*_OFFLINE=1 enabled.

Useful environment variables include:

| Variable | Default | Description |
|---|---|---|
| FAST_ROOT | /path/to/scratch | Fast scratch storage root |
| HF_HOME | $FAST_ROOT/hf_cache | Hugging Face cache directory |
| MODEL_ROOT | $FAST_ROOT/hf_models/VLM3R | Local model directory |
| NUM_PROCESSES | 4 | Number of GPUs |
| MAX_FRAMES_NUM | 32 | Maximum video frames per sample |

## Evaluation 📏

### VSiBench (thinking-in-space native script)

Run from the thinking-in-space directory with the vsibench environment activated:

```bash
conda activate vsibench
cd thinking-in-space
bash eval_vlm_3r_vsibench.sh
```

### VSTiBench (thinking-in-space native script)

```bash
conda activate vsibench
cd thinking-in-space
bash eval_vlm_3r_vstibench.sh
```

### VSiBench (cluster parity script at repository root)

```bash
conda activate vsibench
sbatch eval_vsi_snellius.sh
```

### VSiBench probe spatial ablation pair

This runs the probe-only controlled comparison with zero-spatial as the baseline and Reproduction_2 as the new model. It keeps sample selection, prompt variant, option shuffle seeds, and generation config paired across both runs, then compares without `--allow-mismatch`.

On Leonardo debug-QoS, prefer the manual three-step workflow so only one debug job runs at a time:

```bash
sbatch submit_vsibench_probe_zero_spatial_dbg.slurm
# After Step 1 finishes:
sbatch submit_vsibench_probe_reproduction2_dbg.slurm
# After Step 2 finishes, on the login node:
bash compare_vsibench_probe_zero_vs_repro_login.sh
```

The three-step workflow defaults to `NUM_SAMPLES=200`, `SAMPLE_SEED=42`, `PROMPT_VARIANT=option_shuffle`, and `OPTION_SHUFFLE_SEEDS=0,1,2`.

```bash
NUM_SAMPLES=200 \
SAMPLE_SEED=42 \
PROMPT_VARIANT=option_shuffle \
OPTION_SHUFFLE_SEEDS=0,1,2 \
bash eval_vsibench_probe_spatial_ablation_pair.sh
```

Outputs default to `outputs/vsibench_probe/`, including `compare_zero_spatial_vs_reproduction2_200_seed42_3seeds/report.md`.

### VSiBench training-data option bias

```bash
python scripts/analyze_vsibench_option_bias.py \
  --split train \
  --output outputs/vsibench_bias/train
```

Use `--local-dataset-path /path/to/file.jsonl` for local JSON, JSONL, CSV, or Parquet data, or `--dataset-name` and optional `--dataset-config` for a Hugging Face dataset.

## Pre-extracting Spatial Features ⚡

Use the extraction pipeline to precompute spatial features before training:

```bash
python scripts/extract_spatial_features.py \
  --input-dir /path/to/video/dataset \
  --output-dir /path/to/save/extracted_features \
  --cut3r-weights-path third_party/CUT3R/src/cut3r_512_dpt_4_64.pth \
  --processor-config-path processor_config.json \
  --gpu-ids 0,1,2,3
```

## Notes 📝

- This repository is still under active development.
- Interfaces, scripts, and directory structures may change as the project evolves.
- Several workflows are currently optimized for offline SLURM-based GPU clusters.
- Depending on your environment, you may need to adjust paths, cache locations, and job settings.

## License 📄

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE).

Some third-party components may be distributed under different licenses. In particular:

- CUT3R is licensed under CC BY-NC-SA 4.0.
- This may impose restrictions on commercial use.

Review all dependency licenses before production or commercial use.

## Acknowledgements 🙏

This project builds on several excellent open-source projects:

- [VLM-3R](https://github.com/VITA-Group/VLM-3R)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space)
