
![Omin Logo](./assets/omni_banner.png)

<div align="center">

<a href="https://sofiene-boutaj.github.io/omni-projectpage/"><img src="https://img.shields.io/badge/OMNI-Project_Page-e85d88.svg" alt="Project Page" /></a>
[![Paper](https://img.shields.io/badge/OMNI-arXiv.2507.07860-purple.svg)](https://arxiv.org/abs/2507.07860)
[![Models](https://img.shields.io/badge/HuggingFace-models-yellow.svg)](https://huggingface.co/collections/sofieneb/omni-models)
[![Datasets](https://img.shields.io/badge/HuggingFace-datasets-yellow.svg)](https://huggingface.co/collections/sofieneb/pathology-foundation-model-features)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)

</div>

<div align="center">

**Published at NeurIPS 2026 Main Track**
</div>


OMNI distills **10 pathology foundation models** into a single compact vision encoder. Instead of compressing all teachers into one CLS token, the student learns **one token per teacher**, using a structured attention mask, Mixture-of-Experts (MoE) layers and a per-teacher contrastive loss. This lets distillation keep improving as more teachers are added. The resulting models match or outperform their much larger teachers on 39 tile- and slide-level tasks.

## 🤗 Pretrained models

### Weights

The pretrained weights are on the Hugging Face Hub. **You do not need this repository to use them**: see the model cards for usage and a THUNDER evaluation script.

| Model | Params | Dim | GFLOPs | Tile LP F1 (%) | Slide LP BAcc (%) | Weights |
| ----- | ------ | --- | ------ | -------------- | ----------------- | ------- |
| OMNI-T | 7.3M | 192 | 1.39 | 83.9 | 63.2 | [sofieneb/omni-tiny](https://huggingface.co/sofieneb/omni-tiny) |
| OMNI-S | 28.7M | 384 | 5.48 | 85.3 | 63.1 | [sofieneb/omni-small](https://huggingface.co/sofieneb/omni-small) |
| OMNI-B | 114M | 768 | 21.8 | 87.1 | 63.2 | [sofieneb/omni-base](https://huggingface.co/sofieneb/omni-base) |
| OMNI-L | 330M | 1024 | 67.45 | 87.8 | 63.5 | [sofieneb/omni-large](https://huggingface.co/sofieneb/omni-large) |

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("sofieneb/omni-base", trust_remote_code=True).eval()
embedding = model(x)                                    # (B, 768): average of the 10 teacher tokens
out = model(x, return_teacher_tokens=True)              # + out["teacher_tokens"]: (B, 10, 768)
```

## Evaluation

The evaluation script is in [`eval/eval_thunder.py`](eval/eval_thunder.py). Install THUNDER and set its data folder, where datasets, embeddings and results are stored:

```bash
pip install thunder-bench
export THUNDER_BASE_DATA_FOLDER=/path/to/thunder_data
```

Then run the benchmark, choosing the model with `--model` (`tiny`, `small`, `base` or `large`) and the datasets with `--datasets`. The datasets are downloaded and their splits generated automatically before benchmarking (add `--no-download` to skip this once they are in place):

```bash
python eval/eval_thunder.py --model base --datasets <dataset> --tasks knn linear_probing
```

For example, on MHIST:

```bash
python eval/eval_thunder.py --model base --datasets mhist --tasks knn linear_probing
```

---

## Installation

Python 3.10 or later is required.

```bash
git clone <this-repo> && cd omni
pip install -e .
```

This installs everything needed for training, building the WebDataset shards and running the THUNDER evaluation.

If you also want to use the FlashAttention backend (`train.py --attn flash_attn`), install it with the `flash` extra. It requires a CUDA GPU and toolkit, and must be built against the PyTorch you already have installed:

```bash
pip install -e ".[flash]" --no-build-isolation
```
## Data

The students are trained on **10M tiles (224×224) from 10k TCGA whole-slide images**. Teachers are frozen, so their embeddings are **pre-computed once** and stored next to each tile in [WebDataset](https://github.com/webdataset/webdataset) shards. `train.py` only reads these shards; no teacher is run during training.

### Building the shards

`write_webdataset.py` builds the shards read by `train.py` from **patch features you have already extracted with [TRIDENT](https://github.com/mahmoodlab/TRIDENT) or [CLAM](https://github.com/mahmoodlab/CLAM)**, one feature extraction per teacher. For every patch, it reads the tile from the slide at the patch coordinates, resizes it to 224×224, and stores it with the features of all teachers at these coordinates.

Install its dependencies with `pip install h5py openslide-python openslide-bin`.

**Inputs**

- `--wsi-dir`: the whole-slide images, in any format readable by OpenSlide (`.svs`, `.tiff`, `.ndpi`, ...).
- `--features teacher=folder ...`: one feature folder per teacher, with one `<slide>.h5` file per slide. These are the standard outputs of the two tools:
  - TRIDENT (`run_batch_of_slides.py --task feat --patch_encoder <encoder>`): `<job_dir>/<mag>x_<size>px_<overlap>px_overlap/features_<encoder>/`
  - CLAM (`extract_features_fp.py`): `<feat_dir>/h5_files/`

Each `<slide>.h5` must contain:

| Dataset | Content | Shape |
| --- | --- | --- |
| `features` | one embedding per patch | `(N, D)` |
| `coords` | level-0 (x, y) coordinates of the top-left corner of each patch | `(N, 2)` |

The slide file and its feature files are matched by name: `TCGA-XX.svs` ↔ `TCGA-XX.h5`.

**Notes**

- **Teacher names.** The names on the left of `=` must be the teacher names of `configs.py` (table below), not the TRIDENT encoder names, e.g. `uni2h=.../features_uni_v2`. Training uses the teachers given to `train.py --teachers` (the 10 of the paper by default), so all of them must be in the shards.
- **Matching across teachers.** Features of different teachers are matched by coordinates. Only patches present for every teacher are kept, so all teachers should come from the same patching (same magnification and patch size).
- **Patch size.** TRIDENT stores the patch size in the `coords` attributes (`patch_size_level0`). CLAM feature files do not, so pass it with `--patch-size-level0` (e.g. 512 for 256 px patches at 20× on a 40× slide).
- **Embedding size.** The features may be longer than `D_cls`: only their first `D_cls` values are used. For example, Virchow2 in TRIDENT gives `[CLS, mean patch token]` (2560), of which the CLS (1280) is kept.
- **Subsampling.** `--tiles-per-slide` keeps a random subset of patches per slide. The paper uses ~1,000 tiles per slide, so 10M tiles from 10k slides.

**Usage.** Shards of 10,000 samples are written to `--out-dir`. Slides can be split over several jobs (e.g. a SLURM array) and are read in parallel within a job:

```bash
F=/data/trident/20x_256px_0px_overlap
python write_webdataset.py --wsi-dir /data/tcga/wsis --out-dir /data/tcga/shards \
    --features hoptimus1=$F/features_hoptimus1 virchow2=$F/features_virchow2 uni2h=$F/features_uni_v2 ... \
    --tiles-per-slide 1000 --num-jobs 100 --job-id $SLURM_ARRAY_TASK_ID --num-workers 16
```

Then train with `--data /data/tcga/shards`.

> **Patch loss.** TRIDENT and CLAM features are one embedding per patch: they contain no teacher patch tokens. Shards built with this script therefore only supervise the teacher-specific tokens (CLS and contrastive losses), and `train.py` turns the patch loss off automatically. The released models were also trained with the patch loss of the paper, using shards that additionally contain sampled teacher patch tokens (see the format below). When these files are in the shards, `train.py` uses them automatically.

### Format of the shards

Shards built with another pipeline must follow this format. `write_webdataset.py` writes the `jpg` and `{t}.npy` files; the two patch files are optional and enable the patch loss.

`--data` is a folder of `.tar` shards (any names ending in `.tar`, e.g. `shard-000000.tar`, `shard-000001.tar`, ...). Shards are sampled with replacement and shuffled with a 1,000-sample buffer, so their size does not matter (a few thousand tiles per shard works well).

Each **sample** is a group of files sharing the same key (the part of the file name before the first dot, so keys must not contain dots). For a tile with key `00000042`, a shard contains:

```
00000042.jpg                                        # the tile
00000042.hoptimus1.npy                              # teacher CLS embedding
00000042.hoptimus1_dense_embs.npy                   # K sampled teacher patch embeddings
00000042.hoptimus1_sampled_indices.npy              # positions of these K patches
00000042.virchow2.npy
00000042.virchow2_dense_embs.npy
00000042.virchow2_sampled_indices.npy
...                                                 # 3 files for each of the 10 teachers
```

| File | Content | Shape | dtype |
| --- | --- | --- | --- |
| `{key}.jpg` | 224×224 RGB tile (see the note on colours below) | `(224, 224, 3)` | uint8 |
| `{key}.{t}.npy` | CLS embedding of teacher `t` | `(≥ D_cls,)` | float16 / float32 |
| `{key}.{t}_dense_embs.npy` (optional) | K patch embeddings of teacher `t` | `(K, D_patch)` | float16 / float32 |
| `{key}.{t}_sampled_indices.npy` (optional) | indices of these K patches in the teacher grid | `(K,)` | int |

- `t` is the teacher name used in `configs.py` (e.g. `hoptimus1`, `dinov3vitl16pretrainlvd1689m`).
- The `.npy` files are standard `numpy.save` outputs.
- The CLS embedding may be longer than `D_cls`: only the first `D_cls` values are used. Virchow2 and H0-mini store `[CLS, mean patch token]`, which `train.py` truncates.
- Patch indices refer to the teacher's **patch tokens only** (no CLS or register tokens), flattened in row-major order over the teacher's √N × √N grid. The values must lie in `[0, N)`; this is checked on the first iteration.
- Only a subset of K patches is stored per teacher, which keeps the shards small. K must be the same for all samples of a given teacher (so they can be batched), but can differ between teachers. The released models use K = 10% of the grid.
- **Colours.** `train.py` decodes tiles with `cv2.imdecode`, which keeps the channel order stored in the file. Encode RGB arrays with `cv2.imencode` (as `write_webdataset.py` does) so that tiles are decoded as RGB.

### Teachers

| Teacher | Name in `configs.py` | D_cls | D_patch | Patch grid N |
| --- | --- | --- | --- | --- |
| H-optimus-1 | `hoptimus1` | 1536 | 1536 | 256 (16×16) |
| Virchow2 | `virchow2` | 1280 | 1280 | 256 (16×16) |
| UNI2-h | `uni2h` | 1536 | 1536 | 256 (16×16) |
| Prov-GigaPath | `provgigapath` | 1536 | 1536 | 196 (14×14) |
| Kaiko-ViT-B/8 | `kaiko_vitb8` | 768 | 768 | 784 (28×28) |
| CONCH v1.5 | `titan` | 768 | 1024 | 784 (28×28) |
| Hibou-L | `hiboul` | 1024 | 1024 | 256 (16×16) |
| H0-mini | `h0mini` | 768 | 768 | 256 (16×16) |
| KEEP | `keep` | 768 | 1024 | 196 (14×14) |
| DINOv3-ViT-L/16 | `dinov3vitl16pretrainlvd1689m` | 1024 | 1024 | 196 (14×14) |

## Training

```bash
python train.py --model tiny  --data /path/to/shards --output outputs/omni-tiny
python train.py --model small --data /path/to/shards --output outputs/omni-small
python train.py --model base  --data /path/to/shards --output outputs/omni-base
python train.py --model large --data /path/to/shards --output outputs/omni-large
```

Training runs on a single GPU in bf16. It uses the patch loss when the shards contain teacher patch embeddings, and prints which mode it uses at start-up. It **resumes automatically** from the latest checkpoint in `--output`. Checkpoints (`iters_*.pth`) are saved every 10k iterations, and TensorBoard logs (per-teacher CLS / patch / contrastive losses, MoE loss) are written to `--output/tb`. A student run takes about 40 H100 GPU hours.

To train on a subset of teachers (e.g. the scaling study of Figure 1b):

```bash
python train.py --model base --data /path/to/shards --output outputs/omni-base-3t \
    --teachers hoptimus1 virchow2 uni2h
```

### FlashAttention

```bash
pip install flash-attn --no-build-isolation
python train.py --model base --data /path/to/shards --output outputs/omni-base --attn flash_attn
```

FlashAttention does not support arbitrary attention masks, so the OMNI mask is applied exactly by splitting the attention in two. Patch tokens attend only to patch tokens: this is plain unmasked attention, run with flash-attn, and it is the dominant Np × Np cost. Each teacher token attends to itself and to the patch tokens: this is a small M × (1 + Np) attention computed explicitly. The result is mathematically identical to the masked `sdpa` backend (default), so checkpoints are interchangeable between the two backends.

### Objective

For each teacher *i*, the teacher-specific token is projected by a linear head, and the student patch grid is bilinearly resized to the teacher grid (Appendix B). The per-teacher loss and the total loss are:

$$\mathcal{L}_i = \mathcal{L}^{\text{cls}}_{\text{MTD},i} + \mathcal{L}^{\text{patch}}_{\text{MTD},i} + \lambda_{\text{ctr}} \mathcal{L}^{\text{ctr}}_i$$

$$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} \mathcal{L}_i + \alpha_{\text{MoE}} \mathcal{L}_{\text{MoE}}$$

where:

- the **CLS** and **patch** losses combine a cosine loss and a SmoothL1 loss between L2-normalized vectors,
- the **contrastive** loss is a symmetric InfoNCE loss,
- the **MoE** loss is the Switch Transformer load-balancing loss.
### Hyper-parameters

| | OMNI-T | OMNI-S | OMNI-B | OMNI-L |
| --- | --- | --- | --- | --- |
| Embedding dim / depth / heads | 192 / 12 / 3 | 384 / 12 / 6 | 768 / 12 / 12 | 1024 / 24 / 16 |
| MLP ratio | 3.0 | 3.0 | 3.0 | 2.5 |
| MoE blocks (last) | 3 | 3 | 3 | 5 |
| Experts / top-k | 5 / 2 | 5 / 2 | 5 / 2 | 5 / 2 |
| Optimizer | Adam, lr 1e-4 | | | |
| Batch size / iterations | 128 / 700k | | | |
| λ_reg (SmoothL1, β=1) / λ_ctr / τ / α_MoE | 1.0 / 0.1 / 0.1 / 0.05 | | | |
| Input / patch size | 224 / 14 | | | |

## Citation

```bibtex
@inproceedings{boutaj2026omni,
  author  = {Boutaj, Sofi{\`e}ne and Marza, Pierre and Belagali, Varun and
             Samaras, Dimitris and Vakalopoulou, Maria and Christodoulidis, Stergios},
  title   = {Scaling Multi-Teacher Distillation for Digital Pathology},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year    = {2026}
}
```
