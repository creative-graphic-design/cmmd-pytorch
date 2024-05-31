# cmmd-pytorch

[![CI](https://github.com/creative-graphic-design/cmmd-pytorch/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/cmmd-pytorch/actions/workflows/ci.yaml)

(Unofficial) PyTorch implementation of CLIP Maximum Mean Discrepancy (CMMD) for evaluating image generation models, proposed in [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603). CMMD stands out to be a better metric than FID and tries to mitigate the longstanding issues of FID.

This implementation is a super simple PyTorch port of the [original codebase](https://github.com/google-research/google-research/tree/master/cmmd). I have only focused on the JAX and TensorFlow specific bits and replaced them PyTorch. Some differences:

* The original codebase relies on [`scenic`](https://github.com/google-research/scenic) for computing CLIP embeddings. This repository uses [`transformers`](https://github.com/huggingface/transformers).
* For the data loading, the original codebase uses TensorFlow, this one uses PyTorch `Dataset` and `DataLoader`.

## Setup

First, install PyTorch following instructions from the [official website](https://pytorch.org/).

Then install the depdencies:

```bash
pip install git+https://github.com/creative-graphic-design/cmmd-pytorch
```

After installation, you will be able to use the command `cmmd-pytorch`:

```shell
❯❯❯ cmmd-pytorch --help
usage: cmmd-pytorch [-h] [--batch-size BATCH_SIZE] [--max-count MAX_COUNT] [--ref-embed-file REF_EMBED_FILE] ref_dir eval_dir

positional arguments:
  ref_dir               Path to the directory containing reference images.
  eval_dir              Path to the directory containing images to be evaluated.

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Batch size for embedding generation.
  --max-count MAX_COUNT
                        Maximum number of images to read from each directory.
  --ref-embed-file REF_EMBED_FILE
                        Path to the pre-computed embedding file for the reference images.
```

## Running

```bash
cmmd-pytorch /path/to/reference/images /path/to/eval/images --batch_size=32 --max_count=30000
```

A working example command:

```bash
cmmd-pytorch reference_images generated_images --batch_size=1
```

It should output:

```bash
The CMMD value is:  7.696
```

This is the same as the original codebase, so, that confirms the implementation correctness 🤗

> [!TIP]
> GPU execution is supported when a GPU is available.

## Results

Below, we report the CMMD metric for some popular pipelines on the COCO-30k dataset, as commonly used by the community. CMMD, like FID, is better when it's lower.

| **Pipeline** | **Inference Steps** | **Resolution** | **CMMD** |
|:------------:|:-------------------:|:--------------:|:--------:|
|   [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)   |     30     |   1024x1024  | 0.696 |
|   [`segmind/SSD-1B`](https://huggingface.co/segmind/SSD-1B)   |     30     |   1024x1024  | 0.669 |
|   [`stabilityai/sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo)   |     1     |   512x512  | 0.548 |
|   [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)   |     50     |   512x512  | 0.582 |
|   [`PixArt-alpha/PixArt-XL-2-1024-MS`](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS)   |     20     |   1024x1024  | 1.140 |
|   [`SPRIGHT-T2I/spright-t2i-sd2`](https://huggingface.co/SPRIGHT-T2I/spright-t2i-sd2)   |     50     |   768x768  | 0.512 |

**Notes**:

* For SDXL Turbo, `guidance_scale` is set to 0 following the [official guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl_turbo) in `diffusers`. 
* For all other pipelines, default `guidace_scale` was used. Refer to the official pipeline documentation pages [here](https://huggingface.co/docs/diffusers/main/en/index) for more details.

> [!CAUTION]
> As per the CMMD authors, with models producing high-quality/high-resolution images, COCO images don't seem to be a good reference set (they are of pretty small resolution). This might help explain why SD v1.5 has a better CMMD than SDXL.

## Obtaining CMMD for your pipelines

One can refer to the `generate_images.py` script that generates images from the [COCO-30k randomly sampled captions](https://huggingface.co/datasets/sayakpaul/sample-datasets/raw/main/coco_30k_randomly_sampled_2014_val.csv) using `diffusers`. 

Once the images are generated, run:

```bash
cmmd-pytorch /path/to/reference/images /path/to/generated/images --batch_size=32 --max_count=30000
```

Reference images are COCO-30k images and can be downloaded from [here](https://huggingface.co/datasets/sayakpaul/coco-30-val-2014).

Pre-computed embeddings for the COCO-30k images can be found [here](https://huggingface.co/datasets/sayakpaul/coco-30-val-2014/blob/main/ref_embs_coco_30k.npy).

To use the pre-computed reference embeddings, run:

```bash
cmmd-pytorch None /path/to/generated/images ref_embed_file=ref_embs.npy --batch_size=32 --max_count=30000
```

## Acknowledgements

Thanks to Sadeep Jayasumana (first author of CMMD) for all the helpful discussions.


