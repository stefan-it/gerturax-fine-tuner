# 🇩🇪 GERTuraX Fine-Tuner

* GERTuraX is a series of pretrained encoder-only language models for German.

* The models are ELECTRA-based and pretrained with the [TEAMS](https://aclanthology.org/2021.findings-acl.219/) approach
  on the [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) corpus. 

* In total, three different models were trained and released with pretraining corpus sizes ranging from 147GB to 1.1TB.

This repository hosts all necessary code to conduct the GERTuraX fine-tuning experiments on various downstream tasks
using the awesome **Flair** library.

# 📋 Changelog

* 09.02.2025: Add initial version of this repository.

# ⚗️ Fine-Tuning

## Dependencies

First, Flair and other dependencies must be installed:

```bash
$ pip3 install -r requirements.txt
```

## Environment Variables

The following environment variables can be set:

| Variable       | Required | Description                                                                                                                                                                          |
| -------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `CONFIG`       | ✔️        | Path to JSON-based configuration file, e.g. `configs/germeval14/gbert_base.json`.                                                                                                    |
| `HF_TOKEN`     | ✔️        | Access Token for Hugging Face Model Hub, find it [here](https://huggingface.co/settings/tokens). Must be set when model should be uploaded to Model Hub or to access private models. |
| `HUB_ORG_NAME` | ✖️        | Organization/User name on Hugging Face Model Hub. Must be set for model upload.                                                                                                      |
| `HF_UPLOAD`    | ✖️        | Defines if model should be uploaded to Model Hub or not. Disabled by default.                                                                                                        |

## Configuration format

Here's an example for the used JSON-based configuration format:

```json
{
    "batch_sizes": [
        32,
        16
    ],
    "learning_rates": [
        1e-05,
        2e-05,
        3e-05,
        4e-05
    ],
    "epochs": [
        20
    ],
    "context_sizes": [
        0
    ],
    "seeds": [
        1,
        2,
        3,
        4,
        5
    ],
    "layers": "-1",
    "subword_poolings": [
        "first"
    ],
    "use_crf": false,
    "use_tensorboard": true,
    "hf_model": "deepset/gbert-base",
    "model_short_name": "gbert_base",
    "task": "ner/germeval14",
    "cuda": "0"
}
```

Hyper-parameter searches are possible, e.g. different batch sizes, learning rates, epochs or seeds can be set. The CUDA device id can be set via `cuda` (expecting the id as string).

## Start!

After environment variables are set, the fine-tuning can be started with:

```bash
$ python3 script.py
```

The `flair-log-parser.py` script can be used to get an overview of best configurations and their correspondig F1-Scores.

# 📈 Evaluation Results

GERTuraX and other German Language Models were fine-tuned on GermEval 2014 (NER), GermEval 2018 (Sentiment analysis) and CoNLL-2003 (NER)

We use the same hyper-parameters for GermEval 2014, GermEval 2018 and CoNLL-2003 as used in the
[GeBERTa](https://arxiv.org/abs/2310.07321) paper (cf. Table 5) using 5 runs with different seed and report the averaged
score, conducted with the awesome Flair library.

## GermEval 2014

### GermEval 2014 - Original version

| Model Name                                                             | Avg. Development F1-Score | Avg. Test F1-Score |
|------------------------------------------------------------------------|---------------------------|--------------------|
| [GBERT Base](https://huggingface.co/deepset/gbert-base)                | 87.53 ± 0.22              | 86.81 ± 0.16       |
| [GERTuraX-1](https://huggingface.co/gerturax/gerturax-1) (147GB)       | 88.32 ± 0.21              | 87.18 ± 0.12       |
| [GERTuraX-2](https://huggingface.co/gerturax/gerturax-2) (486GB)       | 88.58 ± 0.32              | 87.58 ± 0.15       |
| [GERTuraX-3](https://huggingface.co/gerturax/gerturax-3) (1.1TB)       | 88.90 ± 0.06              | 87.84 ± 0.18       |
| [GeBERTa Base](https://huggingface.co/ikim-uk-essen/geberta-base)      | 88.79 ± 0.16              | 88.03 ± 0.16       |
| [ModernGBERT 134M](https://huggingface.co/LSX-UniWue/ModernGBERT_134M) | 87.86 ± 0.29              | 86.79 ± 0.29       |

### GermEval 2014 - [Without Wikipedia](https://huggingface.co/datasets/stefan-it/germeval14_no_wikipedia)

| Model Name                                                             | Avg. Development F1-Score | Avg. Test F1-Score |
|------------------------------------------------------------------------|---------------------------|--------------------|
| [GBERT Base](https://huggingface.co/deepset/gbert-base)                | 90.48 ± 0.34              | 89.05 ± 0.21       |
| [GERTuraX-1](https://huggingface.co/gerturax/gerturax-1) (147GB)       | 91.27 ± 0.11              | 89.73 ± 0.27       |
| [GERTuraX-2](https://huggingface.co/gerturax/gerturax-2) (486GB)       | 91.70 ± 0.28              | 89.98 ± 0.22       |
| [GERTuraX-3](https://huggingface.co/gerturax/gerturax-3) (1.1TB)       | 91.75 ± 0.17              | 90.24 ± 0.27       |
| [GeBERTa Base](https://huggingface.co/ikim-uk-essen/geberta-base)      | 91.74 ± 0.23              | 90.28 ± 0.21       |
| [ModernGBERT 134M](https://huggingface.co/LSX-UniWue/ModernGBERT_134M) | 90.64 ± 0.21              | 89.13 ± 0.31       |

## GermEval 2018

### GermEval 2018 - Fine Grained

| Model Name                                                             | Avg. Development F1-Score | Avg. Test F1-Score |
|------------------------------------------------------------------------|---------------------------|--------------------|
| [GBERT Base](https://huggingface.co/deepset/gbert-base)                | 63.66 ± 4.08              | 51.86 ± 1.31       |
| [GERTuraX-1](https://huggingface.co/gerturax/gerturax-1) (147GB)       | 62.87 ± 1.95              | 50.61 ± 0.36       |
| [GERTuraX-2](https://huggingface.co/gerturax/gerturax-2) (486GB)       | 64.37 ± 1.31              | 51.02 ± 0.90       |
| [GERTuraX-3](https://huggingface.co/gerturax/gerturax-3) (1.1TB)       | 66.39 ± 0.85              | 49.94 ± 2.06       |
| [GeBERTa Base](https://huggingface.co/ikim-uk-essen/geberta-base)      | 65.81 ± 3.29              | 52.45 ± 0.57       |
| [ModernGBERT 134M](https://huggingface.co/LSX-UniWue/ModernGBERT_134M) | 59.69 ± 2.12              | 48.75 ± 3.33       |
| [GeistBERT Base](https://huggingface.co/GeistBERT/GeistBERT_base)      | 64.84 ± 1.59              | 53.47 ± 1.12       |

### GermEval 2018 - Coarse Grained

| Model Name                                                             | Avg. Development F1-Score | Avg. Test F1-Score |
|------------------------------------------------------------------------|---------------------------|--------------------|
| [GBERT Base](https://huggingface.co/deepset/gbert-base)                | 83.15 ± 1.83              | 76.39 ± 0.64       |
| [GERTuraX-1](https://huggingface.co/gerturax/gerturax-1) (147GB)       | 83.72 ± 0.68              | 77.11 ± 0.59       |
| [GERTuraX-2](https://huggingface.co/gerturax/gerturax-2) (486GB)       | 84.51 ± 0.88              | 78.07 ± 0.91       |
| [GERTuraX-3](https://huggingface.co/gerturax/gerturax-3) (1.1TB)       | 84.33 ± 1.48              | 78.44 ± 0.74       |
| [GeBERTa Base](https://huggingface.co/ikim-uk-essen/geberta-base)      | 83.54 ± 1.27              | 78.36 ± 0.79       |
| [ModernGBERT 134M](https://huggingface.co/LSX-UniWue/ModernGBERT_134M) | 83.16 ± 2.05              | 76.01 ± 0.89       |

## CoNLL-2003 - German, Revised

| Model Name                                                             | Avg. Development F1-Score | Avg. Test F1-Score |
|------------------------------------------------------------------------|---------------------------|--------------------|
| [GBERT Base](https://huggingface.co/deepset/gbert-base)                | 92.15 ± 0.10              | 88.73 ± 0.21       |
| [GERTuraX-1](https://huggingface.co/gerturax/gerturax-1) (147GB)       | 92.32 ± 0.14              | 90.09 ± 0.12       |
| [GERTuraX-2](https://huggingface.co/gerturax/gerturax-2) (486GB)       | 92.75 ± 0.20              | 90.15 ± 0.14       |
| [GERTuraX-3](https://huggingface.co/gerturax/gerturax-3) (1.1TB)       | 92.77 ± 0.28              | 90.83 ± 0.16       |
| [GeBERTa Base](https://huggingface.co/ikim-uk-essen/geberta-base)      | 92.87 ± 0.21              | 90.94 ± 0.24       |
| [ModernGBERT 134M](https://huggingface.co/LSX-UniWue/ModernGBERT_134M) | 91.49 ± 0.15              | 89.64 ± 0.29       |

# ❤️ Acknowledgements

GERTuraX is the outcome of the last 12 months of working with TPUs from the awesome [TRC program](https://sites.research.google/trc/about/)
and the [TensorFlow Model Garden](https://github.com/tensorflow/models) library.

Many thanks for providing TPUs!

Made from Bavarian Oberland with ❤️ and 🥨.
