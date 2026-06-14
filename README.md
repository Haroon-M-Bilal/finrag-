# FinRAG — Domain-Adapted RAG for Financial Report Question Answering

**Authors:** Haroon Bilal ([@Haroon-M-Bilal](https://github.com/Haroon-M-Bilal)) and Muhammad Sarwar ([@Muhammad2481632](https://github.com/Muhammad2481632)) — joint project, both contributed throughout.

Fine-tuning the retrieval and generation components of a RAG system on financial
filings (S&P 500 10-Ks) and measuring the gains with proper retrieval metrics.
Built and evaluated locally on a single RTX 4080 (12 GB) — no paid APIs.

## Summary

We take an off-the-shelf financial RAG pipeline and domain-adapt three components
on the **FinDER** benchmark: the embedding model, the cross-encoder reranker, and
the generator (Qwen2.5-7B via QLoRA). We report retrieval metrics (MRR, NDCG,
Recall) — which prior work on this setup omitted — and end-to-end generation
quality, with an ablation isolating each component.

## Results

### Retrieval (5,625 questions, vs gold evidence)

| System | MRR@10 | NDCG@10 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|---|
| Dense (off-the-shelf) | 0.0638 | 0.0764 | 0.0911 | 0.1215 | 0.1510 |
| Hybrid | 0.0564 | 0.0704 | 0.0828 | 0.1206 | 0.1642 |
| Hybrid + Rerank | 0.0870 | 0.1024 | 0.1269 | 0.1592 | 0.1976 |
| **FT Dense (ours)** | **0.2602** | **0.2920** | **0.3415** | **0.4040** | **0.4667** |
| FT Hybrid | 0.1934 | 0.2311 | 0.2840 | 0.3617 | 0.4326 |
| FT Hybrid + Rerank | 0.1887 | 0.2233 | 0.2666 | 0.3459 | 0.4358 |

Fine-tuned dense retrieval improves MRR@10 ~3x over the best off-the-shelf baseline.

### Generation (300 held-out questions, same retrieval)

| System | ROUGE-L | BLEU | BERTScore-F1 |
|---|---|---|---|
| Qwen2.5-7B (base) | 0.0767 | 0.327 | 0.8236 |
| **Qwen2.5-7B + QLoRA (ours)** | **0.2027** | **4.153** | **0.8655** |

### Key finding

Once the embedder is domain-tuned, BM25 fusion and reranking no longer help at the
top ranks (they only marginally aid deep recall). This extends prior work, which
found reranking beneficial for *generic* embedders — domain fine-tuning of the
retriever is the dominant lever.

## Pipeline

Question -> fine-tuned embedder (dense) + BM25 -> RRF fusion -> fine-tuned reranker
-> QLoRA Qwen-7B -> answer. (Best config drops fusion/rerank; see ablation.)

## Repo structure

```
src/data/      prepare_finder.py        # build corpus / queries / qrels
src/train/     build_training_data.py   # hard-negative mining
               finetune_embeddings.py   # contrastive embedding FT (MNRL)
               mine_negatives_ft.py     # retriever-consistent negatives
               finetune_reranker.py     # cross-encoder FT
src/eval/      run_baseline.py          # off-the-shelf retrieval metrics
               run_finetuned.py         # fine-tuned vs baseline (comparison.md)
               diag_reranker.py         # reranker diagnostic
src/generate/  prepare_qa_data.py       # build QA SFT data
               finetune_qlora.py        # QLoRA fine-tune Qwen-7B
               run_generation_eval.py   # ROUGE/BLEU/BERTScore
app.py         Gradio demo (live, local)
results/       tables, JSON, figures, make_figures.py
```

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

FinDER (S&P 500 10-Ks + questions + gold evidence), from Linq AI Research:
https://huggingface.co/datasets/Linq-AI-Research/FinDER
Place `10-k.zip` (unzip to `data/finder/10k/`) and `train-00000-of-00001.parquet`
in `data/finder/`. (Data and trained checkpoints are gitignored — regenerate with
the scripts below.)

## Reproduce

```bash
python src/data/prepare_finder.py
python src/train/build_training_data.py
python src/train/finetune_embeddings.py
python src/eval/run_baseline.py
python src/train/mine_negatives_ft.py
python src/train/finetune_reranker.py
python src/eval/run_finetuned.py
python src/generate/prepare_qa_data.py
python src/generate/finetune_qlora.py
python src/generate/run_generation_eval.py
python results/make_figures.py
python app.py            # demo
```

## Base reference

Builds on the FinDER finance-RAG setup of Cheng et al., "Enhancing Financial
Report Question-Answering: A RAG System with Reranking Analysis" (arXiv:2603.16877),
which used off-the-shelf models + a paid LLM and reported only answer quality.
We add domain fine-tuning, a fully-local stack, and retrieval-side metrics.

## Notes

FinDER is a hard benchmark (terse queries, long filings, multi-step reasoning), so
absolute scores are modest; the contribution is the *relative* improvement from
domain fine-tuning, measured on metrics prior work omitted.