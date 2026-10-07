"""
EFFICIENCY BENCHMARK - throughput and peak VRAM for the deployed pipeline.

WHY THIS EXISTS

The paper's central claim is that a domain-adapted financial RAG pipeline runs
on a single consumer GPU. That claim is currently unsupported: no throughput or
memory figure has been reported.

The honest measurement is the FULL PIPELINE CO-RESIDENT - embedder, reranker and
generator all loaded at once, as they would be in deployment - not each
component measured alone. A per-component figure understates the requirement,
because a deployed system cannot swap models in and out between stages without
paying the load cost on every query.

WHAT IS MEASURED

  Stage latency, per query, at the deployed settings:
    embed      query encoding with the fine-tuned bge-small
    route      ticker match + similarity over the resolved filing's chunks
    rerank     cross-encoder over the top-50
    generate   QLoRA generator, top-3 context, greedy decoding
  Peak VRAM after each model is added, and for the whole pipeline under load.
  Generation throughput in tokens/second.

Corpus embeddings are memory-mapped rather than loaded into RAM, matching how a
deployment would hold a 182k-chunk index.

Run:  python -u src/eval/benchmark_efficiency.py
      python -u src/eval/benchmark_efficiency.py Qwen/Qwen2.5-3B-Instruct 50
Output: results/efficiency.md / .json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

PROC = Path("data/finder/processed_v4")
CKPT = Path("checkpoints")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

GEN = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
N_QUERIES = int(sys.argv[2]) if len(sys.argv) > 2 else 50
TAG = GEN.split("/")[-1].lower().replace(".", "-")

EMBED_FT = str(CKPT / "bge-small-finder-v4")
RERANK_FT = str(CKPT / "bge-reranker-finder-v4")
ADAPTER = str(CKPT / f"qlora-{TAG}-v4")

PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_K = 50
TOPK_CTX = 3
MAX_NEW = 768
PASSAGE_TOKENS = 460
TICKRE = re.compile(r"\b[A-Z][A-Z0-9\-]{1,5}\b")

SYS = ("You are a financial analyst. Answer the question using only the "
       "provided context from SEC 10-K filings. If the context does not "
       "contain the information needed, say so explicitly. End every "
       "response with a line beginning 'ANSWER:'.")


def gb(x=None):
    torch.cuda.synchronize()
    return (x if x is not None else torch.cuda.memory_allocated()) / 1e9


def peak_gb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    if not Path(ADAPTER).exists():
        raise SystemExit(f"adapter not found: {ADAPTER}")
    print(f"generator: {GEN}\nqueries: {N_QUERIES}\n")

    dev_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {dev_name}  ({total_vram:.1f} GB)\n")

    corpus = jl(PROC / "corpus.jsonl")
    cids = [c["_id"] for c in corpus]
    ctexts = [c["text"] for c in corpus]
    cid2text = dict(zip(cids, ctexts))
    by_tick = {}
    for i, c in enumerate(cids):
        by_tick.setdefault(c.split("::")[0], []).append(i)
    all_ticks = set(by_tick)
    del corpus

    queries = jl(PROC / "queries_test.jsonl")[:N_QUERIES]
    print(f"corpus: {len(cids)} chunks, {len(by_tick)} filings")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem = {}

    # ---------------------------------------------------------------- embedder
    t0 = time.time()
    emb = SentenceTransformer(EMBED_FT, device="cuda")
    load_embed = time.time() - t0
    mem["after_embedder"] = gb()

    # corpus vectors are memory-mapped: a deployment holds the index on disk,
    # not in RAM, and only the resolved filing's rows are touched per query
    cemb = np.load(PROC / "corpus_emb_ft_v4.npy", mmap_mode="r")
    print(f"embedder loaded ({load_embed:.1f}s), "
          f"VRAM {mem['after_embedder']:.2f} GB")

    # ---------------------------------------------------------------- reranker
    t0 = time.time()
    rr = CrossEncoder(RERANK_FT, device="cuda", max_length=512)
    load_rerank = time.time() - t0
    mem["after_reranker"] = gb()
    print(f"reranker loaded ({load_rerank:.1f}s), "
          f"VRAM {mem['after_reranker']:.2f} GB")

    # --------------------------------------------------------------- generator
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(GEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(GEN, quantization_config=bnb,
                                                 device_map={"": 0})
    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    load_gen = time.time() - t0
    mem["after_generator"] = gb()
    print(f"generator loaded ({load_gen:.1f}s), "
          f"VRAM {mem['after_generator']:.2f} GB  "
          f"<- all three models co-resident\n")

    rr_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    clip_cache: dict[str, str] = {}

    def clip(t):
        hit = clip_cache.get(t)
        if hit is not None:
            return hit
        ids = rr_tok.encode(t, add_special_tokens=False)
        out = (rr_tok.decode(ids[:PASSAGE_TOKENS], skip_special_tokens=True)
               if len(ids) > PASSAGE_TOKENS else t)
        clip_cache[t] = out
        return out

    # ------------------------------------------------------------- warm-up
    print("warming up...")
    q0 = queries[0]["text"]
    _ = emb.encode([PREFIX + q0], normalize_embeddings=True,
                   convert_to_numpy=True)
    _ = rr.predict([[q0, ctexts[0][:2000]]], show_progress_bar=False)
    ids = tok(SYS + q0, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**ids, max_new_tokens=8, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------- measure
    print(f"benchmarking {len(queries)} queries...\n")
    t_embed, t_route, t_rerank, t_gen = [], [], [], []
    new_tokens, routed_n = [], 0

    for i, q in enumerate(queries):
        qt = q["text"]

        t0 = time.time()
        qv = emb.encode([PREFIX + qt], normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)[0]
        t_embed.append(time.time() - t0)

        t0 = time.time()
        named = set(TICKRE.findall(qt)) & all_ticks
        if len(named) == 1:
            idx = by_tick[named.pop()]
            sims = np.asarray(cemb[idx]) @ qv
            order = np.argsort(-sims)[:RERANK_K]
            cand = [cids[idx[int(o)]] for o in order]
            routed_n += 1
        else:
            sims = np.asarray(cemb) @ qv          # full-corpus fallback
            order = np.argsort(-sims)[:RERANK_K]
            cand = [cids[int(o)] for o in order]
        t_route.append(time.time() - t0)

        t0 = time.time()
        sc = rr.predict([[qt, clip(cid2text[c])] for c in cand],
                        batch_size=64, show_progress_bar=False)
        top = [cand[j] for j in np.argsort(sc)[::-1][:TOPK_CTX]]
        t_rerank.append(time.time() - t0)

        ctx = "\n\n".join(f"[{j + 1}] {cid2text[c]}"
                          for j, c in enumerate(top))
        msgs = [{"role": "system", "content": SYS},
                {"role": "user",
                 "content": f"Context:\n{ctx}\n\nQuestion: {qt}"}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        t_gen.append(time.time() - t0)
        new_tokens.append(int(out.shape[1] - enc.shape[1]))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)}", end="\r")
    print()

    mem["peak_under_load"] = peak_gb()

    def stat(v):
        a = np.array(v)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90))}

    tps = float(np.sum(new_tokens) / np.sum(t_gen))
    e2e = np.array(t_embed) + np.array(t_route) + np.array(t_rerank) \
        + np.array(t_gen)

    report = {
        "gpu": dev_name, "total_vram_gb": total_vram,
        "generator": GEN, "n_queries": len(queries),
        "routed_fraction": routed_n / len(queries),
        "load_seconds": {"embedder": load_embed, "reranker": load_rerank,
                         "generator": load_gen},
        "vram_gb": mem,
        "latency_seconds": {"embed": stat(t_embed), "route": stat(t_route),
                            "rerank": stat(t_rerank), "generate": stat(t_gen),
                            "end_to_end": stat(e2e)},
        "generation_tokens_per_second": tps,
        "mean_new_tokens": float(np.mean(new_tokens)),
    }

    L = [f"# Efficiency: {GEN}", "",
         f"GPU: {dev_name} ({total_vram:.1f} GB). {len(queries)} test queries, "
         f"{100 * routed_n / len(queries):.0f}% ticker-routed. All three "
         f"models (embedder, reranker, generator) are loaded SIMULTANEOUSLY, "
         f"as they would be in deployment - per-component figures understate "
         f"the requirement. Corpus vectors ({len(cids)} chunks) are "
         f"memory-mapped, not held in VRAM.", "",
         "## Peak VRAM", "", "| stage | GB |", "|---|---|",
         f"| + embedder (33M, fp32) | {mem['after_embedder']:.2f} |",
         f"| + reranker (278M, fp32) | {mem['after_reranker']:.2f} |",
         f"| + generator (4-bit QLoRA) | {mem['after_generator']:.2f} |",
         f"| **peak under load** | **{mem['peak_under_load']:.2f}** |", "",
         f"Headroom on a {total_vram:.0f} GB card: "
         f"{total_vram - mem['peak_under_load']:.2f} GB.", "",
         "## Latency per query (seconds)", "",
         "| stage | mean | median | p90 |", "|---|---|---|---|"]
    for k in ("embed", "route", "rerank", "generate", "end_to_end"):
        s = report["latency_seconds"][k]
        L.append(f"| {k} | {s['mean']:.3f} | {s['median']:.3f} | "
                 f"{s['p90']:.3f} |")

    L += ["", "## Generation throughput", "",
          f"- **{tps:.1f} tokens/second** (greedy, batch size 1)",
          f"- mean {np.mean(new_tokens):.0f} new tokens per answer",
          f"- model load time: embedder {load_embed:.1f}s, "
          f"reranker {load_rerank:.1f}s, generator {load_gen:.1f}s "
          f"(one-off, not per query)"]

    out = "\n".join(L) + "\n"
    print("\n" + out)
    (RESULTS / f"efficiency_{TAG}.md").write_text(out, encoding="utf-8")
    json.dump(report, open(RESULTS / f"efficiency_{TAG}.json", "w"), indent=2)
    print(f"saved -> results/efficiency_{TAG}.md / .json")


if __name__ == "__main__":
    main()