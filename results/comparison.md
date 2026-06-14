# Baseline vs Fine-tuned (full eval, 5,625 questions)

| System | mrr@10 | ndcg@10 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|
| dense_only | 0.0638 | 0.0764 | 0.0911 | 0.1215 | 0.1510 |
| hybrid | 0.0564 | 0.0704 | 0.0828 | 0.1206 | 0.1642 |
| hybrid_rerank | 0.0870 | 0.1024 | 0.1269 | 0.1592 | 0.1976 |
| ft_dense | 0.2602 | 0.2920 | 0.3415 | 0.4040 | 0.4667 |
| ft_hybrid | 0.1934 | 0.2311 | 0.2840 | 0.3617 | 0.4326 |
| ft_hybrid_rerank | 0.1887 | 0.2233 | 0.2666 | 0.3459 | 0.4358 |
