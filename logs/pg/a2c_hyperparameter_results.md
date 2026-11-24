# A2C hyperparameter results

| Run | Learning rate | Gamma | Lambda | Batch (steps) | Entropy bonus | Eval return | Crash rate | Boxes | Observed behaviour |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0007 | 0.99 | 0.95 | 256 | 0.01 | -18.1 | 0.37 | 1.9 | Baseline (lr 7e-4, gamma 0.99, lambda 0.95, 256-step batches): learned slowly. Eval -18.1, 1.9 boxes, 37% crashes. The curve is low and noisy, so A2C was still weak after 150k steps. |
| 2 | 0.0003 | 0.99 | 0.95 | 256 | 0.01 | -2.6 | 0.90 | 1.3 | Lr 3e-4: the slowest learner, with the curve barely above zero. Eval -2.7, 1.3 boxes, 90% crashes. The step size is too small for the budget, so it never learned battery management. |
| 3 | 0.002 | 0.99 | 0.95 | 256 | 0.01 | 0.9 | 0.97 | 1.7 | Lr 2e-3: a faster start but no real improvement. Eval 0.9, 1.7 boxes and 97% crashes. The higher learning rate did not help, and A2C's single update per batch is noisy. |
| 4 | 0.0007 | 0.95 | 0.95 | 256 | 0.01 | -7.0 | 0.03 | 4.7 | Gamma 0.95: the clearest improvement over the baseline among single changes. Eval -7.0, 4.7 boxes and only 3% crashes, and the curve rises after 80k steps. A shorter horizon gives a clearer credit signal, as it did for PPO. |
| 5 | 0.0007 | 0.9 | 0.95 | 256 | 0.01 | -21.1 | 0.07 | 2.7 | Gamma 0.90: very safe (7% crashes) with 2.7 boxes, but eval -21.1. It is too short-sighted, so it delivers and avoids crashes without using its steps well, and the empty-clinic penalty adds up. |
| 6 | 0.0007 | 0.99 | 0.95 | 128 | 0.01 | -17.4 | 0.43 | 1.6 | 128-step batches: more frequent but noisier updates. Eval -17.4, 1.6 boxes, 43% crashes. Not better than the baseline, so smaller batches did not help. |
| 7 | 0.0007 | 0.99 | 0.95 | 512 | 0.01 | 1.5 | 1.00 | 1.3 | 512-step batches: smoother updates but fewer of them. Eval 1.5 and 1.3 boxes, with a 100% crash rate. In this step budget it learned very little. |
| 8 | 0.0007 | 0.99 | 0.95 | 256 | 0.05 | -8.8 | 0.60 | 1.9 | Entropy 0.05: eval -8.8, 1.9 boxes, 60% crashes. The extra randomness did not fix the crashes. The result is slightly better than the baseline, but the curve is no better. |
| 9 | 0.0007 | 0.99 | 0.8 | 256 | 0.01 | -39.8 | 0.00 | 1.0 | Lambda 0.8: 0% crashes but only 1.0 box and the worst eval (-39.8). A less noisy but more biased advantage made the drone stay safe without delivering. |
| 10 | 0.002 | 0.95 | 0.95 | 512 | 0.01 | 15.2 | 0.07 | 6.7 | Combination (lr 2e-3, gamma 0.95, 512-step batches): the best A2C run. Eval 15.2, 6.7 boxes (the most of any run), 7% crashes, and the highest curve from 80k steps. Gamma 0.95 gave the main improvement, and the larger batch made the faster learning rate stable. |
