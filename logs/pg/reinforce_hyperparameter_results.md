# REINFORCE hyperparameter results

| Run | Learning rate | Gamma | Episodes per update | Entropy bonus | Baseline | Eval return | Crash rate | Boxes | Observed behaviour |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 0.99 | 8 | 0.01 | yes | 0.9 | 0.97 | 1.5 | Baseline: rose quickly from -8 to about 0, then plateaued. Eval 0.9 and 1.5 boxes, but it crashed in 97% of games, so it learned to deliver but not to manage its battery. |
| 2 | 0.0003 | 0.99 | 8 | 0.01 | yes | -1.6 | 1.00 | 0.6 | Lr 3e-4: slowest start, with the lowest curve for the first 60k steps, as expected with a small step size. Eval -1.6 and only 0.6 boxes; it needs more steps than the others. |
| 3 | 0.003 | 0.99 | 8 | 0.01 | yes | 0.0 | 0.97 | 1.5 | Lr 3e-3: fast start but the jumpiest curve. Eval 0.0 and 1.5 boxes. The higher learning rate did not help. |
| 4 | 0.001 | 0.95 | 8 | 0.01 | yes | -10.6 | 0.70 | 1.1 | Gamma 0.95: eval -10.6, worse than the baseline, though the crash rate fell to 0.70. A shorter horizon means it cares less about the distant reward of going back to recharge. |
| 5 | 0.001 | 0.9 | 8 | 0.01 | yes | -29.4 | 0.03 | 2.3 | Gamma 0.90: the only run that mostly avoided crashes (0.03) but with the worst score (-29.4). Short-sighted: it stays safe but delivers inefficiently, and the empty-clinic penalty adds up. Boxes 2.3 is the best of all runs. |
| 6 | 0.001 | 0.99 | 4 | 0.01 | yes | 0.7 | 1.00 | 1.2 | 4 episodes per update: similar to the baseline (eval 0.7, crash 1.0). More frequent but noisier updates did not change the outcome. |
| 7 | 0.001 | 0.99 | 16 | 0.01 | yes | -3.0 | 0.93 | 1.0 | 16 episodes per update: eval -3.0 and fewer boxes (1.0). Smoother updates, but only half as many of them in the same number of steps. |
| 8 | 0.001 | 0.99 | 8 | 0.05 | yes | 1.8 | 1.00 | 1.5 | Entropy 0.05: eval 1.8, among the better runs. The extra randomness kept it exploring, but it still crashed in every game. |
| 9 | 0.001 | 0.99 | 8 | 0.01 | no | -0.7 | 1.00 | 1.2 | No baseline: eval -0.7, close to the baseline run, so normalising returns made little difference here. The curve is a bit noisier and ends lower than most. |
| 10 | 0.003 | 0.95 | 16 | 0.01 | yes | 2.7 | 1.00 | 1.6 | Combination (lr 3e-3, gamma 0.95, 16 episodes): best eval return (2.7) and the best curve at the end, but still a 100% crash rate. The best of the 10, but not by much. |
