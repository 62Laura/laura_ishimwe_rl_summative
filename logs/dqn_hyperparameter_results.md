# DQN hyperparameter results

| Run | Learning rate | Gamma | Batch size | Epsilon decay (steps) | Target tau | Eval return | Crash rate | Boxes | Observed behaviour |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 0.99 | 64 | 30000 | 0.005 | 36.9 | 0.07 | 5.6 | Baseline (lr 0.001, gamma 0.99). It stayed flat for the first ~50k steps, then climbed to about 40 and scored 36.9 in evaluation, delivering 5.6 boxes per game. |
| 2 | 0.0005 | 0.99 | 64 | 30000 | 0.005 | 46.0 | 0.03 | 4.6 | Halving the learning rate to 0.0005 delayed the take-off (~65k steps) but it ended steadier and scored 46.0, better than the baseline. |
| 3 | 0.0001 | 0.99 | 64 | 30000 | 0.005 | 20.3 | 0.00 | 6.4 | The smallest learning rate (0.0001) learned far too slowly: its curve only reached about 10-15 by 150k steps and it scored the lowest (20.3). It was still improving, so it needs many more steps. |
| 4 | 0.001 | 0.95 | 64 | 30000 | 0.005 | 51.2 | 0.00 | 5.6 | Best run (51.2). A lower gamma of 0.95 made the agent focus on nearer rewards, which suits short delivery trips. It started improving early (~25k steps), reached about 40 by 60k steps and never crashed the drone. |
| 5 | 0.001 | 0.9 | 64 | 30000 | 0.005 | 46.7 | 0.10 | 5.6 | Gamma 0.90 learned fastest of all, rising from ~20k steps, and scored 46.7. But a crash rate of 0.10 shows it sometimes ignores the battery, since it looks too little into the future. |
| 6 | 0.001 | 0.99 | 128 | 30000 | 0.005 | 38.2 | 0.10 | 6.2 | A larger batch (128) learned early but scored only 38.2 and lost the drone in 10% of games. A bigger batch gives smoother updates, but here it did not give a better final policy. |
| 7 | 0.001 | 0.99 | 32 | 30000 | 0.005 | 41.6 | 0.03 | 5.7 | A smaller batch (32) gave noisier learning: the take-off came late (~75k steps), but it caught up and scored 41.6. |
| 8 | 0.001 | 0.99 | 64 | 60000 | 0.005 | 45.8 | 0.00 | 5.0 | Exploring for longer (60,000 steps) kept the curve flat until about 60k steps, because the agent acted randomly for longer. Once epsilon fell it rose quickly and scored 45.8 with no crashes. |
| 9 | 0.001 | 0.99 | 64 | 10000 | 0.005 | 48.8 | 0.03 | 6.9 | Very short exploration (10,000 steps) made the curve dip below zero at first, as the agent settled on a poor policy too early. It recovered after ~55k steps and scored 48.8, delivering the most boxes (6.9). |
| 10 | 0.0005 | 0.95 | 128 | 30000 | 0.01 | 46.0 | 0.13 | 5.0 | The combination (lr 0.0005, gamma 0.95, batch 128, tau 0.01) learned early like the other low-gamma runs and scored 46.0, but it lost the drone in 13% of games, the highest crash rate. Combining changes did not beat the single change in run 4. |
