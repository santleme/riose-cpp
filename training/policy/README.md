# Synthetic forage and comfort policy

This experiment trains a small DQN for a Gymnasium `reset`/`step` environment. The 9 normalized observation fields are hunger, thirst, fatigue, heat load, grass, water, shade, neighbor presence, and social stress. Actions are graze, drink, rest, seek shade, and follow the herd.

## Kaggle run v2

- Kernel: [Campo Vivo Synthetic Grazing Policy](https://www.kaggle.com/code/pascalcoimbra/campo-vivo-synthetic-grazing-policy), private kernel version 2.
- Accelerator: NVIDIA Tesla T4 (`cuda`, confirmed at runtime); GPU and internet enabled; Gymnasium 1.2.1 available.
- Training: 50,000 environment steps, 12,373 optimizer updates, seed 41, horizon 240, DQN MLP 9→64→64→5.
- Evaluation: 20 held-out episode seeds 1,000,000–1,000,019. DQN and heuristic are each evaluated on the same seed set; training uses a different seed sequence.
- DQN return: -17.350 ± 3.417; heuristic return: -50.199 ± 4.880; paired return delta (DQN minus heuristic): +32.849 ± 2.857. Final mean need (hunger, thirst, fatigue, heat load, social stress): DQN 0.133 ± 0.021, heuristic 0.307 ± 0.013. Values are mean ± sample standard deviation over 20 episodes.
- Outputs: [`evaluation-v2.json`](evaluation-v2.json), [`kaggle-v2.log`](kaggle-v2.log), [`policy-v2.pt`](policy-v2.pt), and browser-portable [`policy_weights-v2.json`](policy_weights-v2.json).

The baseline is an explicit one-step utility heuristic using the authored action response rates. The learned policy scored better on this particular synthetic environment and fixed seed set; this is not evidence of superiority in real cattle or in the visual application. Reward combines `0.35 * (potential_after - potential_before) + 0.65 * potential_after`, resource depletion cost, unavailable-action penalty, and additional penalties for high hunger, thirst, and heat. Potential weights and all transition coefficients are visible in `train_policy.py`.

## Scope and integration

This environment is intentionally synthetic. Need accrual, grass and water regeneration, cooling under shade, social effects, transition rates, and reward weights are authored game mechanics. They have not been fitted to observed cattle outcomes, and this policy does not establish causal responses to farm changes or biological validity. It is a policy-learning systems experiment only.

`infer.ts` is dependency-free browser inference for the exported JSON MLP and returns one of the five action names. `src/learned-policy.ts` loads the exported weights into the game. `src/simulation.ts` supplies normalized proxies for fatigue, shade, neighbor presence, and social stress, and uses the DQN as its current game policy with the interpretable utility policy as an error fallback. These mappings are game approximations, not measured bovine states; this integration does not extend the training result's scientific scope.

To reproduce locally with dependencies installed: `python train_policy.py --steps 50000 --seed 41 --eval-episodes 20 --output-dir outputs`. Kaggle uses `kernel-metadata.json` and installs `requirements.txt` when needed.
