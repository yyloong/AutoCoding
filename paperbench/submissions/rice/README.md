# RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation

This repository contains a reproduction of the RICE algorithm from the paper "RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation".

## Implementation Overview

This implementation includes:

1. **StateMask explanation method**: A mask network trained via PPO with reward R' = R + α * a_t^m (α = 0.0001)
2. **RND module**: Fixed target + trainable predictor with normalization for exploration bonus
3. **RICE algorithm**: Uses mixed initial state distribution (reset prob p = 0.25 or 0.5) and RND bonus (λ = 0.01)
4. **Baselines**: PPO fine-tuning, StateMask-R, simplified JSRL
5. **Environments**: MuJoCo (Hopper-v5, Walker2d-v5, Reacher-v4, HalfCheetah-v5) with dense/sparse rewards

## Directory Structure

- `rice/`: Main implementation directory
  - `statemask.py`: StateMask explanation method implementation
  - `rnd.py`: Random Network Distillation module
  - `rice_algorithm.py`: Main RICE algorithm implementation
  - `baselines.py`: Baseline methods (PPO fine-tuning, StateMask-R, JSRL)
  - `environments.py`: Environment wrappers and utilities
- `reproduce.sh`: Main reproduction script
- `requirements.txt`: Python dependencies

## How to Run

Execute the reproduction script:
```bash
bash reproduce.sh
```

The script will:
1. Train pre-trained policies for MuJoCo environments
2. Train StateMask explanation models
3. Run RICE refinement and baseline comparisons
4. Save results to the `results/` directory

## Expected Results

The implementation should reproduce the key findings from the paper:
- StateMask explanation method achieves comparable fidelity to the original with improved efficiency
- RICE outperforms baselines (PPO fine-tuning, StateMask-R, JSRL) in MuJoCo environments
- Mixed initial state distribution (p = 0.25 or 0.5) prevents overfitting
- RND exploration bonus (λ = 0.01) is crucial for performance improvement

## Limitations

- This reproduction focuses on the core MuJoCo environments mentioned in the paper (using v4/v5 versions compatible with modern mujoco package)
- Real-world applications (cryptocurrency mining, cyber defense, etc.) are not implemented due to complexity
- Some hyperparameters may need adjustment based on available computational resources
