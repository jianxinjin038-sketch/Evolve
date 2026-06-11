# EVOLVE: Multivariate Time Series Anomaly Prediction with Uncertainty-Aware, Adaptive Prototype Memories

## Introduction
Unsupervised multivariate time series anomaly prediction (AP) aims to estimate future anomaly risks from current observations, providing early-warning signals for proactive intervention and maintenance in complex systems. Existing unsupervised AP methods follow a perturbation-based paradigm, where predefined or random perturbations are injected into historical windows to simulate abnormal future evolution. However, these methods share perturbations across different variables and contexts, and learn deterministic history-to-future evolution. In reality, anomaly evolution is variable- and context-dependent: the same perturbation may be amplified into abnormal future dynamics in one variable or context, while being absorbed as normal fluctuation in another. In addition, future evolution is not uniquely determined by a historical window, since similar histories may lead to multiple plausible trajectories, such as  gradual degradation or abrupt changes. To address these problems, we propose $\textbf{EVOLVE}$, a unified framework that models unsupervised AP via variable- and context-conditioned perturbations and uncertainty-aware future reasoning. EVOLVE constructs context-specific anomaly perturbations through an adaptive anomaly prototype memory, which can dynamically update and retrieve relevant perturbations at the variable level. EVOLVE models uncertainty by sampling multiple latent historical states and decoding through complementary smooth and deviation-amplifying evolution hypotheses, and the disagreement among decoded future trajectories is used to estimate future anomaly risk. Extensive experiments on six real-world benchmark datasets demonstrate that EVOLVE consistently outperforms state-of-the-art anomaly prediction baselines.


<div style="text-align: center;">
    <img src="docs/overall_architecture.png" alt="Evolve" style="zoom:80%;" />
</div>

## Quickstart

### Installation
Create and activate the Conda environment:

   ```bash
   conda create -n Evolve python=3.8 -y
   conda activate Evolve
   ```
Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data preparation
The pre-processed MSL and GECCO datasets are already included in the `./dataset` folder.

## Train and evaluate model
- To see the model structure of Evolve, [click here](./ts_benchmark/baselines/Evolve/models/Evolve_model.py).

- For example you can reproduce a experiment result as the following:

```bash
bash ./scripts/multivariate/label/Evolve.sh
```

## Results
We systematically evaluated our method on six public multivariate datasets with a fixed 96-step look-back window. Results are averaged across four prediction horizons (32, 64, 128, 192) and three independent runs to ensure reliability:

<div style="text-align: center;">
    <img src="docs/main_results.png" alt="Evolve" style="zoom:80%;" />
</div>
