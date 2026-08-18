<div align="center">

# Self-Healing Neural Network for Fault-Tolerant AI

An adaptive, lightweight framework for real-time detection and autonomous repair of hardware-induced neural network faults.
</div>

---

## Overview

Deep Neural Networks (DNNs) deployed in real-world Edge AI systems and resource-constrained environments are vulnerable to hardware-induced faults—such as memory bit-flips, voltage fluctuations, and environmental disturbances. Standard hardware redundancy techniques like Triple Modular Redundancy (TMR) or Error-Correcting Code (ECC) memory add significant computational, power, and memory overheads.

**Self-Healing Neural Network** addresses this challenge by introducing an inline fault-detection and self-repair mechanism operating during inference. By analyzing gradient dynamics and applying dynamic repair strategies, the framework restores model accuracy without requiring full retraining or heavy physical redundancy.

---

## Key Features

* **Runtime Fault Detection:** Rapidly pinpoints corrupted parameters during inference via gradient-based dynamics.
* **Multi-Strategy Healing Mechanism:** Supports 5 distinct recovery techniques targeting different fault topologies:
  * *Gradient-Based Weight Correction*
  * *Autoencoder-Based Weight Reconstruction*
  * *Neuron Regrowth Mechanics*
  * *Dynamic Computation Rerouting*
  * *Hybrid Ensembles*
* **Adaptive Healing Controller:** Dynamically selects the optimal repair strategy based on the detected fault severity.
* **Micro Fine-Tuning:** Performs minimal parameter updates (focused on target dense layers) for rapid stabilization.
* **Controlled Fault Injection:** Built-in simulation suite introducing $10\% - 40\%$ synthetic hardware fault rates.
* **Edge-First Design:** Optimized for low memory and low power consumption suitable for embedded systems.

---

## System Architecture

```text
               ┌───────────────────────┐
               │      Input Image      │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │  CNN Baseline Model   │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │ Fault Injection Layer │ (10% - 40% Weight Corruption)
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │ Gradient Fault Detect │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │ Adaptive Controller   │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │ Healing Strategy Exec │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │  Micro Fine-Tuning    │
               └───────────┬───────────┘
                           ▼
               ┌───────────────────────┐
               │ Recovered Model Output│
               └───────────────────────┘
