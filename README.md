# LLM Internal Physics Benchmark (IPB)

This project introduces the Internal Physics Benchmark (IPB), a suite of tools to analyze and quantify the internal dynamics of Large Language Models (LLMs). Instead of just measuring what a model *does* (its output), we measure *how* it does it by observing its behavior during backpropagation.

This provides a "physical personality" profile for each model, revealing its computational efficiency, stability, and internal organization. The results are visualized in a final radar chart, offering an at-a-glance comparison of different models' architectural character.

## The "Internal Physics" Metaphor

We treat the backpropagation process as a physical system. The gradient of the loss is a signal or "energy" that flows backward through the network. Each layer acts as a medium that transforms this signal. By measuring how the signal changes at each layer, we can deduce properties of the system as a whole.

## The 5 Core Metrics

The benchmark is composed of five novel metrics. For each metric, a lower score indicates more efficiency, stability, or structure. For visualization on the radar chart, the scores are inverted and normalized, so a larger area is better.

| Metric | Measures | Interpretation (Low Score) | Interpretation (High Score) | Analogy |
| :--- | :--- | :--- | :--- | :--- |
| **EC** (Error of Conservation) | Signal transformation intensity per layer. | **Transparent**: Layers preserve the gradient signal. | **Transformative**: Layers aggressively alter the signal. | Careful Restorer vs. Re-painter |
| **ISC** (Index of Computational Strain) | Total gradient "energy" spent per unit of loss. | **Efficient**: Low internal effort for the result. | **Wasteful**: High internal effort for the result. | Well-oiled Machine vs. Strained Engine |
| **ISG** (Index of Magnitude Sync) | Consistency of transformation *intensity* across layers. | **Homogeneous**: All layers work equally hard. | **Specialized**: Some layers are "hot spots". | Roman Phalanx vs. Swarm of Bees |
| **ISA** (Index of Angular Sync) | Consistency of transformation *direction* across layers. | **Coherent**: All layers push in the same direction. | **Flexible**: Layers push in diverse directions. | Crystal vs. Ecosystem |
| **ISF** (Index of Field Structure) | How well global gradient correlations fit a power law. | **Organized**: Hierarchical, long-range structure. | **Disorganized**: Local, random-like correlations. | Gravitational Field vs. Random Gas |


## How to Run

### 1. Prerequisites
This project requires Python 3.8+ and a CUDA-enabled GPU with sufficient VRAM for the models you wish to test. 4-bit quantization is used to lower VRAM requirements, but models like `Mistral-7B` still require ~6-7 GB of VRAM.

### 2. Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/your-username/llm-internal-physics.git
cd llm-internal-physics
pip install -r requirements.txt
