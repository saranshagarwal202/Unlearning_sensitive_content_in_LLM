# Selective Unlearning for Large Language Models

This project implements and evaluates selective unlearning techniques for Large Language Models, specifically focusing on removing sensitive personal information while retaining other similar data patterns. The implementation uses the OLMO 7B model as a base architecture.

## Overview

The project explores three different unlearning algorithms to selectively remove learned information from a pre-trained language model while preserving desired knowledge. Our approach focuses on maintaining model performance on retained information while effectively forgetting targeted sensitive content.

## Architecture

The project is structured into two main components:

### 1. Evaluate
Contains evaluation scripts that measure model performance using:
- Rouge Score metrics
- Perplexity measurements

### 2. Unlearn
Implements three distinct unlearning algorithms:
1. Basic Gradient Ascent (GA) on forget set
2. GA on forget set + KL Divergence preservation on retain set
3. Combined GA/GD approach (Primary Method)

## Primary Unlearning Method

Our main approach uses a combined loss function that simultaneously:
- Maximizes loss on forgotten data (gradient ascent)
- Minimizes loss on retained data (gradient descent)

The loss function is formally defined as:

```
L_total = -CE_forget + CE_retain

where:
CE_forget = Cross Entropy Loss on forget set
CE_retain = Cross Entropy Loss on retain set
```

We minimize this combined loss function during training.

## Results

### Training Progress
![Loss over epochs](loss_epochs.jpeg)

### Performance Metrics
![ROUGE-L Score over epochs](rouge_l_epochs.jpeg)

## Hardware Requirements

The model was trained and evaluated using:
- 2x NVIDIA A100 GPUs
- CUDA compatibility required

## Project Structure
```
.
├── evaluate/
│   └── evaluate_lora_checkpoints.py
├── unlearn/
│   ├── GA_only.py
│   ├── GA_plus_KL_divergence.py
│   └── GA_plus_GD.py
├── loss_epochs.jpeg
├── rouge_l_epochs.jpeg
└── README.md
```

## Usage

To run the primary unlearning algorithm (GA+GD):

```bash
python unlearn/GA_plus_GD.py \
    --model_dir /path/to/pretrained/model \
    --output_dir /path/to/save/model \
    --forget_dir /path/to/forget/dataset \
    --retain_dir /path/to/retain/dataset \
    --trainable_layers all
```

### Arguments
- `model_dir`: Path to pretrained model directory
- `output_dir`: Directory to save the unlearned model
- `forget_dir`: Directory containing the dataset to be forgotten
- `retain_dir`: Directory containing the dataset to be retained
- `trainable_layers`: Layers to train (default: 'all')

## Citation

This work is currently in progress. A formal citation will be provided upon publication of the research paper.

> **Note**: This is an active research project. Methods and results are subject to change as the research progresses.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
