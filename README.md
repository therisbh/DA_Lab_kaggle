- **Name**: RISHABH GUPTA
- **Roll Number**: DA25M024

# Metric-Based Text Scoring Model

A deep learning solution for evaluating text responses across multiple evaluation metrics using large-capacity neural networks.

## Overview

This project implements a neural network model that scores text responses on a 1-10 scale based on different evaluation metrics. The solution uses pre-trained embeddings and achieves strong performance through balanced data augmentation and a large-capacity architecture.

## Requirements

```bash
pip install torch transformers scikit-learn numpy pandas tqdm
```

**Required versions:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU recommended

## Setup

### 1. HuggingFace Token

This project uses Google's `embeddinggemma-300m` model for text embeddings. You need a HuggingFace token with access to this model.

**Get your token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" access
3. Accept the license at https://huggingface.co/google/embeddinggemma-300m

**Set the token:**

```bash
export HF_TOKEN="your_token_here"
```


## Usage

### Complete Pipeline

Run the entire pipeline with a single command:

```bash
kaggle_solution.ipynb
```

This will:
1. Load training and test data
2. Generate Gemma embeddings (requires HF token)
3. Create balanced augmented dataset (50K samples)
4. Train neural network model
5. Generate predictions
6. Create `submission.csv`

### Configuration

Edit the `Config` class in `kaggle_solution.ipynb` to adjust parameters:

```python
class Config:
    TOTAL_SAMPLES = 50000        # Augmented training samples
    BATCH_SIZE = 256             # Training batch size
    EPOCHS = 30                  # Training epochs
    LEARNING_RATE = 1e-4         # Learning rate
    HIDDEN_DIMS = [1024, 768, 512, 256]  # Model architecture
```

## Model Architecture

**Large Capacity Neural Network:**
- **Input:** Concatenated metric and text embeddings
- **Hidden Layers:** 1024 → 768 → 512 → 256 neurons
- **Output:** 10-class softmax (scores 1-10)
- **Parameters:** ~5 million
- **Regularization:** Batch normalization, dropout (30-20%)

## Data Augmentation

The model uses a balanced augmentation strategy:
- Target: 50,000 training samples
- Distribution: Balanced across scores 1-10
- Method: Oversampling/undersampling based on original distribution
- Ensures all score classes are well-represented




