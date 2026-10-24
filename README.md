# Bigram

Character-level name generation with two models, compared on the same held-out data:

1. **Bigram model**: predicts the next character from the previous one, using smoothed counts.
2. **MLP language model**: predicts the next character from the previous 3, using learned character embeddings and a hidden `tanh` layer (in the spirit of Bengio et al., 2003), trained with PyTorch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishwakumar1311/Bigram/blob/main/Bigram.ipynb)

The dataset is the 32,033-name `names.txt` from Andrej Karpathy's *makemore* project, included in this repo. The approach follows his makemore lectures.

## Setup

- Names are shuffled (seed 42) and split 80% / 10% / 10% into train (25,626), dev (3,203) and test (3,204).
- `.` marks both the start and end of a name and is added exactly once per name when training examples are built.
- MLP: 3-character context, 10-dimensional embeddings, 300 hidden units, 27 outputs (about 18k parameters). 70,000 steps of mini-batch gradient descent, batch size 64, learning rate 0.1 and then 0.01 after step 50,000. The learning rate was picked with an exponential sweep (included in the notebook).

## Results

Average negative log-likelihood per character (lower is better; uniform guessing would score about 3.30):

| Model | Train | Dev | Test |
| --- | --- | --- | --- |
| Bigram (counts, +1 smoothing) | n/a | 2.453 | 2.458 |
| MLP, 3-character context | 2.026 | 2.094 | 2.094 |

The MLP improves on the bigram baseline by about 0.36, and its train and dev losses are close, so it is not overfitting much.

Sample names from the MLP: `carmah`, `amelle`, `khy`, `mili`, `kaleigh`, `deliah`, `jareei`, `nellara`.

The notebook also plots the learned character embeddings (first 2 of 10 dimensions).

## Run it

Open the notebook in Colab with the badge above, or locally:

```bash
pip install torch matplotlib jupyter
jupyter notebook Bigram.ipynb
```

The notebook reads `names.txt` from the working directory, and downloads it from this repo if it isn't there (for example on Colab).

## Limitations

- Fixed 3-character context, so it cannot model longer-range structure.
- Plain mini-batch gradient descent with a hand-set learning-rate schedule; no batch normalization or tuned initialization beyond a simple scaling.
- Generated names are plausible-sounding but not guaranteed to be real or distinct from names in the dataset.
