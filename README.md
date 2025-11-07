# 🧠 Deep Learning Challenge — CNN + LSTM from Scratch

This project is part of a deep learning challenge designed to test **fundamental understanding and coding skills** — no shortcuts allowed! 🚀

---

## 🎯 Objective

Build a **Convolutional Neural Network (CNN)** combined with a **Long Short-Term Memory (LSTM)** model **entirely from scratch**, including:
- Manual implementation of **forward** and **backward propagation**
- Implementation of convolution operations, pooling, ReLU activation, and fully connected layers
- Sequence modeling using an LSTM cell coded from the ground up
- Text prediction task on a custom **medical corpus about breast cancer**


---

## 🧩 Project Overview

### 🏗️ 1. Preprocessing
**Goal:** Convert raw text sentences into numerical sequences usable by neural networks.

- **`build_vocab()`**  
  Builds a vocabulary from the corpus and assigns each word a unique integer index.  
  Special tokens:  
  - `<PAD>` → padding  
  - `<UNK>` → unknown words  

- **`texts_to_sequences()`**  
  Converts text sentences into sequences of word indices.

- **`create_examples()`**  
  Generates `(X, Y)` training pairs for next-word prediction tasks.  
  Example:  
  `["breast", "cancer", "screening"] → predict("methods")`

---

### 🧱 2. Model Components (NumPy Implementations)

#### 🔸 Convolution Layer — `Conv1D`
Manually implements 1D convolution:
- Sliding filters over embeddings
- Weight updates with backpropagation
- Includes **ReLU activation** and **global max pooling**

#### 🔸 Dense Layer — `Dense`
A fully connected layer implemented using:
- Matrix multiplication: `y = xW + b`
- Backpropagation via the chain rule

#### 🔸 LSTM Layer — `LSTM`
Implements a recurrent neural network that learns temporal dependencies:
- Manual computation of **input**, **forget**, **output** gates
- **Cell state** and **hidden state** updates
- Backpropagation Through Time (BPTT)

---

### ⚙️ 3. Training Functions

#### 🧠 `train_cnn()`
Trains a standalone CNN for text prediction:
- Learns from context windows of size `seq_len`
- Uses ReLU + Global Max Pooling + Softmax
- Optimized with **stochastic gradient descent (SGD)**

#### 🔄 `train_cnn_lstm()`
Trains a **hybrid CNN + LSTM** model:
- CNN extracts spatial features from word embeddings
- LSTM captures sequential dependencies
- Dense layer predicts the next word
- Tracks training & validation metrics (`loss` and `accuracy`)

---

### 🔍 4. Prediction

**`predict_next()`**  
Given a list of words (prefix), predicts the next most probable tokens:  
```python
predict_next(model, ["breast", "cancer", "is"], top_k=5)

---

## 🚀 How to Run
1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dEY450KxEOGVozh8mxL0GibG1IhmYvX_)
2. Clone this repository:

   ```bash
   git clone https://github.com/meliisse/CNN-LSTM-from-scratch.git
   cd CNN-LSTM-from-scratch
   

