# CustomTransformerChatbot-100M-Parameters

# AIM

***To develop a Transformer-based chatbot model integrating Rotary Positional Embeddings (RoPE) and optimized attention mechanisms for improved conversational quality.***

---

# OBJECTIVES

1. **Design a modified Transformer architecture** that enhances word–position interactions for more context-aware responses.  
2. **Train the chatbot on conversational datasets** using a custom tokenizer and transformer architecture.  
3. **Evaluate chatbot performance** using training metrics and model outputs.  
4. **Provide an interface for interacting with the trained chatbot.**

---
# MODEL ARCHITECTURE

The chatbot is implemented using a **custom Transformer architecture built from scratch**.

### Key Components

- **Transformer architecture**
- **Rotary Positional Embeddings (RoPE)**
- **Multi-head self-attention**
- **Custom tokenizer**
- **PyTorch implementation**

The design focuses on **improving the relationship between tokens and positional information** to generate more context-aware responses.

---

# REPOSITORY STRUCTURE

The repository is organized into two main sections:

## 1. First Model — Initial Training Version

## 2. Best Model — Improved Version After Additional Training

---

# FIRST MODEL (Initial Training)

This section contains files related to the **first trained version of the chatbot model**.

### Files

**config.json**

Configuration file containing model parameters such as:

- architecture settings  
- embedding dimensions  
- number of layers  
- training hyperparameters

---

**tokenizer.json**

Custom tokenizer used to convert input text into tokens before feeding them into the model.

---

**conv-model-training.ipynb**

Main notebook used for **training the first version of the chatbot model**.

---

**Model Metrics Visualization**

A visual representation showing **training performance and metrics after approximately 8 hours of training**.

---

**firstmodel.pth**

Saved weights of the first trained model.

**Due to file size limitations, this file is stored on Hugging Face.**

---

# BEST MODEL (Improved Version)

The remaining files correspond to the **best-performing version of the chatbot model**, obtained after longer training.

### Files

**conv_model_training.ipynb**

Main notebook used for **training the improved model**.

---

**CHATBOT.ipynb**

Notebook used for **testing the chatbot architecture and inference process**.

---

**chat_ui.ipynb**

Notebook demonstrating the **chatbot user interface structure for testing**.

---

**chat_ui.py**

Python script allowing **users to interact with the chatbot through a simple interface**.

---

**config.json**

Configuration file containing **architecture and training parameters used for the final model**.

---

**tokenizer.json**

Tokenizer used during **training and inference of the final model**.

---

**training_stats.json**

Contains **training statistics such as loss values and training progress**.

---

**model output after 12 hours of training.png**

Visualization showing **training metrics after approximately 12 hours of training**.

---

**bestmodel.pth**

Saved weights of the **best-performing model**.

---

**model_epoch_40.pth**

Model weights saved at the **final training epoch (Epoch 40)**.

---

**model_weights.pth**

Additional model weights required to **run the testing and chatbot interaction scripts**.

**These files are hosted on Hugging Face because of their large size.**

---

# MODEL WEIGHTS

Some model files are too large to store directly in this repository.

Download them from **Hugging Face** and place them in the same directory as the testing scripts.

Required files:
firstmodel.pth  (for first model testing only).
bestmodel.pth
model_epoch_40.pth
model_weights.pth

---

# NOTES

Some scripts require **model weight files hosted on Hugging Face**.

Ensure all `.pth` files are downloaded before running the chatbot.

---
