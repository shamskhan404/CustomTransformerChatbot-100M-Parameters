#CustomTransformerChatbot-100M-Parameters
Overview

This project implements a Transformer-based chatbot trained from scratch with approximately 100 million parameters. The model integrates Rotary Positional Embeddings (RoPE) and optimized attention mechanisms to improve contextual understanding in conversations.

The repository contains:

The initial trained model

The improved best-performing model

Training notebooks

Testing notebooks

A simple chatbot interaction interface

Training metrics and statistics

Large model weight files are hosted separately on Hugging Face due to their size.

Aim

To develop a Transformer-based chatbot model integrating Rotary Positional Embeddings (RoPE) and optimized attention mechanisms for improved conversational quality.

Objectives

To design a modified Transformer architecture that enhances word–position interactions for more context-aware responses.

To train the chatbot on conversational datasets using a custom tokenizer and transformer architecture.

To evaluate chatbot performance using training metrics and model outputs.

To create a testing interface for interacting with the trained chatbot.

Repository Structure

The repository is organized into two main parts:

First Model (Initial Training Version)

Best Model (Improved Version After Additional Training)

First Model (Initial Training)

This section contains files related to the first trained version of the chatbot model.

Files

config.json
Contains configuration parameters used during model training such as architecture settings, embedding dimensions, number of layers, and other hyperparameters.

tokenizer.json
Custom tokenizer used to convert input text into tokens before feeding them into the model.

conv-model-training.ipynb
Main notebook used for training the first version of the chatbot model.

Model Metrics Visualization
A visual representation showing training performance and metrics after approximately 8 hours of training.

firstmodel.pth
Saved weights of the first trained model.

Due to file size limitations, this model file is stored on Hugging Face.

Best Model (Improved Version)

The remaining files in the repository correspond to the best-performing version of the chatbot model, trained for a longer duration and achieving better results.

Files

conv_model_training.ipynb
Main notebook used for training the improved model.

CHATBOT.ipynb
Notebook used to test the chatbot structure and inference process.

chat_ui.ipynb
Notebook that demonstrates the chatbot user interface structure for testing.

chat_ui.py
Python script that allows users to interact with the chatbot through a simple interface.

config.json
Configuration file containing architecture and training parameters for the final model.

tokenizer.json
Tokenizer used during training and inference of the final model.

training_stats.json
File containing recorded training statistics such as training loss and progress.

model output after 12 hours of training.png
Visualization of model training metrics after approximately 12 hours of training.

bestmodel.pth
Weights of the best-performing trained model.

model_epoch_40.pth
Model weights saved at the final training epoch (epoch 40).

model_weights.pth
Additional model weights required to run the testing and chatbot interaction scripts.

Due to large file sizes, these files are hosted on Hugging Face and must be downloaded separately.

Model Weights

Some model weight files are not stored directly in this repository because of their size.

Download them from Hugging Face and place them in the same directory as the testing scripts.

Required files include:

firstmodel.pth

bestmodel.pth

model_epoch_40.pth

model_weights.pth

Hugging Face model repository:

[Insert Hugging Face link here]
Running the Chatbot
1. Clone the repository
git clone https://github.com/yourusername/CustomTransformerChatbot-100M-Parameters.git
cd CustomTransformerChatbot-100M-Parameters
2. Install dependencies
pip install torch numpy
3. Download required model weights

Download the .pth files from Hugging Face and place them in the repository directory.

4. Run the chatbot interface
python chat_ui.py
Model Architecture

The chatbot is implemented using a custom Transformer architecture built from scratch.

Key components include:

Transformer architecture

Rotary Positional Embeddings (RoPE)

Multi-head self-attention

Custom tokenizer

PyTorch implementation

The design focuses on improving the relationship between tokens and positional information to generate more context-aware responses.

Training Details

First Model

Training Time: ~8 hours

Output: firstmodel.pth

Best Model

Training Time: ~12 hours

Final Epoch: 40

Output: bestmodel.pth

Training statistics and visualizations are included in the repository.

Notes

Some testing scripts require model weight files that are hosted on Hugging Face.
Ensure all required .pth files are downloaded before running the chatbot.

Author

Shams Tabrez Khan
