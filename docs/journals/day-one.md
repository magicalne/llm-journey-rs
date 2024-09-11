---
title: "Day One"
date: 2024-09-11
---

# Day One

The first milestone is running inference with CPU. A good start with `DeepSeek-V2-Lite (MoE-16B)`.

## Useful links:

- [GGML - Large Language Models for Everyone](https://github.com/rustformers/llm/tree/main/crates/ggml) Explain terms like `hyperparameters`, `vocabulary`, `weights`, etc.

## Cute Claude made a plan for my day

 For Day 1, you'll be focusing on foundational concepts in neural networks, deep learning, and transformer models. Here's a detailed breakdown of the materials and resources you'll need, along with explanations for each topic:
Day 1 Materials:

Neural Networks and Deep Learning Fundamentals (4 hours)

Materials:

Online course: "Neural Networks and Deep Learning" by Andrew Ng (Coursera)
Book: "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (particularly chapters 1-4)
Article: "A Beginner's Guide to Neural Networks and Deep Learning" on AI Wiki

Explanation:
This section will cover the basics of neural networks, including:

Neurons and activation functions
Feedforward networks
Backpropagation and gradient descent
Common architectures (MLPs, CNNs, RNNs)
Loss functions and optimization algorithms

Understanding these concepts is crucial as they form the building blocks of more complex models like LLMs. Your previous experience with sklearn, xgboost, and lightgbm will help you grasp these concepts more quickly, as you're already familiar with some machine learning principles.

Transformer Model Architecture (4 hours)

Materials:

Paper: "Attention Is All You Need" by Vaswani et al. (2017)
Article: "The Illustrated Transformer" by Jay Alammar
Video: "Transformer Neural Networks - EXPLAINED! (Attention is all you need)" by CodeEmporium on YouTube
Tutorial: "The Annotated Transformer" by Harvard NLP

Explanation:
This section focuses on the architecture that revolutionized NLP and forms the basis of modern LLMs:

Overview of the transformer architecture
Encoder and decoder structures
Multi-head attention mechanism
Positional encoding
Feed-forward networks within transformers
Layer normalization and residual connections

Understanding the transformer architecture is crucial for your project, as it will directly inform how you implement the inference engine in Rust. Pay special attention to the attention mechanism, as it's a key component you'll be implementing in the later stages of your project.
Additional Resources:

Rust Programming Language book (official documentation)
While you have a background in Rust, it might be helpful to keep this as a reference, especially for the sections on performance and unsafe Rust, which might come in handy when optimizing your inference engine.
Linear Algebra refresher
A good understanding of linear algebra is crucial for implementing LLMs. Khan Academy or MIT OpenCourseWare have excellent resources if you need to brush up on concepts like matrix multiplication, vector spaces, and eigenvalues/eigenvectors.
"Mathematics for Machine Learning" by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong
This book provides a solid mathematical foundation for machine learning, which will be valuable as you dive deeper into LLM architectures.
Hugging Face's Transformers library documentation
While you're building your own implementation, looking at how established libraries structure their code can provide valuable insights.

As you go through these materials, I recommend:

Taking notes, especially on the transformer architecture
Trying to sketch out the basic structure of a transformer model
Writing pseudocode for key operations like self-attention

This will help solidify your understanding and prepare you for the implementation phase. Remember, the goal of Day 1 is to build a strong theoretical foundation. Don't worry if you don't understand everything perfectly - you'll be applying these concepts in the coming days, which will reinforce your learning.

