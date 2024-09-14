---
title: "Day One"
date: 2024-09-11
---

# Day One

The first milestone is running inference with CPU. A good start with `DeepSeek-V2-Lite (MoE-16B)`.

## Useful links:

- [GGML - Large Language Models for Everyone](https://github.com/rustformers/llm/tree/main/crates/ggml) Explain terms like `hyperparameters`, `vocabulary`, `weights`, etc.

## Sweet Claude makes a plan for my day

 For Day 1, you'll be focusing on foundational concepts in neural networks, deep learning, and transformer models. Here's a detailed breakdown of the materials and resources you'll need, along with explanations for each topic:
Day 1 Materials:

Neural Networks and Deep Learning Fundamentals (4 hours)

Materials:

Online course: "Neural Networks and Deep Learning" by Andrew Ng (Coursera)
Book: "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (particularly chapters 1-4)
Article: "A Beginner's Guide to Neural Networks and Deep Learning" on AI Wiki(I found the one: https://wiki.pathmind.com/neural-network)
(Also found this link: http://neuralnetworksanddeeplearning.com/)

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

(More:)
- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

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

--- 

Mixture of Experts (MoE) Models (2 hours)

Materials:

Paper: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" by Shazeer et al. (2017)
Article: "Understanding Mixture of Experts Architectures" on Hugging Face blog
Paper: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" by Lepikhin et al. (2020)
Video: "Mixture of Experts Explained" by Yannic Kilcher on YouTube

Explanation:
This section introduces the concept of Mixture of Experts (MoE) models, which have become increasingly important in scaling language models:

Basic concept of MoE: combining multiple "expert" networks
Gating mechanisms for selecting experts
Sparsity in MoE models
Scaling advantages of MoE architectures
Applications in large language models
Challenges in training and deploying MoE models

Understanding MoE models is crucial for your project as they represent a significant direction in LLM research, offering ways to scale models to unprecedented sizes while managing computational resources efficiently.
Key concepts to focus on:

The structure of an MoE layer
How MoE differs from traditional dense networks
The role of the gating network
Load balancing and expert utilization
Inference considerations for MoE models

Additional MoE-specific resources:

Paper: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" by Fedus et al. (2021)
This paper introduces Switch Transformers, a simplified MoE approach that's particularly relevant for your inference project.
Blog post: "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale" on Microsoft's DeepSpeed blog
This provides insights into practical implementations and optimizations of MoE models.
GitHub repository: "fairseq" by Facebook Research
Examine the implementation of MoE models in this popular NLP library to get ideas for your Rust implementation.

As you go through these materials, I recommend:

Sketching the architecture of an MoE model
Comparing the computational graphs of a standard Transformer and an MoE Transformer
Thinking about how you might implement expert selection in Rust
Considering the memory implications of MoE models during inference

By adding this MoE-specific content to Day 1, you'll be better prepared to incorporate these advanced concepts into your Rust-based LLM inference project. The MoE architecture will likely influence your design decisions in the later phases of the project, especially when you're optimizing for performance and scalability.
Remember, MoE models introduce additional complexity, especially in terms of routing and load balancing. As you progress through your project, you'll need to decide how to handle these aspects efficiently in your Rust implementation.

## Sketching the architecture of an MoE model

graph TD
    I[Input] --> E[Embedding Layer]
    E --> FF1[Feed Forward Layer]
    FF1 --> A[Attention Layer]
    A --> G[Gating Network]
    G --> |Route 1| E1[Expert 1]
    G --> |Route 2| E2[Expert 2]
    G --> |Route 3| E3[Expert 3]
    G --> |Route N| EN[Expert N]
    E1 --> C[Combine Outputs]
    E2 --> C
    E3 --> C
    EN --> C
    C --> FF2[Feed Forward Layer]
    FF2 --> O[Output]

## Comparing computational graphs of a standard Transformer and an MoE Transformer

graph TD
    subgraph "Standard Transformer"
    I1[Input] --> E1[Embedding]
    E1 --> A1[Self-Attention]
    A1 --> N1[Layer Norm]
    N1 --> FF1[Feed Forward]
    FF1 --> N2[Layer Norm]
    N2 --> O1[Output]
    end

    subgraph "MoE Transformer"
    I2[Input] --> E2[Embedding]
    E2 --> A2[Self-Attention]
    A2 --> N3[Layer Norm]
    N3 --> G[Gating Network]
    G --> |Route| EX[Expert Layer]
    EX --> N4[Layer Norm]
    N4 --> O2[Output]
    end
