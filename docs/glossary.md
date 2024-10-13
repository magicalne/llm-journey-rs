# Glossary

- [GGML - Large Language Models for Everyone](https://github.com/rustformers/llm/tree/main/crates/ggml) Explain terms like `hyperparameters`, `vocabulary`, `weights`, etc.
- [transformer glossary from huggingface](https://huggingface.co/docs/transformers/main/glossary) This glossary defines general machine learning and 🤗 Transformers terms to help you better understand the documentation.
- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [ Large Language Model (LLM)🤖: In and Out ](https://medium.com/towards-artificial-intelligence/large-language-model-llm-in-and-out-84a2d7361022) A great post about attention and ffn
- [ A Visual Walkthrough of DeepSeek’s Multi-Head Latent Attention (MLA) 🧟](https://pub.towardsai.net/a-visual-walkthrough-of-deepseeks-multi-head-latent-attention-mla-%EF%B8%8F-24f56586ca6a) And this one from the same author
- [llama.cpp #7519](https://github.com/ggerganov/llama.cpp/pull/7519/files) is a good reference to implement MLA.
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)

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
