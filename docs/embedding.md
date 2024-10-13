## Embeddings in LLMs

### What Are Embeddings?

**Embeddings** are vector representations of words, tokens, or other textual units that capture semantic and syntactic information. Instead of treating words as discrete symbols (e.g., "cat" as a string), embeddings represent them as continuous vectors in a high-dimensional space. These vectors can be thought of as points in a multi-dimensional space where similar words are located closer to each other.

### How Embeddings Work

1. **Word Embeddings**:
   - **Definition**: Word embeddings are vector representations of individual words. Each word is mapped to a fixed-size vector, typically in a high-dimensional space (e.g., 300 dimensions).
   - **Example**: The word "cat" might be represented as a vector `[0.2, -0.5, 0.8, ...]`.
   - **Training**: Word embeddings can be trained using various algorithms like Word2Vec, GloVe, or FastText. These algorithms learn the embeddings by analyzing the context in which words appear in large corpora.

2. **Token Embeddings**:
   - **Definition**: In the context of LLMs, tokens are the basic units of text after tokenization (e.g., words, subwords, or characters). Token embeddings are vector representations of these tokens.
   - **Example**: After tokenizing the sentence "The cat sat on the mat," each token (e.g., "The", "cat", "sat") is represented as a vector.
   - **Training**: Token embeddings are typically learned during the training of the LLM itself, often as part of the model's initial layers.

3. **Position Embeddings**:
   - **Definition**: Since Transformer models like GPT do not have a built-in notion of sequence order (they process tokens in parallel), position embeddings are added to capture the order of tokens in a sequence.
   - **Example**: If the sentence "The cat sat on the mat" is tokenized into ["The", "cat", "sat", "on", "the", "mat"], position embeddings are added to each token to indicate its position in the sequence.
   - **Training**: Position embeddings are usually learned parameters of the model, or they can be generated using predefined functions (e.g., sinusoidal functions).

TODO: rotary embedding
