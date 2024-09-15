---
title: "Day Four"
date: 2024-09-14
---

# Day Four

Clearly I didn't finish the target of day three. So I continue on the work of yesterday.

## RoPE

I need to read [ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING](https://arxiv.org/pdf/2104.09864)
And the [code](https://github.com/ZhuiyiTechnology/roformer).

Alright, I'm done. I have to ignore the part of math. So `Position Embedding` is a method to provide positional information in a sequence.
It looks like RoPE is the only option:
> To the best of our knowledge, RoPE is the only relative position embeddings that can be used in linear attentions.

I created all the necessary structs today. I got stucked on cuda.
