# PerceiverIO

Implementation of the attention mechanisms necessary for a PerceiverIO.
It is a more efficient Transformer based model.

Heavily influenced by:
[0] : https://arxiv.org/pdf/2107.14795.pdf
[1] : t.ly/nh\_m : Github repository for this

Brief Explanation:
    - Attention allows a network to attenuate to different parts of an
      input depending on what's learnt to be  useful, and takes the form
      of a set of weights stored in a "attention matrix".

    - In self-attention, Q and K and V are all the same: it
      learns to attenuate to different parts of the current
      representation.

    - In cross-attention, Q and K are a different thing altogether to
      V: It learns how to attenuate the current representation based
      off another representation or set of information altogether.
      (See alphafold2 and sequence / structural information)

    - The Perceiver has a latent space representation that is formed by
      using cross-attention from the latent dimensions` to the input data.
      This is then refined using self-attention for a few
      layers like a standard transformer.

    - This is useful because in a usual transformer model the attention
      requires LM**2 computations (each input must attend to all other
      inputs at each layer). By only working with the latent representation
      for self iterative attention we instead scale O(MN + LN**2),
      where M in the input data size, N is the latent dimension (N << M),
      and L is the number of transformer layers.

    - Read the original perceiver paper for more info:
        https://arxiv.org/pdf/2103.03206.pdf

## TODOs:
    - Make a good example project on some interesting data
    - Format like the GAP, as more of a python package
