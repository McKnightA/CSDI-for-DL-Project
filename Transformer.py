import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    A module that computes multi-head attention given query, key, and value tensors.
    """

    def __init__(self, input_dim: int, num_heads: int):
        """
        Constructor.

        Inputs:
        - input_dim: Dimension of the input query, key, and value. Here we assume they all have
          the same dimensions. But they could have different dimensions in other problems.
        - num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()

        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        ###########################################################################
        # TODO: Define the linear transformation layers for key, value, and query.#
        # Also define the output layer.
        ###########################################################################

        self.linear_k = nn.Linear(input_dim, input_dim * num_heads)
        self.linear_q = nn.Linear(input_dim, input_dim * num_heads)
        self.linear_v = nn.Linear(input_dim, input_dim * num_heads)
        self.linear_o = nn.Linear(input_dim * num_heads, input_dim)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None):
        """
        Compute the attended feature representations.

        Inputs:
        - query: Tensor of the shape BxLxFxC, where B is the batch size, L is the sequence length,
          F is the number of features, and C is the channel dimension
        - key: Tensor of the shape BxLxFxC
        - value: Tensor of the shape BxLxFxC
        - mask: Tensor indicating where the attention should *not* be performed
        """
        b, l, f, c = query.shape

        dot_prod_scores = None
        ###########################################################################
        # TODO: Compute the scores based on dot product between transformed query,#
        # key, and value. You may find torch.matmul helpful, whose documentation  #
        # can be found at                                                         #
        # https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul#
        # Remember to divide the dot product similarity scores by square root of #
        # the channel dimension per head.
        #                                                                         #
        # Since no for loops are allowed here, think of how to use tensor reshape #
        # to process multiple attention heads at the same time.                   #
        ###########################################################################

        q = self.linear_q(query)  # should give BxLxKxC*H
        q = torch.reshape(q, (b, l, f, c, self.num_heads))  # should give BxLxKxCxH
        q = torch.permute(q, (0, 4, 1, 2, 3))  # should give BxHxLxKxC
        q = torch.reshape(q, (b * self.num_heads, l, f, c))  # should give B*HxLxKxC

        k = self.linear_k(key)
        k = torch.reshape(k, (b, l, f, c, self.num_heads))
        k = torch.permute(k, (0, 4, 1, 2, 3))
        k = torch.reshape(k, (b * self.num_heads, l, f, c))

        v = self.linear_v(value)
        v = torch.reshape(v, (b, l, f, c, self.num_heads))
        v = torch.permute(v, (0, 4, 1, 2, 3))
        v = torch.reshape(v, (b * self.num_heads, l, f, c))

        # Need to calculate the similarity scores across both time and feature space
        # and torch.tensordot doesn't handle a batch dimension
        dot_prod_scores = []
        for i in range(q.shape[0]):
            dot_prod_scores.append(torch.tensordot(q[i], k[i], dims=([2], [2])) / math.sqrt(c))
        dot_prod_scores = torch.stack(dot_prod_scores)  # should give B*HxLxFxLxF

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        if mask is not None:
            # We simply set the similarity scores to be near zero for the positions
            # where the attention should not be done. Think of why we do this.
            dot_prod_scores = dot_prod_scores.masked_fill(mask == 0, -1e9)

        out = None
        ###########################################################################
        # TODO: Compute the attention scores, which are then used to modulate the #
        # value tensor. Finally concate the attended tensors from multiple heads  #
        # and feed it into the output layer. You may still find torch.matmul      #
        # helpful.                                                                #
        #                                                                         #
        # Again, think of how to use reshaping tensor to do the concatenation.    #
        ###########################################################################

        # we care about the relations between points and time and individual features
        attention_weights = F.softmax(dot_prod_scores, dim=3)  # BxLxFxLxF
        attention_weights = F.softmax(attention_weights, dim=4)

        z = []
        for i in range(attention_weights.shape[0]):
            z.append(torch.tensordot(attention_weights[i], v[i], dims=([2, 3], [0, 1])))  # LxFxLxF X LxFxC
        z = torch.stack(z)
        z = torch.reshape(z, (b, self.num_heads, l, f, c))
        z = torch.permute(z, (0, 2, 3, 4, 1))
        z = torch.reshape(z, (b, l, f, c * self.num_heads))

        out = self.linear_o(z)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return out


class FeedForwardNetwork(nn.Module):
    """
    A simple feedforward network. Essentially, it is a two-layer fully-connected
    neural network.
    """

    def __init__(self, input_dim, ff_dim, dropout):
        """
        Inputs:
        - input_dim: Input dimension
        - ff_dim: Hidden dimension
        """
        super(FeedForwardNetwork, self).__init__()

        ###########################################################################
        # TODO: Define the two linear layers and a non-linear one.
        ###########################################################################

        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)
        self.non_linear = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def forward(self, x: torch.Tensor):
        """
        Input:
        - x: Tensor of the shape BxLxFxC, where B is the batch size, L is the sequence length,
         F is the number of features, and C is the channel dimension

        Return:
        - y: Tensor of the shape BxLxKxC
        """

        y = None
        ###########################################################################
        # TODO: Process the input.                                                #
        ###########################################################################

        y = self.linear1(x)
        y = self.non_linear(y)
        y = self.linear2(y)
        y = self.dropout(y)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return y


class TransformerEncoderCell(nn.Module):
    """
    A single cell (unit) for the Transformer encoder.
    """

    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, dropout: float):
        """
        Inputs:
        - input_dim: Input dimension for each feature in a sequence
        - num_heads: Number of attention heads in a multi-head attention module
        - ff_dim: The hidden dimension for a feedforward network
        - dropout: Dropout ratio for the output of the multi-head attention and feedforward
          modules.
        """
        super(TransformerEncoderCell, self).__init__()

        ###########################################################################
        # TODO: A single Transformer encoder cell consists of
        # 1. A multi-head attention module
        # 2. Followed by dropout
        # 3. Followed by layer norm (check nn.LayerNorm)
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
        #                                                                         #
        # At the same time, it also has
        # 1. A feedforward network
        # 2. Followed by dropout
        # 3. Followed by layer norm
        ###########################################################################

        self.multi_head = MultiHeadAttention(input_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(input_dim)

        self.ff = FeedForwardNetwork(input_dim, ff_dim, dropout)
        self.ln2 = nn.LayerNorm(input_dim)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Inputs:
        - x: Tensor of the shape BxLxFxC, where B is the batch size, L is the sequence length,
          F is the number of features, and C is the channel dimension
        - mask: Tensor for multi-head attention
        """

        y = None
        ###########################################################################
        # TODO: Get the output of the multi-head attention part (with dropout     #
        # and layer norm), which is used as input to the feedforward network (    #
        # again, followed by dropout and layer norm).                             #
        #                                                                         #
        # Don't forget the residual connections for both parts.                   #
        ###########################################################################

        z = self.multi_head.forward(x, x, x, mask)
        z = self.dropout(z)
        z = z + x
        z = self.ln1(z)

        y = self.ff.forward(z)
        y = y + z
        y = self.ln2(y)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return y


class TransformerEncoder(nn.Module):
    """
    A full encoder consisting of a set of TransformerEncoderCell.
    """

    def __init__(self, input_dim: int, num_heads: int, ff_dim: int, num_cells: int, dropout: float = 0.1):
        """
        Inputs:
        - input_dim: Input dimension for each token in a sequence
        - num_heads: Number of attention heads in a multi-head attention module
        - ff_dim: The hidden dimension for a feedforward network
        - num_cells: Number of TransformerEncoderCells
        - dropout: Dropout ratio for the output of the multi-head attention and feedforward
          modules.
        """
        super(TransformerEncoder, self).__init__()

        self.norm = None
        ###########################################################################
        # TODO: Construct a nn.ModuleList to store a stack of                     #
        # TransformerEncoderCells. Check the documentation here of how to use it  #
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList

        # At the same time, define a layer normalization layer to process the     #
        # output of the entire encoder.                                           #
        ###########################################################################

        self.cells = nn.ModuleList([TransformerEncoderCell(input_dim, num_heads, ff_dim, dropout)])

        for _ in range(num_cells - 1):
            self.cells.append(TransformerEncoderCell(input_dim, num_heads, ff_dim, dropout))

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Inputs:
        - x: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
          and C is the channel dimension
        - mask: Tensor for multi-head attention

        Return:
        - y: Tensor of the shape of BxLxC, which is the normalized output of the encoder
        """

        y = None
        ###########################################################################
        # TODO: Feed x into the stack of TransformerEncoderCells and then         #
        # normalize the output with layer norm.                                   #
        ###########################################################################

        y = x
        for cell in self.cells:
            y = cell.forward(y, mask)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return y
