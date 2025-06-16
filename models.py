import torch
from torch import nn
from lstm import MyLSTMCell, ChildSumTreeLSTMCell, NAryTreeLSTMCell
from batch_utils import batch, unbatch

def print_model_parameters(model):
    """
    Prints the parameters of the model with their shapes and trainable status.

    Args:
        model (nn.Module): The PyTorch model.
    """
    total_params = 0
    param_details = []
    for name, p in model.named_parameters():
        total_params += p.numel()
        param_details.append((name, list(p.shape), p.requires_grad))
    return total_params, param_details


class BOW(nn.Module):
    """A simple bag-of-words model."""
    def __init__(self, vocab_size, embedding_dim, vocab):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            vocab: The vocabulary object.
        """
        super(BOW, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

    def forward(self, inputs):
        """
        Forward pass of the BOW model.
        
        Args:
            inputs (Tensor): Input tensor containing word IDs.

        Returns:
            Tensor: Logits after summing embeddings and adding bias.
        """
        embeds = self.embed(inputs)
        logits = embeds.sum(1) + self.bias
        return logits

class CBOW(nn.Module):
    "A simple Continuous Bag of Words model"

    def __init__(self, vocab_size, embedding_dim, output_dim, vocab):
        super(CBOW, self).__init__()
        self.vocab = vocab

        # this is a trainable look-up table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # linear transformation layer (has inherent bias terms)
        self.linear = nn.Linear(embedding_dim, output_dim)

        # init layer weights and biases
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, inputs):
        embeds = self.embed(inputs)
        logits = self.linear(embeds.sum(1))
        return logits
    
class DeepCBOW(nn.Module):
    """A simple Continuous Bag of Words model with hidden layers."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(DeepCBOW, self).__init__()
        self.vocab = vocab

        # Trainable lookup table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # Output layer with two hidden layers
        self.output = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),  # E -> D
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),  # D -> D
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)  # D -> output_dim (e.g., 5 classes)
        )

        # Initialize layer weights and biases
        self._init_weights()

    def _init_weights(self):
        for layer in self.output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        embeds = self.embed(inputs)  # Lookup embeddings
        logits = self.output(embeds.sum(1))  # Aggregate embeddings and pass through MLP
        return logits

class PTDeepCBOW(DeepCBOW):
    """
    A DeepCBOW model that supports pre-trained embeddings.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(PTDeepCBOW, self).__init__(
            vocab_size, embedding_dim, hidden_dim, output_dim, vocab
        )

class LSTMClassifier(nn.Module):
  """Encodes sentence with an LSTM and projects final hidden state"""

  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
    super(LSTMClassifier, self).__init__()
    self.vocab = vocab
    self.hidden_dim = hidden_dim
    self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
    self.rnn = MyLSTMCell(embedding_dim, hidden_dim)

    self.output_layer = nn.Sequential(
        nn.Dropout(p=0.5),  # explained later
        nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, x):

    B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
    T = x.size(1)  # timesteps (the number of words in the sentence)

    input_ = self.embed(x)

    # here we create initial hidden states containing zeros
    # we use a trick here so that, if input is on the GPU, then so are hx and cx
    hx = input_.new_zeros(B, self.rnn.hidden_size)
    cx = input_.new_zeros(B, self.rnn.hidden_size)

    # process input sentences one word/timestep at a time
    # input is batch-major (i.e., batch size is the first dimension)
    # so the first word(s) is (are) input_[:, 0]
    outputs = []
    for i in range(T):
      hx, cx = self.rnn(input_[:, i], (hx, cx))
      outputs.append(hx)

    # if we have a single example, our final LSTM state is the last hx
    if B == 1:
      final = hx
    else:
      #
      # This part is explained in next section, ignore this else-block for now.
      #
      # We processed sentences with different lengths, so some of the sentences
      # had already finished and we have been adding padding inputs to hx.
      # We select the final state based on the length of each sentence.

      # two lines below not needed if using LSTM from pytorch
      outputs = torch.stack(outputs, dim=0)           # [T, B, D]
      outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]

      # to be super-sure we're not accidentally indexing the wrong state
      # we zero out positions that are invalid
      pad_positions = (x == 1).unsqueeze(-1)

      outputs = outputs.contiguous()
      outputs = outputs.masked_fill_(pad_positions, 0.)

      mask = (x != 1)  # true for valid positions [B, T]
      lengths = mask.sum(dim=1)                 # [B, 1]

      indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
      final = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]

    # we use the last hidden state to classify the sentence
    logits = self.output_layer(final)
    return logits

class TreeLSTM(nn.Module):
  """Encodes a sentence using a TreeLSTMCell"""

  def __init__(self, input_size, hidden_size, bias=True, tree="nary"):
    """Creates the weights for this LSTM"""
    super(TreeLSTM, self).__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    if tree == "childsum":
      self.reduce = ChildSumTreeLSTMCell(input_size, hidden_size, bias=bias)
    elif tree == "nary":
      self.reduce = NAryTreeLSTMCell(input_size, hidden_size, bias=bias)

    # project word to initial c
    self.proj_x = nn.Linear(input_size, hidden_size)
    self.proj_x_gate = nn.Linear(input_size, hidden_size)

    self.buffers_dropout = nn.Dropout(p=0.5)

  def forward(self, x, transitions):
    """
    WARNING: assuming x is reversed!
    :param x: word embeddings [B, T, E]
    :param transitions: [2T-1, B]
    :return: root states
    """

    SHIFT = 0
    REDUCE = 1

    B = x.size(0)  # batch size
    T = x.size(1)  # time

    # compute an initial c and h for each word
    # Note: this corresponds to input x in the Tai et al. Tree LSTM paper.
    # We do not handle input x in the TreeLSTMCell itself.
    buffers_c = self.proj_x(x)
    buffers_h = buffers_c.tanh()
    buffers_h_gate = self.proj_x_gate(x).sigmoid()
    buffers_h = buffers_h_gate * buffers_h

    # concatenate h and c for each word
    buffers = torch.cat([buffers_h, buffers_c], dim=-1)

    D = buffers.size(-1) // 2

    # we turn buffers into a list of stacks (1 stack for each sentence)
    # first we split buffers so that it is a list of sentences (length B)
    # then we split each sentence to be a list of word vectors
    buffers = buffers.split(1, dim=0)  # Bx[T, 2D]
    buffers = [list(b.squeeze(0).split(1, dim=0)) for b in buffers]  # BxTx[2D]

    # create B empty stacks
    stacks = [[] for _ in buffers]

    # t_batch holds 1 transition for each sentence
    for t_batch in transitions:

      child_l = []  # contains the left child for each sentence with reduce action
      child_r = []  # contains the corresponding right child

      # iterate over sentences in the batch
      # each has a transition t, a buffer and a stack
      for transition, buffer, stack in zip(t_batch, buffers, stacks):
        if transition == SHIFT:
          stack.append(buffer.pop())
        elif transition == REDUCE:
          assert len(stack) >= 2, \
            "Stack too small! Should not happen with valid transition sequences"
          child_r.append(stack.pop())  # right child is on top
          child_l.append(stack.pop())

      # if there are sentences with reduce transition, perform them batched
      if child_l:
        reduced = iter(unbatch(self.reduce(batch(child_l), batch(child_r))))
        for transition, stack in zip(t_batch, stacks):
          if transition == REDUCE:
            stack.append(next(reduced))

    final = [stack.pop().chunk(2, -1)[0] for stack in stacks]
    final = torch.cat(final, dim=0)  # tensor [B, D]

    return final

class TreeLSTMClassifier(nn.Module):
  """Encodes sentence with a TreeLSTM and projects final hidden state"""

  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab, tree):
    super(TreeLSTMClassifier, self).__init__()
    self.vocab = vocab
    self.hidden_dim = hidden_dim
    self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
    self.treelstm = TreeLSTM(embedding_dim, hidden_dim, tree=tree)
    self.output_layer = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(hidden_dim, output_dim, bias=True)
    )

  def forward(self, x):
    # x is a pair here of words and transitions; we unpack it here.
    # x is batch-major: [B, T], transitions is time major [2T-1, B]
    x, transitions = x
    emb = self.embed(x)

    # we use the root/top state of the Tree LSTM to classify the sentence
    root_states = self.treelstm(emb, transitions)

    # we use the last hidden state to classify the sentence
    logits = self.output_layer(root_states)
    return logits