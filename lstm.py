import torch
import torch.nn as nn
import math

class MyLSTMCell(nn.Module):
  """Our own LSTM cell"""

  def __init__(self, input_size, hidden_size, bias=True):
    """Creates the weights for this LSTM"""
    super(MyLSTMCell, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    self.input_layer = nn.Linear(input_size, 4 * hidden_size, bias=bias)
    self.hidden_layer = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    self.reset_parameters()

  def reset_parameters(self):
    """This is PyTorch's default initialization method"""
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, input_, hx, mask=None):
    """
    input is (batch, input_size)
    hx is ((batch, hidden_size), (batch, hidden_size))
    """
    prev_h, prev_c = hx

    # project input and prev state
    combined_h = self.hidden_layer(prev_h)
    combined_i = self.input_layer(input_)

    # main LSTM computation
    # gates: i, f, g, o
    i_h, f_h, g_h, o_h = torch.chunk(combined_h, 4, dim=1)
    i_i, f_i, g_i, o_i = torch.chunk(combined_i, 4, dim=1)

    # gate activations
    i = torch.sigmoid(i_h + i_i)
    f = torch.sigmoid(f_h + f_i)
    g = torch.tanh(g_h + g_i)
    o = torch.sigmoid(o_h + o_i)

    c = f * prev_c + i * g
    h = o * torch.tanh(c)

    if mask is not None:
      h = h * mask
      c = c * mask

    return h, c

  def __repr__(self):
    return "{}({:d}, {:d})".format(
        self.__class__.__name__, self.input_size, self.hidden_size)

class TreeLSTMCell(nn.Module):
  """A Binary Tree LSTM cell"""

  def __init__(self, input_size, hidden_size, bias=True):
    """Creates the weights for this LSTM"""
    super(TreeLSTMCell, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    self.reduce_layer = nn.Linear(2 * hidden_size, 5 * hidden_size)
    self.dropout_layer = nn.Dropout(p=0.25)

    self.reset_parameters()

  def reset_parameters(self):
    """This is PyTorch's default initialization method"""
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, hx_l, hx_r, mask=None):
    """
    hx_l is ((batch, hidden_size), (batch, hidden_size))
    hx_r is ((batch, hidden_size), (batch, hidden_size))
    """
    prev_h_l, prev_c_l = hx_l  # left child
    prev_h_r, prev_c_r = hx_r  # right child

    B = prev_h_l.size(0)

    # we concatenate the left and right children
    # you can also project from them separately and then sum
    children = torch.cat([prev_h_l, prev_h_r], dim=1)

    # project the combined children into a 5D tensor for i,fl,fr,g,o
    # this is done for speed, and you could also do it separately
    proj = self.reduce_layer(children)  # shape: B x 5D

    # each shape: B x D
    i, f_l, f_r, g, o = torch.chunk(proj, 5, dim=-1)

    # main Tree LSTM computation
    # The shape of each of these is [batch_size, hidden_size]

    i = torch.sigmoid(i) # input gate
    f_l = torch.sigmoid(f_l) # forget gate left
    f_r = torch.sigmoid(f_r) # forget gate right
    g = torch.tanh(g) # block input
    o = torch.sigmoid(o) # output gate

    c = f_l * prev_c_l + f_r * prev_c_r + i * g
    h = o * torch.tanh(c)

    if mask is not None:
      h = h * mask
      c = c * mask

    return h, c

  def __repr__(self):
    return "{}({:d}, {:d})".format(
        self.__class__.__name__, self.input_size, self.hidden_size)
  
