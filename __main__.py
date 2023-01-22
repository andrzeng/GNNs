import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from torch_geometric.data import Data

if __name__ == '__main__':
    edge_index = torch.Tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]])
    x = torch.Tensor([[-1], [0], [1]])                
    data = Data(x=x, edge_index=edge_index)
    data.validate(raise_on_error=True)
    