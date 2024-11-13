import torch
from torch import nn
conv1 = nn.Conv1d(in_channels=3, out_channels=4, kernel_size=2)
input = torch.randn(5, 6, 3)
input = input.permute(0, 2, 1)
# input =[[[1, 2, 3, 4, 5, 6, 7],
#          [4, 5, 6, 7, 8, 9, 10],
#          [1, 2, 3, 4, 5, 6, 7]],
#         [[1, 2, 3, 4, 5, 6, 7],
#          [4, 5, 6, 7, 8, 9, 10],
#          [1, 2, 3, 4, 5, 6, 7]],
#         [[1, 2, 3, 4, 5, 6, 7],
#          [4, 5, 6, 7, 8, 9, 10],
#          [1, 2, 3, 4, 5, 6, 7]],
#         [[1, 2, 3, 4, 5, 6, 7],
#          [4, 5, 6, 7, 8, 9, 10],
#          [1, 2, 3, 4, 5, 6, 7]],
#         [[1, 2, 3, 4, 5, 6, 7],
#          [4, 5, 6, 7, 8, 9, 10],
#          [1, 2, 3, 4, 5, 6, 7]]]
output = conv1(input)

ouput =conv1(input)