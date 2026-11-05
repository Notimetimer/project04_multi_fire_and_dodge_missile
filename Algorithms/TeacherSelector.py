"""
只提供selector的策略蒸馏梯度，不管pilot
1、输入和selector相同的输入，主要根据rule_inputs做决策
2、输出teacher选择的enm_idx

"""

import os
from math import *
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F

_MASK_VAL = 1e8                   # 掩码填充值（-inf 量级，保证 softmax 后概率为 0）

