import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np

import math
SEQ_LEN = 10

class PLSTM_Z(nn.Module):

    def __init__(self, hidden_size, impute_weight, reg_weight, label_weight, classes=False):
        super(PLSTM_Z, self).__init__()
        self.input_size = 9
        self.W_f = Parameter(torch.Tensor(hidden_size, self.input_size))
        self.U_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))

        self.W_i = Parameter(torch.Tensor(hidden_size, self.input_size))
        self.U_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))

        self.W_c = Parameter(torch.Tensor(hidden_size, self.input_size))
        self.U_c = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = Parameter(torch.Tensor(hidden_size))

        self.W_o = Parameter(torch.Tensor(hidden_size, self.input_size))
        self.U_o = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.V_o = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))

        nn.init.uniform_(self.W_f, -0.05, 0.05)
        nn.init.uniform_(self.U_f, -0.05, 0.05)
        nn.init.uniform_(self.V_f, -0.05, 0.05)
        nn.init.uniform_(self.b_f, -0.05, 0.05)
        nn.init.uniform_(self.W_i, -0.05, 0.05)
        nn.init.uniform_(self.U_i, -0.05, 0.05)
        nn.init.uniform_(self.V_i, -0.05, 0.05)
        nn.init.uniform_(self.b_i, -0.05, 0.05)
        nn.init.uniform_(self.W_c, -0.05, 0.05)
        nn.init.uniform_(self.U_c, -0.05, 0.05)
        nn.init.uniform_(self.b_c, -0.05, 0.05)
        nn.init.uniform_(self.W_o, -0.05, 0.05)
        nn.init.uniform_(self.U_o, -0.05, 0.05)
        nn.init.uniform_(self.V_o, -0.05, 0.05)
        nn.init.uniform_(self.b_o, -0.05, 0.05)

        self.rnn_hid_size = hidden_size
        self.impute_weight = impute_weight
        self.reg_weight = reg_weight
        self.label_weight = label_weight
        self.out_reg = nn.Linear(hidden_size, 6)
        self.out_reg2 = nn.Linear(self.rnn_hid_size, 1)
        self.out_reg3 = nn.Linear(self.rnn_hid_size, 1)
        self.out_reg4 = nn.Linear(self.rnn_hid_size, 1)
        self.classes = classes

    def forget_gate_peep(self, x, h, c):
        f = torch.sigmoid(F.linear(h,self.U_f) + F.linear(h,self.U_f) + F.linear(c,self.V_f) + self.b_f)
        return f

    def input_gate_peep(self, x, h, c):
        i = torch.sigmoid(F.linear(x,self.W_i) + F.linear(h,self.U_i) + F.linear(c,self.V_i) + self.b_i)
        return i

    def modulation_peep(self, x, h):
        z = torch.tanh(F.linear(x, self.W_c) + F.linear(h, self.U_c) + self.b_c)
        return z

    def cell_state_peep(self, f, c, i, z):
        c_t = (f*c + i*z)
        c_tilda = torch.tanh(c_t)
        return c_t, c_tilda

    def output_gate_peep(self, x, h, c):
        o = torch.sigmoid(F.linear(x,self.W_o) + F.linear(h,self.U_o) + F.linear(c,self.V_o) + self.b_o)
        return o

    def Peephole_internal(self, x_t, h_t, c_t):
        f = self.forget_gate_peep(x_t, h_t, c_t)
        i = self.input_gate_peep(x_t, h_t, c_t)
        z = self.modulation_peep(x_t, h_t)
        c_t, c_tilda = self.cell_state_peep(f, c_t, i, z)
        o_tilda = self.output_gate_peep(x_t, h_t, c_t)
        h_t = o_tilda * c_tilda

        return h_t, c_t
