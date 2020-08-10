import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Cog_Model(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight, reg_weight, label_weight, classes=False):
        super(Cog_Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.reg_weight = reg_weight
        self.label_weight = label_weight
        self.classes = classes
        self.features = 9
        self.seq_length = 10
        self.build()

    def build(self):
        # Call the Recurrent model
        self.rnn_cell = nn.LSTMCell(self.features * 2, self.rnn_hid_size)
        # Call the information of temporal relation
        self.temp_decay_h = TemporalDecay(input_size = self.features, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = self.features, output_size = self.features, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.features)
        # Call the information of feature-wise relation
        self.feat_reg = FeatureRegression(self.features)
        self.weight_combine = nn.Linear(self.features * 2, self.features)

        self.dropout = nn.Dropout(p = 0.25)
        # Output of the proposed model
        self.out_reg1 = nn.Linear(self.rnn_hid_size, 6) # MRI-biomarkers
        self.out_reg2 = nn.Linear(self.rnn_hid_size, 1) # MMSE
        self.out_reg3 = nn.Linear(self.rnn_hid_size, 1) # ADAS-cog11
        self.out_reg4 = nn.Linear(self.rnn_hid_size, 1) # ADAS-cog13

        if self.classes == True:
            self.out_cls1 = nn.Linear(self.rnn_hid_size, 3) # For classification task

    def forward(self, data, direct, criterion_reg, criterion_cls, multi_flag=False):

        values = data[direct]['values'][:, :10, :]
        masks = data[direct]['masks'][:, :10, :]
        deltas = data[direct]['deltas'][:, :10, :]

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        # labels = data['labels'][:,1:].contiguous().view(-1, 1)
        labels = data['labels'][:, :10].contiguous().view(-1, 1)
        labels_indicator = torch.ones_like(labels)
        labels_indicator[torch.where(labels == -2)[0]] = 0

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0

        imputations = []
        output_reg, output_cls = [], []
        output_probs = []
        analyze_cell = []
        output_mmse, output_ad11, output_ad13 = [], [], []

        observe_t = 10
        n = observe_t
        for t in range(self.seq_length):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            x_h = self.hist_reg(h)

            h = h * gamma_h

            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))


            imputations.append(c_c.unsqueeze(dim=1))
            y_h = self.out_reg1(h)

            y_mmse = self.out_reg2(h)
            y_ad11 = self.out_reg3(h)
            y_ad13 = self.out_reg4(h)


            output_reg.append(y_h.unsqueeze(dim=1))
            output_mmse.append(y_mmse.unsqueeze(dim=1))
            output_ad11.append(y_ad11.unsqueeze(dim=1))
            output_ad13.append(y_ad13.unsqueeze(dim=1))

            if multi_flag == True:
                y_cls = self.out_cls1(h)
                output_prob = torch.softmax(y_cls, dim=1)
                output_cls.append(y_cls.unsqueeze(dim=1))
                output_probs.append(output_prob.unsqueeze(dim=1))
                analyze_cell.append(c.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim = 1)
        output_reg = torch.cat(output_reg, dim=1)
        output_mmse = torch.cat(output_mmse, dim=1)
        output_ad11 = torch.cat(output_ad11, dim=1)
        output_ad13 = torch.cat(output_ad13, dim=1)
        analyze_cell = torch.cat(analyze_cell, dim=1)

        shifted_data = data[direct]['values'][:, 1:, :6]
        shifted_mask = data[direct]['masks'][:, 1:, :6]

        if multi_flag == True:
            output_cls = torch.cat(output_cls,  dim = 1)
            output_probs = torch.cat(output_probs, dim = 1)

        y_reg_loss = criterion_reg(output_reg.contiguous().view(-1,6) * shifted_mask.contiguous().view(-1, 6),
                                   (shifted_data * shifted_mask).contiguous().view(-1, 6))
        y_mmse_loss = criterion_reg(output_mmse.contiguous()*data[direct]['masks'][:, 1:, 6:7],
                                    (data[direct]['values'][:, 1:, 6:7]*data[direct]['masks'][:, 1:, 6:7]))
        y_ad11_loss = criterion_reg(output_ad11.contiguous()* data[direct]['masks'][:, 1:, 7:8],
                                    (data[direct]['values'][:, 1:, 7:8] * data[direct]['masks'][:, 1:, 7:8]))
        y_ad13_loss = criterion_reg(output_ad13.contiguous()* data[direct]['masks'][:, 1:, 8:9],
                                    (data[direct]['values'][:, 1:, 8:9] * data[direct]['masks'][:, 1:, 8:9]))
        if multi_flag == True:
            y_cls_loss = criterion_cls(output_probs.contiguous().view(-1,3), labels.squeeze().long())

            return {'loss': x_loss * self.impute_weight + (y_mmse_loss + y_ad11_loss + y_ad13_loss + y_reg_loss) * self.reg_weight + y_cls_loss * self.label_weight,
                    'predictions': output_probs.contiguous().view(-1, 3),
                    'predictions_feature': output_reg.contiguous().view(-1, 6), \
                    'imputations': imputations, 'labels': labels, 'is_train': labels_indicator, \
                    'evals': evals, 'eval_masks': eval_masks, 'shifted_data': shifted_data, 'shifted_mask': shifted_mask, 'analy':analyze_cell, 'analy_label': data['labels'].contiguous().view(-1, 1),
                    'predict_mmse': output_mmse.contiguous(), 'predict_ad11': output_ad11.contiguous(), 'predict_ad13': output_ad13.contiguous()}

    def run_on_batch(self, data, optimizer, criterion_reg, criterion_cls, multi_flag, epoch = None):
        ret = self(data, direct = 'forward', criterion_reg = criterion_reg, criterion_cls = criterion_cls, multi_flag = multi_flag)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
