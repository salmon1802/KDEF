# =========================================================================
# Copyright (C) 2024 salmon@github
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch
from torch import nn
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.pytorch.layers import InnerProductInteraction, MLP_Block

class ExponentialCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 batch_norm=True,
                 num_cross_layers=3,
                 net_dropout=0.1):
        super(ExponentialCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.layer_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(1))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.masker = nn.ReLU()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        multihead_feature_emb = torch.tensor_split(x, 1, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)  # B × H × F × D/H
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(multihead_feature_emb, 2,
                                                                            dim=-1)  # B × H × F × D/2H
        multihead_feature_emb1, multihead_feature_emb2 = multihead_feature_emb1.flatten(start_dim=2), \
                                                         multihead_feature_emb2.flatten(
                                                             start_dim=2)  # B × H × FD/2H; B × H × FD/2H
        x = torch.cat([multihead_feature_emb1, multihead_feature_emb2], dim=-1)
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            norm_H = self.layer_norm[i](H)
            mask = self.masker(norm_H)
            H = torch.cat([H, H * mask], dim=-1)
            x = x * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x).mean(dim=1)


class FinalNet(nn.Module):
    def __init__(self, num_fields, input_dim, hidden_units=[], hidden_activations="ReLU",
                 dropout_rates=0.1, batch_norm=True):
        # Replacement of MLP_Block, identical when order=1
        super(FinalNet, self).__init__()
        if type(dropout_rates) != list:
            dropout_rates = [dropout_rates] * len(hidden_units)
        if type(hidden_activations) != list:
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        self.field_gate = FinalGate(num_fields)
        for idx in range(len(hidden_units) - 1):
            self.layer.append(FinalLinear(hidden_units[idx], hidden_units[idx + 1]))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))
        self.fc = nn.Linear(hidden_units[-1], 1)

    def forward(self, X):
        X = self.field_gate(X)
        X = X.flatten(start_dim=1)
        X_i = X
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return self.fc(X_i)

class FinalLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        """ A replacement of nn.Linear to enhance multiplicative feature interactions.
            `residual_type="concat"` uses the same number of parameters as nn.Linear
            while `residual_type="sum"` doubles the number of parameters.
        """
        super(FinalLinear, self).__init__()
        assert output_dim % 2 == 0, "output_dim should be divisible by 2."
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        h = self.linear(x)
        h2, h1 = torch.chunk(h, chunks=2, dim=-1)
        h = torch.cat([h2, h1 * h2], dim=-1)
        return h


class FinalGate(nn.Module):
    def __init__(self, num_fields):
        super(FinalGate, self).__init__()
        self.linear = nn.Linear(num_fields, num_fields)

    def reset_custom_params(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.ones_(self.linear.bias)

    def forward(self, feature_emb):
        gates = self.linear(feature_emb.transpose(1, 2)).transpose(1, 2)
        out = torch.cat([feature_emb, feature_emb * gates], dim=1)  # b x 2f x d
        return out

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[],
                 hidden_activations=None,
                 dropout_rates=0.1,
                 batch_norm=True):
        super(MLP, self).__init__()
        if type(dropout_rates) != list:
            dropout_rates = [dropout_rates] * len(hidden_units)
        if type(hidden_activations) != list:
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.layer.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1]))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))
        self.fc = nn.Linear(hidden_units[-1], 1)

    def forward(self, X):
        X_i = X.flatten(start_dim=1)
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        logit = self.fc(X_i)
        return logit

class ProductNetwork(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim=16,
                 hidden_units=[400, 400, 400],
                 batch_norm=False,
                 net_dropout=0.1,
                 hidden_activations="ReLU"):
        super(ProductNetwork, self).__init__()
        self.inner_product_layer = InnerProductInteraction(num_fields, output="inner_product")
        input_dim = int(num_fields * (num_fields - 1) / 2) + num_fields * embedding_dim
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

    def forward(self, x):
        inner_products = self.inner_product_layer(x)
        dense_input = torch.cat([x.flatten(start_dim=1), inner_products], dim=1)
        x = self.dnn(dense_input)
        return x

class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))

    def forward(self, X_0):
        X_0 = X_0.flatten(start_dim=1)
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i

class CompressedInteractionNet(nn.Module):
    def __init__(self, num_fields, cin_hidden_units, output_dim=1):
        super(CompressedInteractionNet, self).__init__()
        self.cin_hidden_units = cin_hidden_units
        self.fc = nn.Linear(sum(cin_hidden_units), output_dim)
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_hidden_units):
            in_channels = num_fields * self.cin_hidden_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1) # kernel output shape

    def forward(self, feature_emb):
        pooling_outputs = []
        X_0 = feature_emb
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_hidden_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                      .view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        output = self.fc(torch.cat(pooling_outputs, dim=-1))
        return output

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.lr_layer = nn.Linear(input_dim, 1)

    def forward(self, X):
        X = X.flatten(start_dim=1)
        output = self.lr_layer(X)
        return output

class xDeepFM(nn.Module):
    def __init__(self,
                 input_dim,
                 num_fields,
                 dnn_hidden_units=[64, 64, 64],
                 dnn_activations="ReLU",
                 cin_hidden_units=[16, 16, 16],
                 net_dropout=0,
                 batch_norm=False):
        super(xDeepFM, self).__init__()
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.lr_layer = LogisticRegression(input_dim)
        self.cin = CompressedInteractionNet(num_fields, cin_hidden_units)

    def forward(self, feature_emb):
        lr_logit = self.lr_layer(feature_emb)
        cin_logit = self.cin(feature_emb)
        dnn_logit = self.dnn(feature_emb.flatten(start_dim=1))
        logit = lr_logit + cin_logit + dnn_logit # LR + CIN + DNN
        return logit


class DCNv2(nn.Module):
    def __init__(self,
                 input_dim,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0,
                 batch_norm=False):
        super(DCNv2, self).__init__()
        self.crossnet = CrossNetV2(input_dim, num_cross_layers)
        self.dnn = MLP_Block(input_dim=input_dim,
                                          output_dim=None,
                                          hidden_units=dnn_hidden_units,
                                          hidden_activations=dnn_activations,
                                          output_activation=None,
                                          dropout_rates=net_dropout,
                                          batch_norm=batch_norm)
        self.fc = nn.Linear(input_dim + dnn_hidden_units[-1], 1)

    def forward(self, feature_emb):
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(flat_feature_emb)
        dnn_out = self.dnn(flat_feature_emb)
        final_out = torch.cat([dnn_out, cross_out], dim=-1)
        logit = self.fc(final_out)
        return logit


class AFN(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_fields,
                 dnn_hidden_units=[64, 64, 64],
                 dnn_activations="ReLU",
                 dnn_dropout=0,
                 afn_hidden_units=[64, 64, 64],
                 afn_activations="ReLU",
                 afn_dropout=0,
                 logarithmic_neurons=5,
                 batch_norm=True):
        super(AFN, self).__init__()
        self.num_fields = num_fields
        self.coefficient_W = nn.Linear(self.num_fields, logarithmic_neurons, bias=False)
        self.dense_layer = MLP_Block(input_dim=embedding_dim * logarithmic_neurons,
                                     output_dim=1,
                                     hidden_units=afn_hidden_units,
                                     hidden_activations=afn_activations,
                                     output_activation=None,
                                     dropout_rates=afn_dropout,
                                     batch_norm=batch_norm)
        self.log_batch_norm = nn.BatchNorm1d(self.num_fields)
        self.exp_batch_norm = nn.BatchNorm1d(logarithmic_neurons)
        self.dnn = MLP_Block(input_dim=embedding_dim * self.num_fields,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=dnn_dropout,
                             batch_norm=batch_norm)
        self.fc = nn.Linear(2, 1)

    def forward(self, feature_emb):
        dnn_input = self.logarithmic_net(feature_emb)
        afn_out = self.dense_layer(dnn_input)
        dnn_out = self.dnn(feature_emb.flatten(start_dim=1))
        logit = self.fc(torch.cat([afn_out, dnn_out], dim=-1))
        return logit

    def logarithmic_net(self, feature_emb):
        feature_emb = torch.abs(feature_emb)
        feature_emb = torch.clamp(feature_emb, min=1e-5)  # ReLU with min 1e-5 (better than 1e-7 suggested in paper)
        log_feature_emb = torch.log(feature_emb)  # element-wise log
        log_feature_emb = self.log_batch_norm(log_feature_emb)  # batch_size * num_fields * embedding_dim
        logarithmic_out = self.coefficient_W(log_feature_emb.transpose(2, 1)).transpose(1, 2)
        cross_out = torch.exp(logarithmic_out)  # element-wise exp
        cross_out = self.exp_batch_norm(cross_out)  # batch_size * logarithmic_neurons * embedding_dim
        concat_out = torch.flatten(cross_out, start_dim=1)
        return concat_out

class WideDeep(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False):
        super(WideDeep, self).__init__()
        self.lr_layer = LogisticRegression(input_dim)
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

    def forward(self, feature_emb):
        feature_emb = feature_emb.flatten(start_dim=1)
        logit = self.lr_layer(feature_emb)
        logit += self.dnn(feature_emb)
        return logit



class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention


class AutoInt(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 num_fields,
                 dnn_hidden_units=[64, 64, 64],
                 dnn_activations="ReLU",
                 attention_layers=2,
                 num_heads=1,
                 attention_dim=8,
                 net_dropout=0,
                 batch_norm=False,
                 layer_norm=False,
                 use_scale=True,
                 use_residual=True):
        super(AutoInt, self).__init__()
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embedding_dim if i == 0 else attention_dim,
                                     attention_dim=attention_dim,
                                     num_heads=num_heads,
                                     dropout_rate=net_dropout,
                                     use_residual=use_residual,
                                     use_scale=use_scale,
                                     layer_norm=layer_norm) \
              for i in range(attention_layers)])
        self.fc = nn.Linear(num_fields * attention_dim, 1)

    def forward(self, feature_emb):
        attention_out = self.self_attention(feature_emb)
        attention_out = torch.flatten(attention_out, start_dim=1)
        logit = self.fc(attention_out)
        logit += self.dnn(feature_emb.flatten(start_dim=1))
        return logit


class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0.,
                 use_residual=True, use_scale=False, layer_norm=False):
        super(MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
            "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        if self.use_residual and input_dim != attention_dim:
            self.W_res = nn.Linear(input_dim, attention_dim, bias=False)
        else:
            self.W_res = None
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim) if layer_norm else None

    def forward(self, X):
        residual = X

        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        if self.W_res is not None:
            residual = self.W_res(residual)
        if self.use_residual:
            output += residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output