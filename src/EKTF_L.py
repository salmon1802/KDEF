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
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
from tqdm import tqdm
import numpy as np
import logging
import sys
from .Backbone import MLP, ExponentialCrossNetwork, ProductNetwork, FinalNet


class EKTF_L(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="EKTF_L",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 batch_norm=True,
                 num_student=3,
                 hidden_units=[64, 64, 64],
                 hidden_activations="Relu",
                 num_cross_layers=3,
                 net1_dropout=0.1,
                 net2_dropout=0.1,
                 net3_dropout=0.1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(EKTF_L, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        num_fields = feature_map.num_fields
        self.num_student = num_student
        self.networks = nn.ModuleList()
        self.networks.append(ProductNetwork(num_fields=num_fields,
                                            embedding_dim=embedding_dim,
                                            hidden_units=hidden_units,
                                            batch_norm=batch_norm,
                                            net_dropout=net1_dropout,
                                            hidden_activations=hidden_activations))
        self.networks.append(FinalNet(input_dim=input_dim * 2,
                                      hidden_units=hidden_units,
                                      hidden_activations=hidden_activations,
                                      dropout_rates=net2_dropout,
                                      batch_norm=batch_norm,
                                      num_fields=num_fields))
        self.networks.append(ExponentialCrossNetwork(input_dim=input_dim,
                                                     num_cross_layers=num_cross_layers,
                                                     batch_norm=batch_norm,
                                                     net_dropout=net3_dropout))

        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        logits_stu = [network(feature_emb) for network in self.networks]
        y_pred = torch.mean(torch.cat(logits_stu, dim=-1), dim=-1, keepdim=True)
        return_dict = {"y_pred": self.output_activation(y_pred),
                       "logits_stu": logits_stu}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred = return_dict["y_pred"]
        logits_stu = return_dict["logits_stu"]

        y_preds_stu = []
        teach_to_stu_loss = []
        label_to_stu_loss = []
        stu_to_stu_loss = []
        stu_study_scores = []

        for stu in logits_stu:
            y_pred_stu = self.output_activation(stu)
            y_preds_stu.append(y_pred_stu)
            label_to_stu_loss.append(self.loss_fn(y_pred_stu, y_true, reduction='mean'))  # stu learn from label
            teach_to_stu_loss.append(
                F.mse_loss(y_pred_stu, y_pred.detach(), reduction='mean'))  # stu learn from teacher
            stu_score = 1 - torch.abs(y_true - y_pred_stu)  # start exam
            stu_study_scores.append(stu_score.mean())

        label_to_stu_loss = torch.stack(label_to_stu_loss, dim=-1).mean()
        teach_to_stu_loss = torch.stack(teach_to_stu_loss, dim=-1)
        teach_to_stu_weight = torch.stack(stu_study_scores, dim=-1)
        teach_to_stu_weight = F.softmin(teach_to_stu_weight, dim=-1)
        teach_to_stu_loss = (teach_to_stu_weight * teach_to_stu_loss).sum(dim=-1)

        for stu1_index in range(self.num_student):
            loss = []
            weight = []
            for stu2_index in range(self.num_student):
                if stu2_index == stu1_index:
                    continue
                y_stu1 = y_preds_stu[stu1_index]
                y_stu2 = y_preds_stu[stu2_index]
                stu1_study_score = stu_study_scores[stu1_index]
                stu2_study_score = stu_study_scores[stu2_index]
                loss.append(F.mse_loss(y_stu1, y_stu2.detach(), reduction='mean'))
                weight.append(stu2_study_score - stu1_study_score)
            weight = torch.softmax(torch.stack(weight, dim=-1), dim=-1)
            loss = torch.stack(loss, dim=-1)
            loss = weight * loss
            stu_to_stu_loss.append(loss.sum(dim=-1))
        stu_to_stu_loss = torch.stack(stu_to_stu_loss, dim=-1).mean()

        return label_to_stu_loss + teach_to_stu_loss + stu_to_stu_loss

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_preds = [[] for _ in range(self.num_student + 1)]
            y_true = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                logits_stu = return_dict["logits_stu"]
                for i, stu in enumerate(logits_stu):
                    y_preds[i].extend(self.output_activation(stu).data.cpu().numpy().reshape(-1))
                y_preds[-1].extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                if self.task.lower() == 'pretrain':
                    y_true.extend(return_dict["y_true"].data.cpu().numpy().reshape(-1))
                else:
                    y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))

            y_true = np.array(y_true, np.float64)
            for i, y_pred in enumerate(y_preds):
                y_pred = np.array(y_pred, np.float64)
                if metrics is not None:
                    val_logs = self.evaluate_metrics(y_true, y_pred, metrics)
                else:
                    val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics)
                if i + 1 == len(y_preds):
                    logging.info(
                        'Teacher: [Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                else:
                    logging.info('Student {}: [Metrics] '.format(i + 1) + ' - '.join(
                        '{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs
