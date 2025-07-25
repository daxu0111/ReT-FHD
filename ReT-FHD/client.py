
from clientbase import Client
import torch.nn as nn
import time
import torch
from collections import defaultdict
import copy
import torch.nn.functional as F
import random
import numpy as np
import math
import os
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import sys
from timm.models.layers import _assert, trunc_normal_
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
class clientHdl(Client):
    def __init__(self, args, id, train_samples, test_samples,pubilc_samples,**kwargs):
        super().__init__(args, id, train_samples, test_samples,pubilc_samples,**kwargs)

        self.logits = None
        self.global_logits = None
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

        # ofa 参数
        self.temperature = args.temperature
        self.eps = args.ofa_eps
        self.stage_count = args.ofa_stage
        self.ofa_loss_weight = args.ofa_loss_weight
        self.ofa_temperature = args.ofa_temperature
        self.gt_loss_weight = args.loss_gt_weight
        self.kd_loss_weight = args.loss_kd_weight
        self.projectors = nn.ModuleDict({
                    str(i): self._create_projector(i, self.modelname)
                    for i in range(1, 5)
                })
        self.projectors.apply(self.init_weights)
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        self.clip_grad = args.clip_grad
        self.begin=0.4
        self.max_tem =3
        self.max_z = 6
        self.class_labels_temp_adv = torch.ones(self.num_classes, dtype=torch.float).cuda()
        self.class_labels_temp_nat = torch.ones(self.num_classes, dtype=torch.float).cuda()
        self.class_labels_temp_stage = torch.ones(3, self.num_classes, dtype=torch.float).cuda()
        self.global_logits_stage = [None] * 3
    def train(self):
        trainloadere = self.load_train_data()
        start_time = time.time()
        self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        logits = defaultdict(list)


        loss_ofa = 0

        loss_kd = 0
        total_loss_gt = 0
        total_loss_kd = 0
        total_loss_ofa = 0




        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloadere):
                self.optimizer.zero_grad()
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss_gt = self.loss(output, y)
                total_loss_gt += loss_gt * self.gt_loss_weight






                if self.global_logits != None:
                    if len(y.shape) != 1:
                        target_mask = F.one_hot(y.argmax(-1).long(), self.num_classes)
                    else:
                        target_mask = F.one_hot(y.long(), self.num_classes)

                    ofa_losses_recent = []
                    logits_dict = {}
                    for stage, eps in zip(self.stage_count, self.eps):
                        model_n = self.stage_info(self.modelname, stage).to(self.device)
                        feat = model_n(x)
                        logits_student = self.projectors[str(stage)].to(self.device)(feat)
                        logits_dict[str(stage)] = logits_student
                        logits_dict[str(stage+1)] = None
                    for stage, eps in zip(self.stage_count, self.eps):
                        model_n = self.stage_info(self.modelname, stage).to(self.device)
                        feat = model_n(x)
                        logits_student = self.projectors[str(stage)].to(self.device)(feat)
                        if stage == 4:
                            teacher_logits = torch.stack([self.global_logits[yi.item()] for yi in y])
                        else:
                            teacher_logits = torch.stack([self.global_logits_stage[stage - 1][yi.item()] for yi in y])
                        if logits_dict[str(stage + 1)] is not None:
                            delta_z = torch.abs(self.normalize(logits_dict[str(stage + 1)]) - self.normalize(teacher_logits)).mean()

                            delta_z = delta_z.item()
                            tem = self.G(self.max_tem, delta_z, self.max_z)
                            kl, _ = self.ofa_loss_t(logits_student, teacher_logits, target_mask, eps, tem)
                            ofa_losses_recent.append(kl)
                        else:
                            kl, _ = self.ofa_loss_t(logits_student, teacher_logits, target_mask, eps, self.ofa_temperature)
                            ofa_losses_recent.append(kl)

                    logit_new = copy.deepcopy(output.detach())
                    
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        logit_new[i,:] = self.global_logits[y_c].data.to(self.device)
                        
                    loss_kd = self.loss_mse(output, logit_new.softmax(dim=1)) * self.kd_loss_weight
                    loss_ofa = sum(ofa_losses_recent) * self.ofa_loss_weight
                    total_loss_kd += loss_kd
                    total_loss_ofa += loss_ofa
                
                for i, yy in enumerate(y):
                    yc = yy.item()
                    logits[yc].append(output[i,:].detach().data)
                (loss_kd + loss_ofa + loss_gt).backward()
                self.optimizer.step()

        self.logits = agg_func(logits)


        torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.learning_rate_decay_gamma)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

    def G(self, gamma, delta_z, delta_z_max):
        """
        Calculate the value of function G(j).

        Parameters:
        j (int): the index variable (may not be used directly in this function based on your formula).
        gamma (float): a parameter of the function.
        delta_z (float): the change ΔZ for the given index j.
        delta_z_max (float): the maximum change ΔZ_max across all indices.

        Returns:
        float: the computed value of G(j).
        """
        if delta_z_max == 0:  # Check to prevent division by zero
            return 1  # or handle as appropriate

        numerator = gamma * math.log(1 + delta_z)
        denominator = math.log(1 + delta_z_max)

        G_j = self.begin + (numerator / denominator)
        return G_j
    
    def set_logits(self, global_logits):
        self.global_logits = copy.deepcopy(global_logits)

    def train_metrics(self):
        trainloader = self.load_train_data()

        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        total_loss_gt = 0
        total_loss_ofa = 0
        total_loss_kd = 0
        train_public_num = 0

        class_labels_temp_rate = 0.1

        temp_max = 5
        temp_min = 0.5

        ce_loss = torch.nn.CrossEntropyLoss(reduce=False).cuda()

        class_labels_total_loss_nat = torch.zeros(self.num_classes, dtype=torch.float).cuda()
        total_loss_stage = torch.zeros(3, self.num_classes, dtype=torch.float).cuda()

        global_logits = self.global_logits

        if self.global_logits is not None:
            for i in range(3):
                self.global_logits_stage[i] = copy.deepcopy(global_logits)
            for i in range(self.num_classes):
                self.global_logits[i] = self.global_logits[i] / self.class_labels_temp_nat[i]
                self.global_logits_stage[0][i] = self.global_logits_stage[0][i] / self.class_labels_temp_stage[0][i]
                self.global_logits_stage[1][i] = self.global_logits_stage[1][i] / self.class_labels_temp_stage[1][i]
                self.global_logits_stage[2][i] = self.global_logits_stage[2][i] / self.class_labels_temp_stage[2][i]
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                # LOSS GT
                loss_gt = self.loss(output, y) * self.gt_loss_weight
                total_loss_gt += loss_gt.item() * y.size(0)
                train_num += y.size(0)

                if self.global_logits is not None:
                    
                    if len(y.shape) != 1:
                        target_mask = F.one_hot(y.argmax(-1).long(), self.num_classes)
                    else:
                        target_mask = F.one_hot(y.long(), self.num_classes)

                    # LOSS    
                    loss_ofa_recent = []
                    logits_dict = {}
                    for stage, eps in zip(self.stage_count, self.eps):
                        model_n = self.stage_info(self.modelname, stage).to(self.device)
                        feat = model_n(x)  # 获取特征
                        logits_student = self.projectors[str(stage)].to(self.device)(feat)  # 获取学生模型输出
                        logits_dict[str(stage)] = logits_student
                        logits_dict[str(stage+1)] = None
                    for stage, eps in zip(self.stage_count, self.eps):
                        model_n = self.stage_info(self.modelname, stage).to(self.device)
                        feat = model_n(x)
                        logits_student = self.projectors[str(stage)].to(self.device)(feat)
                        if stage == 4:
                            teacher_logits = torch.stack([self.global_logits[yi.item()] for yi in y])
                        else:
                            teacher_logits = torch.stack([self.global_logits_stage[stage - 1][yi.item()] for yi in y])
                        if logits_dict[str(stage+1)] is not None:
                            delta_z = torch.abs(self.normalize(logits_dict[str(stage+1)]) - self.normalize(teacher_logits)).mean()
                            # Convert the resulting 0-dimensional tensor to a Python float
                            delta_z = delta_z.item()
                            tem = self.G(self.max_tem, delta_z, self.max_z)
                            kl, kls = self.ofa_loss_t(logits_student, teacher_logits, target_mask, eps, tem)
                            loss_ofa_recent.append(kl)

                            for sample_index in range(y.shape[0]):
                                total_loss_stage[stage - 1][y[sample_index]] = total_loss_stage[stage - 1][
                                                                                   y[sample_index]] + \
                                                                               kls[sample_index]
                        else:
                            kl, _ = self.ofa_loss_t(logits_student, teacher_logits, target_mask, eps, self.ofa_temperature)
                            loss_ofa_recent.append(kl)




                    logit_new = copy.deepcopy(output.detach())
                    for i,yy in enumerate(y):    
                        y_c = yy.item()
                        logit_new[i,:] = self.global_logits[y_c].data.to(self.device)
                    loss_kd = self.loss_mse(output, logit_new.softmax(dim=1)) * self.kd_loss_weight
                          


                    loss_ofa = sum(loss_ofa_recent) * self.ofa_loss_weight
                        
                    train_public_num += y.size(0)
                    total_loss_ofa += loss_ofa.item() * y.size(0)
                    total_loss_kd += loss_kd.item() * y.size(0)

                    logit_1 = output.detach()
                    kl_Loss1_record = ce_loss(logit_1.detach(), y.detach())

                    for sample_index in range(y.shape[0]):
                        class_labels_total_loss_nat[y[sample_index]] = class_labels_total_loss_nat[
                                                                           y[
                                                                               sample_index]] + \
                                                                       kl_Loss1_record[sample_index]


        if self.global_logits is not None:
            loss_gap = class_labels_total_loss_nat - torch.mean(class_labels_total_loss_nat)
            self.class_labels_temp_nat = self.class_labels_temp_nat - class_labels_temp_rate * loss_gap / torch.max(
                torch.abs(loss_gap))
            self.class_labels_temp_nat = torch.clamp(self.class_labels_temp_nat, temp_min, temp_max)
            loss_stage = total_loss_stage - torch.mean(total_loss_stage, dim=1, keepdim=True)
            for i in range(3):
                self.class_labels_temp_stage[i] = self.class_labels_temp_stage[
                                                      i] - class_labels_temp_rate * loss_gap / torch.max(
                    torch.abs(loss_stage[i]))
                self.class_labels_temp_stage[i] = torch.clamp(self.class_labels_temp_stage[i], temp_min, temp_max)


        average_loss_gt = total_loss_gt / train_num if train_num > 0 else 0
        average_loss_ofa = total_loss_ofa / train_public_num if train_public_num > 0 else 0
        average_loss_kd = total_loss_kd / train_public_num if train_public_num > 0 else 0

        print(f"client{self.id}")
        print(f"Ground Truth Loss (loss_gt): {average_loss_gt}")
        print(f"OFA Loss (loss_ofa): {average_loss_ofa}")
        print(f"Knowledge Distillation Loss (loss_kd): {average_loss_kd}")

        total_average_loss = (average_loss_gt + average_loss_ofa + average_loss_kd) * train_num

    
        return total_average_loss, train_num

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []


        class_correct = np.zeros(self.num_classes)
        class_total = np.zeros(self.num_classes)

        with torch.no_grad():
            for x, y in testloaderfull:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                # 总体准确率计算
                pred = torch.argmax(output, dim=1)
                correct = (pred == y)

                test_acc += correct.sum().item()
                test_num += y.shape[0]


                for sample_index in range(y.shape[0]):
                    class_correct[y[sample_index]] += (pred[sample_index] == y[sample_index]).item()
                    class_total[y[sample_index]] += 1




                y_prob.append(output.detach().cpu().numpy())

                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)


        class_accuracy = class_correct / (class_total + 1e-10)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc, class_correct, class_total

    def stage_info(self, modelname, stage):
        # Retrieve the model

        # Define stages for each model
        if modelname == 'resnet' or modelname == 'noisy':
            stages = {
                1: nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool),
                2: nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool, self.model.layer1),
                3: nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool, self.model.layer1, self.model.layer2),
                4: nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool, self.model.layer1, self.model.layer2, self.model.layer3),
            }
        elif modelname == 'shufflenet':
            # ShuffleNet stages might be defined differently as they have a different block structure.
            stages = {
                1: nn.Sequential(self.model.conv1, self.model.maxpool),
                2: nn.Sequential(self.model.conv1, self.model.maxpool, self.model.stage2),
                3: nn.Sequential(self.model.conv1, self.model.maxpool, self.model.stage2, self.model.stage3),
                4: nn.Sequential(self.model.conv1, self.model.maxpool, self.model.stage2, self.model.stage3, self.model.stage4),
            }
        elif modelname == 'googlenet':
            stages = {
                1: nn.Sequential(
                    self.model.conv1,
                    self.model.maxpool1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.maxpool2
                ),
                2: nn.Sequential(
                    self.model.conv1,
                    self.model.maxpool1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.maxpool2,
                    self.model.inception3a,
                    self.model.inception3b,
                    self.model.maxpool3
                ),
                3: nn.Sequential(
                    self.model.conv1,
                    self.model.maxpool1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.maxpool2,
                    self.model.inception3a,
                    self.model.inception3b,
                    self.model.maxpool3,
                    self.model.inception4a,
                    self.model.inception4b,
                    self.model.inception4c,
                    self.model.inception4d,
                    self.model.inception4e,
                    self.model.maxpool4
                ),
                4: nn.Sequential(
                    self.model.conv1,
                    self.model.maxpool1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.maxpool2,
                    self.model.inception3a,
                    self.model.inception3b,
                    self.model.maxpool3,
                    self.model.inception4a,
                    self.model.inception4b,
                    self.model.inception4c,
                    self.model.inception4d,
                    self.model.inception4e,
                    self.model.maxpool4,
                    self.model.inception5a,
                    self.model.inception5b,
                    self.model.avgpool
                )
}


        elif modelname == 'alexnet':
            stages = {
                1: nn.Sequential(
                    self.model.conv1,
                ),
                2: nn.Sequential(
                    self.model.conv1,
                    self.model.conv2,
                ),
                3: nn.Sequential(
                    self.model.conv1,
                    self.model.conv2,
                    self.model.conv3,
                ),
                4: nn.Sequential(
                    self.model.conv1,
                    self.model.conv2,
                    self.model.conv3,
                    self.model.avgpool,
                ),
            }

        else:
            raise ValueError(f"Model {modelname} not supported")

        # Return the requested stage
        if stage in stages:
            return stages[stage]
        else:
            raise ValueError(f"Stage {stage} not defined for model {modelname}")
    
    def normalize(self, logit):

        mean = logit.mean(dim=-1, keepdims=True)

        stdv = logit.std(dim=-1, keepdims=True) + 1e-6

        normalized_logit = (logit - mean) / stdv
        normalized_logit += 1e-6
        return normalized_logit


    def ofa_loss_t(self,logits_student, logits_teacher, target_mask, eps, temperature):

        logits_teacher -= torch.max(logits_teacher, dim=1, keepdim=True).values
        logits_student -= torch.max(logits_student, dim=1, keepdim=True).values


        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        prod = (pred_teacher + target_mask) ** eps
        loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
        return loss.mean(), loss


    def write_to_txt(self, data, stage_id,type, filename):
        with open(filename, 'a') as f:
            f.write(f"Stage {stage_id} Type {type}:\n")
            for line in data:
                f.write('\t'.join(map(str, line)) + '\n')

    def ofa_loss_draw(self, logits_student, logits_teacher, target_mask, eps, temperature, stage_id):

        logits_teacher -= torch.max(logits_teacher, dim=1, keepdim=True).values
        logits_student -= torch.max(logits_student, dim=1, keepdim=True).values


        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
       

        pred_student = self.normalize(pred_student)
        pred_teacher = self.normalize(pred_teacher)

# ————————————————————————————————————————————————————————————————

        pred_student_np = pred_student.cpu().detach().numpy()
        pred_teacher_np = pred_teacher.cpu().detach().numpy()

        self.write_to_txt(pred_student_np, stage_id, "pred_student_np",f'logits_new.txt')
        self.write_to_txt(pred_teacher_np, stage_id, "pred_teacher_np",f'logits_new.txt')
# ————————————————————————————————————————————————————————————————
        prod = (pred_teacher + target_mask) ** eps
        loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)

        return loss.mean()

    def _get_feature_dim(self, model_n, modelname, stage):
        if modelname == 'googlenet':
            stage_dims = {
                1: 192,  # After conv3, before maxpool2
                2: 480,  # Output of inception3a
                3: 832,  # Output of inception4a
                4: 1024, # Output before avgpool, after inception5b
            }
            return stage_dims[stage]
        elif modelname == 'resnet' or modelname == 'noisy':
            stage_dims = {
                1: 64,   # After maxpool
                2: 64,   # Output of layer1
                3: 128,  # Output of layer2
                4: 256,  # Output of layer3
                5: 512,  # Output of layer4, before avgpool
            }
            return stage_dims[stage]
        elif modelname == 'shufflenet':
            stage_dims = {
                1: 24,  # After maxpool
                2: 116, # Output of stage2
                3: 232, # Output of stage3
                4: 464, # Output of stage4, before conv5
            }
            return stage_dims[stage]
        elif modelname == 'alexnet':
            stage_dims = {
                1: 64,   # After conv1
                2: 128,  # After conv2
                3: 256,  # After conv3
                4: 256,  # After avgpool, before entering fc (assuming fc input feature size)
            }
            return stage_dims[stage]
        else:
            raise NotImplementedError(f"Model {modelname} not supported for feature dimension extraction.")
    
    def _create_projector(self, stage, modelname):
        model_n = self.stage_info(self.modelname, stage)

        feature_dim = self._get_feature_dim(model_n, modelname, stage)
        intermediate_dim = feature_dim // 2

        projector = nn.Sequential(
            nn.Conv2d(feature_dim, intermediate_dim, kernel_size=3, stride=1, padding=1),  # 3x3卷积层
            nn.BatchNorm2d(intermediate_dim),
            nn.ReLU(inplace=True),      
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(intermediate_dim, intermediate_dim // 2),  
            nn.ReLU(),
            nn.Linear(intermediate_dim // 2, self.num_classes)
    )
        return projector
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
            
# https://github.com/yuetan031/fedlogit/blob/main/lib/utils.py#L205
def agg_func(logits):
    """
    Returns the average of the weights.
    """

    for [label, logit_list] in logits.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            logits[label] = logit / len(logit_list)
        else:
            logits[label] = logit_list[0]

    return logits

def kl_loss(a,b):
    return -a*b + torch.log(b+1e-5)*b

