import numpy as np
import torch
import wandb
import time
from client import clientHdl
from serverbase import Server
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class Hdl(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientHdl)
        
        print("\nevaluate global model on all clients")
        self.global_logits = [None for _ in range(args.num_classes)]

    def train(self):
        cls_tab = wandb.Table(columns=["Epoch"] + [f"Class_{i+1}" for i in range(10)])
        for i in range(self.global_round+1):
            self.selected_clients = self.select_clients()
            if i == 4:
                print('')
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate(i, cls_tab)
            for client in self.selected_clients:
                if client.id not in [21]:
                    client.train()
            s_t = time.time()
            self.receive_logits()
            e_t = time.time()
            sizes_per_client_kb, total_KB, total_MB = compute_uploaded_logit_size(self.uploaded_logits)
            print(f"time:{e_t - s_t:.2f}s, total memory:{total_KB:.2f}")
            self.global_logits = logit_aggregation(self.uploaded_logits)
            print(self.global_logits)
            self.send_logits()
        print(max(self.rs_test_acc))


    def send_logits(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            if client.id not in [21]:
                client.set_logits(self.global_logits)

    def receive_logits(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_logits = []
        for client in self.selected_clients:
            if client.id not in [21]:
                print(f'{client.id}+') 
                self.uploaded_ids.append(client.id)
                self.uploaded_logits.append(client.logits)

        # self.uploaded_logits = self.validate()


    def evaluate(self, round, cls_tab, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        cls_acc = np.zeros(self.num_classes)
        cls_num = np.zeros(self.num_classes)
        cacc = np.zeros(self.num_classes)
        cls_client_acc = np.zeros((len(self.clients), self.num_classes))
        for i in range(self.num_classes):
            for j in range(len(self.selected_clients)):
                if stats[4][j][i] != 0:
                    cls_num[i] += stats[4][j][i]
                    cls_acc[i] += stats[5][j][i]
                    cls_client_acc[j][i] = stats[5][j][i] / stats[4][j][i]
            cacc[i] = cls_acc[i] / cls_num[i]
        cacc = cacc.tolist()

        cls_client_acc = cls_client_acc.tolist()

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))

        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_class_correct = []
        tot_class_num = []
        tot_auc = []
        ids = []
        for c in self.clients:
            if c.id not in [21]:
                ct, ns, auc, cls_ct, cls_ns = c.test_metrics()
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)
                tot_class_correct.append(cls_ct * 1.0)
                tot_class_num.append(cls_ns)
                ids.append(c.id)


        return ids, num_samples, tot_correct, tot_auc, tot_class_num, tot_class_correct


    def validate(self):

        psi = 0.5
        phi = 0.5
        eta = 0.5
        tau = 1.0

        # 获取数据
        z_E = self.uploaded_logits
        z_global = self.global_logits


        if len(z_global.shape) == 2 and z_global.shape[1] == 2:
            z_g = z_global[:, 0]
            z_V = z_global[:, 1]
        else:

            mid = len(z_global) // 2
            z_g = z_global[:mid]
            z_V = z_global[mid:]

        H = min(len(z_E), len(z_g), len(z_V))

        def Z_function(z, tau):

            return z / tau

        def compute_delta_Z(z_a, z_b, tau):

            return Z_function(z_a, tau) - Z_function(z_b, tau)


        valid_indices = []
        validation_details = []


        for i in range(H):

            delta_Z_E_V = compute_delta_Z(z_E[i], z_V[i], tau)
            delta_Z_E_g = compute_delta_Z(z_E[i], z_g[i], tau)
            delta_Z_V_g = compute_delta_Z(z_V[i], z_g[i], tau)


            term1 = np.exp((delta_Z_E_V) ** psi)
            term2 = np.exp((delta_Z_E_g) ** phi)
            numerator_i = term1 + term2


            denominator_i = np.exp((delta_Z_V_g) ** eta)


            numerator_total = 0
            denominator_total = 0
            for j in range(H):
                delta_Z_E_V_j = compute_delta_Z(z_E[j], z_V[j], tau)
                delta_Z_E_g_j = compute_delta_Z(z_E[j], z_g[j], tau)
                delta_Z_V_g_j = compute_delta_Z(z_V[j], z_g[j], tau)

                numerator_total += (np.exp((delta_Z_E_V_j) ** psi) +
                                    np.exp((delta_Z_E_g_j) ** phi))
                denominator_total += np.exp((delta_Z_V_g_j) ** eta)

            left_side = numerator_total / denominator_total if denominator_total != 0 else float('inf')


            Z_E = Z_function(z_E[i], tau)
            Z_g = Z_function(z_g[i], tau)
            Z_V = Z_function(z_V[i], tau)

            exponent = (Z_E ** psi) - (Z_g ** psi + Z_V ** psi) / 2
            right_side_i = 2 * np.exp(exponent)


            right_side_total = 0
            for j in range(H):
                Z_E_j = Z_function(z_E[j], tau)
                Z_g_j = Z_function(z_g[j], tau)
                Z_V_j = Z_function(z_V[j], tau)

                exponent_j = (Z_E_j ** psi) - (Z_g_j ** psi + Z_V_j ** psi) / 2
                right_side_total += np.exp(exponent_j)
            right_side_total *= 2


            is_valid = left_side > right_side_total



            if is_valid:
                valid_indices.append(i)

            validation_details.append({
                'index': i,
                'is_valid': is_valid,
                'left_side': left_side,
                'right_side': right_side_total,
                'sample_contribution': {
                    'numerator': numerator_i,
                    'denominator': denominator_i,
                    'right_side': right_side_i
                }
            })


        if valid_indices:
            self.uploaded_logits = self.uploaded_logits[valid_indices]


            if len(z_global.shape) == 2 and z_global.shape[1] == 2:
                self.global_logits = self.global_logits[valid_indices]
            else:

                valid_z_g = z_g[valid_indices]
                valid_z_V = z_V[valid_indices]
                self.global_logits = np.concatenate([valid_z_g, valid_z_V])


        return self.uploaded_logits

    def save_models(self):
        models = []
        optimizers = []

        for client in self.clients:
            models.append(client.model.state_dict())
            optimizers.append(client.optimizer.state_dict())
        checkpoints = {
            'models': models,
            'optimizers': optimizers
        }
        torch.save(checkpoints, '')
        print(f"\nFinished saving model to ?")
# https://github.com/yuetan031/fedlogit/blob/main/lib/utils.py#L221
def logit_aggregation(local_logits_list):
    agg_logits_label = defaultdict(list)

    for local_logits in local_logits_list:
        for label in local_logits.keys():
            agg_logits_label[label].append(local_logits[label])

    for [label, logit_list] in agg_logits_label.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            agg_logits_label[label] = logit / len(logit_list)
        else:
            agg_logits_label[label] = logit_list[0].data

    return agg_logits_label
            
def compute_uploaded_logit_size(uploaded_logits, dtype_size=4):

    tensor_size_per_class = 10 * dtype_size
    

    sizes_per_client = [len(logits) * tensor_size_per_class for logits in uploaded_logits]
    total_size = sum(sizes_per_client)

    sizes_per_client_kb = [size / 1024 for size in sizes_per_client]
    return sizes_per_client_kb, total_size / 1024, total_size / (1024 ** 2)
