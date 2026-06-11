import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
import torch.nn.functional as F
from ts_benchmark.baselines.Evolve.models.Evolve_model import EVOLVEModel
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, anomaly_prediction_data_provider
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.baselines.Evolve.utils.tools import EarlyStopping, adjust_learning_rate
from torch import optim
import time
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_EVOLVE_BASED_HYPER_PARAMS = {
    "top_k": 3,
    "enc_in": 4,
    "dec_in": 4,
    "c_out": 4,
    "e_layers": 1,
    "d_layers": 1,
    "d_model": 256,
    "d_ff": 256,
    "embed": "timeF",
    "freq": "h",
    "lradj": "type1",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 72,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 6,
    "patch_size": 6,
    "stride": 6,
    "dropout": 0.1,
    "batch_size": 16,
    "lr": 0.0001,
    "num_epochs": 3,
    "num_workers": 0,
    "loss": "MSE",
    "itr": 1,
    "distil": True,
    "patience": 3,
    "task_name": "anomaly_detection",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "mem_dim": 32,
    "anomaly_ratio": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    "conv_kernel": [12, 16],
    "use_norm": True,
    "parallel_strategy": "DP",
    "num_epochs": 3,
    "seq_len": 72,
    "pre_len": 72,
    "dilation": 3,
    "n_clusters": 3,
    "c_in": 1,
    "latent_size": 128,
    "K": 10,
    "hidden_dim": 64,
    "lamda": 1,
}


def f_abnormal_strength(strength, alpha=0.5, mode='quadratic'):
    if isinstance(strength, torch.Tensor):
        strength = strength.mean()
        
    if mode == 'linear':
        return alpha * strength
    elif mode == 'quadratic':
        return alpha * (strength ** 2)
    elif mode == 'exponential':
        return alpha * (torch.exp(strength) - 1)
    else:
        return strength

def compute_adversarial_loss(logits_normal, logits_abnormal):
    log_prob_abnormal = F.log_softmax(logits_abnormal, dim=-1)
    prob_normal = F.softmax(logits_normal, dim=-1).detach()
    adv_loss = F.kl_div(log_prob_abnormal, prob_normal, reduction='batchmean')
    
    return adv_loss


class EVOLVEConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_EVOLVE_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.parallel_strategy not in [None, 'DP']:
            raise ValueError("Invalid value for parallel_strategy. Supported values are 'DP' and None.")

    @property
    def pred_len(self):
        return self.pre_len

    @property
    def learning_rate(self):
        return self.lr
    
    @property
    def model_name(self):
        return "Evolve"
    

class Evolve:
    def __init__(self, **kwargs):
        super(Evolve, self).__init__()
        self.config = EVOLVEConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.seq_len = self.config.seq_len
        self.pre_len = self.config.pre_len

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

    def detect_validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for input_x, input_y, label in valid_data_loader:
                flag = 'val'
                input_x = input_x.to(device)
                input_y = input_y.to(device)

                y_normal, y_abnormal, logits_normal, logits_abnormal, strength = self.model(flag, input_x, input_y)
                true = input_y.detach() 

                # --- Normal Branch Loss ---
                var_normal = torch.var(y_normal, dim=0, unbiased=False)
                loss_var_normal = var_normal.mean()

                mean_normal = torch.mean(y_normal, dim=0)
                loss_mse_normal = criterion(mean_normal, input_y)

                # --- Abnormal Branch Loss ---
                raw_var_abnormal = torch.var(y_abnormal, dim=0, unbiased=False).mean()
                raw_mse_abnormal = torch.mean((torch.mean(y_abnormal, dim=0) - input_y) ** 2)

                margin = f_abnormal_strength(strength.mean(), alpha=1.0, mode='quadratic')
                loss_var_abnormal = F.relu(margin - raw_var_abnormal)
                loss_mse_abnormal = F.relu(margin - raw_mse_abnormal)

                # --- Logits Loss ---
                BC, K, _ = logits_normal.shape
                target_zeros = torch.zeros((BC, K), dtype=torch.long, device=device)
                target_ones = torch.ones((BC, K), dtype=torch.long, device=device)

                loss_logits_normal = F.cross_entropy(logits_normal.view(-1, 2), target_zeros.view(-1))
                loss_logits_abnormal = F.cross_entropy(logits_abnormal.view(-1, 2), target_ones.view(-1))        

                # --- Total Loss ---
                loss_normal = loss_var_normal + loss_mse_normal
                loss_abnormal = loss_var_abnormal + loss_mse_abnormal
                loss_logits = loss_logits_normal + loss_logits_abnormal

                loss = loss_normal + loss_abnormal + loss_logits
                total_loss.append(loss.item())

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss
    
    def detect_fit(self, train_data: pd.DataFrame, train_label: pd.DataFrame):
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.model = EVOLVEModel(self.config)

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_label_value, valid_label = train_val_split(train_label, 0.8, None)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_prediction_data_provider(
            valid_data,
            valid_label,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            pre_len=config.pre_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_prediction_data_provider(
            train_data_value,
            train_label_value,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            pre_len=config.pre_len,
            step=1,
            mode="train",
        )

        time_now = time.time()

        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(self.device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )


        adv_param_names = ['noise_strength', 'pattern_selector', 'memory_attention']
        adv_params = []
        main_params = []

        for name, param in self.model.named_parameters():
            if any(k in name for k in adv_param_names):
                adv_params.append(param)
            else:
                main_params.append(param)

        optimizer_main = optim.Adam(main_params, lr=config.lr)
        optimizer_adv = optim.Adam(adv_params, lr=config.lr)        

        _, T, _ = next(iter(self.train_data_loader))[0].size()

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.Anomaly_Pattern_Memory_Initialization(T)
        else:
            self.model.Anomaly_Pattern_Memory_Initialization(T)

        for epoch in range(config.num_epochs):
            iter_count = 0
            self.model.train()
            start_fit_time = time.time()
            for i, (input_x, input_y, label) in enumerate(self.train_data_loader):
                iter_count += 1
                train_steps = len(self.train_data_loader)
                flag = 'train'
                input_x = input_x.float().to(self.device)
                input_y = input_y.float().to(self.device)

                # =========================================================
                optimizer_main.zero_grad()
                y_normal, y_abnormal, logits_normal, logits_abnormal, strength = self.model(flag, input_x, input_y)
                
                # --- Normal Branch Loss ---
                var_normal = torch.var(y_normal, dim=0, unbiased=False)
                loss_var_normal = var_normal.mean()
                
                mean_normal = torch.mean(y_normal, dim=0)
                loss_mse_normal = self.criterion(mean_normal, input_y)
                
                # --- Abnormal Branch Loss ---
                raw_var_abnormal = torch.var(y_abnormal, dim=0, unbiased=False).mean()
                raw_mse_abnormal = torch.mean((torch.mean(y_abnormal, dim=0) - input_y) ** 2)

                margin = f_abnormal_strength(strength.mean(), alpha=1.0, mode='quadratic')
                loss_var_abnormal = F.relu(margin - raw_var_abnormal)
                loss_mse_abnormal = F.relu(margin - raw_mse_abnormal)
                
                # --- Logits Loss (Classification) ---
                BC, K, _ = logits_normal.shape
                target_zeros = torch.zeros((BC, K), dtype=torch.long, device=self.device)
                target_ones = torch.ones((BC, K), dtype=torch.long, device=self.device)
                
                loss_logits_n = F.cross_entropy(logits_normal.view(-1, 2), target_zeros.view(-1))
                loss_logits_a = F.cross_entropy(logits_abnormal.view(-1, 2), target_ones.view(-1))
                
                # --- Main Total Loss & Update ---
                loss_main = (loss_var_normal + loss_mse_normal) + (loss_var_abnormal + loss_mse_abnormal) + (loss_logits_n + loss_logits_a)

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss_main: {2:.7f}".format(i + 1, epoch + 1, loss_main.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((config.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    time_now = time.time()

                loss_main.backward()
                optimizer_main.step()
                
                # =========================================================
                optimizer_adv.zero_grad()
                y_normal_2, y_abnormal_2, logits_normal_2, logits_abnormal_2, strength_2 = self.model(flag, input_x, input_y)
                
                # --- Adversarial Loss ---
                loss_adv = self.config.lamda * compute_adversarial_loss(logits_normal_2, logits_abnormal_2)

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss_adv: {2:.7f}".format(i + 1, epoch + 1, loss_adv.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((config.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss_adv.backward()
                optimizer_adv.step()
            valid_loss = self.detect_validate(self.valid_data_loader, self.criterion)

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

            adjust_learning_rate(optimizer_main, epoch + 1, config)
            adjust_learning_rate(optimizer_adv, epoch + 1, config)

        end_fit_time = time.time()
        fit_time = end_fit_time - start_fit_time
  

    def detect_score(self, test: pd.DataFrame, test_label: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = anomaly_prediction_data_provider(
            test,
            test_label,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            pre_len=config.pre_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        test_labels = []

        for i, (batch_x, batch_y, batch_l) in enumerate(self.thre_loader):
            flag = 'test'
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            # reconstruction
            outputs = self.model(flag, batch_x, batch_y)
            # criterion
            score = torch.mean(torch.var(outputs, dim=0, unbiased=False), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    
    def detect_label(self, test: pd.DataFrame, test_label: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = anomaly_prediction_data_provider(
            test,
            test_label,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            pre_len=config.pre_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_prediction_data_provider(
            test,
            test_label,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            pre_len=config.pre_len,
            step=config.pre_len,
            mode="thre",
        )

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        test_labels = []

        start_infer_time = time.time()
        for i, (batch_x, batch_y, batch_l) in enumerate(self.thre_loader):
            flag = 'test'
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            # reconstruction
            outputs = self.model(flag, batch_x, batch_y)
            # criterion
            score = torch.mean(torch.var(outputs, dim=0, unbiased=False), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_l)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        end_infer_time = time.time()
        infer_time = end_infer_time - start_infer_time


        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            # threshold = np.percentile(combined_energy, 100 - ratio)
            threshold = np.percentile(test_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        # pad test_energy
        test_energy = np.pad(
            test_energy,
            (self.seq_len, 0),
            mode="constant",
            constant_values=0,
        )

        # pad preds for each ratio
        for ratio in preds:
            preds[ratio] = np.pad(
                preds[ratio],
                (self.seq_len, 0),
                mode="constant",
                constant_values=0,
            )

        return preds, test_energy

    
    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name
