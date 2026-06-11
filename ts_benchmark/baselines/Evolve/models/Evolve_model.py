import torch
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from ts_benchmark.baselines.Evolve.layers.Embed import WarriorsEmbedding, DataEmbedding
from ts_benchmark.baselines.Evolve.layers.Transformer_EncDec import Encoder, EncoderLayer
from ts_benchmark.baselines.Evolve.layers.SelfAttention_Family import FullAttention, AttentionLayer
from einops import rearrange
from transformers import AutoModel
from sklearn.cluster import KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnomalyPatternMemory(nn.Module):
    def __init__(self, max_dynamic=100):
        super().__init__()
        self.max_dynamic = max_dynamic

        self.point_memory = None
        self.subseq_memory = None

        self.dynamic_memory = None

    def init_memory(self, point_feats, subseq_feats):
        self.point_memory = point_feats.clone().detach()
        self.subseq_memory = subseq_feats.clone().detach()

    def update_dynamic(self, dynamic_feats):
        dynamic_feats = dynamic_feats.clone().detach().unsqueeze(1)

        if self.dynamic_memory is None:
            self.dynamic_memory = dynamic_feats[-self.max_dynamic:]
            return

        new_memory = torch.cat([self.dynamic_memory, dynamic_feats], dim=0)

        if new_memory.shape[0] > self.max_dynamic:
            new_memory = new_memory[-self.max_dynamic:]

        self.dynamic_memory = new_memory


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features):
        attn_output, _ = self.attention(features, features, features)
        x = self.norm1(features + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        out = self.norm2(x + self.dropout(ff_output))
        
        return out
    

class SmoothTransformerBlocks(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )

        self.smooth_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ln2(x)
        x_conv = self.smooth_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        return x
    

class DilatedConvBlock(nn.Module):
    def __init__(self, embed_dim, dilation=2):
        super().__init__()
        self.dconv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            groups=embed_dim
        )

        self.proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_trans = x.transpose(1, 2)
        out = self.dconv(x_trans)
        out = self.proj(out)
        out = out.transpose(1, 2)

        return self.ln(out + x)


class ContrastiveBlock(nn.Module):
    def __init__(self, embed_dim, kernel=5):
        super().__init__()
        self.smooth = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=kernel,
            padding=kernel//2,
            groups=embed_dim
        )

        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_smooth = self.smooth(x.transpose(1, 2)).transpose(1, 2)

        high_freq = x - x_smooth
        high_freq = self.ln(high_freq)
        out = x + high_freq

        return out


class EVOLVEModel(nn.Module):
    def __init__(self, configs):
        super(EVOLVEModel, self).__init__()

        # Parameters
        self.device = device
        self.configs = configs    
        self.batch_size = configs.batch_size
        self.c_in = configs.c_in
        self.seq_len = configs.seq_len
        self.pre_len = configs.pre_len
        self.d_model = configs.d_model
        self.hidden_dim = configs.hidden_dim
        self.dropout = configs.dropout
        self.n_clusters = configs.n_clusters
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.latent_size = configs.latent_size
        self.K = configs.K
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.dilation = configs.dilation

        # Anomaly Pattern Memory
        self.Anomaly_Pattern_Memory = AnomalyPatternMemory()

        # Embedding layers
        self.normal_data_embedding = DataEmbedding(self.configs.seq_len, self.configs.hidden_dim, self.configs.dropout)
        self.anomaly_pattern_embedding = DataEmbedding(self.configs.seq_len, self.configs.hidden_dim, self.configs.dropout)
        self.pre_series_embedding = DataEmbedding(self.configs.pre_len, self.configs.hidden_dim, self.configs.dropout)

        # Foundation model encoder
        self.Foundation_Model = AutoModel.from_pretrained("ts_benchmark/baselines/pre_train/submodules/DADA", trust_remote_code=True).model
        self.Foundation_Model_Encoder = self.Foundation_Model.encoder

        # MLP
        self.Head = nn.Linear(self.d_model, self.pre_len, bias=True)
        self.Mu_H = nn.Linear(self.d_model, self.latent_size, bias=True)
        self.Sigma_H = nn.Linear(self.d_model, self.latent_size, bias=True)
        self.Z_MLP = nn.Linear(self.latent_size, self.d_model, bias=True)
        self.Z_Normal_MLP = nn.Linear(self.d_model, self.d_model, bias=True)
        self.Z_Abnormal_MLP = nn.Linear(self.d_model, self.d_model, bias=True)
        self.Logits_MLP = nn.Linear(self.d_model + self.latent_size, 2, bias=True)
        self.pattern_selector = nn.Sequential(nn.Linear(self.d_model, 64),nn.ReLU(),nn.Linear(64, 3))

        # Transformer
        self.Z_Transformer = TransformerBlock(self.d_model, self.n_heads, self.d_ff)
        self.Smooth_Transformer = SmoothTransformerBlocks(self.d_model, self.n_heads, self.d_ff)
        self.DilatedConv = DilatedConvBlock(self.d_model, self.dilation)
        self.Contrastive_Block = ContrastiveBlock(self.d_model)
        self.memory_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, batch_first=True)


    def generate_point_anomaly(self, T, n_samples, intensity=3.0):
        data = []
        for _ in range(n_samples):
            x = torch.randn(T)
            pos = torch.randint(0, T, (1,))
            direction = 1 if torch.rand(1) > 0.5 else -1
            x[pos] += direction * intensity * (0.5 + torch.rand(1))
            data.append(x)
        return torch.stack(data)


    def generate_multi_pattern_anomaly(self, T, n_samples, max_length_ratio=0.2):
        data = []
        labels = []
        
        for _ in range(n_samples):
            x = torch.randn(T)
            max_L = max(3, int(T * max_length_ratio))
            L = torch.randint(2, max_L, (1,)).item()
            start = torch.randint(0, T - L, (1,)).item()

            mode = np.random.choice(['trend', 'stuck', 'sine', 'variance'])
            
            if mode == 'trend':
                slope = torch.randn(1) * 0.5
                trend = torch.linspace(0, L * slope.item(), L)
                x[start:start+L] += trend
                
            elif mode == 'stuck':
                stuck_val = x[start]
                x[start:start+L] = stuck_val + torch.randn(L) * 0.05
                
            elif mode == 'sine':
                freq = np.random.uniform(1, 4)
                subseq = torch.sin(torch.linspace(0, freq * torch.pi, L))
                x[start:start+L] += subseq * 2.0
                
            elif mode == 'variance':
                x[start:start+L] *= 4.0
                
            data.append(x)
        
        return torch.stack(data)

    
    def Anomaly_Pattern_Memory_Initialization(self, T, n_point=100, n_subseq=100, subseq_ratio=0.1):
        point_seqs = self.generate_point_anomaly(T, n_point).to(device)
        subseq_seqs = self.generate_multi_pattern_anomaly(T, n_subseq, subseq_ratio).to(device)
        point_seqs = point_seqs.unsqueeze(1)
        subseq_seqs = subseq_seqs.unsqueeze(1)

        with torch.no_grad():
            point_emb = self.anomaly_pattern_embedding(point_seqs)
            subseq_emb = self.anomaly_pattern_embedding(subseq_seqs) 

            point_feats = self.Foundation_Model_Encoder(point_emb)
            subseq_feats = self.Foundation_Model_Encoder(subseq_emb)

        self.Anomaly_Pattern_Memory.init_memory(point_feats, subseq_feats)


    def retrieve_from_memory(self, x_query, memory_bank):
        if memory_bank is None:
            return torch.zeros_like(x_query)

        N, T, D = memory_bank.shape
        BC = x_query.shape[0]
        q = torch.mean(x_query, dim=1, keepdim=True) 
            
        mem_repr = torch.mean(memory_bank, dim=1)
        k = mem_repr.unsqueeze(0).expand(BC, -1, -1)
        v = mem_repr.unsqueeze(0).expand(BC, -1, -1)

        attn_output, attn_weights = self.memory_attention(q, k, v)
        best_idx = torch.argmax(attn_weights.squeeze(1), dim=1)
        best_match = memory_bank[best_idx]
            
        return best_match

    
    def Dynamic_Noise_Injection(self, x_abnormal):
        B, T, D = x_abnormal.shape

        cand_point = self.retrieve_from_memory(x_abnormal, self.Anomaly_Pattern_Memory.point_memory)
        cand_subseq = self.retrieve_from_memory(x_abnormal, self.Anomaly_Pattern_Memory.subseq_memory)
        cand_dynamic = self.retrieve_from_memory(x_abnormal, self.Anomaly_Pattern_Memory.dynamic_memory)
        candidates = torch.stack([cand_point, cand_subseq, cand_dynamic], dim=1)

        x_repr = torch.mean(x_abnormal, dim=1)
        scores = self.pattern_selector(x_repr)
        selection_mask = F.gumbel_softmax(scores, tau=1.0, hard=True, dim=1)
        mask_expanded = selection_mask.unsqueeze(-1).unsqueeze(-1)
        final_anomaly_feat = torch.sum(candidates * mask_expanded, dim=1)

        strength = torch.sigmoid(self.noise_strength)
        x_noisy = x_abnormal + strength * final_anomaly_feat
            
        return strength, x_noisy


    def ALR_Module(self, x_encoder):
        mu = self.Mu_H(x_encoder) 
        sigma = self.Sigma_H(x_encoder)

        eps = torch.randn((self.K,) + sigma.shape, device=sigma.device, dtype=sigma.dtype)
        z = mu.unsqueeze(0) + eps * sigma.unsqueeze(0) 
        K, BC, T, Dz = z.shape
        Dx = x_encoder.shape[-1]
        x_expanded = x_encoder.unsqueeze(0).expand(K, BC, T, Dx)
        zx = torch.cat([z, x_expanded], dim=-1) 

        logits = self.Logits_MLP(zx)
        logits = torch.mean(logits, dim=2)

        return z, logits


    def Dual_Structured_Decoder(self, z, logits):
        # Shared Branch
        z = rearrange(z, 'k (b c) t dz -> (k b c) t dz', k = self.K, c = self.c_in, t = 1, dz = self.latent_size)
        logits = rearrange(logits, 'k (b c) m -> (k b c) m', k = self.K, c = self.c_in, m = 2)
        z_mlp = self.Z_MLP(z)
        z_transformer = self.Z_Transformer(z_mlp)
        z_normal = z_transformer.clone()
        z_abnormal = z_transformer.clone()

        # Normal Branch
        z_normal = self.Z_Normal_MLP(z_normal)
        z_normal = self.Smooth_Transformer(z_normal)

        # Abnormal Branch
        z_abnormal = self.Z_Abnormal_MLP(z_abnormal)
        z_abnormal = self.DilatedConv(z_abnormal)
        z_abnormal = self.Contrastive_Block(z_abnormal)

        # Weighted Sum
        weights = torch.softmax(logits, dim=-1)
        w_normal   = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        w_abnormal = weights[:, 1].unsqueeze(-1).unsqueeze(-1)
        y_decoder = w_normal * z_normal + w_abnormal * z_abnormal

        y_decoder = rearrange(y_decoder, '(k b c) t dx -> k (b c) t dx', k = self.K, c = self.c_in, t = 1, dx = self.d_model)

        return y_decoder


    def Anomaly_Pattern_Analysis(self, y_abnormal_decoder, y_true, n_clusters):
        y_abnormal_decoder = rearrange(y_abnormal_decoder, 'k (b c) h d -> k b c h d', k = self.K, c = self.c_in, h = 1, d = self.d_model)
        y_abnormal_decoder = y_abnormal_decoder.mean(dim=1, keepdim=True)
        N, B, C, H, D = y_abnormal_decoder.shape
        y_true = y_true.mean(dim=0, keepdim=True)

        actual_n_clusters = min(n_clusters, N)
        if actual_n_clusters < 1:
            return
        
        features = y_abnormal_decoder.detach().cpu().reshape(N, -1).numpy()
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=0, n_init=10)
        kmeans.fit(features)

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=y_abnormal_decoder.device)
        centers = centers.reshape(actual_n_clusters, C, H, D)

        for i in range(actual_n_clusters):
            feat_proto = centers[i].squeeze(1)
            proto_seq = self.Head(feat_proto).squeeze(-1) 
            proto_seq = rearrange(proto_seq, '(b c) h -> b h c', c = self.c_in, h = self.pre_len)
            pure_anomaly_seq = proto_seq - y_true
            input_seq = pure_anomaly_seq.mean(dim=0, keepdim=True)
            input_seq = rearrange(input_seq, 'b h c -> (b c) h', b = 1, c = self.c_in, h = self.pre_len).unsqueeze(1)
            emb = self.pre_series_embedding(input_seq)
            pure_feat = self.Foundation_Model_Encoder(emb).squeeze(1)
            self.Anomaly_Pattern_Memory.update_dynamic(pure_feat)

    
    def Anomaly_Prediction_Train(self, x, y):
        # -------------------------------------------------------------Input data normalization--------------------------------------------------------------------
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        B, T, C = x.size()
        x = x.permute(0, 2, 1).contiguous()
        x = rearrange(x, 'b c l -> (b c) l').unsqueeze(1)

        # --------------------------------------------------------------------Embedding----------------------------------------------------------------------------
        x_embedding = self.normal_data_embedding(x)

        # ---------------------------------------------------------------------Encoder-----------------------------------------------------------------------------
        x_encoder = self.Foundation_Model_Encoder(x_embedding)

        # --------------------------------------------------------------------Add Noise----------------------------------------------------------------------------
        x_normal_encoder = x_encoder.clone()
        x_abnormal_encoder = x_encoder.clone()
        strength, x_abnormal_encoder = self.Dynamic_Noise_Injection(x_abnormal_encoder)

        # --------------------------------------------------------------------ALR Module---------------------------------------------------------------------------
        z_normal, logits_normal = self.ALR_Module(x_normal_encoder)
        z_abnormal, logits_abnormal = self.ALR_Module(x_abnormal_encoder)

        # ---------------------------------------------------------------------Decoder-----------------------------------------------------------------------------
        y_normal_decoder = self.Dual_Structured_Decoder(z_normal, logits_normal)
        y_abnormal_decoder = self.Dual_Structured_Decoder(z_abnormal, logits_abnormal)

        # --------------------------------------------------------------Anomaly Pattern Analysis-------------------------------------------------------------------
        self.Anomaly_Pattern_Analysis(y_abnormal_decoder, y, self.n_clusters)

        # ----------------------------------------------------------------------Head-------------------------------------------------------------------------------
        y_normal = self.Head(y_normal_decoder).squeeze(2)
        y_abnormal = self.Head(y_abnormal_decoder).squeeze(2)

        # ----------------------------------------------------------------Inverse normalization--------------------------------------------------------------------
        y_normal = rearrange(y_normal, 'n (b c) t -> n b t c', t = self.pre_len, c = self.c_in)
        y_abnormal = rearrange(y_abnormal, 'n (b c) t -> n b t c', t = self.pre_len, c = self.c_in)
        
        std_y  = stdev[:, 0, :].unsqueeze(0).unsqueeze(2)
        mean_y = means[:, 0, :].unsqueeze(0).unsqueeze(2)

        y_normal   = y_normal * std_y + mean_y
        y_abnormal = y_abnormal * std_y + mean_y       

        return y_normal, y_abnormal, logits_normal, logits_abnormal, strength
    

    def Anomaly_Prediction_Test(self, x, y):
        # -------------------------------------------------------------Input data normalization--------------------------------------------------------------------
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        B, T, C = x.size()
        x = x.permute(0, 2, 1).contiguous()
        x = rearrange(x, 'b c l -> (b c) l').unsqueeze(1)

        # --------------------------------------------------------------------Embedding----------------------------------------------------------------------------
        x_embedding = self.normal_data_embedding(x)

        # ---------------------------------------------------------------------Encoder-----------------------------------------------------------------------------
        x_encoder = self.Foundation_Model_Encoder(x_embedding)

        # --------------------------------------------------------------------ALR Module---------------------------------------------------------------------------
        z, logits = self.ALR_Module(x_encoder)

        # ---------------------------------------------------------------------Decoder-----------------------------------------------------------------------------
        y_decoder = self.Dual_Structured_Decoder(z, logits)

        # ----------------------------------------------------------------------Head-------------------------------------------------------------------------------
        y_output = self.Head(y_decoder).squeeze(2)

        # ----------------------------------------------------------------Inverse normalization--------------------------------------------------------------------
        y_output = rearrange(y_output, 'n (b c) t -> n b t c', t = self.pre_len, c = self.c_in)
        
        std_y  = stdev[:, 0, :].unsqueeze(0).unsqueeze(2)
        mean_y = means[:, 0, :].unsqueeze(0).unsqueeze(2)

        y_output   = y_output * std_y + mean_y 

        return y_output

    
    def forward(self, flag, x, y):
        if flag in ['train', 'val']:
            y_normal, y_abnormal, logits_normal, logits_abnormal, strength = self.Anomaly_Prediction_Train(x, y)
            return y_normal, y_abnormal, logits_normal, logits_abnormal, strength
        else:
            y_output = self.Anomaly_Prediction_Test(x, y)
            return y_output
