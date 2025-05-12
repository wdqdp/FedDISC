from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from .scoremodel import ScoreNet, loss_fn, Euler_Maruyama_sampler, ScoreNet_U
# from .diffusionmodel import UNet_conditional, EMA, loss_diff
# from .conditional_ddpm import UNetModel, GaussianDiffusion
from .conditional_ddim import UNetModel, GaussianDiffusion
import functools
from .rcan import Group
from random import sample
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import umap.umap_ as umap
from torch_geometric.nn import RGCNConv, GraphConv

from .module import *
from .graph import batch_graphify



class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse
    
# Set up the SDE (SDE is used to define Diffusion Process)
device = 'cuda'
def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.as_tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.as_tensor(sigma ** t, device=device)

class Reshape(nn.Module):
    def __init__(self, out_channels, out_length):
        super(Reshape, self).__init__()
        self.out_channels = out_channels
        self.out_length = out_length

    def forward(self, x):
        # x 的形状为 (bs, out_channels * out_length)
        return x.view(x.size(0), self.out_channels, self.out_length)

# class Diffusion:
#     def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, dim=32, seq=48, device="cuda"):
#         self.noise_steps = noise_steps
#         self.beta_start = beta_start
#         self.beta_end = beta_end

#         self.beta = self.prepare_noise_schedule().to(device)
#         self.alpha = 1. - self.beta
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0)

#         self.dim = dim
#         self.seq = seq
#         self.device = device

#     def prepare_noise_schedule(self):
#         return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

#     def noise_images(self, x, t):
#         sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
#         sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
#         Ɛ = torch.randn_like(x)
#         return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

#     def sample_timesteps(self, n):
#         return torch.randint(low=1, high=self.noise_steps, size=(n,))

#     def sample(self, model, n, condition, cfg_scale=0):
#         model.eval()
#         with torch.no_grad():
#             x = torch.randn((n, self.dim, self.seq)).to(self.device)
#             # for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
#             for i in reversed(range(1, self.noise_steps)):
#                 t = (torch.ones(n) * i).long().to(self.device)
#                 predicted_noise = model(x, t, condition)
#                 if cfg_scale > 0:
#                     uncond_predicted_noise = model(x, t, None)
#                     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
#                 alpha = self.alpha[t][:, None, None]
#                 alpha_hat = self.alpha_hat[t][:, None, None]
#                 beta = self.beta[t][:, None, None]
#                 if i > 1:
#                     noise = torch.randn_like(x)
#                 else:
#                     noise = torch.zeros_like(x)
#                 x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
#         model.train()
#         # x = (x.clamp(-1, 1) + 1) / 2
#         # x = (x * 255).type(torch.uint8)
#         x = x.clamp(-1,1)
#         return x


class FedDISC_train_diff(nn.Module):
    def __init__(self, args, multi_type=0, need_tsne=0):
        super(FedDISC_train_diff, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                                pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads, self.length = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims   
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout   
        self.embed_dropout = args.embed_dropout 
        self.res_dropout = args.res_dropout
        self.text_dropout = args.text_dropout
        self.output_dropout = args.output_dropout
        self.attn_mask = args.attn_mask
        self.generate_internal = args.generate_internal
        self.train_start_epoch = args.train_start_epoch
        self.local_epoch = args.local_epoch
        self.MSE = MSE()
        self.mse = nn.MSELoss()
        self.multi_type = multi_type  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 6:lva
        self.label_embedding = nn.Embedding(num_embeddings=args.num_classes, embedding_dim=32)
        self.temp = args.temperature
        self.need_tsne = need_tsne
        self.l_max, self.l_min = args.l_max_min[0], args.l_max_min[1]
        self.v_max, self.v_min = args.v_max_min[0], args.v_max_min[1]
        self.a_max, self.a_min = args.a_max_min[0], args.a_max_min[1]
        self.timesteps = 1000
        self.num_class = args.num_classes

        sigma = 15.0 # 25.0
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)  # used for sample
        # self.score_l = ScoreNet_U(marginal_prob_std=self.marginal_prob_std_fn)
        # self.score_v = ScoreNet_U(marginal_prob_std=self.marginal_prob_std_fn)
        # self.score_a = ScoreNet_U(marginal_prob_std=self.marginal_prob_std_fn)

        # self.score_l_va = UNet_conditional(num_classes=args.num_classes)
        # self.score_v_la = UNet_conditional(num_classes=args.num_classes)
        # self.score_a_lv = UNet_conditional(num_classes=args.num_classes)

        self.score_l_va = UNetModel(label_num=args.num_classes)
        self.score_v_la = UNetModel(label_num=args.num_classes)
        self.score_a_lv = UNetModel(label_num=args.num_classes)

        self.diffusion = GaussianDiffusion(timesteps=self.timesteps)

        combined_dim_3 = 2 * (self.d_l + self.d_a + self.d_v)
        combined_dim_2 = 2 * (self.d_l + self.d_a)
        combined_dim_1 = 2 * (self.d_l)

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # 1. Temporal convolutional layers
        # self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)
        self.proj_l = nn.Sequential(
            nn.Linear(self.orig_d_l, self.d_l * self.length),
            Reshape(self.d_l, self.length)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(self.orig_d_a, self.d_a * self.length),
            Reshape(self.d_a, self.length)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(self.orig_d_v, self.d_v * self.length),
            Reshape(self.d_v, self.length)
        )

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=2)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=2)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=2)

        # Projection layers
                # 单模态
        self.out_proj_l = nn.Sequential(
            nn.Linear(combined_dim_1, combined_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_1, combined_dim_1)
        )
        self.out_proj_v = nn.Sequential(
            nn.Linear(combined_dim_1, combined_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_1, combined_dim_1)
        )
        self.out_proj_a = nn.Sequential(
            nn.Linear(combined_dim_1, combined_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout+0.1),
            nn.Linear(combined_dim_1, combined_dim_1)
        )
        # 双模态
        self.out_proj_la = nn.Sequential(
            nn.Linear(combined_dim_2, combined_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_2, combined_dim_2)
        )
        self.out_proj_va = nn.Sequential(
            nn.Linear(combined_dim_2, combined_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_2, combined_dim_2)
        )
        self.out_proj_lv = nn.Sequential(
            nn.Linear(combined_dim_2, combined_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_2, combined_dim_2)
        )

        self.proj1_3 = nn.Linear(combined_dim_3, combined_dim_3)
    
        self.proj2_3 = nn.Linear(combined_dim_3, combined_dim_3)

        self.out_layer_1_l = nn.Linear(combined_dim_1, output_dim)
        self.out_layer_1_v = nn.Linear(combined_dim_1, output_dim)
        self.out_layer_1_a = nn.Linear(combined_dim_1, output_dim)

        self.out_layer_2_la = nn.Linear(combined_dim_2, output_dim)
        self.out_layer_2_va = nn.Linear(combined_dim_2, output_dim)
        self.out_layer_2_lv = nn.Linear(combined_dim_2, output_dim)

        self.out_layer_3 = nn.Linear(combined_dim_3, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, labels, local_epoch, epoch, pseudo_labels, batch_idx, pre_data, num_modal=None):
        # with torch.no_grad():
        # if self.use_bert:
        #     text = self.text_model(text)  # (bs, 3, 50)->(bs, 50, 768)
        # x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)  # (bs, 768, 50)
        # x_a = audio.transpose(1, 2)   # (bs, 5, 50)
        # x_v = video.transpose(1, 2)   # (bs, 20, 50)
        pre_x_v = None
        pre_x_l = None
        pre_x_a = None

        condition_label = torch.nn.functional.one_hot(labels, num_classes=self.num_class).to(torch.float)
        
        x_l = text
        x_a = audio
        x_v = video

        generate_en = False
        output = torch.tensor(0)
        # Project the textual/visual/audio features
        # with torch.no_grad():
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)  # (bs, 32, 48)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)  # (bs, 32, 48)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)  # (bs, 32, 48)

        # proj_x_l = self.ln_l(proj_x_l.transpose(1, 2)).transpose(1, 2) if self.orig_d_l != self.d_l else proj_x_l
        # proj_x_a = self.ln_a(proj_x_a.transpose(1, 2)).transpose(1, 2) if self.orig_d_a != self.d_a else proj_x_a
        # proj_x_v = self.ln_v(proj_x_v.transpose(1, 2)).transpose(1, 2) if self.orig_d_v != self.d_v else proj_x_v

        gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a

            
        if self.multi_type == 3: # text & vision  lva
            condition_l = F.softmax(pseudo_labels[1]/self.temp,dim=1)
            condition_v = F.softmax(pseudo_labels[0]/self.temp,dim=1)
            # loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=label_emb)
            # loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=label_emb)
            loss_score_l = loss_diff(self.score_l, proj_x_l, condition=condition_l)
            loss_score_v = loss_diff(self.score_v, proj_x_v, condition=condition_v)
            loss_score_a = torch.tensor(0)

            # Generate samples from score-based models with the Euler_Maruyama_sampler
            if (epoch+1)%self.generate_internal == 0 and local_epoch+1 == self.local_epoch:
                pre_x_v = self.diffusion.sample(self.score_v, n=len(labels), condition=condition_v)
                pre_x_l = self.diffusion.sample(self.score_l, n=len(labels), condition=condition_l)
                proj_x_a = proj_x_a
            # pre_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
            #                                     device='cuda', condition=label_emb)
            # pre_x_a = None
            # pre_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
            #                                     device='cuda', condition=label_emb)
                # pre_x_v = self.rec_v(pre_x_v)
                # pre_x_l = self.rec_l(pre_x_l)
                # loss_rec = self.MSE(pre_x_v, gt_v)+self.MSE(pre_x_l, gt_l)

                proj_x_v = pre_x_v.permute(2, 0, 1)
                proj_x_l = pre_x_l.permute(2, 0, 1)
                proj_x_a = proj_x_a.permute(2, 0, 1)

                # classifier
                # h_l_with_vs = self.trans_l_with_v(pre_x_l, pre_x_v, pre_x_v)
                # h_v_with_ls = self.trans_v_with_l(pre_x_v, pre_x_l, pre_x_l)
                # h_ls = torch.cat([pre_x_l, h_l_with_vs], dim=2)
                # h_vs = torch.cat([pre_x_v, h_v_with_ls], dim=2)
                # h_ls = self.trans_l_mem(h_ls)
                # h_vs = self.trans_v_mem(h_vs)
                # if type(h_ls) == tuple:
                #     h_ls = h_ls[0]
                # last_h_l = h_ls[-1] 
                # if type(h_vs) == tuple:
                #     h_vs = h_vs[0]
                # last_h_v = h_vs[-1] 
                # last_hs = torch.cat([last_h_l, last_h_v], dim=1)

                # # A residual block
                # last_hs_proj = self.proj2_2(
                #     F.dropout(F.relu(self.proj1_2(last_hs), inplace=True), p=self.output_dropout, training=self.training))
                # last_hs_proj += last_hs

                # output = self.out_layer_2(last_hs_proj)
                
                 # (V,A) --> L
                h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
                h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
                h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
                h_ls = self.trans_l_mem(h_ls)
                if type(h_ls) == tuple:
                    h_ls = h_ls[0]
                last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

                # (L,V) --> A
                h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
                h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
                h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
                h_as = self.trans_a_mem(h_as)
                if type(h_as) == tuple:
                    h_as = h_as[0]
                last_h_a = last_hs = h_as[-1]

                # (L,A) --> V
                h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
                h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
                h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
                h_vs = self.trans_v_mem(h_vs)
                if type(h_vs) == tuple:
                    h_vs = h_vs[0]
                last_h_v = last_hs = h_vs[-1]

                last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

                # A residual block
                last_hs_proj = self.proj2_3(
                    F.dropout(F.relu(self.proj1_3(last_hs), inplace=True), p=self.output_dropout, training=self.training))
                last_hs_proj += last_hs

                output = self.out_layer_3(last_hs_proj)
                generate_en = True
                     
        elif self.multi_type == 4: # text & audio
            condition_l = F.softmax(pseudo_labels[2]/self.temp,dim=1)
            condition_a = F.softmax(pseudo_labels[0]/self.temp,dim=1)
            # loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=condition_l)
            # loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=condition_a)
            loss_score_l = loss_diff(self.score_l, proj_x_l, condition=condition_l)
            loss_score_a = loss_diff(self.score_a, proj_x_a, condition=condition_a)
            loss_score_v = torch.tensor(0)
            # Generate samples from score-based models with the Euler_Maruyama_sampler
            pre_x_v = None
            # pre_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
            #                                     device='cuda', condition=label_emb)
            # pre_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
            #                                     device='cuda', condition=label_emb)

            # pre_x_a = self.rec_a(pre_x_a)
            # pre_x_l = self.rec_l(pre_x_l)
            # loss_rec = self.MSE(pre_x_a, gt_a)+self.MSE(pre_x_l, gt_l)
            if (epoch+1)%self.generate_internal == 0 and local_epoch+1 == self.local_epoch:
                pre_x_a = self.diffusion.sample(self.score_a, n=len(labels), condition=condition_a)
                pre_x_l = self.diffusion.sample(self.score_l, n=len(labels), condition=condition_l)

                proj_x_a = pre_x_a.permute(2, 0, 1)
                proj_x_l = pre_x_l.permute(2, 0, 1)
                proj_x_v = proj_x_v.permute(2, 0, 1)
                # classifier
                # h_l_with_as = self.trans_l_with_a(pre_x_l, pre_x_a, pre_x_a)
                # h_a_with_ls = self.trans_a_with_l(pre_x_a, pre_x_l, pre_x_l)
                # h_ls = torch.cat([pre_x_l, h_l_with_as], dim=2)
                # h_as = torch.cat([pre_x_a, h_a_with_ls], dim=2)
                # h_ls = self.trans_l_mem(h_ls)
                # h_as = self.trans_v_mem(h_as)
                # if type(h_ls) == tuple:
                #     h_ls = h_ls[0]
                # last_h_l = h_ls[-1] 
                # if type(h_as) == tuple:
                #     h_as = h_as[0]
                # last_h_a = h_as[-1] 
                # last_hs = torch.cat([last_h_l, last_h_a], dim=1)

                # # A residual block
                # last_hs_proj = self.proj2_2(
                #     F.dropout(F.relu(self.proj1_2(last_hs), inplace=True), p=self.output_dropout, training=self.training))
                # last_hs_proj += last_hs

                # output = self.out_layer_2(last_hs_proj)
                 # (V,A) --> L
                h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
                h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
                h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
                h_ls = self.trans_l_mem(h_ls)
                if type(h_ls) == tuple:
                    h_ls = h_ls[0]
                last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

                # (L,V) --> A
                h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
                h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
                h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
                h_as = self.trans_a_mem(h_as)
                if type(h_as) == tuple:
                    h_as = h_as[0]
                last_h_a = last_hs = h_as[-1]

                # (L,A) --> V
                h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
                h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
                h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
                h_vs = self.trans_v_mem(h_vs)
                if type(h_vs) == tuple:
                    h_vs = h_vs[0]
                last_h_v = last_hs = h_vs[-1]

                last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

                # A residual block
                last_hs_proj = self.proj2_3(
                    F.dropout(F.relu(self.proj1_3(last_hs), inplace=True), p=self.output_dropout, training=self.training))
                last_hs_proj += last_hs

                output = self.out_layer_3(last_hs_proj)
                generate_en = True
                
        elif self.multi_type == 5: # audiuo & vision
            condition_v = F.softmax(pseudo_labels[2]/self.temp,dim=1)
            condition_a = F.softmax(pseudo_labels[1]/self.temp,dim=1)
            # loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=condition_v)
            # loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=condition_a)
            loss_score_v = loss_diff(self.score_v, proj_x_v, condition=condition_v)
            loss_score_a = loss_diff(self.score_a, proj_x_a, condition=condition_a)
            loss_score_l = torch.tensor(0)
            # Generate samples from score-based models with the Euler_Maruyama_sampler
            # pre_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
            #                                     device='cuda', condition=label_emb)
            # pre_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
            #                                     device='cuda', condition=label_emb)
            # pre_x_l = None
            # pre_x_a = self.rec_a(pre_x_a)
            # pre_x_v = self.rec_v(pre_x_v)
            # loss_rec = self.MSE(pre_x_a, gt_a)+self.MSE(pre_x_v, gt_v)

            if (epoch+1)%self.generate_internal == 0 and local_epoch+1 == self.local_epoch:
                pre_x_a = self.diffusion.sample(self.score_a, n=len(labels), condition=condition_a)
                pre_x_v = self.diffusion.sample(self.score_v, n=len(labels), condition=condition_v)
                proj_x_a = pre_x_a.permute(2, 0, 1)
                proj_x_v = pre_x_v.permute(2, 0, 1)
                proj_x_l = proj_x_l.permute(2, 0, 1)
                # classifier
                # h_v_with_as = self.trans_v_with_a(pre_x_v, pre_x_a, pre_x_a)
                # h_a_with_vs = self.trans_a_with_v(pre_x_a, pre_x_v, pre_x_v)
                # h_vs = torch.cat([pre_x_v, h_v_with_as], dim=2)
                # h_as = torch.cat([pre_x_a, h_a_with_vs], dim=2)
                # h_ls = self.trans_l_mem(h_vs)
                # h_as = self.trans_v_mem(h_as)
                # if type(h_vs) == tuple:
                #     h_vs = h_vs[0]
                # last_h_v = h_vs[-1] 
                # if type(h_as) == tuple:
                #     h_as = h_as[0]
                # last_h_a = h_as[-1] 
                # last_hs = torch.cat([last_h_v, last_h_a], dim=1)

                # # A residual block
                # last_hs_proj = self.proj2_2(
                #     F.dropout(F.relu(self.proj1_2(last_hs), inplace=True), p=self.output_dropout, training=self.training))
                # last_hs_proj += last_hs

                # output = self.out_layer_2(last_hs_proj)
                 # (V,A) --> L
                h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
                h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
                h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
                h_ls = self.trans_l_mem(h_ls)
                if type(h_ls) == tuple:
                    h_ls = h_ls[0]
                last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

                # (L,V) --> A
                h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
                h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
                h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
                h_as = self.trans_a_mem(h_as)
                if type(h_as) == tuple:
                    h_as = h_as[0]
                last_h_a = last_hs = h_as[-1]

                # (L,A) --> V
                h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
                h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
                h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
                h_vs = self.trans_v_mem(h_vs)
                if type(h_vs) == tuple:
                    h_vs = h_vs[0]
                last_h_v = last_hs = h_vs[-1]

                last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

                # A residual block
                last_hs_proj = self.proj2_3(
                    F.dropout(F.relu(self.proj1_3(last_hs), inplace=True), p=self.output_dropout, training=self.training))
                last_hs_proj += last_hs

                output = self.out_layer_3(last_hs_proj)
                generate_en = True
        else:  
            condition_lv = F.softmax(pseudo_labels[0]/self.temp,dim=1)  
            condition_la = F.softmax(pseudo_labels[1]/self.temp,dim=1) 
            condition_va = F.softmax(pseudo_labels[2]/self.temp,dim=1)
            condition_va = F.softmax((condition_va+condition_label)/self.temp, dim=1)

            # 输出acc
            # pre = torch.argmax(condition_la, dim=1)
            # acc = (pre==labels).float().mean()
            # print(acc.item())
            # pre = torch.argmax(condition_va, dim=1)
            # acc = (pre==labels).float().mean()
            # print(acc.item())

            # condition_va = None  # cat two avail modalities as conditions
            # condition_lv = None  # cat two avail modalities as conditions
            # condition_la = None

            # l_max_val = torch.max(proj_x_l)  # 最大值 7
            # l_min_val = torch.min(proj_x_l)  # 最小值 1
            # v_max_val = torch.max(proj_x_v)  # 最大值 7
            # v_min_val = torch.min(proj_x_v)  # 最小值 1
            # a_max_val = torch.max(proj_x_a)  # 最大值 7
            # a_min_val = torch.min(proj_x_a)  # 最小值 1
            # print('l:', l_max_val.item(), l_min_val.item())
            # print('v:', v_max_val.item(), v_min_val.item())
            # print('a:', a_max_val.item(), a_min_val.item())

            # 对特征归一化
            proj_x_l_norm = 2 * (proj_x_l - self.l_min) / (self.l_max - self.l_min) - 1
            proj_x_a_norm = 2 * (proj_x_a - self.a_min) / (self.a_max - self.a_min) - 1
            proj_x_v_norm = 2 * (proj_x_v - self.v_min) / (self.v_max - self.v_min) - 1

            t = torch.randint(0, self.timesteps, (labels.shape[0],), device=device).long()

            loss_score_l = self.diffusion.train_losses(self.score_l_va, proj_x_l_norm, t, condition_va)    # condition_va / labels
            loss_score_v = self.diffusion.train_losses(self.score_v_la, proj_x_v_norm, t, condition_la)    # condition_la
            loss_score_a = self.diffusion.train_losses(self.score_a_lv, proj_x_a_norm, t, condition_lv)    # condition_lv

            if (epoch+1)>=self.train_start_epoch and (epoch+1)%self.generate_internal==0 and local_epoch+1 == self.local_epoch:    # and batch_idx==0
                # pre_x_v = self.diffusion.sample(self.score_v_la, n=len(labels), condition=condition_la)
                # pre_x_l = self.diffusion.sample(self.score_l_va, n=len(labels), condition=condition_va)
                # pre_x_a = self.diffusion.sample(self.score_a_lv, n=len(labels), condition=condition_lv)

                pre_x_v = self.diffusion.sample(self.score_v_la, condition_la, self.length, batch_size=len(labels), channels=self.d_l)   # condition_la / labels
                pre_x_l = self.diffusion.sample(self.score_l_va, condition_va, self.length, batch_size=len(labels), channels=self.d_l)   # condition_va
                pre_x_a = self.diffusion.sample(self.score_a_lv, condition_lv, self.length, batch_size=len(labels), channels=self.d_l)   # condition_lv

                # 对生成的视觉特征进行逆归一化
                pre_x_v = (pre_x_v + 1) / 2 * (self.v_max - self.v_min) + self.v_min
                # 对生成的文本特征进行逆归一化
                pre_x_l = (pre_x_l + 1) / 2 * (self.l_max - self.l_min) + self.l_min
                # 对生成的音频特征进行逆归一化
                pre_x_a = (pre_x_a + 1) / 2 * (self.a_max - self.a_min) + self.a_min

                # pre_x_l_cat = torch.cat([proj_x_l, pre_x_l],dim=0)
                # pre_x_v_cat = torch.cat([proj_x_v, pre_x_v],dim=0)
                # pre_x_a_cat = torch.cat([proj_x_a, pre_x_a],dim=0)
                # labels_cat = torch.cat([labels, labels], dim=0)

                if self.need_tsne == 1:
                    bs = proj_x_l.shape[0]

                    # 将特征从 (bs, 32, 48) 展平为 (bs, 32*48)
                    feat_l_orig = proj_x_l.reshape(bs, -1).detach().cpu().numpy()
                    feat_a_orig = proj_x_a.reshape(bs, -1).detach().cpu().numpy()
                    feat_v_orig = proj_x_v.reshape(bs, -1).detach().cpu().numpy()

                    feat_l_gen = pre_x_l.reshape(bs, -1).detach().cpu().numpy()
                    feat_a_gen = pre_x_a.reshape(bs, -1).detach().cpu().numpy()
                    feat_v_gen = pre_x_v.reshape(bs, -1).detach().cpu().numpy()

                    # 将所有特征拼接起来进行 t-SNE 降维
                    all_features = np.concatenate([
                        feat_l_orig, feat_a_orig, feat_v_orig,
                        feat_l_gen, feat_a_gen, feat_v_gen
                    ], axis=0)


                    # 对所有特征进行标准化（归一化为零均值和单位方差）
                    scaler = StandardScaler()
                    all_features_norm = scaler.fit_transform(all_features)

                    tsne = TSNE(n_components=2, random_state=42)
                    features_2d = tsne.fit_transform(all_features_norm)

                    # # 使用 UMAP 进行降维，将特征降到 2D 空间
                    # reducer = umap.UMAP(n_components=2, random_state=42)
                    # features_2d = reducer.fit_transform(all_features_norm)

                    # 根据顺序划分 t-SNE 后的特征：
                    feat_l_orig_2d = features_2d[:bs]
                    feat_a_orig_2d = features_2d[bs:2*bs]
                    feat_v_orig_2d = features_2d[2*bs:3*bs]
                    feat_l_gen_2d  = features_2d[3*bs:4*bs]
                    feat_a_gen_2d  = features_2d[4*bs:5*bs]
                    feat_v_gen_2d  = features_2d[5*bs:6*bs]

                    # 绘制 t-SNE 图
                    plt.figure(figsize=(8, 8))
                    plt.scatter(feat_l_orig_2d[:, 0], feat_l_orig_2d[:, 1],
                                color='darkred', label='Text Orig')
                    plt.scatter(feat_l_gen_2d[:, 0], feat_l_gen_2d[:, 1],
                                color='salmon', label='Text Gen')

                    plt.scatter(feat_a_orig_2d[:, 0], feat_a_orig_2d[:, 1],
                                color='darkblue', label='Audio Orig')
                    plt.scatter(feat_a_gen_2d[:, 0], feat_a_gen_2d[:, 1],
                                color='lightblue', label='Audio Gen')

                    plt.scatter(feat_v_orig_2d[:, 0], feat_v_orig_2d[:, 1],
                                color='darkgreen', label='Visual Orig')
                    plt.scatter(feat_v_gen_2d[:, 0], feat_v_gen_2d[:, 1],
                                color='lightgreen', label='Visual Gen')

                    plt.legend()
                    plt.title("t-SNE of original and diffusion-generated features")

                    # 保存图像到本地，不显示图像
                    save_path = Path('result') / 'FedDMER' / 'IEMOCAP6' / 'tsne_condition_11'   # MOSI/IEMOCAP4/IEMOCAP6
                    save_path.mkdir(parents=True, exist_ok=True)
                    model_save_path = save_path / ('tsne_modal_'+str(epoch)+'.png')
                    plt.savefig(model_save_path, dpi=300)
                    plt.close()
            
                proj_x_a = pre_x_a.permute(2, 0, 1)
                proj_x_v = pre_x_v.permute(2, 0, 1)
                proj_x_l = pre_x_l.permute(2, 0, 1)
                # classifier
                # (V,A) --> L
                h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
                h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
                h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
                h_ls = self.trans_l_mem(h_ls)
                if type(h_ls) == tuple:
                    h_ls = h_ls[0]
                last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

                # (L,V) --> A
                h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
                h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
                h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
                h_as = self.trans_a_mem(h_as)
                if type(h_as) == tuple:
                    h_as = h_as[0]
                last_h_a = last_hs = h_as[-1]

                # (L,A) --> V
                h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
                h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
                h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
                h_vs = self.trans_v_mem(h_vs)
                if type(h_vs) == tuple:
                    h_vs = h_vs[0]
                last_h_v = last_hs = h_vs[-1]

                last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

                # A residual block
                last_hs_proj = self.proj2_3(
                    F.dropout(F.relu(self.proj1_3(last_hs), inplace=True), p=self.output_dropout, training=self.training))
                last_hs_proj += last_hs

                output = self.out_layer_3(last_hs_proj)
                generate_en = True

            elif (epoch+1)>=self.train_start_epoch and local_epoch+1 == self.local_epoch:
                pre_x_l = pre_data['pre_x_l'] 
                pre_x_v = pre_data['pre_x_v'] 
                pre_x_a = pre_data['pre_x_a']
                labels = pre_data['pre_label']

                # pre_x_l_cat = torch.cat([proj_x_l, pre_x_l],dim=0)
                # pre_x_v_cat = torch.cat([proj_x_v, pre_x_v],dim=0)
                # pre_x_a_cat = torch.cat([proj_x_a, pre_x_a],dim=0)

                proj_x_a = pre_x_a.permute(2, 0, 1)
                proj_x_v = pre_x_v.permute(2, 0, 1)
                proj_x_l = pre_x_l.permute(2, 0, 1)
                # classifier
                # (V,A) --> L
                h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
                h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
                h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
                h_ls = self.trans_l_mem(h_ls)
                if type(h_ls) == tuple:
                    h_ls = h_ls[0]
                last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

                # (L,V) --> A
                h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
                h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
                h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
                h_as = self.trans_a_mem(h_as)
                if type(h_as) == tuple:
                    h_as = h_as[0]
                last_h_a = last_hs = h_as[-1]

                # (L,A) --> V
                h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
                h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
                h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
                h_vs = self.trans_v_mem(h_vs)
                if type(h_vs) == tuple:
                    h_vs = h_vs[0]
                last_h_v = last_hs = h_vs[-1]

                last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

                # A residual block
                last_hs_proj = self.proj2_3(
                    F.dropout(F.relu(self.proj1_3(last_hs), inplace=True), p=self.output_dropout, training=self.training))
                last_hs_proj += last_hs

                output = self.out_layer_3(last_hs_proj)
                generate_en = True

            
        res = {
            'pre_x_l': pre_x_l,
            'pre_x_a': pre_x_a,
            'pre_x_v': pre_x_v,
            'labels': labels,
            'loss_score_l': loss_score_l,
            'loss_score_v': loss_score_v,
            'loss_score_a': loss_score_a,
            # 'loss_rec': loss_rec,
            'output': output,
            'generate_en': generate_en
        }
        return res
            

class FedDISC_train_all(nn.Module):
    def __init__(self, args, multi_type=0):
        super(FedDISC_train_all, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.llen, self.alen, self.vlen = tuple(x-2 for x in args.seq_lens)
        self.prompt_dim = dst_feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.MSE = MSE()
        self.multi_type = multi_type  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 5:lva

        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        sigma = 25.0
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)  # used for sample
        self.score_l = ScoreNet_U(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_v = ScoreNet_U(marginal_prob_std=self.marginal_prob_std_fn)
        self.score_a = ScoreNet_U(marginal_prob_std=self.marginal_prob_std_fn)

        self.cat_lv = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_la = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        self.cat_va = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)

        self.rec_l = nn.Sequential(
            nn.Conv1d(self.d_l, self.d_l*2, 1),
            Group(num_channels=self.d_l*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_l*2, self.d_l, 1)
        )

        self.rec_v = nn.Sequential(
            nn.Conv1d(self.d_v, self.d_v*2, 1),
            Group(num_channels=self.d_v*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_v*2, self.d_v, 1)
        )

        self.rec_a = nn.Sequential(
            nn.Conv1d(self.d_a, self.d_a*2, 1),
            Group(num_channels=self.d_a*2, num_blocks=20, reduction=16),
            nn.Conv1d(self.d_a*2, self.d_a, 1)
        )

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 1.5 modality-signal prompts
        self.promptl_m = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_m = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_m = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))
        self.promptl_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, num_modal=None):
        # with torch.no_grad():
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        # Project the textual/visual/audio features
        # with torch.no_grad():
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a

        #  random select modality  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 5:lva
        # modal_idx = [0, 1, 2]  # (0:text, 1:vision, 2:audio)
        # ava_modal_idx = sample(modal_idx, num_modal)  # sample available modality
        if self.multi_type == 0: # text only
            conditions = None
            loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
            loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
            loss_score_l = torch.tensor(0).cuda()
            # Generate samples from score-based models with the Euler_Maruyama_sampler
            pre_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                device='cuda', condition=conditions)
            pre_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                  device='cuda', condition=conditions)
            proj_x_a = self.rec_a(pre_x_a)
            proj_x_v = self.rec_v(pre_x_v)
    
            proj_x_a = proj_x_a + self.prompta_m
            proj_x_v = proj_x_v + self.promptv_m
            proj_x_l = proj_x_l + self.promptl_nm

        elif self.multi_type == 1: # vision only
            conditions = None
            loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
            loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
            loss_score_v = torch.tensor(0).cuda()
            # Generate samples from score-based models with the Euler_Maruyama_sampler
            pre_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                device='cuda', condition=conditions)
            pre_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                device='cuda', condition=conditions)
            proj_x_a = self.rec_a(pre_x_a)
            proj_x_l = self.rec_l(pre_x_l)

            proj_x_a = proj_x_a + self.prompta_m
            proj_x_v = proj_x_v + self.promptv_nm
            proj_x_l = proj_x_l + self.promptl_m

        elif self.multi_type == 2: # audio only
            conditions = None
            loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
            loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
            loss_score_a = torch.tensor(0).cuda()
            # Generate samples from score-based models with the Euler_Maruyama_sampler
            pre_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                device='cuda', condition=conditions)
            pre_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                device='cuda', condition=conditions)
            proj_x_v = self.rec_v(pre_x_v)
            proj_x_l = self.rec_l(pre_x_l)

            proj_x_a = proj_x_a + self.prompta_nm
            proj_x_v = proj_x_v + self.promptv_m
            proj_x_l = proj_x_l + self.promptl_m

        elif self.multi_type == 3: # text & vision
            # conditions = self.cat_lv(torch.cat([proj_x_l, proj_x_v], dim=1))  # cat two avail modalities as conditions
            conditions = None
            loss_score_a = loss_fn(self.score_a, proj_x_a, self.marginal_prob_std_fn, condition=conditions)
            loss_score_l, loss_score_v = torch.tensor(0).cuda(), torch.tensor(0).cuda()
            # Generate samples from score-based models with the Euler_Maruyama_sampler
            pre_x_a = Euler_Maruyama_sampler(self.score_a, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                device='cuda', condition=conditions)
            proj_x_a = self.rec_a(pre_x_a)

            proj_x_a = proj_x_a + self.prompta_m
            proj_x_v = proj_x_v + self.promptv_nm
            proj_x_l = proj_x_l + self.promptl_nm

        elif self.multi_type == 4: # text & audio
            # conditions = self.cat_la(torch.cat([proj_x_l, proj_x_a], dim=1))  # cat two avail modalities as conditions
            conditions = None
            loss_score_v = loss_fn(self.score_v, proj_x_v, self.marginal_prob_std_fn, condition=conditions)
            loss_score_l, loss_score_a = torch.tensor(0).cuda(), torch.tensor(0).cuda()
            # Generate samples from score-based models with the Euler_Maruyama_sampler
            pre_x_v = Euler_Maruyama_sampler(self.score_v, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                device='cuda', condition=conditions)
            proj_x_v = self.rec_v(pre_x_v)
            proj_x_a = proj_x_a + self.prompta_nm
            proj_x_v = proj_x_v + self.promptv_m
            proj_x_l = proj_x_l + self.promptl_nm

        elif self.multi_type == 5: # audiuo & vision
            # conditions = self.cat_va(torch.cat([proj_x_v, proj_x_a], dim=1))  # cat two avail modalities as conditions
            conditions = None
            loss_score_l = loss_fn(self.score_l, proj_x_l, self.marginal_prob_std_fn, condition=conditions)
            loss_score_v, loss_score_a = torch.tensor(0).cuda(), torch.tensor(0).cuda()
            # Generate samples from score-based models with the Euler_Maruyama_sampler
            pre_x_l = Euler_Maruyama_sampler(self.score_l, self.marginal_prob_std_fn, self.diffusion_coeff_fn, text.size(0),
                                                device='cuda', condition=conditions)
            proj_x_l = self.rec_l(pre_x_l)
            proj_x_a = proj_x_a + self.prompta_nm
            proj_x_v = proj_x_v + self.promptv_nm
            proj_x_l = proj_x_l + self.promptl_m

        else:
            loss_score_l, loss_score_v, loss_score_a = torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()
            proj_x_a = proj_x_a + self.prompta_nm
            proj_x_v = proj_x_v + self.promptv_nm
            proj_x_l = proj_x_l + self.promptl_nm


        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)



        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        # A residual block
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        res = {
            'Feature_t': last_h_l,
            'Feature_a': last_h_a,
            'Feature_v': last_h_v,
            'Feature_f': last_hs,
            'loss_score_l': loss_score_l.type(torch.float32),
            'loss_score_v': loss_score_v.type(torch.float32),
            'loss_score_a': loss_score_a.type(torch.float32),
            'M': output
        }
        return res
    

class FedDISC_train_class(nn.Module):
    def __init__(self, args, multi_type=0, need_tsne=0):
        super(FedDISC_train_class, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads, self.length = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.llen, self.alen, self.vlen = tuple(x-2 for x in args.seq_lens)
        self.prompt_dim = dst_feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.MSE = MSE()
        self.multi_type = multi_type  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 5:lva
        self.temp = args.temperature
        self.need_tsne = need_tsne
        self.l_max, self.l_min = args.l_max_min[0], args.l_max_min[1]
        self.v_max, self.v_min = args.v_max_min[0], args.v_max_min[1]
        self.a_max, self.a_min = args.a_max_min[0], args.a_max_min[1]
        self.timesteps = 1000
        self.num_class = args.num_classes
        self.diffusion_model = args.diffusion_model

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        sigma = 25.0
        self.marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        self.diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)  # used for sample

        # self.score_l = UNet_conditional(num_classes=args.num_classes)
        # self.score_v = UNet_conditional(num_classes=args.num_classes)
        # self.score_a = UNet_conditional(num_classes=args.num_classes)

        self.score_l_va = UNetModel(label_num=args.num_classes)
        self.score_v_la = UNetModel(label_num=args.num_classes)
        self.score_a_lv = UNetModel(label_num=args.num_classes)

        self.diffusion = GaussianDiffusion(timesteps=self.timesteps)

        combined_dim_3 = 2 * (self.d_l + self.d_a + self.d_v)
        combined_dim_2 = 2 * (self.d_l + self.d_a)
        combined_dim_1 = 2 * (self.d_l)

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # self.precomputed_data_train = {}  # 新增缓存属性
        # self.precomputed_data_test = {}  # 新增缓存属性

        # 1. Temporal convolutional layers
        self.proj_l = nn.Sequential(
            nn.Linear(self.orig_d_l, self.d_l * self.length),
            Reshape(self.d_l, self.length)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(self.orig_d_a, self.d_a * self.length),
            Reshape(self.d_a, self.length)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(self.orig_d_v, self.d_v * self.length),
            Reshape(self.d_v, self.length)
        )

        # 1.5 modality-signal prompts
        self.promptl_m = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_m = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_m = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))
        self.promptl_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers

        self.proj1_3 = nn.Linear(combined_dim_3, combined_dim_3)
    
        self.proj2_3 = nn.Linear(combined_dim_3, combined_dim_3)

        self.out_layer_3 = nn.Linear(combined_dim_3, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, labels, pseudo_labels, epoch=-1, num_modal=None):
        # with torch.no_grad():
        x_l = text
        x_a = audio
        x_v = video
        feat_tsne_dict = None
        modalities = None

        condition_label = torch.nn.functional.one_hot(labels, num_classes=self.num_class).to(torch.float)
        # Project the textual/visual/audio features
        # with torch.no_grad():
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)  # (bs, 32, 48)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)  # (bs, 32, 48)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)  # (bs, 32, 48)

        gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a
        
        #  random select modality  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 5:lva

        if self.multi_type == 0: # text only
            condition_l = F.softmax(pseudo_labels[0]/self.temp,dim=1)
            condition_mix = F.softmax((condition_l+condition_label)/self.temp, dim=1)
            if self.diffusion_model == 'ddpm':
                pre_x_a = self.diffusion.sample(self.score_a_lv, condition_label, self.length, batch_size=len(labels), channels=self.d_l)   # condition_lv
                pre_x_v = self.diffusion.sample(self.score_v_la, condition_label, self.length, batch_size=len(labels), channels=self.d_l)   # condition_la / labels  ddim_sample
            elif self.diffusion_model == 'ddim':
                pre_x_a = self.diffusion.ddim_sample(self.score_a_lv, condition_mix, self.length, batch_size=len(labels), channels=self.d_l)   # condition_lv
                pre_x_v = self.diffusion.ddim_sample(self.score_v_la, condition_mix, self.length, batch_size=len(labels), channels=self.d_l)   # condition_la / labels  

            # 对生成的视觉特征进行逆归一化
            pre_x_v = (pre_x_v + 1) / 2 * (self.v_max - self.v_min) + self.v_min
            pre_x_a = (pre_x_a + 1) / 2 * (self.a_max - self.a_min) + self.a_min

            if self.need_tsne == 1:
                bs = proj_x_l.shape[0]

                # 将特征从 (bs, 32, 48) 展平为 (bs, 32*48)
                feat_l_orig = proj_x_l.reshape(bs, -1).detach().cpu().numpy()
                feat_a_orig = proj_x_a.reshape(bs, -1).detach().cpu().numpy()
                feat_v_orig = proj_x_v.reshape(bs, -1).detach().cpu().numpy()

                feat_a_gen = pre_x_a.reshape(bs, -1).detach().cpu().numpy()
                feat_v_gen = pre_x_v.reshape(bs, -1).detach().cpu().numpy()

                # feat_l_orig = proj_x_l.detach().cpu().numpy()
                # feat_a_orig = proj_x_a.detach().cpu().numpy()
                # feat_v_orig = proj_x_v.detach().cpu().numpy()

                # feat_a_gen = pre_x_a.detach().cpu().numpy()
                # feat_v_gen = pre_x_v.detach().cpu().numpy()


                # 将所有特征拼接起来进行 t-SNE 降维
                all_features = np.concatenate([
                    feat_l_orig, feat_a_orig, feat_v_orig,
                    feat_a_gen,feat_v_gen
                ], axis=0)

                modalities = ['l_orig', 'a_orig', 'v_orig', 'a_gen', 'v_gen']
                feat_tsne_dict={name:all_features[i*bs:(i+1)*bs] for i, name in enumerate(modalities)}

            proj_x_a = pre_x_a + self.prompta_m
            proj_x_v = pre_x_v + self.promptv_m
            proj_x_l = proj_x_l + self.promptl_nm

        elif self.multi_type == 1: # vision only
            condition_v = F.softmax(pseudo_labels[1]/self.temp,dim=1)
            condition_mix = F.softmax((condition_v+condition_label)/self.temp, dim=1)

            if self.diffusion_model == 'ddpm':
                pre_x_l = self.diffusion.sample(self.score_l_va, condition_label, self.length, batch_size=len(labels), channels=self.d_l)   # condition_va
                pre_x_a = self.diffusion.sample(self.score_a_lv, condition_label, self.length, batch_size=len(labels), channels=self.d_l)   # condition_lv  ddim_sample
            elif self.diffusion_model == 'ddim':
                pre_x_l = self.diffusion.ddim_sample(self.score_l_va, condition_mix, self.length, batch_size=len(labels), channels=self.d_l)   # condition_va
                pre_x_a = self.diffusion.ddim_sample(self.score_a_lv, condition_mix, self.length, batch_size=len(labels), channels=self.d_l)   # condition_lv  ddim_sample
            # 对生成的音频特征进行逆归一化
            pre_x_l = (pre_x_l + 1) / 2 * (self.l_max - self.l_min) + self.l_min
            pre_x_a = (pre_x_a + 1) / 2 * (self.a_max - self.a_min) + self.a_min

            if self.need_tsne == 1:
                bs = proj_x_l.shape[0]

                # 将特征从 (bs, 32, 48) 展平为 (bs, 32*48)
                feat_l_orig = proj_x_l.reshape(bs, -1).detach().cpu().numpy()
                feat_a_orig = proj_x_a.reshape(bs, -1).detach().cpu().numpy()
                feat_v_orig = proj_x_v.reshape(bs, -1).detach().cpu().numpy()

                feat_a_gen = pre_x_a.reshape(bs, -1).detach().cpu().numpy()
                feat_l_gen = pre_x_l.reshape(bs, -1).detach().cpu().numpy()

                # feat_l_orig = proj_x_l.detach().cpu().numpy()
                # feat_a_orig = proj_x_a.detach().cpu().numpy()
                # feat_v_orig = proj_x_v.detach().cpu().numpy()

                # feat_a_gen = pre_x_a.detach().cpu().numpy()
                # feat_l_gen = pre_x_l.detach().cpu().numpy()

                # 将所有特征拼接起来进行 t-SNE 降维
                all_features = np.concatenate([
                    feat_l_orig, feat_a_orig, feat_v_orig,
                    feat_a_gen,feat_l_gen
                ], axis=0)

                modalities = ['l_orig', 'a_orig', 'v_orig', 'a_gen', 'l_gen']
                feat_tsne_dict={name:all_features[i*bs:(i+1)*bs] for i, name in enumerate(modalities)}

                # # 对所有特征进行标准化（归一化为零均值和单位方差）
                # scaler = StandardScaler()
                # all_features_norm = scaler.fit_transform(all_features)

                # tsne = TSNE(n_components=2, random_state=42)
                # features_2d = tsne.fit_transform(all_features_norm)

                # # 根据顺序划分 t-SNE 后的特征：
                # feat_l_orig_2d = features_2d[:bs]
                # feat_a_orig_2d = features_2d[bs:2*bs]
                # feat_v_orig_2d = features_2d[2*bs:3*bs]
                # feat_a_gen_2d  = features_2d[3*bs:4*bs]
                # feat_l_gen_2d  = features_2d[4*bs:5*bs]


                # # 绘制 t-SNE 图
                # plt.figure(figsize=(8, 8))
                # plt.scatter(feat_l_orig_2d[:, 0], feat_l_orig_2d[:, 1],
                #             color='darkred', label='Text Orig')
                # plt.scatter(feat_l_gen_2d[:, 0], feat_l_gen_2d[:, 1],
                #                 color='salmon', label='Text Gen')
                # plt.scatter(feat_a_orig_2d[:, 0], feat_a_orig_2d[:, 1],
                #             color='darkblue', label='Audio Orig')
                # plt.scatter(feat_a_gen_2d[:, 0], feat_a_gen_2d[:, 1],
                #             color='lightblue', label='Audio Gen')
                # plt.scatter(feat_v_orig_2d[:, 0], feat_v_orig_2d[:, 1],
                #             color='darkgreen', label='Visual Orig')
                

                # plt.legend()
                # plt.title("t-SNE of original and diffusion-generated features")

                # # 保存图像到本地，不显示图像
                # save_path = Path('result') / 'FedDMER' / 'IEMOCAP4' / 'tsne_ddim_mix'/ 'classifier_v'   # MOSI/IEMOCAP4
                # save_path.mkdir(parents=True, exist_ok=True)
                # model_save_path = save_path / ('tsne_modal_'+str(epoch)+'.png')
                # plt.savefig(model_save_path, dpi=300)
                # plt.close()

            proj_x_a = pre_x_a + self.prompta_m
            proj_x_v = proj_x_v + self.promptv_nm
            proj_x_l = pre_x_l + self.promptl_m

        elif self.multi_type == 2: # audio only
            condition_a = F.softmax(pseudo_labels[2]/self.temp,dim=1)
            condition_mix = F.softmax((condition_a+condition_label)/self.temp, dim=1)

            if self.diffusion_model == 'ddpm':
                pre_x_l = self.diffusion.sample(self.score_l_va, condition_label, self.length, batch_size=len(labels), channels=self.d_l)   # condition_va
                pre_x_v = self.diffusion.sample(self.score_v_la, condition_label, self.length, batch_size=len(labels), channels=self.d_l)   # condition_lv  ddim_sample
            elif self.diffusion_model == 'ddim':
                pre_x_l = self.diffusion.ddim_sample(self.score_l_va, condition_mix, self.length, batch_size=len(labels), channels=self.d_l)   # condition_va
                pre_x_v = self.diffusion.ddim_sample(self.score_v_la, condition_mix, self.length, batch_size=len(labels), channels=self.d_l)   # condition_lv  ddim_sample

            # 对生成的视觉特征进行逆归一化
            pre_x_v = (pre_x_v + 1) / 2 * (self.v_max - self.v_min) + self.v_min
            pre_x_l = (pre_x_l + 1) / 2 * (self.l_max - self.l_min) + self.l_min

            if self.need_tsne == 1:
                bs = proj_x_l.shape[0]

                # 将特征从 (bs, 32, 48) 展平为 (bs, 32*48)
                feat_l_orig = proj_x_l.reshape(bs, -1).detach().cpu().numpy()
                feat_a_orig = proj_x_a.reshape(bs, -1).detach().cpu().numpy()
                feat_v_orig = proj_x_v.reshape(bs, -1).detach().cpu().numpy()

                feat_v_gen = pre_x_v.reshape(bs, -1).detach().cpu().numpy()
                feat_l_gen = pre_x_l.reshape(bs, -1).detach().cpu().numpy()

                # feat_l_orig = proj_x_l.detach().cpu().numpy()
                # feat_a_orig = proj_x_a.detach().cpu().numpy()
                # feat_v_orig = proj_x_v.detach().cpu().numpy()

                # feat_v_gen = pre_x_v.detach().cpu().numpy()
                # feat_l_gen = pre_x_l.detach().cpu().numpy()

                # 将所有特征拼接起来进行 t-SNE 降维
                all_features = np.concatenate([
                    feat_l_orig, feat_a_orig, feat_v_orig,
                    feat_v_gen,feat_l_gen
                ], axis=0)

                modalities = ['l_orig', 'a_orig', 'v_orig', 'v_gen', 'l_gen']
                feat_tsne_dict={name:all_features[i*bs:(i+1)*bs] for i, name in enumerate(modalities)}

                # # 对所有特征进行标准化（归一化为零均值和单位方差）
                # scaler = StandardScaler()
                # all_features_norm = scaler.fit_transform(all_features)

                # tsne = TSNE(n_components=2, random_state=42)
                # features_2d = tsne.fit_transform(all_features_norm)

                # # 根据顺序划分 t-SNE 后的特征：
                # feat_l_orig_2d = features_2d[:bs]
                # feat_a_orig_2d = features_2d[bs:2*bs]
                # feat_v_orig_2d = features_2d[2*bs:3*bs]
                # feat_v_gen_2d  = features_2d[3*bs:4*bs]
                # feat_l_gen_2d  = features_2d[4*bs:5*bs]


                # # 绘制 t-SNE 图
                # plt.figure(figsize=(8, 8))
                # plt.scatter(feat_l_orig_2d[:, 0], feat_l_orig_2d[:, 1],
                #             color='darkred', label='Text Orig')
                # plt.scatter(feat_l_gen_2d[:, 0], feat_l_gen_2d[:, 1],
                #                 color='salmon', label='Text Gen')
                # plt.scatter(feat_a_orig_2d[:, 0], feat_a_orig_2d[:, 1],
                #             color='darkblue', label='Audio Orig')
                # plt.scatter(feat_v_orig_2d[:, 0], feat_v_orig_2d[:, 1],
                #             color='darkgreen', label='Visual Orig')
                # plt.scatter(feat_v_gen_2d[:, 0], feat_v_gen_2d[:, 1],
                #                 color='lightgreen', label='Visual Gen')
                

                # plt.legend()
                # plt.title("t-SNE of original and diffusion-generated features")

                # # 保存图像到本地，不显示图像
                # save_path = Path('result') / 'FedDMER' / 'IEMOCAP4' / 'tsne_ddim_mix'/ 'classifier_a'   # MOSI/IEMOCAP4
                # save_path.mkdir(parents=True, exist_ok=True)
                # model_save_path = save_path / ('tsne_modal_'+str(epoch)+'.png')
                # plt.savefig(model_save_path, dpi=300)
                # plt.close()

            proj_x_a = proj_x_a + self.prompta_nm
            proj_x_v = pre_x_v + self.promptv_m
            proj_x_l = pre_x_l + self.promptl_m

        elif self.multi_type == 3: # text & vision
            # conditions = self.cat_lv(torch.cat([proj_x_l, proj_x_v], dim=1))  # cat two avail modalities as conditions
            condition_lv = F.softmax(pseudo_labels[0]/self.temp,dim=1)
            condition_mix = F.softmax((condition_lv+condition_label)/self.temp, dim=1)
            
            if self.diffusion_model == 'ddpm':
                pre_x_a = self.diffusion.sample(self.score_a_lv, condition_label, self.length, batch_size=len(labels), channels=self.d_l)   # condition_lv
            elif self.diffusion_model == 'ddim':
                pre_x_a = self.diffusion.ddim_sample(self.score_a_lv, condition_mix, self.length, batch_size=len(labels), channels=self.d_l)   # condition_lv  
            # 对生成的音频特征进行逆归一化
            pre_x_a = (pre_x_a + 1) / 2 * (self.a_max - self.a_min) + self.a_min

            if self.need_tsne == 1:
                bs = proj_x_l.shape[0]

                # 将特征从 (bs, 32, 48) 展平为 (bs, 32*48)
                feat_l_orig = proj_x_l.reshape(bs, -1).detach().cpu().numpy()
                feat_a_orig = proj_x_a.reshape(bs, -1).detach().cpu().numpy()
                feat_v_orig = proj_x_v.reshape(bs, -1).detach().cpu().numpy()

                feat_a_gen = pre_x_a.reshape(bs, -1).detach().cpu().numpy()

                # 将所有特征拼接起来进行 t-SNE 降维
                all_features = np.concatenate([
                    feat_l_orig, feat_a_orig, feat_v_orig,
                    feat_a_gen
                ], axis=0)

                modalities = ['l_orig', 'a_orig', 'v_orig', 'a_gen']
                feat_tsne_dict={name:all_features[i*bs:(i+1)*bs] for i, name in enumerate(modalities)}
 
            proj_x_a = pre_x_a + self.prompta_m
            proj_x_v = proj_x_v + self.promptv_nm
            proj_x_l = proj_x_l + self.promptl_nm

        elif self.multi_type == 4: # text & audio
            condition_la = F.softmax(pseudo_labels[1]/self.temp,dim=1)
            condition_mix = F.softmax((condition_la+condition_label)/self.temp, dim=1)

            # Generate samples from score-based models with the Euler_Maruyama_sampler
            if self.diffusion_model == 'ddpm':
                pre_x_v = self.diffusion.sample(self.score_v_la, condition_label, self.length, batch_size=len(labels), channels=self.d_l)   # condition_la / labels  ddim_sample
            elif self.diffusion_model == 'ddim':
                pre_x_v = self.diffusion.ddim_sample(self.score_v_la, condition_mix, self.length, batch_size=len(labels), channels=self.d_l)   # condition_la / labels  
            # 对生成的视觉特征进行逆归一化
            pre_x_v = (pre_x_v + 1) / 2 * (self.v_max - self.v_min) + self.v_min
            
            if self.need_tsne == 1:
                bs = proj_x_l.shape[0]

                # 将特征从 (bs, 32, 48) 展平为 (bs, 32*48)
                feat_l_orig = proj_x_l.reshape(bs, -1).detach().cpu().numpy()
                feat_a_orig = proj_x_a.reshape(bs, -1).detach().cpu().numpy()
                feat_v_orig = proj_x_v.reshape(bs, -1).detach().cpu().numpy()

                feat_v_gen = pre_x_v.reshape(bs, -1).detach().cpu().numpy()

                # 将所有特征拼接起来进行 t-SNE 降维
                all_features = np.concatenate([
                    feat_l_orig, feat_a_orig, feat_v_orig,
                    feat_v_gen
                ], axis=0)

                modalities = ['l_orig', 'a_orig', 'v_orig', 'v_gen']
                feat_tsne_dict={name:all_features[i*bs:(i+1)*bs] for i, name in enumerate(modalities)}

            proj_x_a = proj_x_a + self.prompta_nm
            proj_x_v = pre_x_v + self.promptv_m
            proj_x_l = proj_x_l + self.promptl_nm

        elif self.multi_type == 5: # audiuo & vision
            condition_va = F.softmax(pseudo_labels[2]/self.temp,dim=1)
            condition_mix = F.softmax((condition_va+condition_label)/self.temp, dim=1)

            # Generate samples from score-based models with the Euler_Maruyama_sampler
            if self.diffusion_model == 'ddpm':
                pre_x_l = self.diffusion.sample(self.score_l_va, condition_label, self.length, batch_size=len(labels), channels=self.d_l)   # condition_va
            elif self.diffusion_model == 'ddim':
                pre_x_l = self.diffusion.ddim_sample(self.score_l_va, condition_mix, self.length, batch_size=len(labels), channels=self.d_l)   # condition_va
            # 对生成的文本特征进行逆归一化
            pre_x_l = (pre_x_l + 1) / 2 * (self.l_max - self.l_min) + self.l_min
            
            if self.need_tsne == 1:
                bs = proj_x_l.shape[0]

                # 将特征从 (bs, 32, 48) 展平为 (bs, 32*48)
                feat_l_orig = proj_x_l.reshape(bs, -1).detach().cpu().numpy()
                feat_a_orig = proj_x_a.reshape(bs, -1).detach().cpu().numpy()
                feat_v_orig = proj_x_v.reshape(bs, -1).detach().cpu().numpy()

                feat_l_gen = pre_x_l.reshape(bs, -1).detach().cpu().numpy()

                # 将所有特征拼接起来进行 t-SNE 降维
                all_features = np.concatenate([
                    feat_l_orig, feat_a_orig, feat_v_orig,
                    feat_l_gen
                ], axis=0)

                modalities = ['l_orig', 'a_orig', 'v_orig', 'l_gen']
                feat_tsne_dict={name:all_features[i*bs:(i+1)*bs] for i, name in enumerate(modalities)}

            proj_x_a = proj_x_a + self.prompta_nm
            proj_x_v = proj_x_v + self.promptv_nm
            proj_x_l = pre_x_l + self.promptl_m

        else:
            # loss_score_l, loss_score_v, loss_score_a = torch.tensor(0).cuda(), torch.tensor(0).cuda(), torch.tensor(0).cuda()
            proj_x_a = proj_x_a + self.prompta_nm
            proj_x_v = proj_x_v + self.promptv_nm
            proj_x_l = proj_x_l + self.promptl_nm


        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        # A residual block
        last_hs_proj = self.proj2_3(
            F.dropout(F.relu(self.proj1_3(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer_3(last_hs_proj)

        res = {
            'Feature_t': last_h_l,
            'Feature_a': last_h_a,
            'Feature_v': last_h_v,
            'Feature_2d': feat_tsne_dict,
            'modalities': modalities,
            # 'loss_score_l': loss_score_l.type(torch.float32),
            # 'loss_score_v': loss_score_v.type(torch.float32),
            # 'loss_score_a': loss_score_a.type(torch.float32),
            'M': output
        }
        return res



class single_client(nn.Module):
    def __init__(self, args, multi_type=0):
        super(single_client, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads, self.length = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims   # [768, 5, 20]
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.MSE = MSE()
        self.multi_type = multi_type  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 5:lva
        self.gru_droupout = args.gru_droupout

        combined_dim_3 = 2 * (self.d_l + self.d_a + self.d_v)
        combined_dim_2 = 2 * (self.d_l + self.d_a)
        combined_dim_1 = 2 * (self.d_l)

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # self.cat_lv = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        # self.cat_la = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)
        # self.cat_va = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0)

        # 1. Temporal convolutional layers
        # self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)
# nn.GRU(input_size=adim + tdim + vdim, hidden_size=D_e, num_layers=2, bidirectional=True,dropout=dropout)
        self.proj_l = nn.Sequential(
            # nn.GRU(input_size=args.feature_dims[0], hidden_size=int(self.d_l * self.length / 2), num_layers=2, bidirectional=True, dropout=self.gru_droupout),
            nn.Linear(self.orig_d_l, self.d_l * self.length),
            Reshape(self.d_l, self.length)
        )
        self.proj_a = nn.Sequential(
            # nn.GRU(input_size=args.feature_dims[1], hidden_size=int(self.d_l * self.length / 2), num_layers=2, bidirectional=True, dropout=self.gru_droupout),
            nn.Linear(self.orig_d_a, self.d_a * self.length),
            Reshape(self.d_a, self.length)
        )
        self.proj_v = nn.Sequential(
            # nn.GRU(input_size=args.feature_dims[2], hidden_size=int(self.d_l * self.length / 2), num_layers=2, bidirectional=True, dropout=self.gru_droupout),
            nn.Linear(self.orig_d_v, self.d_v * self.length),
            Reshape(self.d_v, self.length)
        )



        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions

        self.trans_l_mem = self.get_network(self_type='l_mem', layers=2)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=2)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=2)

        # Projection layers
        # 单模态
        self.out_proj_l = nn.Sequential(
            nn.Linear(combined_dim_1, combined_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_1, combined_dim_1)
        )
        self.out_proj_v = nn.Sequential(
            nn.Linear(combined_dim_1, combined_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_1, combined_dim_1)
        )
        self.out_proj_a = nn.Sequential(
            nn.Linear(combined_dim_1, combined_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout+0.1),
            nn.Linear(combined_dim_1, combined_dim_1)
        )
        # 双模态
        self.out_proj_la = nn.Sequential(
            nn.Linear(combined_dim_2, combined_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_2, combined_dim_2)
        )
        self.out_proj_va = nn.Sequential(
            nn.Linear(combined_dim_2, combined_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_2, combined_dim_2)
        )
        self.out_proj_lv = nn.Sequential(
            nn.Linear(combined_dim_2, combined_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_2, combined_dim_2)
        )

        self.proj1_3 = nn.Linear(combined_dim_3, combined_dim_3)
        
        self.proj2_3 = nn.Linear(combined_dim_3, combined_dim_3)

        self.out_layer_1_l = nn.Linear(combined_dim_1, output_dim)
        self.out_layer_1_v = nn.Linear(combined_dim_1, output_dim)
        self.out_layer_1_a = nn.Linear(combined_dim_1, output_dim)

        self.out_layer_2_la = nn.Linear(combined_dim_2, output_dim)
        self.out_layer_2_va = nn.Linear(combined_dim_2, output_dim)
        self.out_layer_2_lv = nn.Linear(combined_dim_2, output_dim)

        self.out_layer_3 = nn.Linear(combined_dim_3, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout =  2*self.d_l, self.attn_dropout   # 2*self.d_l
        elif self_type == 'a_mem':
            embed_dim, attn_dropout =  2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout =  2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    def forward(self, text, audio, video, labels=0, batch_index=0, is_train=False, num_modal=None):
        # with torch.no_grad():
        # if self.use_bert:
        #     text = self.text_model(text)
        # x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        # x_a = audio.transpose(1, 2)
        # x_v = video.transpose(1, 2)
        x_l = text
        x_a = audio
        x_v = video

        # Project the textual/visual/audio features
        # with torch.no_grad():
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        # output, hid = self.proj_l[0](x_l)
        # proj_x_l = self.proj_l[1](output)

        # output, hid = self.proj_a[0](x_a)
        # proj_x_a = self.proj_a[1](output)

        # output, hid = self.proj_v[0](x_v)
        # proj_x_v = self.proj_v[1](output)

        gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        #  random select modality  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 5:lva
        if self.multi_type == 0: # text only
            # h_ls = torch.cat([proj_x_l, proj_x_l], dim=2)
            h_l_with_ls = self.trans_l_with_v(proj_x_l, proj_x_l, proj_x_l)
            h_ls = torch.cat([proj_x_l, h_l_with_ls], dim=2)

            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]

            # A residual block
            # last_hs_proj = self.proj2_1(
            #     F.dropout(F.relu(self.proj1_1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj = self.out_proj_l(last_hs)
            last_hs_proj += last_hs

            output = self.out_layer_1_l(last_hs_proj)

        elif self.multi_type == 1: # vision only
            # h_vs = torch.cat([proj_x_v, proj_x_v], dim=2)
            h_v_with_vs = self.trans_v_with_l(proj_x_v, proj_x_v, proj_x_v)
            h_vs = torch.cat([proj_x_v, h_v_with_vs], dim=2)

            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_v_l = last_hs = h_vs[-1]

            # A residual block
            # last_hs_proj = self.proj2_1(
            #     F.dropout(F.relu(self.proj1_1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj = self.out_proj_v(last_hs)
            last_hs_proj += last_hs

            output = self.out_layer_1_v(last_hs_proj)
        
        elif self.multi_type == 2: # audio only
            h_as = torch.cat([proj_x_a, proj_x_a], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_a_l = last_hs = h_as[-1]

            # A residual block
            # last_hs_proj = self.proj2_1(
            #     F.dropout(F.relu(self.proj1_1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj = self.out_proj_a(last_hs)
            last_hs_proj += last_hs

            output = self.out_layer_1_a(last_hs_proj)

        elif self.multi_type == 3: # text & vision
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_ls = torch.cat([proj_x_l, h_l_with_vs], dim=2)
            h_vs = torch.cat([proj_x_v, h_v_with_ls], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls[-1] 
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = h_vs[-1] 
            last_hs = torch.cat([last_h_l, last_h_v], dim=1)

            # A residual block
            last_hs_proj = self.out_proj_lv(last_hs)
            
            last_hs_proj += last_hs

            output = self.out_layer_2_lv(last_hs_proj)
        
        elif self.multi_type == 4: # text & audio
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_ls = torch.cat([proj_x_l, h_l_with_as], dim=2)
            h_as = torch.cat([proj_x_a, h_a_with_ls], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            h_as = self.trans_a_mem(h_as)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls[-1] 
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as[-1] 
            last_hs = torch.cat([last_h_l, last_h_a], dim=1)

            # A residual block
            last_hs_proj = self.out_proj_la(last_hs)
            last_hs_proj += last_hs

            output = self.out_layer_2_la(last_hs_proj)
        
        elif self.multi_type == 5: # audiuo & vision
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_vs = torch.cat([proj_x_v, h_v_with_as], dim=2)
            h_as = torch.cat([proj_x_a, h_a_with_vs], dim=2)
            h_vs = F.dropout(h_vs, p=self.output_dropout, training=self.training)
            h_as = F.dropout(h_as, p=self.output_dropout, training=self.training)
            h_vs = self.trans_v_mem(h_vs)
            h_as = self.trans_a_mem(h_as)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = h_vs[-1] 
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as[-1] 
            last_hs = torch.cat([last_h_v, last_h_a], dim=1)
            last_hs = F.dropout(last_hs, p=self.output_dropout, training=self.training)

            # A residual block
            last_hs_proj = self.out_proj_va(last_hs)
            last_hs_proj += last_hs
            last_hs_proj = F.layer_norm(last_hs_proj, last_hs_proj.shape[1:])

            output = self.out_layer_2_va(last_hs_proj)

        elif self.multi_type == 6: # text & audiuo & vision
             # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

            # A residual block
            last_hs_proj = self.proj2_3(
                F.dropout(F.relu(self.proj1_3(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj += last_hs

            output = self.out_layer_3(last_hs_proj)

        res = {
            'M': output
        }

        return res
    


class pseudo(nn.Module):
    def __init__(self, args, multi_type, base_model, adim, tdim, vdim, D_e, graph_hidden_size, n_speakers, window_past, window_future,
                 n_classes ,dropout=0.5, time_attn=True, no_cuda=False):
        super(pseudo, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads, self.length = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims   # [768, 5, 20]
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.MSE = MSE()
        self.multi_type = multi_type  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 5:lva
        self.gru_droupout = args.gru_droupout

        combined_dim_3 = 2 * (self.d_l + self.d_a + self.d_v)
        combined_dim_2 = 2 * (self.d_l + self.d_a)
        combined_dim_1 = 2 * (self.d_l)

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        self.proj_l = nn.Sequential(
            # nn.GRU(input_size=args.feature_dims[0], hidden_size=int(self.d_l * self.length / 2), num_layers=2, bidirectional=True, dropout=self.gru_droupout),
            nn.Linear(self.orig_d_l, self.d_l * self.length),
            Reshape(self.d_l, self.length)
        )
        self.proj_a = nn.Sequential(
            # nn.GRU(input_size=args.feature_dims[1], hidden_size=int(self.d_l * self.length / 2), num_layers=2, bidirectional=True, dropout=self.gru_droupout),
            nn.Linear(self.orig_d_a, self.d_a * self.length),
            Reshape(self.d_a, self.length)
        )
        self.proj_v = nn.Sequential(
            # nn.GRU(input_size=args.feature_dims[2], hidden_size=int(self.d_l * self.length / 2), num_layers=2, bidirectional=True, dropout=self.gru_droupout),
            nn.Linear(self.orig_d_v, self.d_v * self.length),
            Reshape(self.d_v, self.length)
        )



        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions

        self.trans_l_mem = self.get_network(self_type='l_mem', layers=2)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=2)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=2)

        # Projection layers
        # 单模态
        self.out_proj_l = nn.Sequential(
            nn.Linear(combined_dim_1, combined_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_1, combined_dim_1)
        )
        self.out_proj_v = nn.Sequential(
            nn.Linear(combined_dim_1, combined_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_1, combined_dim_1)
        )
        self.out_proj_a = nn.Sequential(
            nn.Linear(combined_dim_1, combined_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout+0.1),
            nn.Linear(combined_dim_1, combined_dim_1)
        )
        # 双模态
        self.out_proj_la = nn.Sequential(
            nn.Linear(combined_dim_2, combined_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_2, combined_dim_2)
        )
        self.out_proj_va = nn.Sequential(
            nn.Linear(combined_dim_2, combined_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_2, combined_dim_2)
        )
        self.out_proj_lv = nn.Sequential(
            nn.Linear(combined_dim_2, combined_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.output_dropout),
            nn.Linear(combined_dim_2, combined_dim_2)
        )

        self.proj1_3 = nn.Linear(combined_dim_3, combined_dim_3)
        
        self.proj2_3 = nn.Linear(combined_dim_3, combined_dim_3)

        self.out_layer_1_l = nn.Linear(combined_dim_1, output_dim)
        self.out_layer_1_v = nn.Linear(combined_dim_1, output_dim)
        self.out_layer_1_a = nn.Linear(combined_dim_1, output_dim)

        self.out_layer_2_la = nn.Linear(combined_dim_2, output_dim)
        self.out_layer_2_va = nn.Linear(combined_dim_2, output_dim)
        self.out_layer_2_lv = nn.Linear(combined_dim_2, output_dim)

        self.out_layer_3 = nn.Linear(combined_dim_3, output_dim)

        self.out_layer = nn.Linear(output_dim, output_dim)

        # dialogue graph network
        self.no_cuda = no_cuda
        self.base_model = base_model

        # The base model is the sequential context encoder.
        # Change input features => 2*D_e
        input_size = [tdim, vdim, adim, tdim+vdim, tdim+adim, vdim+adim]
        self.lstm = nn.LSTM(input_size=input_size[multi_type], hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
       
        ## Defination for graph model
        ## [modality_type=3(AVT); time_order=3(past, now, future)]
        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future
        self.time_attn = time_attn

        ## gain graph models for 'temporal' and 'speaker'
        n_relations = 3
        self.graph_net_temporal = GraphNetwork(2*D_e, n_relations, self.time_attn, graph_hidden_size, dropout, self.no_cuda)
        n_relations = n_speakers ** 2
        self.graph_net_speaker = GraphNetwork(2*D_e, n_relations, self.time_attn, graph_hidden_size, dropout, self.no_cuda)

        ## classification and reconstruction
        D_h = 2*D_e + graph_hidden_size
        self.smax_fc  = nn.Linear(D_h, n_classes)
        self.linear_rec = nn.Linear(D_h, adim+tdim+vdim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout =  2*self.d_l, self.attn_dropout   # 2*self.d_l
        elif self_type == 'a_mem':
            embed_dim, attn_dropout =  2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout =  2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    def connect_uttr(self, data, lengths):
        # data:(bs, seq_len, dim)
        node_features = []
        length_sum = 0 # for unique node index
        batch_size = data.size(1)
        for j in range(batch_size):
            # gain node_features
            node_feature = data[:lengths[j], j, :] # [Time, Batch, ?, Feat] -> [Time, ?, Feat]
            node_feature = torch.reshape(node_feature, (-1, node_feature.size(-1))) # [Time*?, Feat]
            node_features.append(node_feature) # [Time*?, Feat]
        node_features = torch.cat(node_features, dim=0)
        return node_features
    
    def forward(self, text, audio, video, inputfeats, qmask, umask, seq_lengths, labels=0, batch_index=0, is_train=False, num_modal=None):
        x_l = text
        x_a = audio
        x_v = video

        # Project the textual/visual/audio features
        # with torch.no_grad():
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # DGN
        if self.base_model == 'LSTM':
            outputs, _ = self.lstm(inputfeats)
            outputs = outputs.unsqueeze(2)
        ## add graph model
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.n_speakers, 
                                                             self.window_past, self.window_future, 'temporal', self.no_cuda)
        assert len(edge_type_mapping) == 3
        hidden1 = self.graph_net_temporal(features, edge_index, edge_type, seq_lengths, umask)
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.n_speakers, 
                                                             self.window_past, self.window_future, 'speaker', self.no_cuda)
        assert len(edge_type_mapping) == self.n_speakers ** 2
        hidden2 = self.graph_net_speaker(features, edge_index, edge_type, seq_lengths, umask)
        hidden = hidden1 + hidden2

        ## for classification
        log_prob = self.smax_fc(hidden) # [seqlen, batch, n_classes]
        log_prob = self.connect_uttr(log_prob, seq_lengths)
        # print('log_prob',log_prob.size())

        #  random select modality  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 5:lva
        if self.multi_type == 0: # text only
            # h_ls = torch.cat([proj_x_l, proj_x_l], dim=2)
            h_l_with_ls = self.trans_l_with_v(proj_x_l, proj_x_l, proj_x_l)
            h_ls = torch.cat([proj_x_l, h_l_with_ls], dim=2)

            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]

            # A residual block
            # last_hs_proj = self.proj2_1(
            #     F.dropout(F.relu(self.proj1_1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj = self.out_proj_l(last_hs)
            last_hs_proj += last_hs

            output = self.out_layer(self.out_layer_1_l(last_hs_proj) + log_prob)
            # print("output:", output.size())

        elif self.multi_type == 1: # vision only
            # h_vs = torch.cat([proj_x_v, proj_x_v], dim=2)
            h_v_with_vs = self.trans_v_with_l(proj_x_v, proj_x_v, proj_x_v)
            h_vs = torch.cat([proj_x_v, h_v_with_vs], dim=2)

            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_v_l = last_hs = h_vs[-1]

            # A residual block
            # last_hs_proj = self.proj2_1(
            #     F.dropout(F.relu(self.proj1_1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj = self.out_proj_v(last_hs)
            last_hs_proj += last_hs

            output = self.out_layer(self.out_layer_1_v(last_hs_proj) + log_prob)
        
        elif self.multi_type == 2: # audio only
            h_as = torch.cat([proj_x_a, proj_x_a], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_a_l = last_hs = h_as[-1]

            # A residual block
            # last_hs_proj = self.proj2_1(
            #     F.dropout(F.relu(self.proj1_1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj = self.out_proj_a(last_hs)
            last_hs_proj += last_hs

            output = self.out_layer(self.out_layer_1_a(last_hs_proj) + log_prob)

        elif self.multi_type == 3: # text & vision
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_ls = torch.cat([proj_x_l, h_l_with_vs], dim=2)
            h_vs = torch.cat([proj_x_v, h_v_with_ls], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls[-1] 
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = h_vs[-1] 
            last_hs = torch.cat([last_h_l, last_h_v], dim=1)

            # A residual block
            last_hs_proj = self.out_proj_lv(last_hs)
            
            last_hs_proj += last_hs

            output = self.out_layer_2_lv(last_hs_proj) + log_prob
        
        elif self.multi_type == 4: # text & audio
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_ls = torch.cat([proj_x_l, h_l_with_as], dim=2)
            h_as = torch.cat([proj_x_a, h_a_with_ls], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            h_as = self.trans_a_mem(h_as)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls[-1] 
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as[-1] 
            last_hs = torch.cat([last_h_l, last_h_a], dim=1)

            # A residual block
            last_hs_proj = self.out_proj_la(last_hs)
            last_hs_proj += last_hs

            output = self.out_layer_2_la(last_hs_proj) + log_prob
        
        elif self.multi_type == 5: # audiuo & vision
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_vs = torch.cat([proj_x_v, h_v_with_as], dim=2)
            h_as = torch.cat([proj_x_a, h_a_with_vs], dim=2)
            h_vs = F.dropout(h_vs, p=self.output_dropout, training=self.training)
            h_as = F.dropout(h_as, p=self.output_dropout, training=self.training)
            h_vs = self.trans_v_mem(h_vs)
            h_as = self.trans_a_mem(h_as)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = h_vs[-1] 
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as[-1] 
            last_hs = torch.cat([last_h_v, last_h_a], dim=1)
            last_hs = F.dropout(last_hs, p=self.output_dropout, training=self.training)

            # A residual block
            last_hs_proj = self.out_proj_va(last_hs)
            last_hs_proj += last_hs
            last_hs_proj = F.layer_norm(last_hs_proj, last_hs_proj.shape[1:])

            output = self.out_layer_2_va(last_hs_proj) + log_prob

        elif self.multi_type == 6: # text & audiuo & vision
             # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

            # A residual block
            last_hs_proj = self.proj2_3(
                F.dropout(F.relu(self.proj1_3(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj += last_hs

            output = self.out_layer_3(last_hs_proj) + log_prob

        res = {
            'M': output
        }

        return res


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_relations, time_attn, hidden_size=64, dropout=0.5, no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()
        self.no_cuda = no_cuda 
        self.time_attn = time_attn
        self.hidden_size = hidden_size

        ## graph modeling
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations)
        self.conv2 = GraphConv(hidden_size, hidden_size)

        ## nodal attention
        D_h = num_features+hidden_size
        self.grufusion = nn.LSTM(input_size=D_h, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout)

        ## sequence attention
        self.matchatt = MatchingAttention(2*D_h, 2*D_h, att_type='general2')
        self.linear = nn.Linear(2*D_h, D_h)


    def forward(self, features, edge_index, edge_type, seq_lengths, umask):
        '''
        features: input node features: [num_nodes, in_channels]
        edge_index: [2, edge_num]
        edge_type: [edge_num]
        '''

        ## graph model: graph => outputs
        out = self.conv1(features, edge_index, edge_type) # [num_features -> hidden_size]
        out = self.conv2(out, edge_index) # [hidden_size -> hidden_size]
        outputs = torch.cat([features, out], dim=-1) # [num_nodes, num_features(16)+hidden_size(8)]

        ## change utterance to conversation: (outputs->outputs)
        outputs = outputs.reshape(-1, outputs.size(1)) # [num_utterance, dim]
        outputs = utterance_to_conversation(outputs, seq_lengths, umask, self.no_cuda) # [seqlen, batch, dim]
        outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1, -1) # [seqlen, batch, ?, dim]

        ## outputs -> outputs:
        seqlen = outputs.size(0)
        batch = outputs.size(1)
        outputs = torch.reshape(outputs, (seqlen, batch, -1)) # [seqlen, batch, dim]
        outputs = self.grufusion(outputs)[0] # [seqlen, batch, dim]

        ## outputs -> hidden:
        ## sequence attention => [seqlen, batch, d_h]
        if self.time_attn:
            alpha = []
            att_emotions = []
            for t in outputs: # [bacth, dim]
                # att_em: [batch, mem_dim] # alpha_: [batch, 1, seqlen]
                att_em, alpha_ = self.matchatt(outputs, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0)) # [1, batch, mem_dim]
                alpha.append(alpha_[:,0,:]) # [batch, seqlen]
            att_emotions = torch.cat(att_emotions, dim=0) # [seqlen, batch, mem_dim]
            hidden = F.relu(self.linear(att_emotions)) # [seqlen, batch, D_h]
        else:
            alpha = []
            hidden = F.relu(self.linear(outputs)) # [seqlen, batch, D_h]

        return hidden # [seqlen, batch, D_h]
    

'''
base_model: LSTM or GRU
adim, tdim, vdim: input feature dim
D_e: hidder feature dimensions of base_model is 2*D_e
D_g, D_p, D_h, D_a, graph_hidden_size
'''
class GraphModel(nn.Module):

    def __init__(self, base_model, adim, tdim, vdim, D_e, graph_hidden_size, n_speakers, window_past, window_future,
                 n_classes ,dropout=0.5, time_attn=True, no_cuda=False):
        
        super(GraphModel, self).__init__()

        self.no_cuda = no_cuda
        self.base_model = base_model

        # The base model is the sequential context encoder.
        # Change input features => 2*D_e
        self.lstm = nn.LSTM(input_size=adim+tdim+vdim, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru = nn.GRU(input_size=adim+tdim+vdim, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
       
        ## Defination for graph model
        ## [modality_type=3(AVT); time_order=3(past, now, future)]
        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future
        self.time_attn = time_attn

        ## gain graph models for 'temporal' and 'speaker'
        n_relations = 3
        self.graph_net_temporal = GraphNetwork(2*D_e, n_relations, self.time_attn, graph_hidden_size, dropout, self.no_cuda)
        n_relations = n_speakers ** 2
        self.graph_net_speaker = GraphNetwork(2*D_e, n_relations, self.time_attn, graph_hidden_size, dropout, self.no_cuda)

        ## classification and reconstruction
        D_h = 2*D_e + graph_hidden_size
        self.smax_fc  = nn.Linear(D_h, n_classes)
        self.linear_rec = nn.Linear(D_h, adim+tdim+vdim)

    def forward(self, inputfeats, qmask, umask, seq_lengths):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        """

        ## sequence modeling
        ## inputfeats -> outputs [seqlen, batch, ?, dim]
        if self.base_model == 'LSTM':
            outputs, _ = self.lstm(inputfeats[0])
            outputs = outputs.unsqueeze(2)

        ## add graph model
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.n_speakers, 
                                                             self.window_past, self.window_future, 'temporal', self.no_cuda)
        assert len(edge_type_mapping) == 3
        hidden1 = self.graph_net_temporal(features, edge_index, edge_type, seq_lengths, umask)
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.n_speakers, 
                                                             self.window_past, self.window_future, 'speaker', self.no_cuda)
        assert len(edge_type_mapping) == self.n_speakers ** 2
        hidden2 = self.graph_net_speaker(features, edge_index, edge_type, seq_lengths, umask)
        hidden = hidden1 + hidden2

        ## for classification
        log_prob = self.smax_fc(hidden) # [seqlen, batch, n_classes]

        ## for reconstruction
        rec_outputs = [self.linear_rec(hidden)]

        return log_prob, rec_outputs, hidden