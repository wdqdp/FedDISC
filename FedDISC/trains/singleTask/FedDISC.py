from curses import noecho
import logging
from tkinter import NO
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from pathlib import Path
import torch.nn.functional as F
logger = logging.getLogger('MMSA')
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# import umap.umap_ as umap
# import plotly.express as px
# from mpl_toolkits.mplot3d import Axes3D


def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = [] 
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2) # [seqlen, batch, featdim=adim+tdim+vdim]
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    tmask = qmask.transpose(0, 1) # [batch, seqlen] -> [seqlen, batch]
    tmask = tmask.unsqueeze(2).repeat(1,1,featdim) # -> [seqlen, batch, featdim]
    select_feat = torch.where(tmask==0, feat1, feat2) # -> [seqlen, batch, featdim]
    # input_features.append(select_feat) # 1 * [seqlen, batch, dim]
    return select_feat


def connect_uttr(data, lengths):
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


def distillation_loss(teacher_logits, student_logits, temperature=1.0):
    """
    - teacher_logits: 教师模型的logits，形状为 (batch_size, num_classes)
    - student_logits: 学生模型的logits，形状为 (batch_size, num_classes)
    - temperature: 温度，控制softmax的平滑程度
    
    返回:
    - soft_loss: 知识蒸馏的软损失
    """
    # 使用温度对logits进行缩放
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    
    # 计算KL散度
    soft_loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    
    return soft_loss


class FedDISC():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)  # 各项指标

    # model:models[client_idx]  data_loader:dataloader[client_idx] S
    def client_train_and_send(self, model, data_loader, optimizer, multi_type, train_state='extra', local_epoch=0, epoch=0, 
                              pseudo_models=None, teacher_model=None, num_classes=3,
                              pre_x_ls=None,
                              pre_x_vs=None,
                              pre_x_as=None,
                              pre_labels=None):
        model.train()
        train_loss = 0.0
        left_epochs = self.args.update_epochs
        y_pred, y_true = [], []
        out_x_ls, out_x_vs, out_x_as, out_labels = [], [], [], []
        train_results = None
        pre_data={}
        Features_tsne_ALL = []

        with tqdm(data_loader['train']) as td:
            for batch_idx, data in enumerate(td):
                if left_epochs == self.args.update_epochs:
                    optimizer.zero_grad()
                left_epochs -= 1
                vidnames = []

                # vision = batch_data['vision'].to(self.args.device)  # (bs, 50, 20)
                # audio = batch_data['audio'].to(self.args.device)    # (bs, 50, 5)
                # text = batch_data['text'].to(self.args.device)      # (bs, 3, 50)
                # labels = batch_data['labels']['M'].to(self.args.device) # (bs, 1)
                # batch_index = batch_data['index'].to(self.args.device)

                 ## read dataloader
                """
                audio_host, text_host, visual_host: [seqlen, batch, dim]
                audio_guest, text_guest, visual_guest: [seqlen, batch, dim]
                qmask: speakers, [batch, seqlen]
                umask: has utt, [batch, seqlen]
                label: [batch, seqlen]

                seqlen为什么每个batch不一样？ seqlen的大小为当前batch中最大的seqlen，不足的用0填充
                umask中1的是有原始数据的，0代表是填充的，通过umask恢复数据。
                qmask用于选择是host的数据还是guest的，最终把host与guest拼接起来[seqlen, batch, dim]

                """
                audio_host, text_host, visual_host = data[0].to(self.args.device), data[1].to(self.args.device), data[2].to(self.args.device)
                audio_guest, text_guest, visual_guest = data[3].to(self.args.device), data[4].to(self.args.device), data[5].to(self.args.device)
                qmask, umask, label = data[6].to(self.args.device), data[7].to(self.args.device), data[8].to(self.args.device)
                vidnames += data[-1]
                adim = audio_host.size(2)
                tdim = text_host.size(2)
                vdim = visual_host.size(2)

                lengths = []
                seqlen = audio_host.size(0)
                batch = audio_host.size(1)
                ## host mask [!!use original audio_feature!!]
                view_num = 3

                for j in range(len(umask)):
                    length = (umask[j] == 1).nonzero().tolist()[-1][0] + 1 
                    lengths.append(length)

                ## generate input_features: ? * [seqlen, batch, adim+tdim+vdim]
                input_features = generate_inputs(audio_host, text_host, visual_host, \
                                         audio_guest, text_guest, visual_guest, qmask)
                audio = input_features[:,:,0:adim]
                text = input_features[:,:,adim:adim+tdim]
                vision = input_features[:,:,adim+tdim:adim+tdim+vdim]
                # print('input_features:', input_features.size()) # [?, 4, 2560]

                if multi_type == 0:
                    pseudo_data = input_features[:,:,adim:adim+tdim]
                elif multi_type == 1:
                    pseudo_data = input_features[:,:,adim+tdim:adim+tdim+vdim]
                elif multi_type == 2:
                    pseudo_data = input_features[:,:,0:adim]
                elif multi_type == 3:  #lv
                    combined = torch.cat((text, vision), dim=2)
                elif multi_type == 4:  #la
                    pseudo_data = input_features[:,:,0:adim+tdim]
                elif multi_type == 5:  #la
                    pseudo_data = input_features[:,:,adim+tdim:adim+tdim+vdim]


                audio = connect_uttr(audio, lengths).unsqueeze(1)
                text = connect_uttr(text, lengths).unsqueeze(1)
                vision = connect_uttr(vision, lengths).unsqueeze(1)
                label = connect_uttr(label.transpose(0,1).unsqueeze(-1), lengths).squeeze()
                # print('CONNECT:', audio.size())  # [?, 1, 512]

                # print(label.size(), label)
                if num_classes == 3:
                    labels = torch.zeros_like(label, dtype=torch.long)
                    # 分别赋值
                    labels[label < 0] = 0
                    labels[label == 0] = 1
                    labels[label > 0] = 2
                elif num_classes == 4:
                    labels = label
                elif num_classes == 6:
                    labels = label


                if self.args.train_mode == 'classification':
                    labels = labels.view(-1).long()
                else:
                    labels = labels.view(-1, 1)

                # batch_idx  保存的生成数据
                if pre_x_ls != [] and pre_x_ls != None:
                    pre_data['pre_x_l'] = pre_x_ls[batch_idx]
                    pre_data['pre_x_v'] = pre_x_vs[batch_idx]
                    pre_data['pre_x_a'] = pre_x_as[batch_idx]
                    pre_data['pre_label'] = pre_labels[batch_idx]
                
                
                if train_state == 'pseudo':
                    outputs = model(text, audio, vision, pseudo_data, qmask, umask, lengths)
                    combine_loss = self.criterion(outputs['M'], labels)
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())

                elif train_state == 'extra':
                    outputs = model(text, audio, vision)
                    if teacher_model is not None:
                        outputs_T = teacher_model(text, audio, vision)
                    # loss
                    # soft_loss = distillation_loss(outputs_T['M'], outputs['M'], temperature=2)
                    combine_loss = self.criterion(outputs['M'], labels)
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
                elif train_state == 'gen':
                    pseudo_labels = {}
                    for modal in range(3):
                        pseudo_model = pseudo_models[modal].eval()
                        pseudo_label = pseudo_model(text, audio, vision, labels, local_epoch, epoch)
                        pseudo_labels[modal] = pseudo_label['M']
                    outputs = model(text, audio, vision, labels, local_epoch, epoch, pseudo_labels, batch_idx, pre_data)
                    # compute loss
                    loss_score_l = outputs['loss_score_l']
                    loss_score_v = outputs['loss_score_v']
                    loss_score_a = outputs['loss_score_a']
                    out_x_ls.append(outputs['pre_x_l'])
                    out_x_vs.append(outputs['pre_x_v'])
                    out_x_as.append(outputs['pre_x_a'])
                    out_labels.append(outputs['labels'])
                    # labels_cat = torch.cat([labels, outputs['labels']], dim=0)

                    # loss_rec = outputs['loss_rec']
                    output = outputs['output']
                    generate_en = outputs['generate_en']
                    if (epoch+1)>=self.args.train_start_epoch and local_epoch+1 == self.args.local_epoch:
                        task_loss = self.criterion(output, outputs['labels'])
                        combine_loss = 0.5*(loss_score_l + loss_score_v + loss_score_a) + task_loss
                        # combine_loss = loss_score_l + loss_score_v + loss_score_a
                    else:
                        combine_loss = loss_score_l + loss_score_v + loss_score_a

                    y_pred.append(output.cpu())
                    y_true.append(outputs['labels'].cpu())
                elif train_state == 'class':
                    pseudo_labels = {}
                    for modal in range(3):
                        pseudo_model = pseudo_models[modal].eval()
                        pseudo_label = pseudo_model(text, audio, vision, labels, local_epoch, epoch)
                        pseudo_labels[modal] = pseudo_label['M']
                    outputs = model(text, audio, vision, labels, pseudo_labels, epoch=epoch)
                    task_loss = self.criterion(outputs['M'], labels)
                    # combine_loss = task_loss + 0.1 * (loss_score_l + loss_score_v + loss_score_a)
                    combine_loss = task_loss 
                    feat_tsne_dict = outputs['Feature_2d']
                    modalities = outputs['modalities']
                    
                    if feat_tsne_dict is not None:
                        Features_tsne_ALL.append(feat_tsne_dict)

                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
                
                # backward
                combine_loss.backward()
                if self.args.grad_clip != -1.0: # 梯度裁剪
                    nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                                self.args.grad_clip)
                    
                # store results
                train_loss += combine_loss.item()
                if not left_epochs:
                    optimizer.step()
                    left_epochs = self.args.update_epochs
            if not left_epochs:
                # update
                optimizer.step()
        train_loss = train_loss / len(data_loader['train'])
        
        if train_state == 'pseudo':
            model_params = model.state_dict()
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            return 1, 1, 1, model_params, train_loss, train_results
        if train_state == 'extra':
            if multi_type == 0:
                proj_l_params = model.proj_l.state_dict()
                proj_v_params = None
                proj_a_params = None
            elif multi_type == 1:
                proj_l_params = None
                proj_v_params = model.proj_v.state_dict()
                proj_a_params = None
            elif multi_type == 2:
                proj_l_params = None
                proj_v_params = None
                proj_a_params = model.proj_a.state_dict()
            elif multi_type == 3:
                proj_l_params = model.proj_l.state_dict()
                proj_v_params = model.proj_v.state_dict()
                proj_a_params = None
            elif multi_type == 4:
                proj_l_params = model.proj_l.state_dict()
                proj_v_params = None
                proj_a_params = model.proj_a.state_dict()
            elif multi_type == 5:
                proj_l_params = None
                proj_v_params = model.proj_v.state_dict()
                proj_a_params = model.proj_a.state_dict()
            else:
                proj_l_params = model.proj_l.state_dict()
                proj_v_params = model.proj_v.state_dict()
                proj_a_params = model.proj_a.state_dict()
            model_params = model.state_dict()
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            return proj_l_params, proj_v_params, proj_a_params, model_params, train_loss, train_results
        
        elif train_state == 'gen':
            # 获取训练后的score模型参数  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 6:lva
            if multi_type == 0:
                score_l_params = model.score_l.state_dict()
                score_v_params = None
                score_a_params = None
                # rec_l_params = model.rec_l.state_dict()
                # rec_v_params = None
                # rec_a_params = None
            elif multi_type == 1:
                score_l_params = None
                score_v_params = model.score_v.state_dict()
                score_a_params = None
                # rec_l_params = None
                # rec_v_params = model.rec_v.state_dict()
                # rec_a_params = None
            elif multi_type == 2:
                score_l_params = None
                score_v_params = None
                score_a_params = model.score_a.state_dict()
                # rec_l_params = None
                # rec_v_params = None
                # rec_a_params = model.rec_a.state_dict()
            elif multi_type == 3:
                score_l_params = model.score_l.state_dict()
                score_v_params = model.score_v.state_dict()
                score_a_params = None
                # rec_l_params = model.rec_l.state_dict()
                # rec_v_params = model.rec_v.state_dict()
                # rec_a_params = None
            elif multi_type == 4:
                score_l_params = model.score_l.state_dict()
                score_v_params = None
                score_a_params = model.score_a.state_dict()
                # rec_l_params = model.rec_l.state_dict()
                # rec_v_params = None
                # rec_a_params = model.rec_a.state_dict()
            elif multi_type == 5:
                score_l_params = None
                score_v_params = model.score_v.state_dict()
                score_a_params = model.score_a.state_dict()
                # rec_l_params = None
                # rec_v_params = model.rec_v.state_dict()
                # rec_a_params = model.rec_a.state_dict()
            else:
                score_l_params = model.score_l_va.state_dict()
                score_v_params = model.score_v_la.state_dict()
                score_a_params = model.score_a_lv.state_dict()
                # rec_l_params = model.rec_l.state_dict()
                # rec_v_params = model.rec_v.state_dict()
                # rec_a_params = model.rec_a.state_dict()
            if generate_en:
                pred, true = torch.cat(y_pred), torch.cat(y_true)
                train_results = self.metrics(pred, true)
            else:
                train_results = None

            return score_l_params, score_v_params, score_a_params, train_loss, train_results, out_x_ls, out_x_vs, out_x_as, out_labels
        
        elif train_state == 'class':
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            if Features_tsne_ALL != []:
                color_dict = {'l_orig':['#931712',  'Text Orig'],  # darkred
                              'a_orig':['#194a7a', 'Audio Orig'],  # darkblue
                              'v_orig':['#2c6e49','Visual Orig'],  # darkgreen
                              'l_gen':['#ff595e','Text Gen'],       # lightred
                              'a_gen':['#a2d2ff','Audio Gen'],     # lightblue
                              'v_gen':['#94bf73','Visual Gen']}    # lightgreen
                feat_all = []

                for modality in modalities:
                    feat_list = [feat_tsne_dict[modality] for feat_tsne_dict in Features_tsne_ALL]
                    feat = np.concatenate(feat_list, axis=0)
                    feat_all.append(feat)
                    bs = feat.shape[0]
                all_features = np.concatenate(feat_all, axis=0)
                # 对所有特征进行标准化（归一化为零均值和单位方差）
                scaler = StandardScaler()
                all_features_norm = scaler.fit_transform(all_features)

                # 3d
                # # 创建三维UMAP模型（参数建议）
                # reducer = umap.UMAP(
                #     n_components=3,          # 投影到三维空间
                #     n_neighbors=10,          # 平衡局部/全局结构的邻域大小
                #     min_dist=1.0,           # 控制点间距的紧密程度
                #     metric='euclidean',      # 标准化后推荐欧式距离
                #     random_state=42,         # 保证结果可复现
                #     low_memory=True          # 大数据集时启用
                # )

                # # 执行降维
                # embedding_3d = reducer.fit_transform(all_features_norm)

                # fig = plt.figure(figsize=(8, 8))
                # ax = fig.add_subplot(111, projection='3d')
                # for i, modality in enumerate(modalities):
                #     feature = embedding_3d[i*bs:int((i+2/3)*bs)]
                #     ax.scatter(
                #             feature[:,0], 
                #             feature[:,1],
                #             feature[:,2],
                #             color=color_dict[modality][0], 
                #             cmap='Spectral',
                #             s=5,
                #             alpha=0.8,
                #             depthshade=False        # 改善颜色显示
                #         )
                # ax.view_init(elev=30, azim=20)  # 仰角30度，方位角45度
                # plt.title('3D UMAP Visualization', pad=20)

                # 2d
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(all_features_norm)

                plt.figure(figsize=(11, 11))
                for i, modality in enumerate(modalities):
                    feature = features_2d[i*bs:int((i+4/5)*bs)]
                    plt.scatter(feature[:, 0], feature[:, 1],
                            color=color_dict[modality][0], 
                            s=15)
                plt.title("t-SNE of original and diffusion-generated features")

                # 保存图像到本地，不显示图像
                save_path = Path('result') / 'FedDISC' / 'IEMOCAP6' / 'tsne_ddpm_mix'/ str(multi_type)   # MOSI/IEMOCAP4/IEMOCAP6
                save_path.mkdir(parents=True, exist_ok=True)
                model_save_path = save_path / ('tsne_modal_'+str(epoch)+'.png')
                plt.savefig(model_save_path, dpi=300)
                plt.close()



            return train_loss, train_results


    def aggregate_models_extra(self, client_models_params, agg_type=None):
        if agg_type == 'proj':
            # client_models_params是一个字典，键是客户端ID，值是一个包含score_l, score_v, score_a参数的字典
            avg_proj_l_params = {}
            avg_proj_v_params = {}
            avg_proj_a_params = {}
            num_l, num_v, num_a = 0, 0, 0

            for client_id in client_models_params:
                proj_l_params, proj_v_params, proj_a_params = client_models_params[client_id]
                
                # 对每个客户端的proj_l进行平均聚合
                if proj_l_params is not None:
                    for key in proj_l_params:
                        if key not in avg_proj_l_params:
                            avg_proj_l_params[key] = proj_l_params[key].clone()
                        else:
                            avg_proj_l_params[key] += proj_l_params[key]
                    num_l += 1
                
                # 对每个客户端的score_v进行平均聚合
                if proj_v_params is not None:
                    for key in proj_v_params:
                        if key not in avg_proj_v_params:
                            avg_proj_v_params[key] = proj_v_params[key].clone()
                        else:
                            avg_proj_v_params[key] += proj_v_params[key]
                    num_v += 1
                
                # 对每个客户端的score_a进行平均聚合
                if proj_a_params is not None:
                    for key in proj_a_params:
                        if key not in avg_proj_a_params:
                            avg_proj_a_params[key] = proj_a_params[key].clone()
                        else:
                            avg_proj_a_params[key] += proj_a_params[key] 
                    num_a += 1

            # 对每个参数进行平均
            for key in avg_proj_l_params:
                avg_proj_l_params[key] /= num_l
            for key in avg_proj_v_params:
                avg_proj_v_params[key] /= num_v
            for key in avg_proj_a_params:
                avg_proj_a_params[key] /= num_a

            return avg_proj_l_params, avg_proj_v_params, avg_proj_a_params
        
        elif agg_type == 'all':
            avg_model_params = {}
            num = 0
            for client_id in client_models_params:
                mdoel_params = client_models_params[client_id]
                
                # 对每个客户端的proj_l进行平均聚合
                if mdoel_params is not None:
                    for key in mdoel_params:
                        if key not in avg_model_params:
                            avg_model_params[key] = mdoel_params[key].clone()
                        else:
                            avg_model_params[key] += mdoel_params[key]
                    num += 1

            for key in avg_model_params:
                avg_model_params[key] /= num
            return avg_model_params


    def aggregate_models_gen(self, client_models_params):
        # client_models_params是一个字典，键是客户端ID，值是一个包含score_l, score_v, score_a参数的字典
        avg_score_l_params = {}
        avg_score_v_params = {}
        avg_score_a_params = {}


        num_l, num_v, num_a = 0, 0, 0

        for client_id in client_models_params:
            score_l_params, score_v_params, score_a_params = client_models_params[client_id]
            
            # 对每个客户端的score_l进行平均聚合
            if score_l_params is not None:
                for key in score_l_params:
                    if key not in avg_score_l_params:
                        avg_score_l_params[key] = score_l_params[key].clone()
                    else:
                        avg_score_l_params[key] += score_l_params[key]
                num_l += 1
            
            # 对每个客户端的score_v进行平均聚合
            if score_v_params is not None:
                for key in score_v_params:
                    if key not in avg_score_v_params:
                        avg_score_v_params[key] = score_v_params[key].clone()
                    else:
                        avg_score_v_params[key] += score_v_params[key]
                num_v += 1
            
            # 对每个客户端的score_a进行平均聚合
            if score_a_params is not None:
                for key in score_a_params:
                    if key not in avg_score_a_params:
                        avg_score_a_params[key] = score_a_params[key].clone()
                    else:
                        avg_score_a_params[key] += score_a_params[key] 
                num_a += 1

        # 对每个参数进行平均
        for key in avg_score_l_params:
            avg_score_l_params[key] /= num_l
        for key in avg_score_v_params:
            avg_score_v_params[key] /= num_v
        for key in avg_score_a_params:
            avg_score_a_params[key] /= num_a

        return avg_score_l_params, avg_score_v_params, avg_score_a_params


    def aggregate_models_class(self, client_models_params):
        # 假设每个客户端的 model 参数包含了多个层
        # 并且你想要聚合这些特定层
        selected_layers = [
            'trans_l_with_a', 'trans_l_with_v', 'trans_l_mem',
            'trans_a_with_v', 'trans_a_with_l', 'trans_a_mem',
            'trans_v_with_a', 'trans_v_with_l', 'trans_v_mem',
            'proj1_3', 'proj2_3', 'out_layer_3'
        ]
        
        avg_params = {}  # 存储聚合后的参数
        num_clients = len(client_models_params)

        # 遍历客户端模型的参数
        for client_id, model in client_models_params.items():
            # 迭代模型的参数
            for name, param in model.named_parameters():
                # 只聚合选中的层
                if any(layer_name in name for layer_name in selected_layers):
                    if name not in avg_params:
                        avg_params[name] = param.clone()
                    else:
                        avg_params[name] += param

        # 计算每个层的平均值
        for name in avg_params:
            avg_params[name] /= num_clients

        return avg_params



    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        # load pretrained
        origin_model = torch.load('pt/pretrained-{}.pth'.format(self.args.dataset_name))
        net_dict = model.state_dict()
        new_state_dict = {}
        for k, v in origin_model.items():
            k = k.replace('Model.', '')
            new_state_dict[k] = v
        net_dict.update(new_state_dict)
        model.load_state_dict(net_dict, strict=False)

        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            miss_one, miss_two = 0, 0  # num of missing one modal and missing two modal
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    # forward
                    miss_2 = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
                    miss_1 = [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0]
                    if miss_two / (np.round(len(dataloader['train']) / 10) * 10) < miss_2[int(self.args.mr*10-1)]:  # missing two modal
                        outputs = model(text, audio, vision, num_modal=1)
                        miss_two += 1
                    elif miss_one / (np.round(len(dataloader['train']) / 10) * 10) < miss_1[int(self.args.mr*10-1)]:  # missing one modal
                        outputs = model(text, audio, vision, num_modal=2)
                        miss_one += 1
                    else:  # no missing
                        outputs = model(text, audio, vision, num_modal=3)

                    # compute loss
                    task_loss = self.criterion(outputs['M'], labels)
                    loss_score_l = outputs['loss_score_l']
                    loss_score_v = outputs['loss_score_v']
                    loss_score_a = outputs['loss_score_a']
                    loss_rec = outputs['loss_rec']
                    combine_loss = task_loss + 0.1 * (loss_score_l + loss_score_v + loss_score_a + loss_rec)

                    # backward
                    combine_loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                                  self.args.grad_clip)
                    # store results
                    train_loss += combine_loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            model_save_path = 'pt/' + str(epochs) + '.pth'
            torch.save(model.state_dict(), model_save_path)
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None


    def do_train_extractor(self, mode, models, dataloader, multi_type, return_epoch_results=False, teacher_model=None, num_classes=3):
        optimizers = {
            idx: optim.Adam(model.parameters(), lr=self.args.learning_rate)
            for idx, model in models.items()
        }
        schedulers = {
            idx: ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)
            for idx, optimizer in optimizers.items()
        }
        # initilize results
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        client_num = self.args.client_num
        best_acc_all = [0.0 for _ in range(client_num)]
        is_better = [0 for _ in range(client_num)]
        best_models = {}

        for i in range(self.args.extractor_epoch):
            client_models_params = {}
            # local train
            for client_idx in range(client_num):
                # train
                losses = []
                data_loader = dataloader[client_idx]
                model = models[client_idx].train()
                # teacher_model.eval()
                train_loss = 0.0

                for epoch in range(self.args.local_epoch):
                    proj_l_params, proj_v_params, proj_a_params, mdoel_params, train_loss, train_results = self.client_train_and_send(model, data_loader, optimizers[client_idx], multi_type[client_idx], 'pseudo', 
                                                                                                                                      teacher_model=teacher_model, 
                                                                                                                                      num_classes=num_classes)
                    logger.info(
                    f"LOCAL_TRAIN-CLIENT({client_idx+1}) -epoch[{epoch+1} / [{i+1}]] "
                    f">> loss: {round(train_loss, 4)} "
                    f"{dict_to_str(train_results)}"
                    )
                    schedulers[client_idx].step(train_loss)
                # client_models_params[client_idx] = [proj_l_params, proj_v_params, proj_a_params]
                # pseudo & extra
                client_models_params[client_idx] = mdoel_params

            if mode == 0:  # extra
                # sever_aggregation
                avg_model_params = self.aggregate_models_extra(client_models_params, 'all')

                # update localmodel
                for idx in range(client_num):
                    # get model
                    model = models[idx]
                    # update model
                    if avg_model_params:
                        model.load_state_dict(avg_model_params)
            
            # test
            for client_idx in range(client_num):
                data_loader = dataloader[client_idx]
                model = models[client_idx].eval()
                test_results = self.do_test(model, data_loader['test'], mode="pseudo", num_classes=self.args.num_classes, multi_type=multi_type[client_idx])
                if best_acc_all[client_idx] < test_results['Acc_4']:
                    is_better[client_idx] = 1
                    best_acc_all[client_idx] = test_results['Acc_4']
                    best_models[client_idx] = client_models_params[client_idx]
            if sum(is_better) >= (client_num//2 + 1):
                # best_proj_l, best_proj_v, best_proj_a = avg_proj_l_params, avg_proj_v_params, avg_proj_a_params
                    best_model = avg_model_params
            is_better = [0 for _ in range(client_num)]
        # save each epoch model
        if mode == -1:  # pseudo
            modality = ['l', 'v', 'a']
            save_path = Path('pt') / 'FedDISC' / 'IEMOCAP6' / 'pseudo'   #extra/pseudo
            save_path.mkdir(parents=True, exist_ok=True)

            for client_idx in range(len(modality)):
                model_save_path = save_path / ('pseudo_'+modality[client_idx]+'.pth')
                torch.save(best_models[client_idx], model_save_path)
            
            logger.info(
                    f"successfully saved pseudo models "
                    )

        elif mode == 0:  # extra
            save_path = Path('pt') / 'FedDISC' / 'IEMOCAP6' / 'extra'   #extra/pseudo
            model_save_path = save_path / 'model_extra.pth'
            torch.save(best_model, model_save_path)

            logger.info(
                    f"successfully saved extra models "
                    )
       
        
            
        return None


    def do_train_generator(self, models, pseudo_models, dataloader, multi_type, return_epoch_results=False, num_classes=3):
        optimizers = {
            idx: optim.AdamW(
                [param for param in model.parameters() if param.requires_grad],  # 只优化需要训练的参数
                lr=self.args.learning_rate
            )
            for idx, model in models.items()
        }
        schedulers = {
            idx: ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)
            for idx, optimizer in optimizers.items()
        }
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        # 最小化损失还是最大化
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        client_num = self.args.client_num
        best_acc = [0]*client_num
        avg_score_l_params, avg_score_v_params, avg_score_a_params = None, None, None
        best_score_l_params, best_score_v_params, best_score_a_params = None, None, None
        pre_x_ls=[]
        pre_x_vs=[]
        pre_x_as=[]
        pre_labels=[]

        for i in range(self.args.generator_epoch):
            client_models_score_params = {}   # client_models_params是一个字典，键是客户端ID，值是一个包含score_l, score_v, score_a参数的字典
            client_models_rec_params = {} 
            is_better = [0]*client_num
            # local train
            for idx in range(client_num):
                # train
                losses = []
                data_loader = dataloader[idx]
                model = models[idx].train()
                train_loss = 0.0
                   
                for epoch in range(self.args.local_epoch):
                    score_l_params, score_v_params, score_a_params, train_loss, results, pre_x_ls, pre_x_vs, pre_x_as, pre_labels = self.client_train_and_send(model, data_loader, optimizers[idx], multi_type[idx], 'gen', 
                                                                                                                                                                local_epoch=epoch, 
                                                                                                                                                                epoch=i, 
                                                                                                                                                                pseudo_models=pseudo_models, 
                                                                                                                                                                num_classes=num_classes,
                                                                                                                                                                pre_x_ls=pre_x_ls,
                                                                                                                                                                pre_x_vs=pre_x_vs,
                                                                                                                                                                pre_x_as=pre_x_as,
                                                                                                                                                                pre_labels=pre_labels)
                    if results != None:
                        logger.info(
                        f"LOCAL_TRAIN-CLIENT({idx+1}) -epoch[{epoch+1} / [{i+1}]] "
                        f">> loss: {round(train_loss, 4)} "
                        f"{dict_to_str(results)}"
                        )
                        if results["Acc_6"] > best_acc[idx]:
                            is_better[idx] = 1
                            best_acc[idx] = results["Acc_6"]
                    else:
                        logger.info(
                        f"LOCAL_TRAIN-CLIENT({idx+1}) -epoch[{epoch+1} / [{i+1}]] "
                        f">> loss: {round(train_loss, 4)} "
                        )
                    schedulers[idx].step(train_loss)
                client_models_score_params[idx] = [score_l_params, score_v_params, score_a_params]
                # client_models_rec_params[idx] = [rec_l_params, rec_v_params, rec_a_params]
            print(is_better)
            if (sum(is_better) >= (client_num//2 + 1)) and (i > 0):
            # if i > 0:
                best_score_l_params, best_score_v_params, best_score_a_params = avg_score_l_params, avg_score_v_params, avg_score_a_params
                modality = ['l_va', 'v_la', 'a_lv']
                # modality = ['l', 'v', 'a']
                for j in range(self.args.modality_num):
                    save_path = Path('pt') / 'FedDISC' / 'IEMOCAP6' / 'diffusion'   # MOSI/IEMOCAP4/IEMOCAP6
                    save_path.mkdir(parents=True, exist_ok=True)
                    model_save_path = save_path / ('score_' + str(modality[j]) + '.pth')
                    if j == 0:
                        torch.save(best_score_l_params, model_save_path)
                    elif j == 1:
                        torch.save(best_score_v_params, model_save_path)
                    elif j == 2:
                        torch.save(best_score_a_params, model_save_path)
                logger.info(
                    f"successfully saved diffusion models "
                    )

            # sever_aggregation
            avg_score_l_params, avg_score_v_params, avg_score_a_params = self.aggregate_models_gen(client_models_score_params)
            # avg_rec_l_params, avg_rec_v_params, avg_rec_a_params = self.aggregate_models_gen(client_models_rec_params)

            # update localmodel
            for idx in range(client_num):
                # get model
                model = models[idx]
                # update model
                if avg_score_l_params:
                    model.score_l_va.load_state_dict(avg_score_l_params)
                    # model.score_l.load_state_dict(avg_score_l_params)
                    # model.rec_l.load_state_dict(avg_rec_l_params)
                if avg_score_v_params:
                    model.score_v_la.load_state_dict(avg_score_v_params)
                    # model.score_v.load_state_dict(avg_score_v_params)
                    # model.rec_v.load_state_dict(avg_rec_v_params)
                if avg_score_a_params:
                    model.score_a_lv.load_state_dict(avg_score_a_params)
                    # model.score_a.load_state_dict(avg_score_a_params)
                    # model.rec_a.load_state_dict(avg_rec_a_params)

            # save each epoch model
        # if best_score_l_params:
        #     modality = ['l_va', 'v_la', 'a_lv']
        #     # modality = ['l', 'v', 'a']
        #     for j in range(self.args.modality_num):
        #         save_path = Path('pt') / 'FedDMER' / 'IEMOCAP4' / 'diffusion'   # MOSI/IEMOCAP4
        #         save_path.mkdir(parents=True, exist_ok=True)
        #         model_save_path = save_path / ('score_' + str(modality[j]) + '.pth')
        #         if j == 0:
        #             torch.save(best_score_l_params, model_save_path)
        #         elif j == 1:
        #             torch.save(best_score_v_params, model_save_path)
        #         elif j == 2:
        #             torch.save(best_score_a_params, model_save_path)
        #     logger.info(
        #             f"successfully saved diffusion models "
        #             )
        return


    def do_train_all(self, models, pseudo_models, dataloader, multi_type, return_epoch_results=False, num_classes=3):
        # optimizers = {
        #     idx: optim.Adam(model.parameters(), lr=self.args.learning_rate)
        #     for idx, model in models.items()
        # }
        optimizers = {
            idx: optim.AdamW(
                [param for param in model.parameters() if param.requires_grad],  # 只优化需要训练的参数
                lr=self.args.learning_rate
            )
            for idx, model in models.items()
        }
       
        schedulers = {
            idx: ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)
            for idx, optimizer in optimizers.items()
        }
        # initilize results
        epochs, best_epoch = 0, 0

        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        # 最小化损失还是最大化
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        client_num = self.args.client_num
        best_non0_acc_2 = [0.0 for _ in range(client_num)]
        best_test = {}

        for i in range(self.args.train_all_epoch):
            attn_params_all = {}  # 键是客户端ID, 值是某个客户端的注意力机制部分的参数，包括cross和self
            # local train
            for client_idx in range(client_num):
                # train
                losses = []
                data_loader = dataloader[client_idx]
                model = models[client_idx].train()
                train_loss = 0.0

                for epoch in range(self.args.local_epoch):
                    train_loss, result = self.client_train_and_send(model, data_loader, optimizers[client_idx], multi_type[client_idx], train_state='class', epoch=i, pseudo_models=pseudo_models, num_classes=num_classes)
                    logger.info(
                    f"LOCAL_TRAIN-CLIENT({client_idx+1}) -epoch[{epoch+1} / [{i+1}]] "
                    f">> loss: {round(train_loss, 4)} "
                    f"{dict_to_str(result)}"
                    )
                    schedulers[client_idx].step(train_loss)

            # sever_aggregation
            aggregated_params = self.aggregate_models_class(models)

            # update model
            for client_idx in range(client_num):
                model = models[client_idx]
                for name, param in aggregated_params.items():
                    if name in model.state_dict():  # 确保只更新在当前模型中存在的层
                        model.state_dict()[name].data.copy_(param.data)
            
            # test
            if (i+1)%self.args.test_internal == 0:
                for client_idx in range(client_num):
                    data_loader = dataloader[client_idx]
                    model = models[client_idx].eval()
                    test_results = self.do_test(model, data_loader['test'], mode="TEST", num_classes=num_classes, pseudo_models=pseudo_models)
                    if test_results['Acc_4'] > best_non0_acc_2[client_idx]:
                        best_non0_acc_2[client_idx] = test_results['Acc_4']
                        best_test[client_idx] = test_results
            
        return best_test if return_epoch_results else None


    def do_train_single_client(self, model, dataloader, return_epoch_results=False, multi_type=0):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        best_acc = 0

        while True:
            epochs += 1
            # train
            y_pre, y_true = [], []
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    output = model(text, audio, vision)

                    # loss
                    task_loss = self.criterion(output['M'], labels)
                    # backward
                    task_loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                                  self.args.grad_clip)
                    # store results
                    train_loss += task_loss
                    y_pre.append(output['M'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss.item() / len(dataloader['train'])

            pred, true = torch.cat(y_pre), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}]-multi_type:{multi_type} "
                f">> loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            cur_acc = test_results['Non0_acc_2']
            scheduler.step(val_results['Loss'])
            # best epoch
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            better = best_acc < cur_acc
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
            if better:
                best_acc, best_test = cur_acc, test_results
                # save model
                model_save_path = 'pt/' + 'singel_model_l' + '.pth'
                model_param = model.state_dict()
                torch.save(model_param, model_save_path)

            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return best_test if return_epoch_results else None


                    

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False, num_classes=3, pseudo_models=None, multi_type=0):
        model.eval()
        y_pred, y_true = [], []
        miss_one, miss_two = 0, 0
        local_epoch = 0
        epoch = 0

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for data in td:
                    # vision = batch_data['vision'].to(self.args.device)
                    # audio = batch_data['audio'].to(self.args.device)
                    # text = batch_data['text'].to(self.args.device)
                    # labels = batch_data['labels']['M'].to(self.args.device)
                    # batch_index = batch_data['index'].to(self.args.device)
                    batch_index = 0
                    vidnames = []

                    audio_host, text_host, visual_host = data[0].to(self.args.device), data[1].to(self.args.device), data[2].to(self.args.device)
                    audio_guest, text_guest, visual_guest = data[3].to(self.args.device), data[4].to(self.args.device), data[5].to(self.args.device)
                    qmask, umask, label = data[6].to(self.args.device), data[7].to(self.args.device), data[8].to(self.args.device)
                    vidnames += data[-1]
                    adim = audio_host.size(2)
                    tdim = text_host.size(2)
                    vdim = visual_host.size(2)

                    lengths = []
                    for j in range(len(umask)):
                        length = (umask[j] == 1).nonzero().tolist()[-1][0] + 1 
                        lengths.append(length)

                    ## generate input_features: ? * [seqlen, batch, adim+tdim+vdim]
                    input_features = generate_inputs(audio_host, text_host, visual_host, \
                                            audio_guest, text_guest, visual_guest, qmask)
                    audio = input_features[:,:,0:adim]
                    text = input_features[:,:,adim:adim+tdim]
                    vision = input_features[:,:,adim+tdim:adim+tdim+vdim]

                    if multi_type == 0:
                        pseudo_data = input_features[:,:,adim:adim+tdim]
                    elif multi_type == 1:
                        pseudo_data = input_features[:,:,adim+tdim:adim+tdim+vdim]
                    elif multi_type == 2:
                        pseudo_data = input_features[:,:,0:adim]
                    elif multi_type == 3:  #lv
                        torch.cat(text,vision,dim=2)
                    elif multi_type == 4:  #la
                        pseudo_data = input_features[:,:,0:adim+tdim]
                    elif multi_type == 5:  #la
                        pseudo_data = input_features[:,:,adim+tdim:adim+tdim+vdim]

                    audio = connect_uttr(audio, lengths).unsqueeze(1)
                    text = connect_uttr(text, lengths).unsqueeze(1)
                    vision = connect_uttr(vision, lengths).unsqueeze(1)
                    label = connect_uttr(label.transpose(0,1).unsqueeze(-1), lengths).squeeze()
                    if num_classes == 3:
                        labels = torch.zeros_like(label, dtype=torch.long)
                        # 分别赋值
                        labels[label < 0] = 0
                        labels[label == 0] = 1
                        labels[label > 0] = 2
                    elif num_classes == 4:
                        labels = label
                    elif num_classes == 6:
                        labels = label



                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    
                    if pseudo_models is not None:
                        pseudo_labels = {}
                        for modal in range(3):
                            pseudo_model = pseudo_models[modal].eval()
                            pseudo_label = pseudo_model(text, audio, vision, labels, local_epoch, epoch)
                            pseudo_labels[modal] = pseudo_label['M']
                        outputs = model(text, audio, vision, labels, pseudo_labels)
                    else:
                        if mode == 'pseudo':
                            outputs = model(text, audio, vision, pseudo_data, qmask, umask, lengths)
                        elif mode == 'extra':
                            outputs = model(text, audio, vision, is_train=False, num_modal=3)
                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        
        return eval_results
    


    