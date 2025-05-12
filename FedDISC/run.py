import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import random
from config import get_config_regression
from dataloader_cmumosi import CMUMOSIDataset
from dataloader_iemocap import IEMOCAPDataset
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import feddisc
import path
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')  # 创建一个日志记录器对象


def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def score_model_load(args, models):
    score_a_lv_pth = 'pt/FedDISC/IEMOCAP4/diffusion/score_a_lv.pth'
    score_l_va_pth = 'pt/FedDISC/IEMOCAP4/diffusion/score_l_va.pth'
    score_v_la_pth = 'pt/FedDISC/IEMOCAP4/diffusion/score_v_la.pth'
    # score_a_pth = 'pt/FedDISC/IEMOCAP4/diffusion/score_a.pth'
    # score_l_pth = 'pt/FedDISC/IEMOCAP4/diffusion/score_l.pth'
    # score_v_pth = 'pt/FedDISC/IEMOCAP4/diffusion/score_v.pth'
    # rec_a_pth = 'pt/rec_a.pth'
    # rec_l_pth = 'pt/rec_l.pth'
    # rec_v_pth = 'pt/rec_v.pth'
    proj_l_pth = 'pt/proj_l.pth'
    proj_v_pth = 'pt/proj_v.pth'
    proj_a_pth = 'pt/proj_a.pth'
    model_extra_pth = 'pt/FedDISC/IEMOCAP4/extra/model_extra.pth'    # MOSI/IEMOCAP4

    for client_idx in range(args.client_num):
        multi_type = args.multi_type[client_idx]  # 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 6:lva
        model = models[client_idx]
        # load params
        # model.proj_l.load_state_dict(torch.load(proj_l_pth))
        # model.proj_v.load_state_dict(torch.load(proj_v_pth))
        # model.proj_a.load_state_dict(torch.load(proj_a_pth))
        # model.load_state_dict(torch.load(model_extra_pth), strict=False)
        model_extra = torch.load(model_extra_pth)
        # model.load_state_dict(model_extra, strict=False)
        # print(model_extra.keys())
        # # # 冻结这些部分的参数
        # for name, param in model.named_parameters():
        #     if name in model_extra:
        #         param.requires_grad = False
        proj_l_dict = {k.replace("proj_l.", ""): v for k, v in model_extra.items() if k.startswith("proj_l.")}
        model.proj_l.load_state_dict(proj_l_dict)

        # 筛选 proj_a 的参数，并去掉前缀 "proj_a."
        proj_a_dict = {k.replace("proj_a.", ""): v for k, v in model_extra.items() if k.startswith("proj_a.")}
        model.proj_a.load_state_dict(proj_a_dict)

        # 筛选 proj_v 的参数，并去掉前缀 "proj_v."
        proj_v_dict = {k.replace("proj_v.", ""): v for k, v in model_extra.items() if k.startswith("proj_v.")}
        model.proj_v.load_state_dict(proj_v_dict)

        # 冻结这些部分的参数
        for param in model.proj_l.parameters():
            param.requires_grad = False
        for param in model.proj_v.parameters():
            param.requires_grad = False
        for param in model.proj_a.parameters():
            param.requires_grad = False
        
        model.score_a_lv.load_state_dict(torch.load(score_a_lv_pth))
        model.score_l_va.load_state_dict(torch.load(score_l_va_pth))
        model.score_v_la.load_state_dict(torch.load(score_v_la_pth))

        # model.score_a.load_state_dict(torch.load(score_a_pth))
        # model.score_l.load_state_dict(torch.load(score_l_pth))
        # model.score_v.load_state_dict(torch.load(score_v_pth))
        # 冻结这些部分的参数
        for param in model.score_l_va.parameters():
            param.requires_grad = False
        for param in model.score_v_la.parameters():
            param.requires_grad = False
        for param in model.score_a_lv.parameters():
            param.requires_grad = False
        # if multi_type == 0:
        #     model.score_a.load_state_dict(torch.load(score_a_pth))
        #     model.score_v.load_state_dict(torch.load(score_v_pth))
        #     # model.rec_a.load_state_dict(torch.load(rec_a_pth))
        #     # model.rec_v.load_state_dict(torch.load(rec_v_pth))
        # elif multi_type == 1:
        #     model.score_a.load_state_dict(torch.load(score_a_pth))
        #     model.score_l.load_state_dict(torch.load(score_l_pth))
        #     # model.rec_a.load_state_dict(torch.load(rec_a_pth))
        #     # model.rec_l.load_state_dict(torch.load(rec_l_pth))
        # elif multi_type == 2:
        #     model.score_v.load_state_dict(torch.load(score_v_pth))
        #     model.score_l.load_state_dict(torch.load(score_l_pth))
        #     # model.rec_v.load_state_dict(torch.load(rec_v_pth))
        #     # model.rec_l.load_state_dict(torch.load(rec_l_pth))
        # elif multi_type == 3:
        #     model.score_a.load_state_dict(torch.load(score_a_pth))
        #     # model.rec_a.load_state_dict(torch.load(rec_a_pth))
        # elif multi_type == 4:
        #     model.score_v.load_state_dict(torch.load(score_v_pth))
        #     # model.rec_v.load_state_dict(torch.load(rec_v_pth))
        # elif multi_type == 5:
        #     model.score_l.load_state_dict(torch.load(score_l_pth))
        #     # model.rec_l.load_state_dict(torch.load(rec_l_pth))
 

def get_loaders(args, audio_root, text_root, video_root, num_folder, dataset, batch_size, num_workers, client_num):

    ###########################################################################
    ###########################################################################
    if dataset in ['CMUMOSI', 'CMUMOSEI']:
        # mosi: conversations：93  utterances: 2199
        dataset = CMUMOSIDataset(label_path=path.PATH_TO_LABEL_Win[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)
        # -Vids里面相当于是被试的名字，每个名字对应一段conversation
        trainNum = args.data_align[0]
        valNum = args.data_align[1]
        testNum = args.data_align[2]
        train_idxs = [list(range(i*trainNum, (i+1)*trainNum)) for i in range(client_num)]
        val_idxs = list(range(trainNum*client_num, trainNum*client_num+valNum))
        test_idxs = list(range(trainNum*client_num+valNum, trainNum*client_num+valNum+testNum))
        dataLoader_all = {}
        for idx in range(client_num):

            train_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(train_idxs[idx]),
                                    collate_fn=dataset.collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=False)
            val_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(val_idxs),
                                    collate_fn=dataset.collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=False)
            test_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(test_idxs),
                                    collate_fn=dataset.collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=False)

            # train_loaders = [train_loader]
            # val_loaders = [val_loader]
            # test_loaders = [test_loader]

            dataloader = {'train':train_loader, 'test':test_loader, 'valid':val_loader}
            dataLoader_all[idx] = dataloader

        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return dataLoader_all, adim, tdim, vdim


    ###########################################################################
    ###########################################################################

    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']: ## five folder cross-validation, each fold contains (train, test)

        dataset = IEMOCAPDataset(label_path=path.PATH_TO_LABEL_Win[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)

        ## gain index for cross-validation
        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):
            session = int(vid[4]) - 1
            if session not in session_to_idx: session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == num_folder, f'Must split into five folder'

        train_test_idxs = []
        for ii in range(num_folder): # ii in [0, 4]
            test_idxs = session_to_idx[ii]
            train_idxs = []
            for jj in range(num_folder):
                if jj != ii: train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])

        ## gain train and test loaders
        train_loaders = []
        test_loaders = []

        train_test_idx = train_test_idxs[4]
        train_idxs = train_test_idx[0]
        test_idxs = train_test_idx[1]

        trainNum = int(len(train_idxs)/3)

        dataLoader_all = {}
        for idx in range(args.client_num):
            
            # print(len(train_idxs), train_idxs)
            # print(len(test_idxs), test_idxs)

            train_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(train_idxs[idx*trainNum: (idx+1)*trainNum]),  # [idx*trainNum: (idx+1)*trainNum]
                                    collate_fn=dataset.collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=False)
            test_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(test_idxs),
                                    collate_fn=dataset.collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=False)

            # train_loaders = [train_loader]
            # val_loaders = [val_loader]
            # test_loaders = [test_loader]

            dataloader = {'train':train_loader, 'test':test_loader}
            dataLoader_all[idx] = dataloader



        ## return loaders
        adim, tdim, vdim = dataset.get_featDim()
        return dataLoader_all, adim, tdim, vdim


def FedDISC_run(
        model_name, dataset_name, config=None, config_file="", seeds=[], mr=0.1, is_tune=False,
        tune_times=500, feature_T="", feature_A="", feature_V="",
        model_save_dir="./pt", res_save_dir="./result", log_dir="./log",
        gpu_ids=[0], num_workers=4, verbose_level=1, mode='train'
):

    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if config_file != "":
        config_file = Path(config_file)
    else:  # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"  # Path(__file__).parent它代表当前脚本文件的路径的父目录
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"   # Path.home() 返回当前用户的主目录路径
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    args = get_config_regression(model_name, dataset_name, config_file)
    args.mode = mode
    args.mr = mr  # missing rate
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'classification'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    random.seed(2010221)
    # random_list = [random.randint(0, 6) for _ in range(args.client_num)]   
    random_list = [3,4,5]
    multi_type = ['l','v','a','lv','la','va','lva']# 0:l, 1:v, 2:a, 3:lv, 4:la, 5:va, 6:lva
    args['multi_type'] = random_list
    print([multi_type[i] for i in random_list])
    args['train_state'] = 2 # -1--pseudo, 0--extract, 1--generate, 2--classify
    
    
    
    if config:
        args.update(config)

    res_save_dir = Path(res_save_dir) / "FedDISC"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []

    # for i, seed in enumerate(seeds):
    #     setup_seed(seed)
    #     args['cur_seed'] = i + 1
    #     result = _run(args, num_workers, is_tune)
    #     model_results.append(result)
    seed = seeds[0]
    setup_seed(seed)
    model_results = _run(args, num_workers, is_tune)

    # save result to csv
    csv_file = res_save_dir / f"{dataset_name}_best.csv"

    if os.path.exists(csv_file):
        print(f"文件 '{csv_file}' 已存在。")
    else:
        # 使用 pandas 的 DataFrame 来存储结果
        df = pd.DataFrame(model_results)

        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(csv_file, index=False)
  
    logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=8, is_tune=False, from_sena=False):
    audio_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.audio_feature)
    text_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.text_feature)
    video_root = os.path.join(path.PATH_TO_FEATURES_Win[args.dataset], args.video_feature)
    # dataloader = MMDataLoader(args, num_workers)
    dataloader, adim, tdim, vdim = get_loaders( args=args,
                                                audio_root = audio_root, 
                                                text_root  = text_root,
                                                video_root = video_root,
                                                num_folder = 5,
                                                batch_size = args.batch_size,
                                                dataset = args.dataset,
                                                num_workers = num_workers,
                                                client_num = args.client_num
                                                )
    if args.model_name == 'disc':
        # =========train extractor=========
        if args.train_state == -1:
            print('=========train pseudo=========')
            models = {}
            D_e = args.hidden
            graph_h = args.hidden // 2
            for client_idx in range(args.client_num):
                models[client_idx] = getattr(feddisc, 'pseudo')(args, 
                                                                args.multi_type[client_idx],
                                                                args.base_model,
                                                                adim, tdim, vdim, D_e, graph_h,
                                                                n_speakers=args.n_speakers,
                                                                window_past=args.windowp,
                                                                window_future=args.windowf,
                                                                n_classes=args.num_classes,
                                                                dropout=args.dropout,
                                                                time_attn=args.time_attn,
                                                                no_cuda=args.no_cuda 
                                                                     )
                models[client_idx] = models[client_idx].cuda()
            trainer = ATIO().getTrain(args)
            trainer.do_train_extractor(args.train_state, models, dataloader, multi_type=args.multi_type, return_epoch_results=from_sena, num_classes=args.num_classes)

        if args.train_state == 0:
            print('=========train extractor=========')
            models = {}
            teacher_model_pth = 'pt/FedDISC/MOSI/pseudo/pseudo_l.pth'
            model_extra_pth = 'pt/FedDISC/IEMOCAP4/extra/model_extra.pth'
            for client_idx in range(args.client_num):
                models[client_idx] = getattr(feddisc, 'single_client')(args, multi_type = args.multi_type[client_idx])
                models[client_idx] = models[client_idx].cuda()

                # model_extra = torch.load(model_extra_pth)
                # models[client_idx].load_state_dict(model_extra, strict=False)

            trainer = ATIO().getTrain(args)
            trainer.do_train_extractor(models, dataloader, multi_type=args.multi_type, return_epoch_results=from_sena, num_classes=args.num_classes)

        # =========train generator=========
        elif args.train_state == 1:
            print('=========train generator=========')
            # proj path
            proj_l_pth = 'pt/proj_l.pth'
            proj_v_pth = 'pt/proj_v.pth'
            proj_a_pth = 'pt/proj_a.pth'
            singel_model_l_pth = 'pt/singel_model_l.pth'
            model_extra_pth = 'pt/FedDISC/IEMOCAP6/extra/model_extra.pth'    # MOSI/IEMOCAP4/IEMOCAP6
            if args.multi_type[0] == 6:
                pseudo_pth = {'lv':'pt/FedDISC/IEMOCAP6/pseudo/pseudo_lv.pth',
                            'la':'pt/FedDISC/IEMOCAP6/pseudo/pseudo_la.pth',
                            'va':'pt/FedDISC/IEMOCAP6/pseudo/pseudo_va.pth'}
                modals = ['lv', 'la','va']
            elif args.multi_type[0] == 3:
                pseudo_pth = {'l':'pt/FedDISC/IEMOCAP4/pseudo/pseudo_l.pth',
                              'v':'pt/FedDISC/IEMOCAP4/pseudo/pseudo_v.pth',
                              'a':'pt/FedDISC/IEMOCAP4/pseudo/pseudo_a.pth'}
                modals = ['l', 'v','a']
            # GET MODEL
            pseudo_models = {}
            
            for modal_idx in range(3):
                pseudo_models[modal_idx] = getattr(feddisc, 'single_client')(args, multi_type = modal_idx+3)
                pseudo_models[modal_idx] = pseudo_models[modal_idx].cuda()
                pseudo_models[modal_idx].load_state_dict(torch.load(pseudo_pth[modals[modal_idx]]))

            models = {}
            for client_idx in range(args.client_num):
                models[client_idx] = getattr(feddisc, 'FedDISC_train_diff')(args, multi_type = args.multi_type[client_idx], need_tsne=1)
                models[client_idx] = models[client_idx].cuda()
                # load params
                # models[client_idx].proj_l.load_state_dict(torch.load(proj_l_pth))
                # models[client_idx].proj_v.load_state_dict(torch.load(proj_v_pth))
                # models[client_idx].proj_a.load_state_dict(torch.load(proj_a_pth))

                model_extra = torch.load(model_extra_pth)

                # 加载整个预训练模型
                # models[client_idx].load_state_dict(model_extra, strict=False)

                # 加载投影部分，分类器不加载
                proj_l_dict = {k.replace("proj_l.", ""): v for k, v in model_extra.items() if k.startswith("proj_l.")}
                models[client_idx].proj_l.load_state_dict(proj_l_dict)
                proj_a_dict = {k.replace("proj_a.", ""): v for k, v in model_extra.items() if k.startswith("proj_a.")}
                models[client_idx].proj_a.load_state_dict(proj_a_dict)
                proj_v_dict = {k.replace("proj_v.", ""): v for k, v in model_extra.items() if k.startswith("proj_v.")}
                models[client_idx].proj_v.load_state_dict(proj_v_dict)

                # 冻结参数
                for param in models[client_idx].proj_l.parameters():
                    param.requires_grad = False
                for param in models[client_idx].proj_v.parameters():
                    param.requires_grad = False
                for param in models[client_idx].proj_a.parameters():
                    param.requires_grad = False

                # for name, param in models[client_idx].named_parameters():
                #     if name in model_extra:
                #         param.requires_grad = False
          


            trainer = ATIO().getTrain(args)

            # train diff
            trainer.do_train_generator(models, pseudo_models, dataloader, multi_type=args.multi_type, return_epoch_results=from_sena, num_classes=args.num_classes)

        # =========train classifier=========
        elif args.train_state == 2:
            print('=========train classifier=========')
            # get model
            model_extra_pth = 'pt/FedDISC/IEMOCAP4/extra/model_extra.pth'    # MOSI/IEMOCAP4/IEMOCAP6
            if args.multi_type[0] >= 3:
                pseudo_pth = {'lv':'pt/FedDISC/IEMOCAP4/pseudo/pseudo_lv.pth',
                            'la':'pt/FedDISC/IEMOCAP4/pseudo/pseudo_la.pth',
                            'va':'pt/FedDISC/IEMOCAP4/pseudo/pseudo_va.pth'}
                modals = ['lv', 'la','va']
            elif args.multi_type[0] < 3:
                pseudo_pth = {'l':'pt/FedDISC/IEMOCAP4/pseudo/pseudo_l.pth',
                            'v':'pt/FedDISC/IEMOCAP4/pseudo/pseudo_v.pth',
                            'a':'pt/FedDISC/IEMOCAP4/pseudo/pseudo_a.pth'}
                modals = ['l', 'v','a']
            # GET MODEL
            pseudo_models = {}
            
            for modal_idx in range(3):
                pseudo_models[modal_idx] = getattr(feddisc, 'single_client')(args, multi_type = args.multi_type[modal_idx])
                pseudo_models[modal_idx] = pseudo_models[modal_idx].cuda()
                pseudo_models[modal_idx].load_state_dict(torch.load(pseudo_pth[modals[modal_idx]]))

            models = {}
            for client_idx in range(args.client_num):
                models[client_idx] = getattr(feddisc, 'FedDISC_train_class')(args, multi_type=args.multi_type[client_idx], need_tsne=0)
                models[client_idx] = models[client_idx].cuda()
            score_model_load(args, models)

            trainer = ATIO().getTrain(args)

            # train all
            best_tests = trainer.do_train_all(models, pseudo_models, dataloader, multi_type=args.multi_type, return_epoch_results=True, num_classes=args.num_classes)
            BEST_RESULT = [best_tests[i] for i in range(args.client_num)]
            return BEST_RESULT
        

    elif args.model_name == 'single_client':
        results = []
        # for multi_type in range(7):
        # get model
        model = getattr(feddisc, 'single_client')(args, multi_type = 0)
        model = model.cuda()

        trainer = ATIO().getTrain(args)

        # train
        epoch_results = trainer.do_train_single_client(model, dataloader[0], return_epoch_results=True, multi_type=0)
        results.append(epoch_results)
        return results
    
        
    return