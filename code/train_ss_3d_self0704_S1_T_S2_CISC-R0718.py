#整合这两个train_interpolation_consistency_training_3D.py和train_regularized_dropout_3D.py并且把make_query_S1_S2_CISR-R.py的内容加上去
#整合这两个train_interpolation_consistency_training_3D.py和train_regularized_dropout_3D.py并且把make_query_S1_S2_CISR-R.py的内容加上去
#整合这两个train_interpolation_consistency_training_3D.py和train_regularized_dropout_3D.py并且把make_query_S1_S2_CISR-R.py的内容加上去

import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from networks.discriminator import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/LA/', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='SSNet', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--batch_size_all', type=int, default=16, help='batch_size_all per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=4, help='trained samples')#换的越来越多试试4的倍数
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

parser.add_argument('--ict_alpha', type=int, default=0.2, help='ict_alpha')
### cost
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model/LA_{}_{}_labeled/{}".format(args.exp, args.labelnum, args.model)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iteration
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)

def Weight_GAP_3D(supp_feat, mask):
    # 确保掩码具有正确的维度
    if len(mask.size()) != 5:
        mask = mask.unsqueeze(1).float()
    if supp_feat.size() != mask.size():
        supp_feat = F.interpolate(supp_feat, size=mask.size()[-3:], mode='trilinear', align_corners=True)
    # 加权特征图
    supp_feat = supp_feat * mask
    feat_d, feat_h, feat_w = supp_feat.shape[-3:]
    area = F.avg_pool3d(mask, (feat_d, feat_h, feat_w)) * feat_d * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool3d(input=supp_feat, kernel_size=supp_feat.shape[-3:]) * feat_d * feat_h * feat_w / area
    return supp_feat

def select_reliable(model=None):
    model.eval()
    '''
    Get labeled-anchors first, index by num_class计算每个类的类向量
    '''
    # db_train = LAHeart(base_dir=train_data_path,
    db_train = LAHeart(base_dir=train_data_path,                   
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(), 
                        ]))
    all_idxs = list(range(args.max_samples))
    batch_sampler_all = BatchSampler(all_idxs, args.batch_size_all)#这里的batch_size_all是32,主要与batch_size的区分
    #太大了爆掉了，改成16

    trainloader_all = DataLoader(db_train, batch_sampler=batch_sampler_all, num_workers=4,pin_memory=True, worker_init_fn=worker_init_fn)#注意这里batch_size 很大

    tbar = tqdm(trainloader_all)#这句
    
    vecs_all = torch.zeros((num_classes, 256)).cuda()#这里的256可能有问题，batch_size_all 32对应512，16对应256？
    num = torch.zeros((num_classes)).cuda()
    
    with torch.no_grad():
        for img, mask, _ in tbar:#batch size 32这里循环了两次
            img = img.cuda()#batch size 32这里尺寸是[32, 1, 112, 112, 80]
            # print("Labeled anchors img.shape is:\n", img.shape)
            mask = mask.cuda()#batch size 32这里尺寸是[32, 112, 112, 80]
            # print("Labeled anchors mask.shape is:\n", mask.shape)
            _, feat = model(img)#这里的feat就是高维特征feature,不知道它的模型return的是什么
    
            for index in range(num_classes):
                mask_cp = mask.clone()
                mask_cp[mask == index] = 1
                mask_cp[mask != index] = 0
                vec = Weight_GAP_3D(feat, mask_cp).view(-1)#[4, 112, 112, 80]
                vecs_all[index, :] += vec#这里计算所有类的向量vec
                num[index] += 1
            
    #mean
    vecs_all = vecs_all / num.view(-1,1)
    print("Labeled anchors vecs_all.shape is:\n", vecs_all.shape)#batch size 32这里尺寸是[2，512]这里没有一点问题

    '''
    CISC-based select labeled images for query
    '''
 
    min_reliability_feats = {}

    for index in range(num_classes):#num_classes是2，这里循环两遍
        anchor = vecs_all[index]
        tbar = tqdm(trainloader_all)

        id_to_reliability = []#这里也就表明

        #初始化变量来存储当前index下最小的reliability和对应的feat值
        min_reliability = float('inf')
        min_feat = None
        min_id = None
        min_simi = None

        with torch.no_grad():
            for img, mask, id in tbar:#这个循环次数与batch size有关80 // 32 + 1 = 3次
                # if index in list(np.unique(mask.cpu().numpy())):#如果index这个类在mask表示的label中，就退出循环，为啥？？？
                #     print("Break......")
                #     break
                img = img.cuda()#batch size 32这里尺寸是[32, 1, 112, 112, 80]
                # print("CISC-based select labeled images for query img.shape is:\n" ,img.shape)
                mask = mask.cuda()#batch size 32这里尺寸是[32, 112, 112, 80]
                # print("CISC-based select labeled images for query mask.shape is:\n" ,mask.shape)
                _, feat = model(img)
                n, c, _, _, _ = feat.size()#此处的高维特征要在VNet网络的编码阶段进行提取
                mask_cp = mask.clone()
                mask_cp[mask == index] = 1#把mask_cp中所有等于index这个类的地方设置为1
                mask_cp[mask != index] = 0#把mask_cp中多有不等于index的这个类的地方设置为0

                simi = F.cosine_similarity(feat, anchor.view(n, c, 1, 1, 1))#计算特征feat和类锚点anchor之间的相似度，先调整anchor的形状以匹配feat
                simi = F.interpolate(simi.unsqueeze(1), size=mask.size()[-3:], mode='trilinear').view(mask.size())
                print("simi.shape is:\n" ,simi.shape)#[32, 112, 112, 80]

                bce_loss = F.binary_cross_entropy(simi, mask_cp.float(), reduction='none')
                print("bce_loss.shape is:\n" ,bce_loss.shape)#[32, 112, 112, 80]

                loss = bce_loss[mask != 255].view(n, -1).mean(dim=1)
                print("loss.shape is:\n" ,loss.shape)#torch.Size([32])

                for i in range(len(id)):
                    reliability = loss[i].item()
                    id_to_reliability.append((id[i],reliability))
                    #更新当前index下最小的relibility和对应的feat值
                    if reliability < min_reliability:
                        min_reliability = reliability
                        min_feat = feat[i].clone()
                        min_id = id[i]
                        min_simi = simi[i].clone()

        id_to_reliability.sort(key=lambda elem: elem[1], reverse=False)#按照可靠性从低到高排序

        min_reliability_feats[index] = (min_reliability, min_feat, min_id, min_simi)

    print("min_feat shape is:\n", min_feat.shape)#torch.Size([16, 112, 112, 80])

    for index, data in min_reliability_feats.items():
        min_reliability, min_feat, min_id, min_simi = data
        print(f"Index: {index}, Min Reliability Feat Shape: {min_feat.shape}, File Name: {min_id}, Min reliability: {min_reliability}")

    return min_reliability_feats


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def create_model(ema=False):
    net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model
def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

if __name__ =="__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
#**********************************************************************************
    net_S1 = create_model()
    net_T = create_model(ema=True)
    net_S2 = create_model()
    model_S1 = kaiming_normal_init_weight(net_S1)
    model_T = kaiming_normal_init_weight(net_T)
    model_S2 = xavier_normal_init_weight(net_S2)
#**********************************************************************************
    db_train = LAHeart(base_dir=train_data_path, 
                    split='train', 
                    transform=transforms.Compose([
                        RandomRotFlip(), 
                        RandomCrop(patch_size), 
                        ToTensor(),
                        ]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
#**********************************************************************************
    model_S1.train()
    model_T.train()
    model_S2.train()
#**********************************************************************************
    optimizer_S1 = optim.SGD(model_S1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_S2 = optim.SGD(model_S2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#**********************************************************************************
    ce_loss = nn.CrossEntropyLoss()
    # ce_loss = losses.soft_ce_loss
    dice_loss = losses.Binary_dice_loss
    # dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
#**********************************************************************************
    best_dice_S1 = 0.0
    best_dice_S2 = 0.0
#**********************************************************************************
    max_epoch = max_iterations //len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            start_time = time.time()
            volume_batch, label_batch = sampled_batch[0], sampled_batch[1]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            print('\n\n\n================>  Select reliable images for the 1st stage re-training')
            min_reliability_feats = select_reliable(model_S2)#这里用到的是模型S2的能力，不对吧，要用教师模型的能力呀！！！
            #用教师模型的能力选类anchor,再从库中选和每个类最接近的数据，然后下面继续
            print('\n\n\n================>  Selection End')

            # ICT mix factors
            ict_mix_factors = np.random.beta(
                args.ict_alpha, args.ict_alpha, size=(args.labeled_bs // 2, 1, 1, 1, 1))
            ict_mix_factors = torch.tensor(
                ict_mix_factors, dtype=torch.float).cuda()
            unlabeled_volume_batch_0 = unlabeled_volume_batch[0:1, ...]
            unlabeled_volume_batch_1 = unlabeled_volume_batch[1:2, ...]

            # Mix imges
            batch_ux_mixed = unlabeled_volume_batch_0 * (1.0 - ict_mix_factors) + unlabeled_volume_batch_1 * ict_mix_factors
            input_volume_batch = torch.cat([labeled_volume_batch, batch_ux_mixed], dim=0)
            outputs_S1 = model_S1(input_volume_batch)[0]
#**********************************************************************************
            outputs_S2 = model_S2(input_volume_batch)[0]            
#**********************************************************************************        
            # print(outputs_S1)
            outputs_soft_S1 = torch.softmax(outputs_S1, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(
                    model_T(unlabeled_volume_batch_0)[0], dim=1)
                ema_output_ux1 = torch.softmax(
                    model_T(unlabeled_volume_batch_1)[0], dim=1)
                batch_pred_mixed = ema_output_ux0 * (1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors
                ###################自己加的0718######################
                embedding_T = model_T(unlabeled_volume_batch)[1]
                real_outputs_T = model_T(unlabeled_volume_batch)[0]
                real_outputs_soft_T = torch.softmax(real_outputs_T, dim=1).float()

############################################自己加的################################
            print('\n\n\n================>  开始逐一比较embedding_S1中的每个样本与两个min_feat分别对应index 0和index 1')
            embedding_S2 = model_S2(volume_batch)[1]#[4, 16, 112, 112, 80]
            real_outputs_S2 = model_S2(volume_batch)[0]#[4, 2, 112, 112, 80]
            real_outputs_soft_S2 = torch.softmax(real_outputs_S2, dim=1).float()

            n, c, d, h, w = embedding_S2.size()#embedding_S1改成了embedding_T
            similarity_losses = {0:[], 1:[]}#用于存储每个样本的相似度损失
            similarity_tensor = torch.zeros((n, 2, d, h, w), device=embedding_T.device)

            for i in range(n):
                sample = embedding_S2[i]
                sample = sample.view(1, c, d, h, w)

                for index in range(2):
                    min_feat = min_reliability_feats[index][1]
                    min_feat = min_feat.view(1, c, d, h, w)

                    #计算余弦相似度
                    simi = F.cosine_similarity(sample, min_feat, dim=1)
                    simi = F.interpolate(simi.unsqueeze(1), size=(d, h, w), mode='trilinear').view(d, h, w)
                    #将余弦相似度存储起来
                    similarity_losses[index].append(simi)
                    #将相似度加过存储到相似度张量中
                    similarity_tensor[i, index] = simi
            print("similarity_tensor shape is:\n", similarity_tensor.shape)#[4, 2, 112, 112, 80]
            print("real_outputs shape is:\n", real_outputs_S2.shape)#[4, 2, 112, 112, 80]
            print("real_outputs_soft_S2 shape is:\n", real_outputs_soft_S2.shape)

            simi_bce_loss = F.binary_cross_entropy_with_logits(similarity_tensor, real_outputs_soft_S2, reduction='none')
            simi_loss = simi_bce_loss.mean()
            print(f"simi_bce_loss shape: {simi_bce_loss.shape}, simi_loss shape: {simi_loss.shape}, simi loss value: {simi_loss}")
            '''
            损失不对啊, 原文公式7对应的是Weighted_CE啊
            重新写！！！

            '''
############################################自己加的################################

#**********************************************************************************
            outputs_soft_S2 = torch.softmax(outputs_S2, dim=1)

            model_S1_loss = 0.5 * (ce_loss(outputs_S1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(outputs_soft_S1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            model_S2_loss = 0.5 * (ce_loss(outputs_S2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(outputs_soft_S2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            r_drop_loss = losses.compute_kl_loss(outputs_S1[args.labeled_bs:], outputs_S2[args.labeled_bs:])
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_loss = torch.mean((outputs_soft_S2[args.labeled_bs:] - batch_pred_mixed)**2)
            loss = model_S1_loss + model_S2_loss  + consistency_weight * (consistency_loss + r_drop_loss) + simi_loss
#**********************************************************************************
            # loss_ce = ce_loss(outputs_S1[:args.labeled_bs],label_batch[:args.labeled_bs][:])
            # loss_dice = dice_loss(outputs_soft_S1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            # supervised_loss = 0.5 * (loss_dice + loss_ce)
            # consistency_weight = get_current_consistency_weight(iter_num // 150)
            # consistency_loss = torch.mean((outputs_soft_S1[args.labeled_bs:] - batch_pred_mixed)**2)
            # loss = supervised_loss + consistency_weight * consistency_loss

#**********************************************************************************
            optimizer_S1.zero_grad()
            optimizer_S2.zero_grad()
            loss.backward()
            optimizer_S1.step()
            optimizer_S2.step()
            update_ema_variables(model_S1, model_T, args.ema_decay, iter_num)
#**********************************************************************************

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_S1.param_groups:
                param_group['lr'] = lr_
#**********************************************************************************
            for param_group1 in optimizer_S1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer_S2.param_groups:
                param_group2['lr'] = lr_
#**********************************************************************************

#**********************************************************************************
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/model_S1_loss', model_S1_loss, iter_num)
            writer.add_scalar('info/model_S2_loss', model_S2_loss, iter_num)
            # writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/r_drop_loss', r_drop_loss, iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/simi_loss', simi_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, model S1 loss: %f, model S2 loss: %f, r drop loss: %f,  consistency loss: %f' %
                (iter_num, loss.item(), model_S1_loss.item(), model_S2_loss.item(), r_drop_loss.item(), consistency_loss.item()))
 #**********************************************************************************           

#**********************************************************************************
            if iter_num > 0 and iter_num % 200 == 0:
                model_S1.eval()
                dice_sample_S1 = test_3d_patch.var_all_case_LA(model_S1, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample_S1 > best_dice_S1:
                    best_dice_S1 = round(dice_sample_S1, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice_S1))
                    save_best_path = os.path.join(snapshot_path,'{}S1_best_model.pth'.format(args.model))
                    torch.save(model_S1.state_dict(), save_mode_path)
                    torch.save(model_S1.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/S1_Dice', dice_sample_S1, iter_num)
                writer.add_scalar('4_Var_dice/S1_Best_dice', best_dice_S1, iter_num)
                model_S1.train()
 
                model_S2.eval()
                dice_sample_S2 = test_3d_patch.var_all_case_LA(model_S2, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample_S2 > best_dice_S2:
                    best_dice_S2 = round(dice_sample_S2, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice_S2))
                    save_best_path = os.path.join(snapshot_path,'{}S2_best_model.pth'.format(args.model))
                    torch.save(model_S2.state_dict(), save_mode_path)
                    torch.save(model_S2.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/S2_Dice', dice_sample_S2, iter_num)
                writer.add_scalar('4_Var_dice/S2_Best_dice', best_dice_S2, iter_num)
                model_S2.train()
#**********************************************************************************
                end_time = time.time()
                iter_time = end_time - start_time  # Calculate iteration time
                print(f"Iteration {iter_num} took {iter_time:.2f} seconds")
            
                if iter_num >= max_iterations:
                    break

            if iter_num >= max_iterations:
                iterator.close()
                break

        writer.close()

            

            




    

###############################################################################
###############################################################################
###############################################################################
# if __name__ == "__main__":
#     if not os.path.exists(snapshot_path):
#         os.makedirs(snapshot_path)
#     if os.path.exists(snapshot_path + '/code'):
#         shutil.rmtree(snapshot_path + '/code')
#     shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

#     logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))

#     model_S1 = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    
#     db_train = LAHeart(base_dir=train_data_path,
#                        split='train',
#                        transform=transforms.Compose([
#                            RandomRotFlip(),
#                            RandomCrop(patch_size),
#                            ToTensor(),
#                        ]))
#     labelnum = args.labelnum
#     labeled_idxs = list(range(labelnum))
#     unlabeled_idxs = list(range(labelnum, args.max_samples))
#     batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    
#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)
        
#     trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

#     model_S1.train()

#     optimizer_S1 = optim.SGD(model_S1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

#     dice_loss = losses.Binary_dice_loss
#     CE_loss = losses.soft_ce_loss
#     KL_loss = losses.kl_loss
#     MSE_loss = losses.mse_loss
#     bce_loss = nn.BCELoss()

#     writer = SummaryWriter(snapshot_path+'/log')
#     logging.info("{} iterations per epoch".format(len(trainloader)))
#     iter_num = 0
#     best_dice = 0
#     max_epoch = max_iterations // len(trainloader) + 1
#     lr_ = base_lr
#     iterator = tqdm(range(max_epoch), ncols=70)

#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             start_time = time.time()
#             volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

#             # Student Model S1
#             model_S1.train()
#             outputs_S1, embedding_S1 = model_S1(volume_batch)#横线_本来是embedding_S1

#             outputs_soft_S1 = F.softmax(outputs_S1, dim=1)
#             y_S1 = outputs_soft_S1[:args.labeled_bs]
#             _, prediction_label_S1 = torch.max(y_S1, dim=1)
#             _, pseudo_label_S1 = torch.max(outputs_soft_S1[args.labeled_bs:], dim=1)
#             pseudo_label_S1 = pseudo_label_S1.float()

#             true_labels = label_batch[:args.labeled_bs]
#             true_labels = F.one_hot(true_labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

#             loss_dice_S1 = dice_loss(y_S1[:, 1, ...], (true_labels == 1))
#             supervised_loss_S1 = loss_dice_S1

#             # Overall loss
#             total_loss = loss_dice_S1

#             optimizer_S1.zero_grad()
#             total_loss.backward()
#             optimizer_S1.step()

#             iter_num = iter_num + 1

#             writer.add_scalar('lr', lr_, iter_num)
#             writer.add_scalar('loss/total_loss', total_loss, iter_num)
#             # writer.add_scalar('loss/loss_adv', loss_adv, iter_num)

#             logging.info('iteration %d : total_loss : %f' % (iter_num, total_loss.item()))

#                 # change lr
#             if iter_num % 2500 == 0:
#                 lr_ = base_lr * 0.1 ** (iter_num // 2500)
#                 for param_group in optimizer_S1.param_groups:
#                     param_group['lr'] = lr_
            
#             # if iter_num >= 400 and iter_num % 200 == 0:
#             model_S1.eval()
#             if iter_num >= 400 and iter_num % 200 == 0:
#                 dice_sample = test_3d_patch.var_all_case_LA(model_S1, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
#                 if dice_sample > best_dice:
#                     best_dice = round(dice_sample, 4)
#                     save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
#                     save_best_path = os.path.join(snapshot_path,'{}S1_best_model.pth'.format(args.model))
#                     torch.save(model_S1.state_dict(), save_mode_path)
#                     torch.save(model_S1.state_dict(), save_best_path)
#                     logging.info("save best model to {}".format(save_mode_path))
#                 writer.add_scalar('4_Var_dice/S1_Dice', dice_sample, iter_num)
#                 writer.add_scalar('4_Var_dice/S1_Best_dice', best_dice, iter_num)
#             model_S1.train()


#             end_time = time.time()
#             iter_time = end_time - start_time  # Calculate iteration time
#             print(f"Iteration {iter_num} took {iter_time:.2f} seconds")
            
#             if iter_num >= max_iterations:
#                 break
            
#         if iter_num >= max_iterations:
#             iterator.close()
#             break

#     writer.close()



