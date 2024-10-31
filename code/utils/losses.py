import torch
from torch.nn import functional as F
import torch.nn as nn
import contextlib
import numpy as np
import math
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot

def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass

class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    # def forward(self, inputs, target, weight=None, softmax=False):
    #     if softmax:
    #         inputs = torch.softmax(inputs, dim=1)
    #     target = self._one_hot_encoder(target)
    #     if weight is None:
    #         weight = [1] * self.n_classes
    #     assert inputs.size() == target.size(), 'predict & target shape do not match'
    #     class_wise_dice = []
    #     loss = 0.0
    #     for i in range(0, self.n_classes):
    #         dice = self._dice_loss(inputs[:, i], target[:, i])
    #         class_wise_dice.append(1.0 - dice.item())
    #         loss += dice * weight[i]
    #     return loss / self.n_classes
    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            # bug found by @CamillerFerros at github issue#25
            mask = mask.repeat(1, self.n_classes, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes): 
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes

def Binary_dice_loss(predictive, target, ep=1e-5):#1e-8改一下试试
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss / self.n_classes
        
@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d

class VAT2d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)

    def forward(self, model, x):
        with torch.no_grad():
            pred= F.softmax(model(x)[0], dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d) 
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                logp_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)[0]
            logp_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(logp_hat, pred)
        return lds

class VAT3d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss
        
    def forward(self, model, x):
        with torch.no_grad():
            pred= F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device) ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d) ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)[0]
            p_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(p_hat, pred)
        return lds

def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]


def mse_Loss(probs, targets, reduction='mean'):
    if reduction == 'mean':
        criteria = nn.MSELoss(reduction='mean')
    elif reduction == 'sum':
        criteria = nn.MSELoss(reduction='sum')
    else:
        criteria = nn.MSELoss(reduction='none')
    mse_loss = criteria(probs, targets)
    return mse_loss


def dice_loss(score, target):
    target = target.to(torch.float32)
    # target = target.float()
    smooth = 1e-10
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def compute_sdf(img_gt, out_shape):
    """
    Semi-supervised Medical Image Segmentation through Dual-task Consistency
    https://ojs.aaai.org/index.php/AAAI/article/view/17066

    code refer to:
    https://github.com/HiLab-git/DTC/blob/master/code/utils/util.py

    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        # posmask = img_gt[b].astype(np.bool)#0813改
        posmask = img_gt[b].astype(bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                    np.max(posdis) - np.min(posdis))
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf

    return normalized_sdf

class DTU_Student_Loss():
    def __init__(self, args, targets, l_output, r_output, l_output_sdf, r_output_sdf, le_output=None, re_output=None,
                 le_output_sdf=None, re_output_sdf=None, epc=0):
        self.args = args
        self.epc = epc * 0.5
        # self.epc = epc
        self.targets = targets
        self.l_output = l_output
        self.r_output = r_output
        self.l_output_sdf = l_output_sdf
        self.r_output_sdf = r_output_sdf
        self.le_output = le_output
        self.re_output = re_output
        self.le_output_sdf = le_output_sdf
        self.re_output_sdf = re_output_sdf
        # self.save_fig_path = './save_figure/'


    def cal_train_loss(self):

        l_output_labeled = self.l_output[:self.args.labeled_bs]
        r_output_labeled = self.r_output[:self.args.labeled_bs]
        l_output_labeled_sdf = self.l_output_sdf[:self.args.labeled_bs]
        r_output_labeled_sdf = self.r_output_sdf[:self.args.labeled_bs]

        l_output_unlabeled = self.l_output[self.args.labeled_bs:]
        r_output_unlabeled = self.r_output[self.args.labeled_bs:]
        le_output_unlabeled = self.le_output[self.args.labeled_bs:]
        re_output_unlabeled = self.re_output[self.args.labeled_bs:]

        target_label = self.targets

        # Segmentation Loss
        l_output_labeled_soft = F.softmax(l_output_labeled, dim=1)
        r_output_labeled_soft = F.softmax(r_output_labeled, dim=1)
        l_dice = 0.0
        r_dice = 0.0

        l_loss_dice = dice_loss(l_output_labeled_soft[:, 0, :, :, :], target_label)
        r_loss_dice = dice_loss(r_output_labeled_soft[:, 0, :, :, :], target_label)
        l_dice += l_loss_dice
        r_dice += r_loss_dice

        # SDF Loss
        b, c, d, w, h = l_output_labeled.size()
        sdf = compute_sdf(target_label.detach().cpu().numpy(), (b, d, w, h))
        sdf = torch.from_numpy(sdf).cuda().float()
        l_output_labeled_sdf_c0 = l_output_labeled_sdf[:, 0, ...]
        r_output_labeled_sdf_c0 = r_output_labeled_sdf[:, 0, ...]
        l_sdfLoss = mse_Loss(l_output_labeled_sdf_c0, sdf)
        r_sdfLoss = mse_Loss(r_output_labeled_sdf_c0, sdf)

        # Consistency Loss (uncertainty) Seg
        consistency_weight = 0.1 * math.exp(-5 * math.pow((1 - self.epc / self.args.nEpochs), 2))
        l_output_soft = F.softmax(self.l_output, dim=1)
        le_output_soft = F.softmax(self.le_output, dim=1)
        r_output_soft = F.softmax(self.r_output, dim=1)
        re_output_soft = F.softmax(self.re_output, dim=1)

        l_p_seg_bool = (l_output_soft > self.args.tau_p) * (le_output_soft > self.args.tau_p)
        l_n_seg_bool = (l_output_soft <= self.args.tau_n) * (le_output_soft <= self.args.tau_n)
        l_certain_seg_bool = l_p_seg_bool + l_n_seg_bool

        r_p_seg_bool = (r_output_soft > self.args.tau_p) * (re_output_soft > self.args.tau_p)
        r_n_seg_bool = (r_output_soft < self.args.tau_n) * (re_output_soft < self.args.tau_n)

        r_certain_seg_bool = r_p_seg_bool + r_n_seg_bool

        l_certain_seg = l_output_soft * (l_certain_seg_bool == True)
        le_certain_seg = le_output_soft * (l_certain_seg_bool == True)
        r_certain_seg = r_output_soft * (r_certain_seg_bool == True)
        re_certain_seg = re_output_soft * (r_certain_seg_bool == True)
        l_uncertain_seg = l_output_soft * (l_certain_seg_bool == False)
        le_uncertain_seg = le_output_soft * (l_certain_seg_bool == False)
        r_uncertain_seg = r_output_soft * (r_certain_seg_bool == False)
        re_uncertain_seg = re_output_soft * (r_certain_seg_bool == False)

        # reliable consLoss seg
        lc_consLoss_seg = torch.mean(torch.pow(l_certain_seg - le_certain_seg, 2))
        rc_consLoss_seg = torch.mean(torch.pow(r_certain_seg - re_certain_seg, 2))

        # unreliable consLoss seg
        le_uncertainty = -1.0 * torch.sum(le_uncertain_seg * torch.log2(le_uncertain_seg + 1e-6), dim=1, keepdim=True)
        re_uncertainty = -1.0 * torch.sum(re_uncertain_seg * torch.log2(re_uncertain_seg + 1e-6), dim=1, keepdim=True)

        luc_consLoss_seg = torch.mean(torch.pow(l_uncertain_seg - le_uncertain_seg, 2) * torch.exp(-le_uncertainty))
        ruc_consLoss_seg = torch.mean(torch.pow(r_uncertain_seg - re_uncertain_seg, 2) * torch.exp(-re_uncertainty))

        # Consistency Loss (uncertainty) SDF
        l_certain_sdf_bool = [(self.l_output_sdf > self.args.sdf_threshold) * (self.le_output_sdf > self.args.sdf_threshold)]\
                             + [(self.l_output_sdf < -self.args.sdf_threshold) * (self.le_output_sdf < -self.args.sdf_threshold)]
        r_certain_sdf_bool = [(self.r_output_sdf > self.args.sdf_threshold) * (self.re_output_sdf > self.args.sdf_threshold)]\
                             + [(self.r_output_sdf < -self.args.sdf_threshold) * (self.re_output_sdf < -self.args.sdf_threshold)]

        # reliable consLoss sdf
        l_certain_sdf = self.l_output_sdf * (l_certain_sdf_bool == True)
        le_certain_sdf = self.le_output_sdf * (l_certain_sdf_bool == True)
        r_certain_sdf = self.r_output_sdf * (r_certain_sdf_bool == True)
        re_certain_sdf = self.re_output_sdf * (r_certain_sdf_bool == True)

        lc_consLoss_sdf = torch.mean(torch.pow(l_certain_sdf - le_certain_sdf, 2))
        rc_consLoss_sdf = torch.mean(torch.pow(r_certain_sdf - re_certain_sdf, 2))

        # unreliable consLoss sdf
        l_uncertain_sdf = self.l_output_sdf * (l_certain_sdf_bool == False)
        le_uncertain_sdf = self.le_output_sdf * (l_certain_sdf_bool == False)
        r_uncertain_sdf = self.r_output_sdf * (r_certain_sdf_bool == False)
        re_uncertain_sdf = self.re_output_sdf * (r_certain_sdf_bool == False)

        le_uncertainty_sdf = le_uncertain_sdf.var()
        re_uncertainty_sdf = re_uncertain_sdf.var()
        luc_consLoss_sdf = torch.mean(torch.pow(l_uncertain_sdf - le_uncertain_sdf, 2) * torch.exp(-le_uncertainty_sdf))
        ruc_consLoss_sdf = torch.mean(torch.pow(r_uncertain_sdf - re_uncertain_sdf, 2) * torch.exp(-re_uncertainty_sdf))

        # Dual-task Cons Loss
        l_output_sdf_c0 = self.l_output_sdf[:, 0, ...]
        r_output_sdf_c0 = self.r_output_sdf[:, 0, ...]
        l_output_c0 = self.l_output[:, 0, ...]
        r_output_c0 = self.r_output[:, 0, ...]
        l_seg_2sigmoid = torch.sigmoid(-2.0 * l_output_c0)
        r_seg_2sigmoid = torch.sigmoid(-2.0 * r_output_c0)

        l_dual_task_consistency = torch.mean(torch.pow((2 * l_seg_2sigmoid - 1) - l_output_sdf_c0, 2))
        r_dual_task_consistency = torch.mean(torch.pow((2 * r_seg_2sigmoid - 1) - r_output_sdf_c0, 2))

        # Stabilization Loss
        l_output_unlabeled_soft = F.softmax(l_output_unlabeled, dim=1)
        r_output_unlabeled_soft = F.softmax(r_output_unlabeled, dim=1)
        le_output_unlabeled_soft = F.softmax(le_output_unlabeled, dim=1)
        re_output_unlabeled_soft = F.softmax(re_output_unlabeled, dim=1)

        l_stable = mse_Loss(l_output_unlabeled_soft, le_output_unlabeled_soft, reduction='none')
        r_stable = mse_Loss(r_output_unlabeled_soft, re_output_unlabeled_soft, reduction='none')

        l_stable_bool = (l_output_unlabeled_soft > self.args.label_threshold) * \
                        (le_output_unlabeled_soft > self.args.label_threshold) * (l_stable < self.args.stable_threshold)
        r_stable_bool = (r_output_unlabeled_soft > self.args.label_threshold) * \
                        (re_output_unlabeled_soft > self.args.label_threshold) * (r_stable < self.args.stable_threshold)

        l_stable_bool_np = l_stable_bool.detach().cpu().numpy()
        r_stable_bool_np = r_stable_bool.detach().cpu().numpy()
        l_stable_np = l_stable.detach().cpu().numpy()
        r_stable_np = r_stable.detach().cpu().numpy()
        l_output_unlabeled_np = l_output_unlabeled_soft.detach().cpu().numpy()
        r_output_unlabeled_np = r_output_unlabeled_soft.detach().cpu().numpy()

        # a1 = np.where(n1 * n2, np.where(s1 < s2, p1, p2), np.where(n1, p1, np.where(n2, p2, p1)))

        l_stabled_output_np = np.where(l_stable_bool_np * r_stable_bool_np,
                                       np.where(l_stable_np < r_stable_np, l_output_unlabeled_np, r_output_unlabeled_np),
                                       np.where(l_stable_bool_np, l_output_unlabeled_np,
                                                np.where(r_stable_bool_np, r_output_unlabeled_np, l_output_unlabeled_np)))
        r_stabled_output_np = np.where(r_stable_bool_np * l_stable_bool_np,
                                       np.where(r_stable_np < l_stable_np, r_output_unlabeled_np, l_output_unlabeled_np),
                                       np.where(r_stable_bool_np, l_output_unlabeled_np,
                                                np.where(l_stable_bool_np, l_output_unlabeled_np, r_output_unlabeled_np)))
        l_stabled_output = torch.from_numpy(l_stabled_output_np).cuda()
        r_stabled_output = torch.from_numpy(r_stabled_output_np).cuda()

        stabilization_weight = 0.1 * math.exp(-5 * math.pow((1 - self.epc / self.args.nEpochs), 2))
        l_stabilization_loss = mse_Loss(l_stabled_output, l_output_unlabeled_soft)
        r_stabilization_loss = mse_Loss(r_stabled_output, r_output_unlabeled_soft)

        l_consLoss = (lc_consLoss_seg + luc_consLoss_seg) + (lc_consLoss_sdf + luc_consLoss_sdf) + l_dual_task_consistency
        r_consLoss = (rc_consLoss_seg + ruc_consLoss_seg) + (rc_consLoss_sdf + ruc_consLoss_sdf) + r_dual_task_consistency

        l_loss = (l_dice + l_sdfLoss) + consistency_weight * l_consLoss + stabilization_weight * l_stabilization_loss
        r_loss = (r_dice + r_sdfLoss) + consistency_weight * r_consLoss + stabilization_weight * r_stabilization_loss

        return l_loss, r_loss, l_dice, r_dice

    def cal_val_loss(self):
        target_label = self.targets
        l_output_labeled_soft = F.softmax(self.l_output, dim=1)
        r_output_labeled_soft = F.softmax(self.r_output, dim=1)
        l_dice = 0.0
        r_dice = 0.0
        l_loss_dice = dice_loss(l_output_labeled_soft[:, 0, :, :, :], target_label)
        r_loss_dice = dice_loss(r_output_labeled_soft[:, 0, :, :, :], target_label)
        l_dice += l_loss_dice
        r_dice += r_loss_dice

        return l_dice, r_dice
