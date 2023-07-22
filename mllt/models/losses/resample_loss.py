import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from ..registry import LOSSES
from .cross_entropy_loss import BCELosswithDiffLogits, cross_entropy, partial_cross_entropy
import numpy as np



@LOSSES.register_module
class ResampleLoss(nn.Module):

    def __init__(self,
                 up_mult=5,dw_mult=3,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 partial=False,
                 focal=dict(
                     focal=False,
                     balance_param=2.0,
                     gamma=2,
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 coef_param=dict(
                      coef_alpha=0.5,
                      coef_beta=0.5
        
                 ),
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 freq_file='./class_freq.pkl'):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.up_mult = up_mult
        self.dw_mult = dw_mult
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = BCELosswithDiffLogits()#binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.balance_param = focal['balance_param']

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        # coef params
        self.coef_alpha = coef_param['coef_alpha']
        self.coef_beta = coef_param['coef_beta']

        self.class_freq = torch.from_numpy(np.asarray(
            mmcv.load(freq_file)['class_freq'])).float().cuda()
        self.neg_class_freq = torch.from_numpy(
            np.asarray(mmcv.load(freq_file)['neg_class_freq'])).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq[0] + self.neg_class_freq[0]
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

        # print('\033[1;35m loading from {} | {} | {} | s\033[0;0m'.format(freq_file, reweight_func, logit_reg))
        # print('\033[1;35m rebalance reweighting mapping params: {:.2f} | {:.2f} | {:.2f} \033[0;0m'.format(self.map_alpha, self.map_beta, self.map_gamma))

    def forward(self,
                norm_prop, 
                nonzero_var_tensor, 
                zero_var_tensor, 
                normalized_sigma_cj, 
                normalized_ro_cj, 
                normalized_tao_cj,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)
        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, norm_prop, nonzero_var_tensor, zero_var_tensor, normalized_sigma_cj, normalized_ro_cj, 
                normalized_tao_cj,weight)
      
        if self.focal:
            logpt = - self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(logpt)
            loss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            loss = ((1 - pt) ** self.gamma) * loss
            loss = self.balance_param * loss
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def pgd_like(self, x, y,step,sign):
        y = y.to(torch.float32)
        iters = int(torch.max(step).item()+1)
        logit=torch.zeros_like(x)
        for k in range(iters):
            grad = torch.sigmoid(x)-y
            x = x + grad*sign/x.shape[1]
            logit = logit + x*(step==k)
        return logit

    def pgd_like_diff_sign(self, x, y, step, sign):


        y = y.to(torch.float32)

        iters = int(torch.max(step).item()+1)
        logit = torch.zeros_like(x)
        for k in range(iters):
            grad = torch.sigmoid(x)-y
            x = x + grad*sign/x.shape[1]
            logit = logit + x*(step==k)
        return logit

    def lpl(self,logits, labels):

        # compute split
        quant = self.train_num*0.5
        split = torch.where(self.class_freq>quant,1,0)

        # compute head bound  
        head_dw_steps = torch.ones_like(split)*self.dw_mult

        # compute tail bound
        max_tail = torch.max(self.class_freq*(1-split))
        tail_up_steps = torch.floor(-torch.log(self.class_freq/max_tail)+0.5)*self.up_mult

        logits_head_dw = self.pgd_like(logits, labels, head_dw_steps, -1.0) - logits   # 极小化（正头部，负尾部）
        logits_tail_up = self.pgd_like(logits, labels, tail_up_steps, 1.0) - logits    # 极大化 （正尾 ，负头）

        head = torch.sum(logits_head_dw*labels*split,dim=0)/(torch.sum(labels*split,dim=0)+1e-6)
        tail = torch.sum(logits_tail_up*labels*(1-split),dim=0)/(torch.sum(labels*(1-split),dim=0)+1e-6)

        # compute perturb
        perturb = head+tail

        return perturb.detach()
    
    def lpl_imbalance(self, logits, labels, prop, nonzero_var_tensor, zero_var_tensor, normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj):
        C = labels.size(1)
  
        coef_cc =(1-(1-self.coef_alpha)*self.coef_beta)*prop+(1-self.coef_alpha)*self.coef_beta*(torch.div(nonzero_var_tensor,zero_var_tensor+torch.full((zero_var_tensor.shape[0],),0.00000001).cuda()))
        coef_cj =self.coef_alpha*normalized_tao_cj+(1-self.coef_alpha)*(self.coef_beta*normalized_sigma_cj+(1-self.coef_beta)*normalized_ro_cj)
        coef_cj = coef_cj * (torch.ones(20, 20).cuda()- torch.eye(20).cuda()) + torch.diag(coef_cc)
  
        coef =coef_cj

        quant = torch.sum(coef)/C**2
        split = torch.where(coef>quant,1,0)
        head_coef=coef*split

        tail_coef =coef*(1-split)

        head_dw_steps = torch.floor(head_coef* self.dw_mult).cuda()
        tail_up_steps = torch.floor(tail_coef* self.up_mult).cuda()

        logits_head_dw = self.pgd_like_diff_sign(logits,labels,head_dw_steps,-1.0) -logits
        logits_tail_up = self.pgd_like_diff_sign(logits,labels,tail_up_steps,1.0) -logits

        perturb = logits_head_dw+logits_tail_up

        return perturb.detach()



    def logit_reg_functions(self, labels, logits, norm_prop, nonzero_var_tensor, zero_var_tensor, normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj, weight=None):
        if not self.logit_reg:
            return logits, weight

        if 'init_bias' in self.logit_reg:

            batch_size = logits.size(0)
            num_classes = logits.size(1)
            logits = logits.view(batch_size,num_classes,1).expand(batch_size,num_classes,num_classes).clone()
            labels = labels.view(batch_size,num_classes,1).expand(batch_size,num_classes,num_classes).clone()
            logits += self.lpl_imbalance(logits, labels, norm_prop,nonzero_var_tensor, zero_var_tensor, normalized_sigma_cj, normalized_ro_cj, normalized_tao_cj)

        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            weight = weight.view(weight.size(0),weight.size(1),1).expand(weight.size(0),weight.size(1),weight.size(1))
            weight = weight / self.neg_scale * (1 - labels) + weight * labels

        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight

    def data_normal(self, data):
        d_min = torch.min(data)
        d_max = torch.max(data)
        dst = d_max-d_min
        norm_data = torch.div(data-d_min,dst)
        reverse_norm_data = torch.div(d_max-data,dst)
        return norm_data, reverse_norm_data

    def none_zero_normal(self, data):
        ones = torch.ones_like(data)
        d_min = torch.min(torch.where(data==0,ones,data))
        d_max = torch.max(data)
        dst = d_max - d_min
        norm_data =torch.div(data-d_min,dst)
        norm_data =torch.clamp(norm_data,min=0.0)
        reverse_norm_data =torch.div(d_max-data,dst)
        zero = torch.zeros_like(reverse_norm_data)
        reverse_norm_data =torch.where(data>1,zero,data)
        return norm_data, reverse_norm_data
