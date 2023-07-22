import torch
import torch.nn as nn
import numpy as np

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda() #前mini-batch的每类样本数
        self.Prop = torch.zeros(class_num).cuda()
        self.Cov_pos = torch.zeros(class_num).cuda()
        self.Cov_neg = torch.zeros(class_num).cuda()
        self.Sigma_cj = torch.zeros(class_num, class_num).cuda()
        self.Ro_cj = torch.zeros(class_num, class_num).cuda()
        self.Tao_cj = torch.zeros(class_num, class_num).cuda()


    def update_CV(self, features, labels, logits): #features 128,640; label:128 


        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = labels



        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)#对应位置的feature

        Amount_CxA = NxCxA_onehot.sum(0) #每一类的数量
        Amount_CxA[Amount_CxA == 0] = 1 #20*256


        pr_C = onehot.sum(0) / N


        nonzero_var_list=[]
        zero_var_list=[]
        sigma_cj=[]
        ro_cj=[]
        tao_cj=[]
        for c in range(labels.size(1)):
            mask_c = labels[:,c]==1
            sigma_j=[]
            ro_j=[]
            tao_j=[]
            for j in range(labels.size(1)):
                mask_j = labels[:,j]==1
                result = logits[mask_c & mask_j, :]
                if result.numel() == 0:
                    sigma_j.append(torch.tensor(0))
                else:
                    sigma_j.append(torch.var(result))
                ro_j.append(result.size(0)/(onehot.sum(0)[j]+torch.tensor(0.0000001)))
                tao_j.append((torch.tensor(N)-onehot.sum(0)[c])/(onehot.sum(0)[j]-torch.tensor(result.size(0))+torch.tensor(0.00000001)))

            sigma_cj.append(sigma_j)
            sigma_cj_ = torch.tensor(np.array(sigma_cj).tolist())
            max_val_sigma = torch.max(sigma_cj_)
            min_val_sigma = torch.min(sigma_cj_)
            normalized_sigma_cj = [(t-min_val_sigma)/(max_val_sigma - min_val_sigma+0.0000001) for t in sigma_cj_]
            ro_cj.append(ro_j)
            ro_cj_ = torch.tensor(np.array(ro_cj).tolist())
            max_val_ro = torch.max(ro_cj_)
            min_val_ro = torch.min(ro_cj_)
            normalized_ro_cj = [(t-min_val_ro)/(max_val_ro - min_val_ro+0.0000001) for t in ro_cj_]
            tao_cj.append(tao_j)
            tao_cj_ = torch.tensor(np.array(tao_cj).tolist())
            max_val_tao = torch.max(tao_cj_)
            min_val_tao = torch.min(tao_cj_)
            normalized_tao_cj = [(t-min_val_tao)/(max_val_tao - min_val_tao+0.0000001) for t in tao_cj_]

            nonzero_indices = torch.nonzero(labels[:, c])#与c共现
            not_labels = torch.logical_not(labels)
            zero_indices = torch.nonzero(not_labels[:, c])

            if nonzero_indices.shape[0]==0:
                nonzero_var = torch.tensor(0).cuda()
            else:
                nonzero_var = torch.var(logits[nonzero_indices.squeeze()])
            nonzero_var_list.append(nonzero_var)
            zero_var = torch.var(logits[zero_indices.squeeze()])
            zero_var_list.append(zero_var)
            
        nonzero_var_tensor = torch.stack(nonzero_var_list,dim=0)
        zero_var_tensor = torch.stack(zero_var_list,dim=0)

        ave_CxA = features_by_sort.sum(0) / Amount_CxA #20,256 #平均特征

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A)) #10，640，640

        sum_weight_PR = onehot.sum(0).view(C)

        sum_weight_PR_neg = N - sum_weight_PR 

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A) #mini-batch中每类样本数量

        sum_weight_CJ = torch.zeros(onehot.shape[1],onehot.shape[1])
        for i in range(20):
            for j in range(i, 20):
                if i == j:
                    sum_weight_CJ[i, j] = (onehot[:, i] == 1).sum()
                else:
                    sum_weight_CJ[i, j] = (onehot[:, i] == 1).logical_and(onehot[:, j] == 1).sum()
                    sum_weight_CJ[j, i] = sum_weight_CJ[i, j]
        sum_weight_CJ = sum_weight_CJ.cuda()

        weight_PR = sum_weight_PR.div(
            sum_weight_PR + self.Amount.view(C)
        )

        weight_PR[weight_PR != weight_PR] = 0

        weight_PR_neg = sum_weight_PR_neg.div(
            sum_weight_PR_neg + self.Amount.view(C)
        )

        weight_PR_neg[weight_PR_neg != weight_PR_neg] = 0

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)#m/n+m  
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        weight_CJ = sum_weight_CJ.div(
            sum_weight_CJ + self.Amount.view(C)
        )
        weight_CJ[weight_CJ != weight_CJ] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )
        
        self.Prop = (self.Prop.mul(1 - weight_PR)+ pr_C.mul(weight_PR)).detach()
        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach() #10,640,640

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach() #10,640

        self.Cov_pos = (self.Cov_pos.mul(1 - weight_PR)+ nonzero_var_tensor.mul(weight_PR)).detach()
        self.Cov_neg = (self.Cov_neg.mul(1 - weight_PR_neg)+ zero_var_tensor.mul(weight_PR_neg)).detach()

      
        self.Amount += onehot.sum(0)
        normalized_sigma_cj = torch.stack(normalized_sigma_cj,dim=1).cuda()
        normalized_ro_cj = torch.stack(normalized_ro_cj,dim=1).cuda()
        normalized_tao_cj = torch.stack(normalized_tao_cj,dim=1).cuda()

        self.Sigma_cj = (self.Sigma_cj.mul(1 - weight_CJ)+ normalized_sigma_cj.mul(weight_CJ)).detach()
        self.Ro_cj = (self.Ro_cj.mul(1 - weight_CJ)+ normalized_ro_cj.mul(weight_CJ)).detach()
        self.Tao_cj = (self.Tao_cj.mul(1 - weight_CJ)+ normalized_tao_cj.mul(weight_CJ)).detach()
        


        return  self.Prop, self.Cov_pos, self.Cov_neg,self.Sigma_cj, self.Ro_cj, self.Tao_cj


