import torch
import torch.nn as nn
import torch.nn.functional as F

class SinkhornDistance(nn.Module):
    def __init__(self, eps=0.1, max_iter=100, reduction='mean', cost_matrix=None, stability_factor=1e-6):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.cost_matrix = cost_matrix
        self.stability_factor = stability_factor
        
    def _cost_matrix_euclidean(self, n_class, device):
        classes = torch.arange(n_class, device=device).float()
        class_idx = classes.unsqueeze(0).expand(n_class, n_class)
        class_idx_t = class_idx.t()
        
        cost_matrix = (class_idx - class_idx_t).pow(2).float()
        
        if cost_matrix.max() > 0:
            cost_matrix = cost_matrix / cost_matrix.max()
        
        return cost_matrix
        
    def forward(self, pred, target):
        batch_size, n_class = pred.shape
        
        pred_probs = F.softmax(pred, dim=1) + self.stability_factor
        pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)
        
        target_one_hot = F.one_hot(target, n_class).float() + self.stability_factor
        target_one_hot = target_one_hot / target_one_hot.sum(dim=1, keepdim=True)
        
        if self.cost_matrix is None:
            cost_matrix = self._cost_matrix_euclidean(n_class, pred.device)
        else:
            cost_matrix = self.cost_matrix.to(pred.device)
            
        ot_dist = self._stabilized_sinkhorn(pred_probs, target_one_hot, cost_matrix)
        
        if self.reduction == 'mean':
            return ot_dist.mean()
        elif self.reduction == 'sum':
            return ot_dist.sum()
        else:  # 'none'
            return ot_dist
            
    def _stabilized_sinkhorn(self, mu, nu, cost_matrix):
        batch_size, n_class = mu.shape
        
        u = torch.zeros(batch_size, n_class, device=mu.device)
        v = torch.zeros(batch_size, n_class, device=mu.device)
        
        K = torch.exp(-cost_matrix / self.eps)
        
        K_stab = K.clone()
        K_stab[K_stab < self.stability_factor] = self.stability_factor
        log_K = torch.log(K_stab)
        
        for _ in range(self.max_iter):
            u_new = torch.log(mu + self.stability_factor) - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)
            v_new = torch.log(nu + self.stability_factor) - torch.logsumexp(log_K.t() + u_new.unsqueeze(2), dim=1)
            
            if ((u - u_new).abs().max() < 1e-4) and ((v - v_new).abs().max() < 1e-4):
                break
                
            u = u_new
            v = v_new
        
        log_P = u.unsqueeze(2) + log_K + v.unsqueeze(1)
        P = torch.exp(log_P)
        
        P = P + self.stability_factor
        P = P / P.sum(dim=2, keepdim=True) * mu.unsqueeze(2)
        P = P / P.sum(dim=1, keepdim=True) * nu.unsqueeze(1)
        
        cost = torch.sum(P * cost_matrix.unsqueeze(0), dim=(1, 2))
        
        if torch.isnan(cost).any():
            print("WARNING: NaN in transport cost, returning alternate cost")
            preds_indices = torch.argmax(mu, dim=1)
            target_indices = torch.argmax(nu, dim=1)
            alt_cost = cost_matrix[preds_indices, target_indices]
            return alt_cost
            
        return cost

class OTCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.3, eps=0.1, max_iter=100, reduction='mean', cost_matrix=None):
        super(OTCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.ot_loss = SinkhornDistance(eps, max_iter, reduction, cost_matrix)
        
    def forward(self, pred, target):
        ce_loss_val = self.ce_loss(pred, target)
        
        if self.alpha == 0:
            return ce_loss_val
            
        ot_loss_val = self.ot_loss(pred, target)
        
        if torch.isnan(ot_loss_val):
            print("WARNING: NaN detected in OT loss, falling back to CE loss only")
            return ce_loss_val
        
        combined_loss = (1 - self.alpha) * ce_loss_val + self.alpha * ot_loss_val
        
        if torch.isnan(combined_loss):
            print("WARNING: NaN in combined loss, falling back to CE loss")
            return ce_loss_val
            
        return combined_loss

def create_robust_ot_loss(mode='combined', alpha=0.3, eps=0.1, max_iter=100, 
                          similarity_matrix=None, n_classes=None):
    if similarity_matrix is None and n_classes is not None:
        similarity_matrix = torch.zeros(n_classes, n_classes)
        for i in range(n_classes):
            for j in range(n_classes):
                similarity_matrix[i, j] = 1.0 / (1.0 + abs(i - j))
    
    cost_matrix = None
    if similarity_matrix is not None:
        cost_matrix = 1.0 - similarity_matrix
    
    if mode == 'combined':
        return OTCrossEntropyLoss(alpha=alpha, eps=eps, max_iter=max_iter, 
                                        cost_matrix=cost_matrix)
    elif mode == 'pure':
        return SinkhornDistance(eps=eps, max_iter=max_iter, 
                                      cost_matrix=cost_matrix)
    else:
        return OTCrossEntropyLoss(alpha=alpha, eps=eps, max_iter=max_iter, 
                                        cost_matrix=cost_matrix)