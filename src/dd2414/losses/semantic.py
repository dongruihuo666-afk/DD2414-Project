import torch
import torch.nn.functional as F
def semantic_loss(prediction, target, confidence):
    prediction = F.normalize(prediction.float(), dim=1)
    target = F.normalize(target.float(), dim=1)
    cosine = (prediction * target).sum(dim=1)
    weights = confidence.float()
    weights = torch.where(weights > 0.05, weights, torch.zeros_like(weights))
    weights = weights / weights.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    loss = ((1.0 - cosine) * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, cosine
