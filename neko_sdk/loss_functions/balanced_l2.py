import torch
def neko_masked_l2(pred,binary_gt):
    # the tanh saturates at -1 and 1, so we slightly expand it for a bit.
    ret = torch.tanh(pred) * 1.5;
    pwl = (binary_gt - ret) * (binary_gt - ret);
    pos = binary_gt > 0;
    neg = binary_gt <= 0;
    pl = torch.sum(pwl * pos) / (torch.sum(pos));
    nl = torch.sum(pwl * neg) / (torch.sum(neg));
    return pl + nl;

