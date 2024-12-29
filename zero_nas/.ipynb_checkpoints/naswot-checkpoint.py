import torch
import numpy as np
from torch import nn


def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return ld


def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    # return jacob, target.detach(), y.detach()
    return jacob, y.detach()


def naswot(model, x, target, criterion):
    batch_size = x.shape[0]
    model.K = np.zeros((batch_size, batch_size))

    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
        except Exception as err:
            print('---- error on model : ')
            print(model)
            raise err


    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    for name, module in model.named_modules():
        # if 'ReLU' in str(type(module)):
        if isinstance(module, nn.ReLU):
            # hooks[name] = module.register_forward_hook(counting_hook)
            module.visited_backwards = True
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)

    jacobs, y = get_batch_jacobian(model, x)
    print(model.K)

    score = logdet(model.K)
    return float(score) / batch_size