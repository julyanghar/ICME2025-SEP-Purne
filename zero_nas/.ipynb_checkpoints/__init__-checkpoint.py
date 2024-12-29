from .jacov import jacov
from .grad_norm import grad_norm
from .naswot import naswot
from .synflow import synflow
from .snip import snip
from .fisher import fisher
from .ntk import nasi
from tqdm import tqdm
from torch import nn
import look2hear.datas
import yaml
from look2hear.utils import tensors_to_device

import numpy as np

ZERO_PROXY = {
    'jacov': jacov,
    'grad_norm': grad_norm,
    'naswot': naswot,
    'synflow': synflow,
    'snip': snip,
    'fisher': fisher,
    'ntk': ntk,
    'nasi': nasi
}





class ZeroNas:
    def __init__(self, dataloader,
                 indicator,
                 criterion=nn.CrossEntropyLoss(),
                 num_batch=1):
        assert isinstance(indicator, str) or isinstance(indicator, list)
        if isinstance(indicator, str):

            self.ntk = True if indicator in ['ntk', 'nasi'] else False
            self.indicator = {indicator: ZERO_PROXY[indicator]}

        elif isinstance(indicator, list):

            self.ntk = True if indicator[0] in ['ntk', 'nasi'] else False
            self.indicator = {ind: ZERO_PROXY[ind] for ind in indicator}

        self.dataloader = dataloader

        self.criterion = criterion
        self.num_batch = num_batch

    def get_score(self, model):
        model = model.cuda()
        model.train()
        scores = dict()
        for i, data in enumerate(self.dataloader):
            # cifar10, cifar100
            # images = data[0].cuda()
            # target = data[1].cuda()

            #lrs2
          
            mixtures, targets, key = tensors_to_device(data,
                                                    device=next(model.parameters()).device)
     
            # # imagenet_dali
            # images = data[0]["data"].cuda()
            # target = data[0]["label"].squeeze().long().cuda()
            
            for k, score_func in self.indicator.items():
                if k in scores.keys():
                    scores[k].append(score_func(model, mixtures, targets, self.criterion))
                else:
                    scores[k] = [score_func(model, mixtures, targets, self.criterion)]

            if i+1 == self.num_batch:
                break

        for k, v in scores.items():
            # Get the mean value
            scores[k] = sum(v)/len(v)

        return scores
