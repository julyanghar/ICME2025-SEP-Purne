import copy
import datetime
import os
from itertools import combinations

from utils.utils import set_random_seed, set_gpu, get_logger
import torch
from zero_nas import ZeroNas



import os
import torch
from look2hear import models
import look2hear.models
import look2hear.datas
import argparse
import yaml
from look2hear.utils import tensors_to_device



#### 语音分离模型部分
parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir",
                    default="Experiments/checkpoint/TDANet/conf.yml",
                    help="Full path to save best validation model")
args = parser.parse_args()
config = dict(vars(args))

# Load training config
with open(args.conf_dir, "rb") as f:
    train_conf = yaml.safe_load(f)
config["train_conf"] = train_conf

config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
    os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
)

model_path = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "best_model.pth")

# print(config.keys())

loss_func = {
        "train": getattr(look2hear.losses, config["train_conf"]["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["train_conf"]["loss"]["train"]["sdr_type"]),
            **config["train_conf"]["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["train_conf"]["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["train_conf"]["loss"]["val"]["sdr_type"]),
            **config["train_conf"]["loss"]["val"]["config"],
        ),
    }


## python Reassembly.py --pretrained --set cifar10 --num_classes 10 --batch_size 64  --arch resnet56 --gpu 0  --zero_proxy grad_norm --evaluate
## python Reassembly.py --pretrained --set cifar100 --num_classes 100 --batch_size 64  --arch resnet56 --gpu 0  --zero_proxy grad_norm --evaluate
## python Reassembly.py --pretrained --set cifar10 --num_classes 10 --batch_size 64  --arch cvgg16_bn --gpu 0  --zero_proxy grad_norm --evaluate
## python Reassembly.py --pretrained --set imagenet_dali --num_classes 1000 --batch_size 64  --arch ResNet50 --gpu 0  --zero_proxy grad_norm --evaluate



tdanet_k6_partition=[1, 1, 1, 1, 1]


tdanet_layer_name =[f'sm.unet.loc_glo_fus.{i}'for i in range(5)]




def get_layer(partition):
    max_part = max(partition)
    print("Max partition: {}".format(max_part))

    block = []
    for i in range(max_part):
        block.append([])

    for i in range(len(partition)):
        for j in range(max_part):
            if partition[i] == j+1:
                block[j].append(i)

    return block, max_part


def main():

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('experiment/tdanet/'):
        os.makedirs('experiment/tdanet/' , exist_ok=True)
    logger = get_logger('experiment/tdanet/'  + 'logger' + now + '.log')
  
    remain_layer = [4]  # 需要调整

    value_list = replace_layer_initialization(remain_layer)
    print(value_list)
    logger.info(value_list)
    best_layer=search_best(value_list)
    logger.info("Remaining layers: {}".format(best_layer))
    
    # torch.save(value_list, "save/value_{}_{}_{}_{}-{}-{}-{}-{}.pth".format(args.arch, args.set, args.zero_proxy, remain_layer[0], remain_layer[1], remain_layer[2], remain_layer[3], remain_layer[4]))


####### TDANet需要剪枝操作的对象层
def generate_layer_names(parts, blocks, sub_parts):
    layers = []
    for part in parts:
        for block in blocks:
            for sub_part in sub_parts:
                layers.append(f'{part}.{block}.{sub_part}.weight')
                layers.append(f'{part}.{block}.{sub_part}.gamma')
                layers.append(f'{part}.{block}.{sub_part}.beta')
    return layers

parts = ['sm.unet.loc_glo_fus']
blocks = range(5)  # 0, 1, 2, 3, 4
sub_parts = ['local_embedding.conv', 'local_embedding.norm', 'global_embedding.conv', 'global_embedding.norm', 'global_act.conv', 'global_act.norm']
layers_to_prune = generate_layer_names(parts, blocks, sub_parts)

def prune_layer_weights(model, param_paths):
        for name, param in model.named_parameters():
            if name in param_paths:
                # print(name,' pruned!')
                param.data.zero_()
        return model


def replace_layer_initialization(remain_layer):
    """ Run the methods on the data and then saves it to out_path. """
    
    model1 = getattr(models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(model_path,**config["train_conf"]["audionet"]["audionet_config"]).cuda()
    # model2 = getattr(models, config["train_conf"]["audionet"]["audionet_name"])(
    #     # sample_rate=config["datamodule"]["data_config"]["sample_rate"],
    #     **config["train_conf"]["audionet"]["audionet_config"],
    # ).cuda()
    model2 = getattr(models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(model_path,**config["train_conf"]["audionet"]["audionet_config"]).cuda()
    
    model1.eval()
    model2.eval()

    datamodule: object = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
    **config["train_conf"]["datamodule"]["data_config"]
)
    datamodule.setup()
    _, _ , test_set = datamodule.make_loader
    
 
    block, max_part = get_layer(tdanet_k6_partition)
    print(block)
    value_list = [[] for i in range(max_part)]
    for iii in range(max_part):
        for p in combinations(block[iii], remain_layer[iii]):
            print(p)
            prune_layer_weights(model2,layers_to_prune)
            for uuu in p:
                # print(uuu)
                for part in ['local_embedding.conv.weight','local_embedding.norm.gamma','local_embedding.norm.beta','global_act.conv.weight','global_act.norm.gamma','global_act.norm.beta']:

                    key_in_model = '{}.'.format(tdanet_layer_name[uuu])+part
                    model2.state_dict()[key_in_model].copy_(model1.state_dict()[key_in_model])
                    # model2['{}.'.format(tdanet_layer_name[uuu]+part)]
                    # prune_layer_weights(model2,['{}.'.format(tdanet_layer_name[uuu]+part)])

            # model2.load_state_dict(model2.state_dict())
            indicator = ZeroNas(dataloader=test_set, indicator='grad_norm', num_batch=5,criterion=loss_func['train'])
            value = indicator.get_score(model2)['grad_norm']
            value_list[iii].append((p, value))

    return value_list


def search_best(value_list):
    num = len(value_list)
  
    for i in range(num):
        best = 1000
        best_layer = None
        for j in value_list[i]:
            if best > j[1]:
                best = j[1]
                best_layer = j[0]

        print("Remaining layers: {}".format(best_layer))
        
    return best_layer

if __name__ == "__main__":
    main()
