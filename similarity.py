import datetime
import os
import torch
import tqdm
# from args import args
from cka import linear_CKA
from utils.get_hook import get_inner_feature_for_tdanet, get_inner_feature_for_vgg, get_inner_feature_for_smallresnet
import os
import torch
from look2hear import models
import look2hear.models
import look2hear.datas
import argparse
import yaml
from look2hear.utils import tensors_to_device
import numpy as np

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
# import pdb; pdb.set_trace()
# conf["train_conf"]["masknet"].update({"n_src": 2})




def main():
 
    model =  getattr(models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(
    model_path,
    # sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
    **config["train_conf"]["audionet"]["audionet_config"],
).cuda()

    # print(list(model.named_modules())[0])

    model_device = next(model.parameters()).device
    model.eval()
    
    batch_count = 0

    datamodule: object = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
        **config["train_conf"]["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _ , test_set = datamodule.make_loader
 

    inter_feature = []
    CKA_matrix_list = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    print(type(test_set))
    with torch.no_grad():
        for i, data in tqdm.tqdm(
                enumerate(test_set), ascii=True, total=len(test_set)
        ):
            batch_count += 1

            mix, sources, key = tensors_to_device(data,device=model_device)

            # print(mix[None].shape)
            handle_list = get_inner_feature_for_tdanet(model, hook, 'tdanet')
                
          
            output = model(mix[None].squeeze())

            inter_feature=[inter_feature[i] for i in [0,16,32,48,64]]
            
            for m in range(len(inter_feature)):
                print('-'*50)
                print(m)
            
                if len(inter_feature[m].shape) != 2:
                    inter_feature[m] = inter_feature[m].reshape(7, -1)

        
            CKA_matrix_for_visualization = CKA_heatmap(inter_feature)
            print(CKA_matrix_for_visualization)
            CKA_matrix_list.append(CKA_matrix_for_visualization)

            inter_feature = []
            for i in range(len(handle_list)):
                handle_list[i].remove()

            if batch_count == 100:
                break


    torch.save(CKA_matrix_list, 'save/CKA_matrix_for_visualization_tdanet.pth')


def CKA_heatmap(inter_feature):
    layer_num = len(inter_feature)
    CKA_matrix = torch.zeros((layer_num, layer_num))
    for ll in range(layer_num):
        for jj in range(layer_num):
            if ll < jj:
                CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = linear_CKA(inter_feature[ll], inter_feature[jj])
                # CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = unbias_CKA(inter_feature[ll], inter_feature[jj])
    # print(CKA_matrix)
    
    CKA_matrix_for_visualization = CKA_matrix + torch.eye(layer_num)
    return CKA_matrix_for_visualization


if __name__ == "__main__":
    main()