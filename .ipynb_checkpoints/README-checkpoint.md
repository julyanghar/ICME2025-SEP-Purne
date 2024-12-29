
# Dataset preprocess
```bash
python DataPreProcess/process_lrs2.py --in_dir=xxxx --out_dir=DataPreProcess/LRS2_bak
```

# Dataset
### 训练：./Look2hear/DataPreProcess/LRS2_bak/tr
### 验证：./Look2hear/DataPreProcess/LRS2_bak/cv
### 测试：./Look2hear/DataPreProcess/LRS2_bak/tt


# Checkpoints 
Experiments/checkpoint


# 代码运行

1. Calculate the CKA matrix and save it to ./save
```bash
python similarity.py 
```
2. Use Fisher algorithm for segmentation
```bash
python Network_Partition.py
```

3. Reload the original model weights in replace_layer_initialization, and finally run the evaluation code to get the layers that need to be retained.
Finally, we get Remaining layers: [0,1,2,3]
```bash
python Reassembly.py   
```

4. Prune layer 4 of TDANet in audio_test.py and test the inference result.
```bash
python audio_test.py  --conf_dir Experiments/checkpoint/TDANet/conf.yml
```


