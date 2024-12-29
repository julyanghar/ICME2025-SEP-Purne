'''
Author: Kai Li
Date: 2020-08-09 17:32:53
LastEditTime: 2021-05-21 22:54:23
'''
import torch
import random
import torch.nn as nn
import math
import torch.nn.functional as F
from .base_model import BaseModel
# import torch.nn.utils.prune as prune
from torchsummary import summary

class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    '''
    This class defines the convolution layer with normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    '''
    This class defines a normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: number of output channels
        '''
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class Blocks(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,
                                    stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,
                                           stride=1, groups=in_channels, d=1))
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=5,
                                               stride=2,
                                               groups=in_channels, d=1))
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j-i == 1:
                    fuse_layer.append(None)
                elif i-j == 1:
                    fuse_layer.append(DilatedConvNorm(in_channels, in_channels,
                                                      kSize=5,
                                                      stride=2,
                                                      groups=in_channels, d=1))
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth-1:
                self.concat_layer.append(ConvNormAct(
                    in_channels*2, in_channels, 1, 1))
            else:
                self.concat_layer.append(ConvNormAct(
                    in_channels*3, in_channels, 1, 1))

        self.last_layer = nn.Sequential(
            ConvNormAct(in_channels*upsampling_depth, in_channels, 1, 1)
        )
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = output[i].shape[-1]
            y = torch.cat((self.fuse_layers[i][0](output[i-1]) if i-1 >= 0 else torch.Tensor().to(output1.device),
                           output[i],
                           F.interpolate(output[i+1], size=wav_length, mode='nearest') if i+1 < self.depth else torch.Tensor().to(output1.device)), dim=1)
            x_fuse.append(self.concat_layer[i](y))

        wav_length = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(
                x_fuse[i], size=wav_length, mode='nearest')

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual
        #return expanded
    def apply_pruning(self, pruning_method, amount=0.5):
        """
        Apply pruning recursively to all layers within Blocks.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                prune.l1_unstructured(module, name="weight", amount=amount)
                print(f"Pruned {name} with {amount} sparsity")

    def remove_blocks(self, block_indices_to_remove=None, random_fraction=0.5):
        """
        Remove entire blocks from `spp_dw` and adjust other dependent layers.
        :param block_indices_to_remove: List of indices specifying which blocks to remove.
        :param random_fraction: Fraction of blocks to remove randomly if indices not provided.
        """
        if block_indices_to_remove is None:
            num_blocks = len(self.spp_dw)
            num_to_remove = max(1, int(num_blocks * random_fraction))
            block_indices_to_remove = sorted(random.sample(range(num_blocks), num_to_remove), reverse=True)

        # Remove specified blocks
        for block_idx in block_indices_to_remove:
            del self.spp_dw[block_idx]
            del self.fuse_layers[block_idx]
            del self.concat_layer[block_idx]

        # Update depth
        self.depth = len(self.spp_dw)
        print(f"Removed {len(block_indices_to_remove)} blocks: {block_indices_to_remove}")

class Recurrent(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4,
                 _iter=4):
        super().__init__()
        self.blocks = Blocks(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        #self.attention = Attention_block(out_channels)
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.blocks(x)
            else:
                #m = self.attention(mixture, x)
                x = self.blocks(self.concat_block(mixture+x))
        return x


class AFRCNN(BaseModel):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2,
                 sample_rate=16000):
        super(AFRCNN, self).__init__(sample_rate=sample_rate)

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
            self.enc_kernel_size // 2,
            2 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(in_channels=1, out_channels=enc_num_basis,
                                 kernel_size=enc_kernel_size,
                                 stride=enc_kernel_size // 2,
                                 padding=enc_kernel_size // 2,
                                 bias=False)
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=enc_num_basis,
            out_channels=out_channels,
            kernel_size=1)

        # Separation module
        self.sm = Recurrent(out_channels, in_channels, upsampling_depth, num_blocks)

        mask_conv = nn.Conv1d(out_channels, num_sources * enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=1, bias=False)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()
    # Forward pass

    def forward(self, input_wav):
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0).unsqueeze(1)
        if input_wav.ndim == 2:
            input_wav = input_wav.unsqueeze(1)
        # Front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = self.remove_trailing_zeros(
            estimated_waveforms, input_wav)
        if was_one_d:
            return estimated_waveforms.squeeze(0)
        return estimated_waveforms

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32).to(x.device)
            padded_x[..., :x.shape[-1]] = x
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args

    def apply_pruning(self, pruning_method, amount=0.5):
        """
        Apply pruning to specific layers in the network.
        :param pruning_method: Pruning method (e.g., L1Unstructured, RandomUnstructured).
        :param amount: Fraction of weights to prune (0 to 1).
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                prune.l1_unstructured(module, name="weight", amount=amount)
                print(f"Pruned {name} with {amount} sparsity")

class WrappedGPT:
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]  # 输出通道数
        self.columns = layer.weight.shape[1]  # 输入通道数

        self.scaler_row = torch.zeros((self.columns), device=self.dev)  # 初始化统计量
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        # 确保 inp 为 [batch_size, channels, features]
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(1)  # 添加通道维度 [batch_size, 1, features]
        elif len(inp.shape) != 3:
            raise ValueError(f"Expected input of shape [batch_size, channels, features], got {inp.shape}")

        batch_size, channels, features = inp.shape
        if channels != self.columns:
            raise ValueError(f"Input channels ({channels}) do not match expected channels ({self.columns})")

        # 更新 scaler_row (范数平方的累积平均)
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)  # 更新历史平均
        self.nsamples += batch_size

        # 按通道计算范数平方，并对特征维度和批量维度求平均
        norm_per_channel = torch.norm(inp, p=2, dim=(0, 2))  # [channels]
        self.scaler_row += (norm_per_channel.to(self.dev) ** 2) / self.nsamples



def prune_wanda_afrcnn(model, dataloader, device, sparsity_ratio=0.5, prune_n=0, prune_m=0):
    """
    WANDA剪枝方法适配AFRCNN模型，仅针对卷积层进行剪枝
    :param model: AFRCNN模型实例
    :param dataloader: 数据加载器，用于输入数据采样
    :param device: 使用的设备 (CPU/GPU)
    :param sparsity_ratio: 剪枝的稀疏率
    :param prune_n: n:m结构化剪枝中的n值
    :param prune_m: n:m结构化剪枝中的m值
    """
    model.eval()  # 切换为评估模式
    # 仅提取卷积层
    layers = [module for module in model.modules() if isinstance(module, nn.Conv1d)]
    wrapped_layers = {id(layer): WrappedGPT(layer) for layer in layers}

    print("收集输入数据...")
    with torch.no_grad():
        for batch in dataloader:
            # 检查 batch 类型
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)  # 假设数据为 (input, label) 格式
            elif isinstance(batch, torch.Tensor):
                inputs = batch.to(device)
            else:
                raise ValueError("Unsupported batch type. Expected Tensor, list, or tuple.")

            # 确保输入形状为 [batch_size, channels=1, sequence_length]
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(1)  # 添加通道维度 [batch_size, 1, sequence_length]

            # 遍历每一层并捕获输入输出
            for layer in layers:
                print(f"Processing Conv1d layer {id(layer)} with input shape {inputs.shape}")
                outputs = layer(inputs)  # 前向传播，确保输入输出的通道数正确
                wrapped_layers[id(layer)].add_batch(inputs, outputs)
                inputs = outputs  # 更新输入为下一层的输出

    print("开始剪枝...")
    for layer in layers:
        layer_id = id(layer)
        wrapped = wrapped_layers[layer_id]
        print(f"剪枝卷积层 {layer_id} 的权重")
        weight = layer.weight.data
        W_metric = torch.abs(weight) * torch.sqrt(wrapped.scaler_row.reshape((1, -1)))
        W_mask = torch.zeros_like(W_metric, dtype=torch.bool)  # 初始化为全False

        if prune_n > 0 and prune_m > 0:
            # n:m结构化剪枝
            for i in range(W_metric.shape[1]):
                if i % prune_m == 0:
                    tmp = W_metric[:, i:i+prune_m].float()
                    W_mask[:, i:i+prune_m] = (tmp < torch.topk(tmp, prune_m-prune_n, dim=1, largest=False)[0])
        else:
            # 非结构化剪枝
            num_prune = int(W_metric.numel() * sparsity_ratio)
            threshold = torch.topk(W_metric.view(-1), num_prune, largest=False)[0][-1]
            W_mask = W_metric < threshold

        # 应用掩码
        weight[W_mask] = 0

    print("剪枝完成！")


def prune_trained_model(model, amount=0.5, pruning_method='l1'):
    """
    对已训练好的模型进行剪枝。
    :param model: 训练好的模型
    :param amount: 剪枝比例
    :param pruning_method: 剪枝方法，支持 L1 或随机剪枝
    """
    for name, module in model.named_modules():
    # for name in model:
    #     module = model[name]
        if isinstance(module, torch.nn.Conv1d):
            if pruning_method == 'l1':
                prune.l1_unstructured(module, name="weight", amount=amount)
            elif pruning_method == 'random':
                prune.random_unstructured(module, name="weight", amount=amount)
            else:
                raise ValueError("Unsupported pruning method")
            print(f"Pruned {amount * 100}% weights in layer {name}")
    return model

# 微调模型（示例）
# 假设有 dataloader 定义训练数据
def fine_tune_model(model, dataloader, optimizer, criterion, epochs=2):
    """
    对剪枝后的模型进行微调。
    """
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

# 定义计算稀疏度函数
def calculate_sparsity(model):
    """
    计算模型的稀疏度。
    :param model: 模型
    :return: 稀疏率
    """
    total_params = 0
    zero_params = 0
    # for name in model:
    #     param = model[name]
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    sparsity = zero_params / total_params
    print(f"Model sparsity: {sparsity * 100:.2f}%")
    return sparsity

# 定义推理时间测量函数
def measure_inference_time(model, input_size=(1, 1, 16000), device='cuda'):
    """
    测量模型的单次推理时间。
    :param model: 模型
    :param input_size: 输入数据的形状
    :param device: 运行设备
    :return: 推理时间
    """
    model.eval()
    input_data = torch.randn(input_size).to(device)
    model = model.to(device)

    # GPU 时间测量工具
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        # 热身运行
        for _ in range(10):
            _ = model(input_data)
        
        # 测量推理时间
        starter.record()
        _ = model(input_data)
        ender.record()
        torch.cuda.synchronize()
        inference_time = starter.elapsed_time(ender)  # 毫秒
    print(f"Inference time: {inference_time:.2f} ms")
    return inference_time

# 定义精度评估函数
def evaluate_accuracy(model, dataloader, criterion, device='cuda'):
    """
    在测试集上评估模型的精度。
    :param model: 模型
    :param dataloader: 测试数据加载器
    :param criterion: 损失函数
    :param device: 运行设备
    :return: 平均损失
    """
    model.eval()
    model = model.to(device)
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    average_loss = total_loss / total_samples
    print(f"Test loss: {average_loss:.4f}")
    return average_loss

# 定义剪枝函数
def prune(W, X, s):
    """
    自定义剪枝方法，计算剪枝指标并移除部分权重。
    :param W: 权重矩阵 (C_out, C_in)
    :param X: 输入矩阵 (N * L, C_in)
    :param s: 剪枝比例 (0 < s < 1)
    :return: 剪枝后的权重矩阵
    """
    # 计算剪枝指标
    metric = W.abs() * X.norm(p=2, dim=0)
    # 根据指标排序
    _, sorted_idx = torch.sort(metric, dim=1)
    # 确定要剪枝的索引
    pruned_idx = sorted_idx[:, :int(W.size(1) * s)]
    # 将对应位置的权重设置为 0
    W.scatter_(dim=1, index=pruned_idx, src=torch.zeros_like(pruned_idx, dtype=W.dtype, device=W.device))
    return W

# 定义在模型中应用剪枝的函数
def apply_custom_pruning(model, input_data, s=0.5):
    """
    对模型中所有Conv1d层应用自定义剪枝方法。
    :param model: 目标模型
    :param input_data: 示例输入数据 (用于计算X)
    :param s: 剪枝比例 (0 < s < 1)
    """
    model.eval()  # 设置为评估模式
    hooks = []

    def forward_hook(module, input, output):
        """
        钩子函数，用于获取中间层的输入。
        """
        if isinstance(module, nn.Conv1d):
            with torch.no_grad():
                # 获取权重和输入
                W = module.weight
                X = input[0]
                # 应用剪枝
                W_pruned = prune(W, X, s)
                module.weight.copy_(W_pruned)

    # 注册前向钩子
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(forward_hook))

    # 运行一次前向传播，触发钩子
    with torch.no_grad():
        _ = model(input_data)

    # 移除钩子
    for hook in hooks:
        hook.remove()



# 测试
if __name__ == '__main__':
    device = torch.device('cuda')
    model = AFRCNN(out_channels=512,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=41,
                 enc_num_basis=512,
                 num_sources=2,
                 sample_rate=16000)
    
    conf = torch.load(
        '/home/zhaoyu/projects/code/Look2hear/Experiments/checkpoint/AFRCNN_2/epoch=208.ckpt', map_location=device
    )  # Attempt to find the model and instantiate it.
    state_dict = conf["state_dict"]
    model.load_state_dict({k.replace('audio_model.', ''): v for k, v in state_dict.items()})
    # model.load_state_dict(state_dict)
    # model = model.to(device)
    # model = torch.load(device)

    # summary(model, input_size=(1, 16000)) # 注意densnet使用summary函数无法打印输出模型结构；应该是库版本问题
    # 模拟输入数据
    index = 0
    
    datamodule = LRS2DataModule(train_dir= 'DataPreProcess/LRS2_bak/tr',
    valid_dir= 'DataPreProcess/LRS2_bak/cv',
    test_dir= 'DataPreProcess/LRS2_bak/tt',
    n_src= 2,
    sample_rate= 16000,
    segment= 2.0,
    normalize_audio= False,
    batch_size= 8,
    num_workers= 8,
    pin_memory= True,
    persistent_workers= False)
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader
    # # input_data = torch.randn(128, 1, 16000)  # 模拟16kHz音频
    # # dataloader = get_dataloader(input_data)

    # prune_wanda_afrcnn(model, test_loader, device, sparsity_ratio=0.5)
    # 定义剪枝函数

    
    # 剪枝前性能评估
    print("Before pruning:")
    calculate_sparsity(model)
    measure_inference_time(model)
    # evaluate_accuracy(model, test_dataloader, criterion)

    
    # 对模型进行剪枝
    # print("Applying pruning to the trained model...")
    # pruned_model = prune_trained_model(model, amount=0.9, pruning_method='l1')

    # 应用自定义剪枝方法
    print("Applying custom pruning...")
    # 创建示例输入数据
    input_data = torch.randn((512, 1, 16000), device=device)
    apply_custom_pruning(model, input_data, s=0.5)

    # 评估剪枝后模型的性能
    print("After pruning:")
    calculate_sparsity(model)
    measure_inference_time(model, input_size=(1, 1, 16000), device=device)

    # 剪枝后性能评估
    print("After pruning:")
    # model.load_state_dict({k.replace('audio_model.', ''): v for k, v in pruned_model.items()})
    calculate_sparsity(model)
    measure_inference_time(model)
    # evaluate_accuracy(model, test_dataloader, criterion)

    # 定义训练超参数（仅供参考）
    # optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-4)
    # criterion = torch.nn.MSELoss()

    # 微调模型（此处假设 dataloader 已定义）
    # fine_tune_model(pruned_model, dataloader, optimizer, criterion, epochs=2)

    # 保存剪枝后的模型
    torch.save(model, "/home/zhaoyu/projects/code/Look2hear/Experiments/checkpoint/AFRCNN_2/pruned_afrcnn_model.pth")
    print("Pruned model saved as pruned_afrcnn_model.pth")