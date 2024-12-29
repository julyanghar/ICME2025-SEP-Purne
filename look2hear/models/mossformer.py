# Copyright (c) Alibaba, Inc. and its affiliates.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseModel

from look2hear.datas.lrs2datamodule import LRS2DataModule
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
from look2hear.utils.parser_utils import (
    prepare_parser_from_dict,
    parse_args_as_dict,
)
import copy
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.audio.separation.mossformer_block import (
    MossFormerModule, ScaledSinuEmbedding)
from modelscope.models.audio.separation.mossformer_conv_module import (
    CumulativeLayerNorm, GlobalLayerNorm)
from modelscope.models.base import Tensor
# from .base_model import BaseModel
EPS = 1e-8


class MossFormer(BaseModel):
    """Library to support MossFormer speech separation.

        Args:
            model_dir (str): the model path.
    """

    def __init__(
            self,
            kernel_size,
            stride,
            bias,
            out_channels,
            in_channels,
            num_blocks,
            d_model,
            attn_dropout,
            group_size,
            query_key_dim,
            expansion_factor,
            causal,
            norm,
            num_spks
        ):
        super().__init__(sample_rate=16000)
        self.encoder = Encoder(
            kernel_size=kernel_size,
            out_channels=out_channels)
        self.decoder = Decoder(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias)
        self.mask_net = MossFormerMaskNet(
            in_channels,
            out_channels,
            MossFormerM(num_blocks, d_model,
                        attn_dropout, group_size,
                        query_key_dim, expansion_factor,
                        causal),
            norm=norm,
            num_spks=num_spks)
        self.num_spks = num_spks

    def forward(self, inputs: Tensor):
        # input shape: (B, T)
        was_one_d = False
        if inputs.ndim == 1:
            was_one_d = True
            inputs = inputs.unsqueeze(0)
        if inputs.ndim == 2:
            inputs = inputs
        if inputs.ndim == 3:
            inputs = inputs.squeeze(1)
        # Separation
        mix_w = self.encoder(inputs)
        # import pdb; pdb.set_trace()
        est_mask = self.mask_net(mix_w)
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask
        # Decoding
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )
        # T changed after conv1d in encoder, fix it here
        t_origin = inputs.size(-1)
        t_est = est_source.size(1)
        if t_origin > t_est:
            est_source = F.pad(est_source, (0, 0, 0, t_origin - t_est))
        else:
            est_source = est_source[:, :t_origin, :]
        return est_source.permute(0, 2, 1).contiguous()
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args

def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Args:
        kernel_size: Length of filters.
        in_channels: Number of  input channels.
        out_channels: Number of output channels.

    Examples:

    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape # torch.Size([2, 64, 499])
    """

    def __init__(self,
                 kernel_size: int = 2,
                 out_channels: int = 64,
                 in_channels: int = 1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor):
        """Return the encoded output.

        Args:
            x: Input tensor with dimensionality [B, L].

        Returns:
            Encoded tensor with dimensionality [B, N, T_out].
            where B = Batchsize
                  L = Number of timepoints
                  N = Number of filters
                  T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Args:
        kernel_size: Length of filters.
        in_channels: Number of  input channels.
        out_channels: Number of output channels.

    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Args:
            x: Input tensor with dimensionality [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError('{} accept 3/4D tensor as input'.format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class IdentityBlock:
    """This block is used when we want to have identity transformation within the Dual_path block.

    Example
    -------
    >>> x = torch.randn(10, 100)
    >>> IB = IdentityBlock()
    >>> xhat = IB(x)
    """

    def _init__(self, **kwargs):
        pass

    def __call__(self, x):
        return x


class MossFormerM(nn.Module):
    """This class implements the transformer encoder.

    Args:
    num_blocks : int
        Number of mossformer blocks to include.
    d_model : int
        The dimension of the input embedding.
    attn_dropout : float
        Dropout for the self-attention (Optional).
    group_size: int
        the chunk size
    query_key_dim: int
        the attention vector dimension
    expansion_factor: int
        the expansion factor for the linear projection in conv module
    causal: bool
        true for causal / false for non causal

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512)) #B, S, N
    >>> net = MossFormerM(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(self,
                 num_blocks,
                 d_model=None,
                 attn_dropout=0.1,
                 group_size=256,
                 query_key_dim=128,
                 expansion_factor=4.,
                 causal=False):
        super().__init__()

        self.mossformerM = MossFormerModule(
            dim=d_model,
            depth=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
            causal=causal,
            attn_dropout=attn_dropout)
        import speechbrain as sb
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(self, src: torch.Tensor):
        """
        Args:
            src: Tensor shape [B, S, N],
            where, B = Batchsize,
                   S = time points
                   N = number of filters
            The sequence to the encoder layer (required).
        """
        output = self.mossformerM(src)
        output = self.norm(output)

        return output


class ComputeAttention(nn.Module):
    """Computation block for dual-path processing.

    Args:
    att_mdl : torch.nn.module
        Model to process within the chunks.
     out_channels : int
        Dimensionality of attention model.
     norm : str
        Normalization type.
     skip_connection : bool
        Skip connection around the attention module.

    Example
    ---------
        >>> att_block = MossFormerM(num_blocks=8, d_model=512)
        >>> comp_att = ComputeAttention(att_block, 512)
        >>> x = torch.randn(10, 64, 512)
        >>> x = comp_att(x)
        >>> x.shape
        torch.Size([10, 64, 512])
    """

    def __init__(
        self,
        att_mdl,
        out_channels,
        norm='ln',
        skip_connection=True,
    ):
        super(ComputeAttention, self).__init__()

        self.att_mdl = att_mdl
        self.skip_connection = skip_connection

        # Norm
        self.norm = norm
        if norm is not None:
            self.att_norm = select_norm(norm, out_channels, 3)

    def forward(self, x: torch.Tensor):
        """Returns the output tensor.

        Args:
            x: Input tensor of dimension [B, S, N].

        Returns:
            out: Output tensor of dimension [B, S, N].
            where, B = Batchsize,
               N = number of filters
               S = time points
        """
        # [B, S, N]
        att_out = x.permute(0, 2, 1).contiguous()

        att_out = self.att_mdl(att_out)

        # [B, N, S]
        att_out = att_out.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            att_out = self.att_norm(att_out)

        # [B, N, S]
        if self.skip_connection:
            att_out = att_out + x

        out = att_out
        return out


class MossFormerMaskNet(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Args:
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    att_model : torch.nn.module
        Attention model to process the input sequence.
    norm : str
        Normalization type.
    num_spks : int
        Number of sources (speakers).
    skip_connection : bool
        Skip connection around attention module.
    use_global_pos_enc : bool
        Global positional encodings.

    Example
    ---------
    >>> mossformer_block = MossFormerM(num_blocks=8, d_model=512)
    >>> mossformer_masknet = MossFormerMaskNet(64, 64, att_model, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = mossformer_masknet(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        att_model,
        norm='ln',
        num_spks=2,
        skip_connection=True,
        use_global_pos_enc=True,
    ):
        super(MossFormerMaskNet, self).__init__()
        self.num_spks = num_spks
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(
            in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.mdl = copy.deepcopy(
            ComputeAttention(
                att_model,
                out_channels,
                norm,
                skip_connection=skip_connection,
            ))

        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1)
        self.conv1_decoder = nn.Conv1d(
            out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        """Returns the output tensor.

        Args:
            x: Input tensor of dimension [B, N, S].

        Returns:
            out: Output tensor of dimension [spks, B, N, S]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               S = the number of time frames
        """

        # before each line we indicate the shape after executing the line
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d_encoder(x)
        if self.use_global_pos_enc:
            base = x
            x = x.transpose(1, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(0, -1)
            x = base + emb
        # [B, N, S]
        x = self.mdl(x)
        x = self.prelu(x)
        # [B, N*spks, S]
        x = self.conv1d_out(x)
        b, _, s = x.shape
        # [B*spks, N, S]
        x = x.view(b * self.num_spks, -1, s)
        # [B*spks, N, S]
        x = self.output(x) * self.output_gate(x)
        # [B*spks, N, S]
        x = self.conv1_decoder(x)
        # [B, spks, N, S]
        _, n, L = x.shape
        x = x.view(b, self.num_spks, n, L)
        x = self.activation(x)
        # [spks, B, N, S]
        x = x.transpose(0, 1)
        return x

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

def prune(W, X, s):
    """
    自定义剪枝方法，修复索引越界问题。
    :param W: 权重矩阵 (C_out, C_in, kernel_size)
    :param X: 输入矩阵 (N, C_in, L)
    :param s: 剪枝比例 (0 < s < 1)
    :return: 剪枝后的权重矩阵
    """
    # 确保输入和权重在 CPU 上
    W = W.cpu()
    X = X.cpu()

    # 计算 X 在时间维度上的范数，形状变为 (C_in,)
    X_norm = X.pow(2).sum(dim=(0, 2)).sqrt()

    # 计算剪枝指标 (C_out, C_in, kernel_size)
    metric = W.abs() * X_norm.unsqueeze(-1)

    # 展平 kernel_size 维度
    metric = metric.view(W.size(0), -1)

    # 根据剪枝指标排序
    _, sorted_idx = torch.sort(metric, dim=1)

    # 确定要剪枝的数量
    num_prune = int(metric.size(1) * s)
    pruned_idx = sorted_idx[:, :num_prune]

    # 创建掩码并剪枝
    mask = torch.ones_like(W.view(W.size(0), -1))

    # 遍历每个输出通道，更新对应的掩码
    for i in range(pruned_idx.size(0)):  # 遍历每个输出通道
        valid_indices = pruned_idx[i]  # 当前通道的索引
        valid_indices = valid_indices[valid_indices < mask.size(1)]  # 确保索引在范围内
        mask[i, valid_indices] = 0  # 设置对应位置为 0

    # 恢复掩码形状并应用
    mask = mask.view_as(W)
    W.mul_(mask)

    return W


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
                W = module.weight  # 卷积核权重
                X = input[0]  # 输入特征图
                W_pruned = prune(W, X, s)  # 调用修复后的剪枝函数
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

import time
def measure_inference_time_cpu(model, input_size=(1, 1, 16000), num_runs=10):
    """
    测量模型在 CPU 上的平均推理时间。
    :param model: PyTorch 模型
    :param input_size: 输入数据的形状
    :param num_runs: 测试的运行次数（取平均）
    :return: 平均推理时间（毫秒）
    """
    # 准备模型和输入数据
    model.eval()
    model = model.to("cpu")  # 确保模型在 CPU 上
    input_data = torch.randn(input_size).to("cpu")  # 创建随机输入数据

    # 热身运行，避免初始化影响
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_data)

    # 测量多次推理的耗时
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()  # 开始计时
            _ = model(input_data)
            end_time = time.time()  # 结束计时
            times.append((end_time - start_time) * 1000)  # 转换为毫秒

    avg_time = sum(times) / len(times)  # 计算平均耗时
    print(f"Average inference time on CPU: {avg_time:.2f} ms")
    return avg_time

# 测试
if __name__ == '__main__':
    device = torch.device('cuda')
    model =  MossFormer(
            kernel_size =16,
            stride=8,
            bias=False,
            out_channels=512,
            in_channels=512,
            num_blocks=24,
            d_model=512,
            attn_dropout=0.1,
            group_size=256,
            query_key_dim=128,
            expansion_factor=4.0,
            causal=False,
            norm='ln',
            num_spks = 2
        )
    conf = torch.load(
        '/home/zhaoyu/projects/code/Look2hear/Experiments/checkpoint/MossFormer/epoch=113.ckpt', map_location=device
    )  # Attempt to find the model and instantiate it.
    state_dict = conf["state_dict"]
    model.load_state_dict({k.replace('audio_model.', ''): v for k, v in state_dict.items()})
    # 模拟输入数据
    index = 0
    
    datamodule = LRS2DataModule(train_dir= 'DataPreProcess/LRS2/tr',
    valid_dir= 'DataPreProcess/LRS2/cv',
    test_dir= 'DataPreProcess/LRS2/tt',
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
    
    # 剪枝前性能评估
    print("Before pruning:")
    calculate_sparsity(model)
    input_data = torch.randn((1,1,16000), device=device)
    measure_inference_time(model)
    

    # 应用自定义剪枝方法
    print("Applying custom pruning...")
    # 创建示例输入数据
    input_data = torch.randn((1,1,16000), device=device)
    apply_custom_pruning(model, input_data, s=0.5)

    print("After pruning:")
    # 评估剪枝后模型的性能
    measure_inference_time(model)
    calculate_sparsity(model)