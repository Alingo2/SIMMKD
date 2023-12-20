# ------------------------------------------------------------------------------
# Modified by Yanjie Li (leeyegy@gmail.com)
# ------------------------------------------------------------------------------
# from models.ops.modules import MSDeformAttn
# import copy
from models.transformer import build_transformer

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from timm.models.layers.weight_init import trunc_normal_
import math
from visualizer import get_local

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn,fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim*fusion_factor)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio, 
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop) 
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, stage_idx)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
        
    def forward(self, x):
        
        x = self.proj(x)
        
        positions = []
        references = []
        for d in range(self.depths):

            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5      #在softmax()之前先降维，否则各个值之间差距太大

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints
        
    @get_local('attn_map')
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads       #*.shape 就是shape 只不过去掉()  执行解包操作 赋值给b,n,_  b=batch,n=1025(=256像素*4+1keypoint?),_=192
        qkv = self.to_qkv(x).chunk(3, dim = -1) #torch.chunk() 执行分块操作 分成q k v 三份 因为to_qkv乘了3 所以q,k,v的shape分别为[batch,n=1025,inner_dim=192]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)   #（b,n,inner_dim）变为 （b,n,heads × dim_head）即[b,8,n,24]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale   #q：（b，heads，n，dim_head）； k：（b，heads，n，dim_head）；
                                                                    #q×k →（b，heads，n，n），相当于是n行dim_head列的特征图，乘以dim_head行n列的特征图。
                                                                    #从图像角度理解（b，heads，n，dim_head）：
                                                                    #b为batchsize；heads为通道数；n为行，就是每个heads之下有n组dim_head；dim_head为列。
                                                                    #q×k是在每个batchsize（b）下，在各个通道下（heads）下的特征图相乘。就是q和k在对应通道上的特征图分别相乘，不同通道上的特征图是不相乘的。对应到通道（heads）这个维度上，意义在于heads实为multi-heads的数量，在每个heads的下面各自分别计算，不同heads之间不交叉。
                                                                    #n相当于每个heads下有几个patches，q×k的矩阵操作意味着q的每一组patch和k的每一组patch都一一相乘了。


        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn_map = attn
        out = torch.einsum('bhij,bhjd->bhid', attn, v)      #（b，heads，n，n）×（b，heads，n，dim_head）→（b，heads，n，dim_head）
                                                            #先对attn（n，n）进行softmax（dim=-1），结果是（n，n）的每一行之和是1。然后再和v相乘，相当于对v的n组patches相应位置上系数之和为1。

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout,num_keypoints=None,all_attn=False, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head))),
                # Residual(PreNorm(dim, MSDeformAttn(dim, n_heads = heads,n_levels=1))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None,pos=None):
        for idx,(attn, ff) in enumerate(self.layers):
            if idx>0 and self.all_attn:
                x[:,self.num_keypoints:] += pos
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TokenPose_S_base(nn.Module):
    def __init__(self, config, image_size, patch_size, num_keypoints, dim, depth, heads, mlp_dim, apply_init=False, apply_multi=True, hidden_heatmap_dim=64*6,heatmap_dim=64*48,heatmap_size=[64,48], channels = 3, dropout = 0., emb_dropout = 0.,pos_embedding_type="learnable"):
        super().__init__()
        
        self.use_decoder = config.USE_DECORDER
        self.use_backbone = config.USE_BACKBONE
        self.use_regression = config.USE_REGRESSION
        self.use_cls = config.USE_CLS
        self.use_detr = config.USE_DETR
        self.use_detr_transformer = False
        self.use_detection = False
        self.use_det_reg = config.USE_DET_REG
        self.use_backbone_cls = config.USE_BACKBONE_CLS
        self.use_insert = config.USE_INSERT
        self.insert_num = config.INSERT_NUM

        if self.use_backbone:
            feature_size = image_size
            assert isinstance(feature_size,list) and isinstance(patch_size,list), 'image_size and patch_size should be list'
            assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
            num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
            patch_dim = channels * patch_size[0] * patch_size[1]
            assert pos_embedding_type in ['sine','learnable','sine-full']
        else:
            assert isinstance(image_size,list) and isinstance(patch_size,list), 'image_size and patch_size should be list'
            assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
            num_patches = (image_size[0] // (4*patch_size[0])) * (image_size[1] // (4*patch_size[1]))
            patch_dim = channels * patch_size[0] * patch_size[1]
            assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
            assert pos_embedding_type in ['sine','none','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")


        self.token_num = self.num_keypoints
        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        if self.use_backbone:
            h,w = feature_size[0] // (self.patch_size[0]), feature_size[1] // ( self.patch_size[1])
        else:
            h,w = image_size[0] // (4*self.patch_size[0]), image_size[1] // (4* self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)


        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # stem net
        if not self.use_backbone:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                                bias=False)
            self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                                bias=False)
            self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(Bottleneck, 64, 4)

        # transformer
        if self.use_detr_transformer:
            self.detr_transformer = build_transformer(hidden_dim=dim, dropout=dropout, nheads=heads, dim_feedforward=extra.DIM_FEEDFORWARD,
                                    enc_layers=depth, dec_layers=depth, pre_norm=extra.PRE_NORM)
        if self.use_decoder:
            self.token_num = 0
            decoder_layer = nn.TransformerDecoderLayer(d_model=192,nhead=heads)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer,num_layers=4)
        
        if self.use_detection:
            self.det_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))   #cls,x1,x2,y1,y2
        if self.use_cls :
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.cls_num = config.CLS_NUM
            self.cls_mlp_head = MLP(dim,dim,self.cls_num,3)
            self.token_num += 1
        if self.use_det_reg:
            self.det_reg_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.det_reg_mlp = MLP(dim,dim,4,3)
            self.token_num += 1
        if self.use_regression:
            num_keypoints = self.num_keypoints
            self.mlp_head = MLP(dim, dim, 3, 3)   #最后的3指3层
            # self.mlp_head = nn.Sequential(
            #     nn.LayerNorm(dim),
            #     nn.Linear(dim, num_keypoints*self.predict_num)
            # )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_heatmap_dim),
                nn.LayerNorm(hidden_heatmap_dim),
                nn.Linear(hidden_heatmap_dim, heatmap_dim)
            ) if (dim <= hidden_heatmap_dim*0.5 and apply_multi) else  nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, heatmap_dim)
            )

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout,num_keypoints=self.token_num,all_attn=self.all_attn)
        
        if self.use_insert:
            self.keypoint_token_front = nn.Parameter(torch.zeros(1, self.num_keypoints-1, dim))
            self.keypoint_token_behind = nn.Parameter(torch.zeros(1, 1, dim))
            self.transformer_front = Transformer(dim, self.insert_num, heads, mlp_dim, dropout,num_keypoints=self.token_num-1,all_attn=self.all_attn)
            self.transformer_behind = Transformer(dim, 12-self.insert_num, heads, mlp_dim, dropout,num_keypoints=self.token_num,all_attn=self.all_attn)
            
        self.to_keypoint_token = nn.Identity()
        self.to_cls_token = nn.Identity()      #恒等映射 f(x) = x
        self.to_det_reg_token = nn.Identity()      #恒等映射 f(x) = x

        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)
            
    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img, mask = None):
        p = self.patch_size

        if not self.use_backbone:
            # stem net 
            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.layer1(x)
        else:
            x = img
        # transformer
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])        #before transformer: [batch,48,64,64]

        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        if self.use_insert:
            keypoint_tokens_front = repeat(self.keypoint_token_front, '() n d -> b n d', b = b)
            keypoint_tokens_behind = repeat(self.keypoint_token_behind, '() n d -> b n d', b = b)
        if self.use_cls:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        if self.use_det_reg:
            det_reg_token = repeat(self.det_reg_token, '() n d -> b n d', b = b)

        if self.use_decoder:
            keypoint_tokens = keypoint_tokens.permute(1,0,2)     #换为Decorder之后输入
            #需要加入use_cls 再加入
            x += self.pos_embedding[:, :n]
        else:
            if self.pos_embedding_type in ["sine","sine-full"] :#
                x += self.pos_embedding[:, :n]
                if self.use_insert:
                    x = torch.cat((keypoint_tokens_front, x), dim=1)
                else:
                    x = torch.cat((keypoint_tokens, x), dim=1)
                if self.use_cls:
                    x = torch.cat((cls_tokens, x), dim=1)   #cls_token 摆在keypoint之前 即 顺序为： cls  keypoint feature
                if self.use_det_reg:
                    x = torch.cat((det_reg_token,x),dim=1) #det_reg_token 摆在keypoint之前 即 顺序为： det_reg_token  keypoint feature
            elif self.pos_embedding_type == "learnable":
                #需要使用learnable再改
                x = torch.cat((keypoint_tokens, x), dim=1)
                x += self.pos_embedding[:, :(n + self.num_keypoints)]

        x = self.dropout(x)
        if self.use_insert:
            x = self.transformer_front(x, mask,self.pos_embedding)
            x = torch.cat((keypoint_tokens_behind, x), dim=1)
            x = self.dropout(x)
            x = self.transformer_behind(x, mask,self.pos_embedding)
        else:
            x = self.transformer(x, mask,self.pos_embedding)
            

        if self.use_decoder:
            x = x.permute(1,0,2)    #换为Decorder之后输入
            x = self.transformer_decoder(keypoint_tokens,x)
            x = self.to_keypoint_token(x) 
        else:
            if self.use_cls:
                y = self.to_cls_token(x[:, 0:1])
                y = self.cls_mlp_head(y)        #得到[batch,channel,cls][b,1,3]
                
                y = rearrange(y,'b c cls -> b (c cls)')
                y = F.softmax(y,dim=1)
                # y = y.sigmoid()       #n分类 映射到n类  每一类的概率？ n类 sigmoid?
                # y = rearrange(y,'b c (p1 p2) -> b c p1 p2',p1=1,p2=self.cls_num)       #1*n类？ 
                x = x[:, 1:]
            if self.use_det_reg:
                z = self.to_det_reg_token(x[:, 0:1])
                z = self.det_reg_mlp(z)        #得到[batch,channel,cls][b,1,4]
                z = z.sigmoid() #[batch,1,4]
                z = z.squeeze(1)
                x = x[:, 1:]
                
            x = self.to_keypoint_token(x[:, 0:self.num_keypoints])
        x = self.mlp_head(x)
        if self.use_regression:
            x = x.sigmoid()

            x = x.unsqueeze(1)  #用这个来替换下一行
            # x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.num_keypoints,p2=3)         
        else:
            x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])

        if self.use_decoder:
            x = x.permute(1,0,2,3)    #换为Decoder之后的
        if self.use_cls:
            return x,y
            # x = torch.cat([x,y],dim=3)
        if self.use_det_reg:
            return x,z
        
        return x      

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TokenPose_TB_base(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth, heads, mlp_dim, apply_init=False, apply_multi=True, hidden_heatmap_dim=64*6,heatmap_dim=64*48,heatmap_size=[64,48], channels = 3, dropout = 0., emb_dropout = 0., pos_embedding_type="learnable"):
        super().__init__()
        assert isinstance(feature_size,list) and isinstance(patch_size,list), 'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['sine','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h,w = feature_size[0] // (self.patch_size[0]), feature_size[1] // ( self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)


        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout,num_keypoints=num_keypoints,all_attn=self.all_attn, scale_with_head=True)
        

        self.to_keypoint_token = nn.Identity()      #恒等映射 f(x) = x

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim <= hidden_heatmap_dim*0.5 and apply_multi) else  nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, heatmap_dim)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask = None):
        p = self.patch_size
        # transformer
        x = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        if self.pos_embedding_type in ["sine","sine-full"] :
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)
        x = self.transformer(x, mask,self.pos_embedding)
        x = self.to_keypoint_token(x[:, 0:self.num_keypoints])
        x = self.mlp_head(x)
        x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
        return x

class TokenPose_L_base(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth, heads, mlp_dim, apply_init=False, hidden_heatmap_dim=64*6,heatmap_dim=64*48,heatmap_size=[64,48], channels = 3, dropout = 0., emb_dropout = 0.,pos_embedding_type="learnable"):
        super().__init__()
        assert isinstance(feature_size,list) and isinstance(patch_size,list), 'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['sine','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h,w = feature_size[0] // (self.patch_size[0]), feature_size[1] // ( self.patch_size[1])

        # for normal 
        self._make_position_embedding(w, h, dim, pos_embedding_type)


        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.use_decoder = True
        if self.use_decoder:
            num_keypoints = 0
            decoder_layer1 = nn.TransformerDecoderLayer(d_model=192,nhead=heads)
            self.transformer_decoder1 = nn.TransformerDecoder(decoder_layer1,num_layers=6)
            decoder_layer2 = nn.TransformerDecoderLayer(d_model=192,nhead=heads)
            self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer2,num_layers=6)
            decoder_layer3 = nn.TransformerDecoderLayer(d_model=192,nhead=heads)
            self.transformer_decoder3 = nn.TransformerDecoder(decoder_layer3,num_layers=6)
        
        self.transformer1 = Transformer(dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints, all_attn=self.all_attn, scale_with_head=True)
        self.transformer2 = Transformer(dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints, all_attn=self.all_attn, scale_with_head=True )
        self.transformer3 = Transformer(dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints, all_attn=self.all_attn, scale_with_head=True)

        self.to_keypoint_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim*3),
            nn.Linear(dim*3, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim*3 <= hidden_heatmap_dim*0.5 and apply_multi) else  nn.Sequential(
            nn.LayerNorm(dim*3),
            nn.Linear(dim*3, heatmap_dim)
        )
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)
            
    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask = None):
        p = self.patch_size
        # transformer
        x = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p[0], p2 = p[1])
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens1 = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        keypoint_tokens2 = repeat(self.keypoint_token, '() n d -> b n d', b = b)
        keypoint_tokens3 = repeat(self.keypoint_token, '() n d -> b n d', b = b)

        if self.use_decoder:
            keypoint_tokens1 = keypoint_tokens1.permute(1,0,2)     #换为Decorder之后输入
            keypoint_tokens2 = keypoint_tokens2.permute(1,0,2)     #换为Decorder之后输入
            keypoint_tokens3 = keypoint_tokens3.permute(1,0,2)     #换为Decorder之后输入
            x += self.pos_embedding[:, :n]
        else:
            if self.pos_embedding_type in ["sine","sine-full"] :
                x += self.pos_embedding[:, :n]
                x = torch.cat((keypoint_tokens, x), dim=1)
            else:
                x = torch.cat((keypoint_tokens, x), dim=1)
                x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)

        x1 = self.transformer1(x, mask,self.pos_embedding)
        x2 = self.transformer2(x1, mask,self.pos_embedding)
        x3 = self.transformer3(x2, mask,self.pos_embedding)

        if self.use_decoder:
            x1 = x1.permute(1,0,2)    #换为Decorder之后输入
            x2 = x2.permute(1,0,2)    #换为Decorder之后输入
            x3 = x3.permute(1,0,2)    #换为Decorder之后输入
            x1 = self.transformer_decoder1(keypoint_tokens1,x1)
            x2 = self.transformer_decoder2(keypoint_tokens2,x2)
            x3 = self.transformer_decoder3(keypoint_tokens3,x3)
            x1_out = self.to_keypoint_token(x1)
            x2_out = self.to_keypoint_token(x2)
            x3_out = self.to_keypoint_token(x3)
        else:
            x1_out = self.to_keypoint_token(x1[:, 0:self.num_keypoints])
            x2_out = self.to_keypoint_token(x2[:, 0:self.num_keypoints])
            x3_out = self.to_keypoint_token(x3[:, 0:self.num_keypoints])

        x = torch.cat((x1_out, x2_out, x3_out), dim=2)
        if self.use_decoder:
            x = x.permute(1,0,2)    #加入Decoder
        x = self.mlp_head(x)
        x = rearrange(x,'b c (p1 p2) -> b c p1 p2',p1=self.heatmap_size[0],p2=self.heatmap_size[1])
        return x
