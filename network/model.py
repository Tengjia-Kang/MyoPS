# import torch
# import torch.nn as nn
# from network.unet import UNet, UNetEncoder, UNetDecoder, UNetDecoderPlus
#
#
# ### Segment
# class MyoPSNet(nn.Module):
#     def __init__(self, in_chs, out_chs):
#         super(MyoPSNet, self).__init__()
#
#         self.unet_C0 = UNet(in_ch = in_chs[0], out_ch = out_chs[0])
#
#         self.encoder_LGE = UNetEncoder(in_ch = in_chs[1])
#         self.decoder_LGE = UNetDecoderPlus(out_ch = out_chs[1])
#
#         self.encoder_T2 = UNetEncoder(in_ch = in_chs[2])
#         self.decoder_T2 = UNetDecoderPlus(out_ch = out_chs[2])
#
#         self.encoder_mapping = UNetEncoder(in_ch = in_chs[3])
#         self.decoder_mapping = UNetDecoderPlus(out_ch = out_chs[3])
#
#     def forward(self, C0, LGE, T2, T1m, T2starm):
#
#         img = torch.cat([C0, LGE, T2, T1m, T2starm], dim=1)
#         seg_C0 = self.unet_C0(img)
#         mask_C0 = torch.argmax(seg_C0, dim=1, keepdim=True)
#
#         img_LGE = torch.cat([LGE, mask_C0.detach()], dim=1)
#         img_T2 = torch.cat([T2, mask_C0.detach()], dim=1)
#         img_mapping = torch.cat([T1m, T2starm, mask_C0.detach()], dim=1)
#
#         f_LGE = self.encoder_LGE(img_LGE)
#         f_T2 = self.encoder_T2(img_T2)
#         f_mapping = self.encoder_mapping(img_mapping)
#
#         seg_LGE_input = []
#         seg_T2_input = []
#         seg_mapping_input = []
#
#         for i in range(5):
#             seg_LGE_input.append(torch.max(f_mapping[i],f_T2[i]))
#             seg_T2_input.append(torch.max(f_LGE[i],f_mapping[i]))
#             seg_mapping_input.append(torch.max(f_LGE[i],f_T2[i]))
#
#         seg_LGE = self.decoder_LGE(seg_LGE_input,f_LGE)
#         seg_T2 = self.decoder_T2(seg_T2_input,f_T2)
#         seg_mapping = self.decoder_mapping(seg_mapping_input,f_mapping)
#
#         return seg_C0, seg_LGE, seg_T2, seg_mapping
#

import torch
import torch.nn as nn
from network.unet import UNet, UNetEncoder, UNetDecoderPlus

class MyoPSNet(nn.Module):
    def __init__(self, modalities, out_chs):
        super(MyoPSNet, self).__init__()

        self.modalities = modalities
        self.use_C0 = 'C0' in modalities
        self.use_LGE = 'LGE' in modalities
        self.use_T2 = 'T2' in modalities
        self.use_mapping = 'mapping' in modalities

        if self.use_C0:
            self.unet_C0 = UNet(in_ch=5, out_ch=out_chs[0])  # 5通道（所有模态拼接）

        if self.use_LGE:
            self.encoder_LGE = UNetEncoder(in_ch=2)  # LGE + mask
            self.decoder_LGE = UNetDecoderPlus(out_ch=out_chs[1])

        if self.use_T2:
            self.encoder_T2 = UNetEncoder(in_ch=2)  # T2 + mask
            self.decoder_T2 = UNetDecoderPlus(out_ch=out_chs[2])

        if self.use_mapping:
            self.encoder_mapping = UNetEncoder(in_ch=3)  # T1m + T2starm + mask
            self.decoder_mapping = UNetDecoderPlus(out_ch=out_chs[3])

    def forward(self, C0=None, LGE=None, T2=None, T1m=None, T2starm=None):
        inputs = []
        if C0 is not None: inputs.append(C0)
        if LGE is not None: inputs.append(LGE)
        if T2 is not None: inputs.append(T2)
        if T1m is not None: inputs.append(T1m)
        if T2starm is not None: inputs.append(T2starm)

        img = torch.cat(inputs, dim=1)
        seg_C0 = self.unet_C0(img) if self.use_C0 else None
        mask_C0 = torch.argmax(seg_C0, dim=1, keepdim=True) if seg_C0 is not None else None

        seg_LGE, seg_T2, seg_mapping = None, None, None

        if self.use_LGE:
            img_LGE = torch.cat([LGE, mask_C0.detach()], dim=1)
            f_LGE = self.encoder_LGE(img_LGE)

        if self.use_T2:
            img_T2 = torch.cat([T2, mask_C0.detach()], dim=1)
            f_T2 = self.encoder_T2(img_T2)

        if self.use_mapping:
            img_mapping = torch.cat([T1m, T2starm, mask_C0.detach()], dim=1)
            f_mapping = self.encoder_mapping(img_mapping)

        if self.use_LGE and self.use_T2 and self.use_mapping:
            seg_LGE_input = [torch.max(f_mapping[i], f_T2[i]) for i in range(len(f_mapping))]
            seg_T2_input = [torch.max(f_LGE[i], f_mapping[i]) for i in range(len(f_LGE))]
            seg_mapping_input = [torch.max(f_LGE[i], f_T2[i]) for i in range(len(f_LGE))]

            seg_LGE = self.decoder_LGE(seg_LGE_input, f_LGE)
            seg_T2 = self.decoder_T2(seg_T2_input, f_T2)
            seg_mapping = self.decoder_mapping(seg_mapping_input, f_mapping)

        return seg_C0, seg_LGE, seg_T2, seg_mapping
