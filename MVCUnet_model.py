import torch
import torch.nn as nn

from  mamba_model import Residualmamba, ModelArgs
from  vmamba import VSSBlock



class TimeChannelEmbed(nn.Module):
    def __init__(self,in_cn=6,out_cn=64,hidden_cn=2,mamba_repeat_num=2):
        super().__init__()
        self.in_cn =  in_cn
        self.out_cn = out_cn
        self.hid_cn = hidden_cn
        self.head_conv = nn.Conv2d(in_channels=self.in_cn, out_channels=3*self.hid_cn,kernel_size=3,stride=1,padding=1)
        self.mambaarg = ModelArgs(d_model=hidden_cn,d_state=2)
        self.block = nn.Sequential(*[Residualmamba(self.mambaarg) for _ in range(mamba_repeat_num)])
        self.bn1 = nn.BatchNorm2d(num_features=self.out_cn)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=self.out_cn)
        self.relu2 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=self.hid_cn*3,out_channels=self.out_cn, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.out_cn, out_channels=self.out_cn, kernel_size=3, stride=1,padding=1)

    def forward(self, x):
        (B, _, H, W) = x.shape
        x = self.head_conv(x) # B 3D H W
        seq_x = x.permute(0, 2, 3, 1).contiguous().view(B*H*W,3,self.hid_cn) # (BHW ,l=3, D)
        seq_out = self.block(seq_x)
        seq_out = seq_out.contiguous().view(-1, 3*self.hid_cn).view(B,H,W,-1).permute(0, 3, 1, 2) # (BHW ,lD) -> (B,H,W,lD)->(B,lD,H,W)
        seq_out = seq_out + x #(B,lD,H,W)
        seq_out= self.conv1(seq_out) #(B,64,H,W)
        res = self.relu1(self.bn1(self.conv2(seq_out))) #(B,64,H,W)
        seq_out = self.relu2(self.bn2(seq_out+res)) #(B,64,H,W)

        return seq_out


class VSSCBlock(nn.Module):
    def __init__(self,in_cn,out_cn):
        super().__init__()
        self.in_cn = in_cn
        self.out_cn = out_cn
        self.conv_star = nn.Conv2d(in_channels=self.in_cn,out_channels=self.out_cn,kernel_size=1,stride=1)
        self.conv_end = nn.Conv2d(in_channels=self.out_cn, out_channels=self.out_cn, kernel_size=1,stride=1)
        self.conv_concat = nn.Sequential(
            nn.Conv2d(self.out_cn // 2, self.out_cn // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.out_cn // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.out_cn // 2, self.out_cn // 2, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.vss = VSSBlock(hidden_dim=self.out_cn // 2,drop_path=0,attn_drop_rate=0,d_state=self.out_cn // 2)
        self.bn = nn.BatchNorm2d(self.out_cn)
        self.relu = nn.ReLU()

    def forward(self,x): # x : B C H W

        vss_x, conv_x = torch.split(self.conv_star(x), (self.out_cn // 2, self.out_cn // 2), dim=1) # vssx  convx: B D/2 H W
        convx =  conv_x + self.conv_concat(conv_x)
        vss_x = self.vss(vss_x)   # vssx  convx: B D/2 H W
        res = self.conv_end(torch.cat([vss_x, convx], dim=1)) # res: B D H W
        x = self.relu(self.bn(x + res))
        return x


class Down_module(nn.Module):
    def __init__(self,in_cn,out_cn):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=in_cn,out_channels=out_cn, kernel_size=2, stride=2)
        self.VSSC = VSSCBlock(in_cn=out_cn, out_cn=out_cn)

    def forward(self, x):
        x = self.downsample(x)
        x = self.VSSC(x)
        return x


class Up_module(nn.Module):
    def __init__(self, in_cn, out_cn, is_skip=True):
        super().__init__()
        self.is_skip = is_skip
        self.VSSC = VSSCBlock(in_cn=out_cn, out_cn=out_cn)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_cn, out_cn, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_cn),
            nn.ReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(in_cn, 4 * in_cn, 3, 1, 1, bias=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(in_cn, out_cn, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_cn),
            nn.ReLU()
        )

    def forward(self,x1, x2):
        x = self.upsample1(x1)
        if self.is_skip:
            x = self.VSSC(x + x2)
        else:
            x = self.VSSC(x)
        return x


# whole multi train quantitative mamba unet
class MVCUnet(nn.Module):
    def __init__(self,in_cn, out_cn, hidden_cn):
        super().__init__()
        self.in_cn = in_cn
        self.out_cn = out_cn
        self.hid_cn = hidden_cn
        self.scale = [64, 128, 256, 512]
        self.head = nn.Sequential(
            TimeChannelEmbed(in_cn=self.in_cn, out_cn=self.scale[0], hidden_cn=self.hid_cn, mamba_repeat_num=2),
            VSSCBlock(in_cn=self.scale[0], out_cn=self.scale[0])
        )
        self.down1 = Down_module(in_cn=self.scale[0], out_cn=self.scale[1])
        self.down2 = Down_module(in_cn=self.scale[1], out_cn=self.scale[2])
        self.down3 = Down_module(in_cn=self.scale[2], out_cn=self.scale[3])
        self.up1 = Up_module(in_cn=self.scale[3], out_cn=self.scale[2], is_skip=True)
        self.up2 = Up_module(in_cn=self.scale[2], out_cn=self.scale[1], is_skip=True)
        self.up3 = Up_module(in_cn=self.scale[1], out_cn=self.scale[0], is_skip=True)
        self.tail = nn.Sequential(
            VSSCBlock(self.scale[0], out_cn=self.scale[0]),
            nn.Conv2d(in_channels=self.scale[0], out_channels=self.out_cn, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.head(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x8 = self.tail(x7)
        return x8