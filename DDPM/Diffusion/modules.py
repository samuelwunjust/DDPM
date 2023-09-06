
import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
    def __init__(self,beta):
        self.beta=beta
        self.step=0
    def update_average_model(self,ma_model,current_model):
        for ma_parameters,current_parameters in zip(ma_model.parameters(),current_model.parameters()):
            old_weight=ma_parameters
            new_weight=current_parameters
            ma_parameters.data=self.update_average(old_weight,new_weight)
    
    def update_average(self,old_weight,new_weight):
        if old_weight is None:
            return new_weight
        else:
            return old_weight*self.beta+(1-self.beta)*new_weight
    
    def step_ema(self,model,ma_model,step_ema=2000):
        if self.step<step_ema:
            self.reset_parameters(ma_model,model)
            self.step+=1
            return
        self.update_average_model(ma_model,model)
        self.step+=1
    def reset_parameters(self,ma_model,model):
        ma_model.load_state_dict(model.state_dict())


class selfattention(nn.Module):
    def __init__(self, channel,size):
        super(selfattention,self).__init__()
        self.channel=channel
        self.size=size
        self.ln=nn.LayerNorm([channel])
        self.mha=nn.MultiheadAttention(channel,4,batch_first=True)
        self.a_ff=nn.Sequential(
            nn.LayerNorm([channel]),
            nn.Linear(channel,channel),
            nn.GELU(),
            nn.Linear(channel,channel)

        )
    
    def forward(self,x):
       
        #x(batch_size,channel,size*size)

        """
        返回值：
        - attention_value: 经过多头自注意力和前馈神经网络处理后的特征张量。

        在前向传播中，输入特征被重塑为 (batch_size, channels, size, size)，然后经过以下步骤：
        1. 使用多头自注意力层处理特征。
        2. 将自注意力的结果与输入特征相加，以获得自注意力增强的特征。
        3. 使用前馈神经网络处理增强特征。
        4. 最后，将结果重塑为原始形状并返回。
        """
        x = x.view(-1, self.channel, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.a_ff(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channel, self.size, self.size)
    
class Doubleconv(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel=None,residual=False):
        super().__init__()
       
        self.residual=residual
        if not mid_channel:
            mid_channel=out_channel
        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channel),
            nn.GELU(),
            nn.Conv2d(mid_channel,out_channel, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channel),
            
        )

    def forward(self,x):
            if self.residual==False:
                 return self.double_conv(x)
            else:
                return F.gelu(x+self.double_conv(x))

class Down(nn.Module):#down sample maxpooling+time-embed
    def __init__(self, in_channel,out_channel,embed_dim=256) -> None:
        super().__init__()   
        self.maxpooling=nn.Sequential(
            nn.MaxPool2d(2),#reduce half of image feature
            Doubleconv(in_channel,out_channel=in_channel,residual=True),
            Doubleconv(in_channel=in_channel,out_channel=out_channel)

        )     
        self.time_embed=nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim,out_channel),

        )
    def forward(self,x,t):
            x=self.maxpooling(x)
            emb= self.time_embed(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
            return x+emb

class Up(nn.Module):#up sample
    def __init__(self, in_channel,out_channel,emb_dim=256) -> None:
        super().__init__()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)#up
        self.conv=nn.Sequential(
            Doubleconv(in_channel,in_channel,residual=True),
            Doubleconv(in_channel,out_channel,in_channel//2)

        )
        self.emb_layer=nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim,out_channel),

        )

    def forward(self,x,skip_x,t):

       x=self.up(x)
       x=torch.cat([skip_x,x],dim=1)
       x=self.conv(x)

       
       emb= self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
       return emb+x
    
    

"""Unet结构Input
           |
        DoubleConv
           |
        Down -> SelfAttention
           |
        DoubleConv
           |
        Down -> SelfAttention
           |
        DoubleConv
           |
        Down -> SelfAttention
           |
           |
        DoubleConv
           |
        Up -> SelfAttention
           |
        DoubleConv
           |
        Up -> SelfAttention
           |
        DoubleConv
           |
        Up -> SelfAttention
           |
           |
        Output"""

class UNet(nn.Module):
    def __init__(self, c_in,c_out,tim_embed=256,device='cuda') -> None:
        super().__init__()
        self.device=device
        self.time_dim=tim_embed
    #encoder
        self.inc=Doubleconv(c_in,64)
        self.down1=Doubleconv(64,128)
        self.sa1=selfattention(128,32)
        self.down2=Doubleconv(128,256)
        self.sa2=selfattention(256,16)
        self.down3=Doubleconv(256,256)
        self.sa4=selfattention(256,8)
    #middle
        self.bot1=Doubleconv(256,512)
        self.bot2=Doubleconv(512,512)
        self.bot3=Doubleconv(512,256)

    #decoder
        self.up1 = Up(512, 128)
        self.sa4 = selfattention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = selfattention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = selfattention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
    def pos_encoding(self, t, channels):
        
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
         
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # 编码器部分的前向传播
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # 中间层的前向传播
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # 解码器部分的前向传播
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = Doubleconv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = selfattention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = selfattention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = selfattention(256, 8)

        self.bot1 = Doubleconv(256, 512)
        self.bot2 = Doubleconv(512, 512)
        self.bot3 = Doubleconv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = selfattention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = selfattention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = selfattention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__=="__main__":
    net=UNet_conditional(num_classes=10,device="cpu")
    print(sum([p.numel()for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)

    # create [500, 500, 500]
    t = x.new_tensor([500] * x.shape[0]).long()

    # create label y [1, 1, 1]
    y = x.new_tensor([1] * x.shape[0]).long()

    # 状
    output = net(x, t, y)
    print(output.shape)


    
        
        
    