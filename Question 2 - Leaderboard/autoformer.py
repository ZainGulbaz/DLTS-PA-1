"""Compact Autoformer encoder-decoder, based on Wu et al. (NeurIPS 2021).
Original implementation; per-example delays are used in both train and eval.
https://arxiv.org/abs/2106.13008
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Decomposition(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0: raise ValueError('positive odd kernel required')
        self.kernel=kernel
    def forward(self,x):
        padded=F.pad(x.transpose(1,2),(self.kernel//2,self.kernel//2),mode='replicate')
        trend=F.avg_pool1d(padded,self.kernel,1).transpose(1,2)
        return x-trend,trend

class AutoCorrelation(nn.Module):
    def __init__(self,width,heads=4,factor=2):
        super().__init__()
        assert width%heads==0
        self.heads=heads;self.factor=factor
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width)
        self.v=nn.Linear(width,width);self.out=nn.Linear(width,width)
    def forward(self,queries,keys,values):
        batch,length,width=queries.shape
        def project(layer,x):
            z=layer(x).reshape(batch,x.shape[1],self.heads,width//self.heads).permute(0,2,3,1)
            # Official Autoformer aligns key/value length to query length by truncation/padding.
            if z.shape[-1]<length: z=F.pad(z,(0,length-z.shape[-1]))
            return z[...,:length]
        q,k,v=project(self.q,queries),project(self.k,keys),project(self.v,values)
        q=q-q.mean(-1,keepdim=True);k=k-k.mean(-1,keepdim=True)
        corr=torch.fft.irfft(torch.fft.rfft(q)*torch.fft.rfft(k).conj(),n=length)
        scores=corr.mean((1,2))/math.sqrt(width//self.heads)
        count=min(length,max(1,int(self.factor*math.log(length))))
        top,delays=scores.topk(count,dim=-1)
        weights=top.softmax(-1)
        out=torch.zeros_like(v);t=torch.arange(length,device=v.device)
        for j in range(count):
            index=((t[None,:]-delays[:,j,None])%length)[:,None,None,:].expand_as(v)
            out=out+weights[:,j,None,None,None]*v.gather(-1,index)
        return self.out(out.permute(0,3,1,2).reshape(batch,length,width))

class SeasonalNorm(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width)
    def forward(self,x):
        z=self.norm(x);return z-z.mean(1,keepdim=True)

class EncoderLayer(nn.Module):
    def __init__(self,width,heads,kernel,dropout):
        super().__init__();self.mix=AutoCorrelation(width,heads)
        self.split1=Decomposition(kernel);self.split2=Decomposition(kernel)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*width,width))
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        x,_=self.split1(x+self.drop(self.mix(x,x,x)))
        x,_=self.split2(x+self.drop(self.ff(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self,width,heads,kernel,dropout):
        super().__init__();self.self_mix=AutoCorrelation(width,heads);self.cross_mix=AutoCorrelation(width,heads)
        self.splits=nn.ModuleList([Decomposition(kernel) for _ in range(3)])
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*width,width))
        self.trend=nn.Conv1d(width,1,3,padding=1,padding_mode='circular',bias=False)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,memory,trend):
        x,t1=self.splits[0](x+self.drop(self.self_mix(x,x,x)))
        x,t2=self.splits[1](x+self.drop(self.cross_mix(x,memory,memory)))
        x,t3=self.splits[2](x+self.drop(self.ff(x)))
        trend=trend+self.trend((t1+t2+t3).transpose(1,2)).transpose(1,2)
        return x,trend

class Autoformer(nn.Module):
    def __init__(self,context=336,horizon=168,label=84,width=32,heads=4,kernel=25,external=0,dropout=.1):
        super().__init__()
        self.context=context;self.horizon=horizon;self.label=label
        self.initial_split=Decomposition(kernel)
        self.enc_embed=nn.Conv1d(1+external,width,3,padding=1,padding_mode='circular',bias=False)
        self.dec_embed=nn.Conv1d(1+external,width,3,padding=1,padding_mode='circular',bias=False)
        self.encoder=EncoderLayer(width,heads,kernel,dropout)
        self.decoder=DecoderLayer(width,heads,kernel,dropout)
        self.enc_norm=SeasonalNorm(width);self.dec_norm=SeasonalNorm(width)
        self.project=nn.Linear(width,1)
    def forward(self,history,past_external,future_external):
        batch=history.shape[0]
        seasonal,trend=self.initial_split(history)
        # Future target values are never fed to the decoder.
        seasonal=torch.cat([seasonal[:,-self.label:],history.new_zeros(batch,self.horizon,1)],1)
        trend=torch.cat([trend[:,-self.label:],history.mean(1,keepdim=True).expand(-1,self.horizon,-1)],1)
        enc=torch.cat([history,past_external],-1)
        dec_cov=torch.cat([past_external[:,-self.label:],future_external],1)
        dec=torch.cat([seasonal,dec_cov],-1)
        memory=self.enc_norm(self.encoder(self.enc_embed(enc.transpose(1,2)).transpose(1,2)))
        decoded=self.dec_embed(dec.transpose(1,2)).transpose(1,2)
        decoded,trend=self.decoder(decoded,memory,trend)
        forecast=self.project(self.dec_norm(decoded))+trend
        return forecast[:,-self.horizon:,0]
