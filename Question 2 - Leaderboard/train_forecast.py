"""Chronological multi-seed Autoformer ablation and final 168-point export."""
import argparse
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from autoformer import Autoformer

ROOT=Path(__file__).resolve().parent
@dataclass
class Config:
    context:int=336
    horizon:int=168
    label:int=84
    width:int=32
    heads:int=4
    kernel:int=25
    external:int=0
    dropout:float=.1

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def metrics(y,p):
    error=p-y
    denominator=np.abs(y)+np.abs(p)
    ratio=np.divide(2*np.abs(error),denominator,out=np.zeros_like(error),where=denominator>0)
    return {'RMSE':float(np.sqrt(np.mean(error**2))),'MAE':float(np.mean(np.abs(error))),
            'sMAPE':float(100*np.mean(ratio))}

class Windows(Dataset):
    def __init__(self,y,cov,origins,cfg): self.y=y;self.cov=cov;self.origins=origins;self.cfg=cfg
    def __len__(self):return len(self.origins)
    def __getitem__(self,i):
        t=int(self.origins[i]);l=self.cfg.context;h=self.cfg.horizon
        return (torch.from_numpy(self.y[t-l:t,None]),torch.from_numpy(self.cov[t-l:t]),
                torch.from_numpy(self.cov[t:t+h]),torch.from_numpy(self.y[t:t+h]))

def load_data():
    train=pd.read_csv(ROOT/'Data/student_train.csv');test=pd.read_csv(ROOT/'Data/student_test.csv')
    ext=pd.read_csv(ROOT/'Data/optional_external_data.csv')
    n=len(train);h=len(test)
    assert n==43656 and h==168 and len(ext)==n+h
    assert np.array_equal(train.time_idx,np.arange(1,n+1))
    assert np.array_equal(test.time_idx,np.arange(n+1,n+h+1))
    assert np.array_equal(ext.time_idx,np.arange(1,n+h+1))
    y=train.value.to_numpy(dtype=np.float32);cov=ext.drop(columns='time_idx').to_numpy(dtype=np.float32)
    assert np.isfinite(y).all() and np.isfinite(cov).all() and (y>=0).all()
    return y,cov,test.time_idx.to_numpy()

def scale(y,cov,end,external):
    mean=float(y[:end].mean());std=max(float(y[:end].std()),1e-6)
    cmean=cov[:end].mean(0);cstd=np.maximum(cov[:end].std(0),1e-6)
    yc=((y-mean)/std).astype(np.float32)
    cc=((cov-cmean)/cstd).astype(np.float32) if external else np.empty((len(cov),0),np.float32)
    return yc,cc,{'target_mean':mean,'target_std':std,'cov_mean':cmean.tolist(),'cov_std':cstd.tolist()}

@torch.no_grad()
def predict(model,loader,device,scaler):
    model.eval();ys=[];ps=[]
    for x,past,future,target in loader:
        out=model(x.to(device),past.to(device),future.to(device)).cpu().numpy()
        ps.append(np.maximum(0,out*scaler['target_std']+scaler['target_mean']))
        ys.append(target.numpy()*scaler['target_std']+scaler['target_mean'])
    return np.concatenate(ys),np.concatenate(ps)

def fit(cfg,y,cov,end,origins,validation,seed,epochs,device,smoke=False):
    seed_all(seed);yc,cc,scaler=scale(y,cov,end,cfg.external)
    model=Autoformer(**asdict(cfg)).to(device)
    train=DataLoader(Windows(yc,cc,origins,cfg),batch_size=32,shuffle=True,num_workers=0)
    valid=DataLoader(Windows(yc,cc,validation,cfg),batch_size=8) if len(validation) else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float('inf');best_state=None;best_epoch=epochs;stale=0;trace=[];started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0.;count=0
        for batch,(x,past,future,target) in enumerate(train):
            optimizer.zero_grad(set_to_none=True)
            out=model(x.to(device),past.to(device),future.to(device))
            loss=(out-target.to(device)).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();loss_sum+=loss.item();count+=1
            if smoke and batch>=1:break
        row={'epoch':epoch,'train_loss':loss_sum/count}
        if valid is not None:
            truth,pred=predict(model,valid,device,scaler);row.update(metrics(truth,pred))
            if row['RMSE']<best:
                best=row['RMSE'];best_epoch=epoch;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
        trace.append(row);print('seed',seed,'external',cfg.external,'width',cfg.width,row,flush=True)
        if valid is not None and stale>=3:break
    spent=epoch
    if best_state is not None:model.load_state_dict(best_state)
    return model,scaler,best_epoch,spent,trace,time.perf_counter()-started

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],default='full')
    parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=parser.parse_args()
    if len(set(args.seeds))<2:raise ValueError('At least two distinct seeds are required.')
    if args.epochs<1:raise ValueError('epochs must be positive')
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smoke=args.mode=='smoke';dest=ROOT/('smoke_results' if smoke else 'results');dest.mkdir(exist_ok=True)
    y,cov,test_idx=load_data();end=len(y)-8*168
    validation=np.arange(end,len(y)-168+1,168)
    assert len(validation)==8 and validation[-1]+168==len(y)
    pd.DataFrame({'origin_time_idx':validation+1,'last_target_time_idx':validation+168}).to_csv(dest/'validation_blocks.csv',index=False)
    # Training-only descriptive periodicity diagnostic. No calendar identities are inferred.
    centered=y[:end].astype(np.float64)-y[:end].mean()
    spectrum=np.abs(np.fft.rfft(centered))**2
    freq=np.fft.rfftfreq(len(centered));keep=(freq>0)&(freq>=1/1000)&(freq<=1/2)
    inds=np.where(keep)[0];top=inds[np.argsort(spectrum[inds])[-10:][::-1]]
    pd.DataFrame({'period_samples':1/freq[top],'power':spectrum[top]}).to_csv(dest/'training_periodogram.csv',index=False)
    base=[]
    for t in validation:
        for name,p in [('last value',np.repeat(y[t-1],168)),('context mean',np.repeat(y[t-336:t].mean(),168))]:
            base.append({'baseline':name,'origin':int(t+1),**metrics(y[t:t+168],p)})
    pd.DataFrame(base).to_csv(dest/'baselines.csv',index=False)
    rows=[];details={}
    widths=[16] if smoke else [32,64]
    for width in widths:
        for external in [0,10]:
            cfg=Config(width=width,external=external)
            origins=np.arange(cfg.context,end-cfg.horizon+1,24)
            assert origins[-1]+cfg.horizon<=end
            for seed in args.seeds:
                tag=f'w{width}-x{external}-s{seed}'
                model,scaler,best_epoch,spent,trace,seconds=fit(cfg,y,cov,end,origins,validation,seed,
                    1 if smoke else args.epochs,device,smoke)
                yc,cc,_=scale(y,cov,end,external)
                valid=DataLoader(Windows(yc,cc,validation,cfg),batch_size=8)
                truth,pred=predict(model,valid,device,scaler)
                row={'width':width,'external':external,'seed':seed,**metrics(truth,pred),
                     'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'best_epoch':best_epoch,'epochs_spent':spent,'fit_seconds':seconds}
                rows.append(row);pd.DataFrame(trace).to_csv(dest/(tag+'-trace.csv'),index=False)
                block=[{'origin':int(t+1),**metrics(truth[j],pred[j])} for j,t in enumerate(validation)]
                pd.DataFrame(block).to_csv(dest/(tag+'-blocks.csv'),index=False)
                torch.save({'state_dict':model.state_dict(),'config':asdict(cfg),'scaler':scaler,'seed':seed,
                            'epochs_spent':spent,'best_epoch':best_epoch},dest/(tag+'.pt'))
                details[(width,external,seed)]=row
    results=pd.DataFrame(rows);results.to_csv(dest/'runs.csv',index=False)
    summary=results.groupby(['width','external'],as_index=False).agg(
        RMSE_mean=('RMSE','mean'),RMSE_std=('RMSE','std'),MAE_mean=('MAE','mean'),MAE_std=('MAE','std'),
        sMAPE_mean=('sMAPE','mean'),sMAPE_std=('sMAPE','std'),parameters=('parameters','first'),
        epochs_mean=('epochs_spent','mean'),fit_seconds_mean=('fit_seconds','mean'))
    summary.to_csv(dest/'summary.csv',index=False);print(summary.to_string(index=False))
    paired=results.pivot(index=['width','seed'],columns='external',values='RMSE')
    paired['RMSE_without_minus_with_external']=paired[0]-paired[10]
    paired.to_csv(dest/'external_ablation_paired.csv')
    if smoke:
        print('SMOKE ONLY: no final predictions generated.');return
    winner=summary.sort_values(['RMSE_mean','parameters']).iloc[0]
    cfg=Config(width=int(winner.width),external=int(winner.external))
    seed=args.seeds[0] # Fixed in advance, not selected by performance.
    matched=results[(results.width==cfg.width)&(results.external==cfg.external)]
    refit_epochs=max(1,int(np.rint(matched.best_epoch.median())))
    selected_run=details[(cfg.width,cfg.external,seed)]
    origins=np.arange(cfg.context,len(y)-cfg.horizon+1,24)
    model,scaler,_,spent,trace,seconds=fit(cfg,y,cov,len(y),origins,[],seed,refit_epochs,device)
    yc,cc,_=scale(y,cov,len(y),cfg.external)
    with torch.no_grad():
        model.eval()
        p=model(torch.from_numpy(yc[-cfg.context:,None])[None].to(device),
                torch.from_numpy(cc[len(y)-cfg.context:len(y)])[None].to(device),
                torch.from_numpy(cc[len(y):])[None].to(device)).cpu().numpy().reshape(-1)
    p=np.maximum(0,p*scaler['target_std']+scaler['target_mean'])
    assert p.shape==(168,) and np.isfinite(p).all()
    parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)
    # Count the selected seed's validation training AND full-data refit, conservatively using
    # all spent epochs, not just the selected checkpoint epoch. Other tuning runs are recorded.
    manifest={'config':asdict(cfg),'seed':seed,'P':parameters,
        'E':int(selected_run['epochs_spent']+spent),'validation_epochs_spent':int(selected_run['epochs_spent']),
        'refit_epochs':spent,'all_tuning_epochs':int(results.epochs_spent.sum()),
        'selection':'seed-mean validation RMSE; fixed first seed; median best epoch for refit',
        'epochs_limit':args.epochs,'validation_seeds':args.seeds,'training_rows':len(y),'forecast_first_index':int(test_idx[0]),'forecast_last_index':int(test_idx[-1]),
        'device':str(device),'torch':torch.__version__,'refit_seconds':seconds}
    pd.DataFrame({'time_idx':test_idx,'prediction':p}).to_csv(dest/'forecast.csv',index=False)
    (dest/'predictions.txt').write_text(', '.join(f'{v:.8f}' for v in p))
    (dest/'final_manifest.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(trace).to_csv(dest/'refit_trace.csv',index=False)
    torch.save({'state_dict':model.state_dict(),'config':asdict(cfg),'scaler':scaler,'manifest':manifest},dest/'final_model.pt')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
    ax.plot(np.arange(len(y)-336+1,len(y)+1),y[-336:],label='observed history')
    ax.plot(test_idx,p,label='168-step Autoformer forecast');ax.set(xlabel='time_idx',ylabel='value');ax.legend()
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb',suffix='.pdf',dir=dest,delete=False) as stream:
        temporary=Path(stream.name)
        fig.savefig(stream,format='pdf')
    if b'%%EOF' not in temporary.read_bytes()[-100:]:raise RuntimeError('Incomplete forecast PDF')
    temporary.replace(dest/'final_forecast.pdf');plt.close(fig)
    print('Final predictions:',dest/'predictions.txt','P=',parameters,'E=',manifest['E'])

if __name__=='__main__':main()
