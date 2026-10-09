"""Generate fictional service telemetry with independent deterministic splits."""
import csv
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
FEATURES=['cpu_percent','latency_ms','error_rate','requests_per_minute']

def normal(rng,n):
    traffic=np.clip(rng.normal(500,100,n),50,1000)
    cpu=np.clip(15+traffic*0.055+rng.normal(0,6,n),1,95)
    latency=np.clip(30+cpu*1.5+rng.normal(0,12,n),1,1000)
    errors=np.clip(rng.beta(2,150,n),0,1)
    return np.column_stack([cpu,latency,errors,traffic])

def generate():
    for split,count,seed in [('train',1000,41),('calibration',300,42),('test',500,43)]:
        rng=np.random.default_rng(seed);data=normal(rng,count);labels=np.zeros(count,dtype=int)
        if split=='test':
            anomalies=normal(rng,100)
            for i in range(100):
                mode=i%4
                if mode==0:anomalies[i,0]=rng.uniform(87,100)
                elif mode==1:anomalies[i,1]=rng.uniform(240,700)
                elif mode==2:anomalies[i,2]=rng.uniform(0.15,0.8)
                else:anomalies[i,3]=rng.uniform(1400,2500)
            data=np.vstack([data,anomalies]);labels=np.r_[labels,np.ones(100,dtype=int)]
            order=rng.permutation(len(data));data=data[order];labels=labels[order]
        path=ROOT/'data'/f'{split}.csv';path.parent.mkdir(exist_ok=True)
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['record_id']+FEATURES+['is_anomaly']);writer.writeheader()
            for i,(values,label) in enumerate(zip(data,labels)):
                writer.writerow({'record_id':f'{split}-{i:04d}',**dict(zip(FEATURES,map(lambda x:round(float(x),5),values))),
                                 'is_anomaly':int(label)})

if __name__=='__main__':generate()
