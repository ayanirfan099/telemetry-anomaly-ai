"""Unsupervised anomaly scoring, separate threshold calibration and PSI drift reports."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,precision_score,recall_score
from generate_data import FEATURES,ROOT

def validate_record(record):
    if not isinstance(record,dict):raise ValueError('Each record must be a JSON object')
    missing=set(FEATURES)-record.keys()
    if missing:raise ValueError('Missing features: '+', '.join(sorted(missing)))
    values=[]
    for name in FEATURES:
        value=record[name]
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ValueError(name+' must be a finite number')
        if value<0:raise ValueError(name+' cannot be negative')
        if name=='cpu_percent' and value>100:raise ValueError('cpu_percent cannot exceed 100')
        if name=='error_rate' and value>1:raise ValueError('error_rate must be a fraction from 0 to 1')
        if name in {'latency_ms','requests_per_minute'} and value>1e9:raise ValueError(name+' exceeds the demo input limit')
        values.append(float(value))
    return values

def read_csv(path):
    with Path(path).open(newline='') as handle:rows=list(csv.DictReader(handle))
    if not rows:raise ValueError('Dataset is empty')
    records=[{name:float(row[name]) for name in FEATURES} for row in rows]
    values=np.array([validate_record(row) for row in records])
    return rows,values

class TelemetryModel:
    def __init__(self,train_path=ROOT/'data/train.csv',calibration_path=ROOT/'data/calibration.csv'):
        train_rows,train=read_csv(train_path);calibration_rows,calibration=read_csv(calibration_path)
        if len(train)<50 or len(calibration)<30:raise ValueError('Need at least 50 training and 30 calibration records')
        self.train_ids={row['record_id'] for row in train_rows};self.calibration_ids={row['record_id'] for row in calibration_rows}
        self.fit_rows={tuple(row) for row in train};self.calibration_rows={tuple(row) for row in calibration}
        if self.train_ids & self.calibration_ids or self.fit_rows & self.calibration_rows:
            raise ValueError('Training and calibration splits overlap')
        self.model=IsolationForest(n_estimators=150,max_samples=256,random_state=42,contamination='auto',n_jobs=1)
        # The is_anomaly label is never part of the model features.
        self.model.fit(train)
        self.threshold=float(np.quantile(-self.model.score_samples(calibration),0.98))
        self.median=np.median(train,axis=0)
        self.spread=np.maximum(np.median(np.abs(train-self.median),axis=0)*1.4826,1e-6)
        self.reference=[]
        for i in range(len(FEATURES)):
            cuts=np.unique(np.quantile(train[:,i],np.linspace(0,1,11)))[1:-1]
            bins=np.r_[-np.inf,cuts,np.inf];counts=np.histogram(train[:,i],bins)[0]
            self.reference.append((bins,counts/len(train)))
        digest=hashlib.sha256(train.tobytes()+calibration.tobytes()+b'iforest-150-seed42-q98').hexdigest()
        self.model_id=digest[:16]

    def predict_batch(self,records):
        if not isinstance(records,list) or not 1<=len(records)<=500:
            raise ValueError('records must be an array containing 1–500 items')
        values=np.array([validate_record(record) for record in records])
        scores=-self.model.score_samples(values);results=[]
        for record,score in zip(values,scores):
            deviation=(record-self.median)/self.spread
            reasons=[{'feature':FEATURES[int(i)],'robust_deviation':round(float(deviation[i]),3)}
                     for i in np.argsort(-np.abs(deviation))[:2]]
            results.append({'is_anomaly':bool(score>self.threshold),'anomaly_score':round(float(score),6),
                            'threshold':round(self.threshold,6),'largest_deviations':reasons})
        drift=None
        if len(values)>=20:
            drift={}
            for i,(bins,expected) in enumerate(self.reference):
                actual=np.histogram(values[:,i],bins)[0]/len(values)
                p=np.maximum(expected,1e-6);q=np.maximum(actual,1e-6)
                p=p/p.sum();q=q/q.sum()
                drift[FEATURES[i]]=round(float(np.sum((q-p)*np.log(q/p))),6)
        return {'model_id':self.model_id,'results':results,'population_stability_index':drift,
                'notes':['Scores are anomaly rankings, not probabilities.',
                         'Largest deviations are descriptive, not causal model explanations.',
                         'PSI is computed only for batches of at least 20; small batches are noisy.']}

    def evaluate(self,test_path=ROOT/'data/test.csv'):
        rows,values=read_csv(test_path)
        if ({row['record_id'] for row in rows}&(self.train_ids|self.calibration_ids)
                or {tuple(row) for row in values}&(self.fit_rows|self.calibration_rows)):
            raise ValueError('Test data overlaps training or calibration data')
        truth=np.array([int(row['is_anomaly']) for row in rows])
        if set(truth)!={0,1}:raise ValueError('Evaluation requires binary labels with both classes')
        scores=-self.model.score_samples(values);predicted=scores>self.threshold
        return {'model_id':self.model_id,'dataset':'Synthetic service telemetry; generated independent train/calibration/test splits',
                'warning':'Artificial anomalies are easier than many real incidents; results do not establish production performance.',
                'threshold_policy':'98th percentile of anomaly scores on separate normal calibration data',
                'threshold':self.threshold,'roc_auc':float(roc_auc_score(truth,scores)),
                'average_precision':float(average_precision_score(truth,scores)),
                'precision':float(precision_score(truth,predicted,zero_division=0)),
                'recall':float(recall_score(truth,predicted,zero_division=0)),
                'false_positive_rate':float(np.mean(predicted[truth==0])),
                'confusion_matrix_labels':[0,1],'confusion_matrix':confusion_matrix(truth,predicted,labels=[0,1]).tolist(),
                'train_rows':len(self.train_ids),'calibration_rows':len(self.calibration_ids),'test_rows':len(rows)}
