"""Explicit synthetic SCM; no records describe real people."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
import json,hashlib,datetime

def generate(n, gamma, seed):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=n); G=rng.binomial(1,.5,n); U=rng.normal(size=n)
    p=expit(-.2+.6*X+.4*G+gamma*U); A=rng.binomial(1,p)
    eM=rng.normal(size=n); eY=rng.normal(scale=2,size=n); eC=rng.normal(size=n); epre=rng.normal(size=n)
    M=.7*A+.4*X+.6*U+eM; C=.8*A+.8*U+eC
    Y=(2+G)*A+.8*M+1.5*X+G+U+eY
    Ypre=.8*X+.5*G+U+epre
    Y0=.8*(.4*X+.6*U+eM)+1.5*X+G+U+eY; Y1=Y0+2.56+G
    assert np.allclose(Y,np.where(A==1,Y1,Y0))
    return pd.DataFrame({'X':X,'G':G,'A':A,'M':M,'C':C,'Y':Y,'Ypre':Ypre,'U':U,'true_propensity':p,'Y0':Y0,'Y1':Y1})

if __name__=='__main__':
    base=Path(__file__).resolve().parent; protocol=json.loads((base/'protocol.json').read_text()); raw=base/'data/raw'; raw.mkdir(parents=True,exist_ok=True)
    oracle=generate(protocol['n'],protocol['confounding_strength'],protocol['seed'])
    oracle.to_csv(raw/'oracle.csv',index=False)
    oracle[['X','G','A','M','C','Y','Ypre']].to_csv(raw/'observed.csv',index=False)
    (raw/'generator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir() if p.is_file() and p.name!='manifest.json'}
    manifest={'kind':'explicitly synthetic structural causal model','version':'1.0','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':protocol['seed'],'n':protocol['n'],'gamma':protocol['confounding_strength'],'files':files,'source':'generate_data.py; no external observations or real participants','oracle_columns':'U, true_propensity, Y0, Y1 are simulation truth, unavailable to the observed-data estimator'}
    (raw/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Generated synthetic data:',len(oracle))
