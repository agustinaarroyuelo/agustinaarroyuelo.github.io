---
yout: post
title:  "Integrating SMC-ABC sampler into PyMC3"
date:   2018-07-06 16:00 +0530
categories: jekyll update
---





```python
%matplotlib inline
import numpy as np
import pymc3 as pm
import arviz as az
from pymc3.step_methods import smc_ABC
from tempfile import mkdtemp
test_folder = mkdtemp(prefix='SMC_TEST')
```

    /home/agustina/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters



```python
data = np.random.normal(0, 5, 1000)
```


```python
epsilon=np.linspace(1, 0.5, 8)
```


```python
step_kwargs = {'minimum_eps':0.55, 'iqr_scale': 2}
```


```python
with pm.Model() as model:
    a = pm.Normal('a', mu=0.5, sd=1)
    trace = pm.sample(step=pm.SMC_ABC(observed=data), step_kwargs=step_kwargs)
```

    /home/agustina/Documents/pymc3/pymc3/step_methods/smc_ABC.py:144: UserWarning: Warning: SMC is an experimental step method, and not yet recommended for use in PyMC3!
      warnings.warn(EXPERIMENTAL_WARNING)
    Adding model likelihood to RVs!
    /home/agustina/Documents/pymc3/pymc3/step_methods/smc_ABC.py:458: UserWarning: Warning: SMC is an experimental step method, and not yet recommended for use in PyMC3!
      warnings.warn(EXPERIMENTAL_WARNING)
    Init new trace!
    Sample initial stage: ...
    Epsilon: 0.7574 Stage: 0
    Initializing chain traces ...
    Sampling ...
    100%|██████████| 100/100 [00:00<00:00, 2460.48it/s]
    Epsilon: 0.7574 Stage: 1
    Initializing chain traces ...
    Sampling ...
    100%|██████████| 100/100 [00:03<00:00, 30.31it/s]
    Epsilon 0.3324 < minimum epsilon 0.5000
    Sample final stage
    Initializing chain traces ...
    Sampling ...
    100%|██████████| 500/500 [00:17<00:00, 29.23it/s]



```python
az.traceplot(trace);
```


![png]({{ "/assets/images/output_5_0_1.png" | absolute_url}})



```python
az.summary(trace)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-0.02</td>
      <td>0.23</td>
      <td>0.01</td>
      <td>-0.43</td>
      <td>0.44</td>
    </tr>
  </tbody>
</table>
</div>


