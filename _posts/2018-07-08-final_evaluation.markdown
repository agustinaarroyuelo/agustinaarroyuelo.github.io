---
yout: post
title:  "Module for Approximate Bayesian Computation-Final Evaluation"
date:   2018-07-08 16:00 +0530
categories: jekyll update
---

# Module for Approximate Bayesian Computation

# Google Summer of Code - Final Evaluation

This blog post contains a detailed review of the development of the Google Summer of Code  of the project [Module for Approximate Bayesian Computation](https://storage.googleapis.com/summerofcode-prod.appspot.com/gsoc/core_project/doc/6546736076554240_1521818788_GSoC_proposal_-_Agustina_Arroyuelo.pdf?Expires=1533748504&GoogleAccessId=summerofcode-prod%40appspot.gserviceaccount.com&Signature=G7P53qJeot85XEiivQsWGyRg33kUpzMDJ7nAQiBKD%2BFXW%2F2Sypto8rJ5JYj9gqEGwrAY5BBXUjwit5cFViQtt3dgR3zz3wIxyJ5yfEdXY0asRTf8Hzmt38HXYnFvuOdmMmRNkOATMCSoYI0r%2BkQijp7dZw4Nucw%2FPBXLKgxcKvIIEvnin%2BQlvwUyCTCSvAzF3q4iI9gRmcUgF12NrUQm57MR1cYF24XghJAq24HoidYTHoh4Hgb5ArCTLZxWKWrIDya1MuSvbk1fvQEZuYRzgupKzNSPWwqZJCRqN7TiNZ3zaeQemNwGwvBKmUDDhdfy8X%2FnZRCxJExgRRTiv%2BAupQ%3D%3D). I will go over key points of the project and provide code examples and links to commits that show the current state of the implementation. Along the way I will point out challeging points and future work to be done. Finally I will test the work product against a common example.


```python
import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)
data = np.random.normal(loc=0, scale=1, size=1000)
plt.style.use(['bmh'])
```

# Project Abstract
Approximate Bayesian Computation (ABC) algorithms, also called likelihood free inference techniques, are a family of methods that can perform inference without the need to define a _likelihood function_. Additionally, the ABC approach has proven to be successful over likelihood based methods in several. We propose to implement a module for ABC in PyMC3, specifically Sequential Monte Carlo-ABC (SMC-ABC). Our work will signify a meaningful increase in the spectrum of models that PyMC3 will be able to perform.

# Main Objective
This project's objective is to implement an ABC module in PyMC3 on the basis of the current implementation of the Sequential Monte-Carlo Algorithm. 

[Link to current PyMC3's SMC code](https://github.com/pymc-devs/pymc3/blob/6fd230fdd032744b1d30a9485403ff0a59288906/pymc3/step_methods/smc.py)

[Link to a recent PR refactoring the SMC code](https://github.com/pymc-devs/pymc3/pull/3124)



# What was done
 
* In this project a complete refactorization of the Sequential Monte Carlo code was made.
* We defined the SMC_ABC class with attributes that are particularly necessary for ABC. [Link to SMC_ABC class code](https://github.com/agustinaarroyuelo/pymc3/blob/1c6b32794162364b3619225ae278844be365da99/pymc3/step_methods/smc_ABC.py#L23)
* A new PyMC3 distribution was added, the Simulator. [Link to simulator](https://github.com/agustinaarroyuelo/pymc3/blob/smcabc/pymc3/distributions/simulator.py)

* We defined a function to manage epsilon thresholds. [Link to epsilon computation function](https://github.com/agustinaarroyuelo/pymc3/blob/1c6b32794162364b3619225ae278844be365da99/pymc3/step_methods/smc_ABC.py#L247)
* We defined a function to compute summary statistics. [Link to summary statistics computation code](https://github.com/agustinaarroyuelo/pymc3/blob/1c6b32794162364b3619225ae278844be365da99/pymc3/step_methods/smc_ABC.py#L308)
* We defined functions to manage the different distance metrics available to the user. [Link to distance metrics code](https://github.com/agustinaarroyuelo/pymc3/blob/1c6b32794162364b3619225ae278844be365da99/pymc3/step_methods/smc_ABC.py#L340)
* With the components metioned above, we defined a sampling function that applies a rejection kernel. [Link to sampling function](https://github.com/agustinaarroyuelo/pymc3/blob/1c6b32794162364b3619225ae278844be365da99/pymc3/step_methods/smc_ABC.py#L80)

# What is left to do
* This implementation would need several improvements in terms of performance
* In many cases the sampler encounters issues with covariance matrix computing or low or zero acceptance rates. 

# Links to core commits

### These commits contain most fo the new code.

[Refactor SMC ABC code](https://github.com/pymc-devs/pymc3/commit/5c2b1361b835b360f14488b808e4b05d90ca2fbb)

[Add simulator distribution and distance functions](https://github.com/pymc-devs/pymc3/commit/6804a847ef8b81a35cb2eb241ce29fb552c4cefd)

[Refactor posterior and proposal distributions](https://github.com/pymc-devs/pymc3/commit/90083e0099941174a2497028ffff263665638e6e)

[Final changes to epsilon schedule computation](https://github.com/pymc-devs/pymc3/commit/f4fada25ec922f51cd1ed1fbb42d1998753bd9cf)



# Main development....


## _How do we define a PyMC3 model without a likelihood?_
This was one of the first challenges we encountered, given that the conceptual basis of an ABC module is to provide the user with the API for perfoming inference on a model with no likelihood function. 

ABC iteratevly compares the summary statistics computed from simulated data, with the same summary statistics computed on the observed data, which presents the need for a variable inside the model that stores the observed data. This variable tipically is the _likelihood_. 

In this SMC-ABC implemetation we constructed the _pm.Simulator()_ distribution. This is a dummy distribution that only stores the observed data and a function to compute the simulated data.

## Defining a _Simulator_ distribution
[Link to simulator](https://github.com/agustinaarroyuelo/pymc3/blob/smcabc/pymc3/distributions/simulator.py)

This is a fraction of the simulator code:

```python 
class Simulator(NoDistribution):

    def __init__(self, function, *args, **kwargs):
   
        self.function = function
        observed = self.data```


As you can see, this variable stores the Simulator function and the observed data. You can define it this way:

```python 
simulator = pm.Simulator('simulator', function, observed=data)```


## _How are Summary Statistics computed_?
The user can choose between a predefined set of summary statistics. As the argument for this function is a list, a combination of summary statistics can be used. The user can define it's own summary statistic function and pass it to the sampler. 


```python

def get_sum_stats(data, sum_stat=['mean']):
    """
    Parameters:
    -----------
    data : array
        Observed or simulated data
    sum_stat : list
        List of summary statistics to be computed. Accepted strings are mean, std, var. 
        Python functions can be passed in this argument.

    Returns:
    --------
    sum_stat_vector : array
        Array contaning the summary statistics.
    """
    if data.ndim == 1:
        data = data[:,np.newaxis]
    sum_stat_vector = np.zeros((len(sum_stat), data.shape[1]))

    for i, stat in enumerate(sum_stat):
        for j in range(sum_stat_vector.shape[1]):
            if stat == 'mean':
                sum_stat_vector[i, j] =  data[:,j].mean()
            elif stat == 'std':
                sum_stat_vector[i, j] =  data[:,j].std()
            elif stat == 'var':
                sum_stat_vector[i, j] =  data[:,j].var()
            else:
                sum_stat_vector[i, j] =  stat(data[:,j])

    return np.atleast_1d(np.squeeze(sum_stat_vector))
```

Default summary statistic is mean:


```python
get_sum_stats(data)
```




    array([-0.04784994])



Here we compute the mean, standard deviation and variance of a one dimensional array:


```python
get_sum_stats(data, sum_stat=['mean', 'std', 'var'])
```




    array([-0.04784994,  1.00437194,  1.008763  ])




```python
# custom summary statistic function
def custom_f(data):
    return np.square(data.mean()+data.var())
```

If the data has more than one dimension, applying a custom defined function:


```python
get_sum_stats(data.reshape(500,2), sum_stat=['mean', 'std', 'var', custom_f])
```




    array([[-0.01314941, -0.08255046],
           [ 1.00339486,  1.00414964],
           [ 1.00680125,  1.0083165 ],
           [ 0.98734398,  0.85704276]])



In this dataframe is easier to observe how the summary statistics are displayed:


```python
sum_stats = get_sum_stats(data.reshape(500,2), sum_stat=['mean', 'std', 'var', custom_f])
pd.DataFrame(sum_stats, index=['mean', 'std', 'var', custom_f], columns=['feature1', 'feature2'])
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
      <th>feature1</th>
      <th>feature2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>-0.013149</td>
      <td>-0.082550</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.003395</td>
      <td>1.004150</td>
    </tr>
    <tr>
      <th>var</th>
      <td>1.006801</td>
      <td>1.008316</td>
    </tr>
    <tr>
      <th>&lt;function custom_f at 0x7f106699ed08&gt;</th>
      <td>0.987344</td>
      <td>0.857043</td>
    </tr>
  </tbody>
</table>
</div>



This functions is used internallly for the sampler for computing distances.

[Link to summary statistics computation code](https://github.com/agustinaarroyuelo/pymc3/blob/1c6b32794162364b3619225ae278844be365da99/pymc3/step_methods/smc_ABC.py#L308)

##  _Distance metrics_
Distance metrics are an argument of the SMC_ABC class. The user can choose any of the following distance metrics :
* absolute difference
* sum of squared distance
* mean absolute error
* mean squared error
* euclidean

Once the argument is read it is used in the rejection kernel

[Link to distance metrics code](https://github.com/agustinaarroyuelo/pymc3/blob/1c6b32794162364b3619225ae278844be365da99/pymc3/step_methods/smc_ABC.py#L340)

## _Epsilon thresholds computation_
An SMC-ABC sampler runs across a series of acceptance-rejection thresholds called epsilon. 
On this implementation the user can provide a sequence in the form of a list, otherwise they are computed taking a factor of the inter quantile range of the last simulated data.

[Link to epsilon computation function](https://github.com/agustinaarroyuelo/pymc3/blob/1c6b32794162364b3619225ae278844be365da99/pymc3/step_methods/smc_ABC.py#L247)

## _Making all of these components work toghether_

This is the [sampler](https://github.com/agustinaarroyuelo/pymc3/blob/1c6b32794162364b3619225ae278844be365da99/pymc3/step_methods/smc_ABC.py#L80) SMC ABC function that combines the components above. 

# Examples

# A trivial example
Trying to estimate the mean and standard deviation of normal data


```python
# true mean and std
data.mean(), data.std()
```




    (-0.04784993519517787, 1.004371943957994)




```python
def normal_sim(a, b):
    return np.random.normal(a, np.abs(b), 1000)
```


```python
with pm.Model() as example:
    a = pm.Normal('a', mu=0, sd=5)
    b = pm.HalfNormal('b', sd=1)
    s = pm.Simulator('s', normal_sim, observed=data)
    trace_example = pm.sample(step=pm.SMC_ABC())
```

    Using absolute difference as distance metric
    Using ['mean'] as summary statistic
    Sample initial stage: ...
    Sampling stage 0 with Epsilon 1.915432
    100%|██████████| 500/500 [00:00<00:00, 836.37it/s]
    Sampling stage 1 with Epsilon 0.939661
    100%|██████████| 500/500 [00:00<00:00, 717.28it/s] 
    Sampling stage 2 with Epsilon 0.452645
    100%|██████████| 500/500 [00:00<00:00, 710.28it/s]



```python
pm.summary(trace_example)
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
      <td>-0.091348</td>
      <td>0.236713</td>
      <td>0.009575</td>
      <td>-0.449605</td>
      <td>0.370533</td>
    </tr>
    <tr>
      <th>b</th>
      <td>1.161391</td>
      <td>1.935439</td>
      <td>0.080310</td>
      <td>0.028291</td>
      <td>3.522031</td>
    </tr>
  </tbody>
</table>
</div>




```python
_, ax = plt.subplots(figsize=(12,6))
pm.kdeplot(data, label='True data', ax=ax, marker='.')
pm.kdeplot(normal_sim(trace_example['a'].mean(), trace_example['b'].mean()), ax=ax)
ax.legend();
```

![png]({{ "/assets/images/output_33_0.png" | absolute_url}})


# Lotka–Volterra
In this example we will try to find parameters for the Lotka-Volterra equations. A common competition model for describing how the number of individuals of each species changes when more than one species uses the same resource (Otto, Day, 2007).


```python
from scipy.integrate import odeint
```


```python
# Definition of parameters
a = 1.
b = 0.1
c = 1.5
d = 0.75

X0 = [10., 5.]
size = 1000
time = 15
t = np.linspace(0, time, size)

def dX_dt(X, t, a, b, c, d):
    """ Return the growth rate of fox and rabbit populations. """

    return np.array([ a*X[0] -   b*X[0]*X[1] , 
                  -c*X[1] + d*b*X[0]*X[1] ])
```


```python
def add_noise(a, b, c, d):
    noise = np.random.normal(size=(size, 2))
    simulated = simulate(a, b, c, d)
    simulated += noise
    indexes = np.sort(np.random.randint(low=0, high=size, size=size))    
    return simulated[indexes]
```


```python
def simulate(a, b, c, d): 
    return odeint(dX_dt, y0=X0, t=t, rtol=0.1, args=(a, b, c, d))
```


```python
observed = add_noise(a, b, c, d )
_, ax = plt.subplots(figsize=(16,7))
ax.plot(observed, 'x')
ax.set_xlabel('time')
ax.set_ylabel('population');
```

![png]({{ "/assets/images/output_39_0.png" | absolute_url}})



```python
with pm.Model() as model:
    a = pm.HalfNormal('a', 1, transform=None)
    b = pm.HalfNormal('b', 0.1, transform=None)
    c = pm.HalfNormal('c', 1.5, transform=None)
    d = pm.HalfNormal('d', 0.75, transform=None)
    simulator = pm.Simulator('simulator', simulate, observed=observed)
    trace = pm.sample(step=pm.SMC_ABC(n_steps=50, min_epsilon=70, iqr_scale=3), draws=500)
```

    Using absolute difference as distance metric
    Using ['mean'] as summary statistic
    Sample initial stage: ...
    Sampling stage 0 with Epsilon 8.542185
     92%|█████████▏| 458/500 [00:09<00:00, 46.22it/s]/home/agustina/anaconda3/lib/python3.6/site-packages/scipy/integrate/odepack.py:218: ODEintWarning: Excess work done on this call (perhaps wrong Dfun type). Run with full_output = 1 to get quantitative information.
      warnings.warn(warning_msg, ODEintWarning)
    100%|██████████| 500/500 [00:10<00:00, 47.18it/s]



```python
pm.traceplot(trace);
```

![png]({{ "/assets/images/output_41_0.png" | absolute_url}})


```python
_, ax = plt.subplots(figsize=(16,7))
ax.plot(observed, 'x')
ax.plot(simulate(trace['a'].mean(), trace['b'].mean(), trace['c'].mean(), trace['d'].mean()))
ax.set_xlabel('time')
ax.set_ylabel('population');
```

![png]({{ "/assets/images/output_42_0.png" | absolute_url}})



```python
pm.summary(trace)
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
      <td>0.974816</td>
      <td>0.491178</td>
      <td>0.021652</td>
      <td>0.212000</td>
      <td>1.815388</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.115175</td>
      <td>0.051615</td>
      <td>0.002292</td>
      <td>0.026255</td>
      <td>0.213313</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1.567371</td>
      <td>0.829644</td>
      <td>0.035822</td>
      <td>0.126548</td>
      <td>3.232692</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.762891</td>
      <td>0.389715</td>
      <td>0.016962</td>
      <td>0.163140</td>
      <td>1.554152</td>
    </tr>
  </tbody>
</table>
</div>

#Ending Remarks



