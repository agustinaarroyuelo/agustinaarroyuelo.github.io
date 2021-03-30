---
yout: post
title:  "Circular Traceplot for Matplotlib"
date:   2020-08-7 16:00 +0530
categories: jekyll update
---

Hi everyone! This week I worked on adapting ArviZ' `plot_trace` function to plot circular variables. This function is mostly based on the modifications done in `plot_dist` to obtain a circular KDE. I basically added the necessary arguments to `plot_trace` to handle circular variables. You can take a look at the code in this [PR](https://github.com/arviz-devs/arviz/pull/1336).

Something that might be interesting to notice is that `fig, ax = plt.subplots()` can not be used, because in that case every subplot must have the same projection. In a circular traceplot only the circular variables need circular projections. Consequently, the plots must be added one by one using:

```
fig = plt.figure()
spec = gridspec.GridSpec(figure=fig)
fig.add_subplot(polar=is_circular)
```

Have a great weekend :)