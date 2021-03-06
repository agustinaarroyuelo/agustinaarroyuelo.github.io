---
yout: post
title:  "GSoC with ArviZ 2020 Final Evaluation"
date:   2020-08-23 11:00 +0530
categories: jekyll update
---

In this blog post I will sum up the work I have done during this Google Summer of Code with ArviZ by linking merged and open PRs. 

* I added a new example to ArviZ that contains circular variables. [PR](https://github.com/arviz-devs/arviz/pull/1265), [commits](https://github.com/arviz-devs/arviz/commit/ba2b6840c8859fffbea0b45e9bd474fd6e67acd4).
* I added a circular histogram (Matplolib and Bokeh) and KDE plot (Matplotlib). [PR](https://github.com/arviz-devs/arviz/pull/1266), [commits](https://github.com/arviz-devs/arviz/commit/04201c5aa7bdde1b2ed9b05d453e45bd0670f2c0).
* I adapted ArviZ `az.summary` function to better handle circular variables. [PR](https://github.com/arviz-devs/arviz/pull/1313), [commits](https://github.com/arviz-devs/arviz/commit/b6339eb56671049e9247d7ef592664dd8c83e874).
* I adapted ArviZ `az.plot_trace` function to circular variables. [PR](https://github.com/arviz-devs/arviz/pull/1336), [commits](https://github.com/arviz-devs/arviz/commit/ea60cd5c1364d36a351f344841421d3d94d420ae). When merged, this code caused some [issues](https://github.com/arviz-devs/arviz/issues/1360) with ArviZ traceplots. In this [PR](https://github.com/arviz-devs/arviz/pull/1361) I am sending a fix for it.

I am really happy that all this code got merged :)

Currently, I am working on adding the [separation plot](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-5907.2011.00525.x) ([PR](https://github.com/arviz-devs/arviz/pull/1359)), a very simple way for evaluating the performance of binary output models.

Some of the items on my original proposal that are left to do are adding the [quantile-dots plot](https://vega.github.io/vega/examples/quantile-dot-plot/#:~:text=A%20quantile%20dot%20plot%20represents,them%20in%20a%20dot%20plot.&text=If%20we%20are%20willing%20to,arrive%20at%20the%20bus%20stop.) and reviewing the computation of R-hat, ESS and MCE diagnostics to assure its correctness in a circular variable setting. 

I want to thank Google for hosting this amazing program and everyone at Arviz, specially to my mentors Osvaldo Martin, Ravin Kumar and Ari Hartikainen, for being so nice and helpfull and overall promoting a positive work environment. 