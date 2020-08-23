---
yout: post
title:  "GSoC with ArviZ 2020 Final Evaluation"
date:   2020-08-23 11:00 +0530
categories: jekyll update
---

In this blog post I will sum up the work I have done during this Google Summer of Code with ArviZ by linking merged and open PRs. 

* I added a new example to ArviZ that contains circular variables. [PR](https://github.com/arviz-devs/arviz/pull/1265), [commits]((https://github.com/arviz-devs/arviz/commit/ba2b6840c8859fffbea0b45e9bd474fd6e67acd4)).
* In the next week's work I focused on adding a circular histogram (Matplolib and Bokeh) and KDE plot (Matplotlib). [PR](https://github.com/arviz-devs/arviz/pull/1266), [commits](https://github.com/arviz-devs/arviz/commit/04201c5aa7bdde1b2ed9b05d453e45bd0670f2c0).
* I adapeted ArviZ `az.summary` function to better handle circular variables. [PR](https://github.com/arviz-devs/arviz/pull/1313), [commits](https://github.com/arviz-devs/arviz/commit/b6339eb56671049e9247d7ef592664dd8c83e874).
* I adapted ArviZ `az.plot_trace` function to circular variables. [PR](https://github.com/arviz-devs/arviz/pull/1336), [commits](https://github.com/arviz-devs/arviz/commit/ea60cd5c1364d36a351f344841421d3d94d420ae). When merged, this code caused some [issues](https://github.com/arviz-devs/arviz/issues/1360) with ArviZ traceplots. In this [PR](https://github.com/arviz-devs/arviz/pull/1361) I am sending a fix for them.

I am really happy that all this code got merged to master :)

Currently, I am working on adding the [separation plot](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-5907.2011.00525.x) to ArviZ: [PR](https://github.com/arviz-devs/arviz/pull/1359).



