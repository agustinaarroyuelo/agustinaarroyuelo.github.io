---
yout: post
title:  "Circular Plots PR"
date:   2020-07-10 16:00 +0530
categories: jekyll update
---


In the last couple of weeks I have been working on this [PR](https://github.com/arviz-devs/arviz/pull/1266). It adds the functionalities for plotting circular histograms and KDE plots with ArviZ. As I mentioned in the previous blog post, the Bokeh plot requiered to place ticks and ticklabels manually. I defined a [function](https://github.com/arviz-devs/arviz/blob/8a995127e42873d6f48574f9d981ea2a59eb0d74/arviz/plots/plot_utils.py#L722) to do so. This way, the `plot_histogram` function looks more tidy. I also updated the tests to cover the new `is_circular` argument.

This week I participated in ArviZ lab-meeting. My fellow GSoC'ers and I gave a quick update on our respective projects. The lab-meeting in general was really interesting and everyone super nice. 

Have a great weekend! :)
