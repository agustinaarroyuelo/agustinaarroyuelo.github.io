---
yout: post
title:  "Circular Histogram with Bokeh"
date:   2020-06-26 16:00 +0530
categories: jekyll update
---

Today I focused on adding a circular histogram for ArviZ using Bokeh as backend. I have worked with Bokeh in the past when developping ArviZ' pairplot function. Back then using Bokeh improved code readability quite a bit compared to Matplotlib. 

The first thing I learned when I started working on a circular histogram using Bokeh as backend, is that Bokeh does not support polar projection, as explained in this [issue](https://github.com/bokeh/bokeh/issues/657). This implied coding a few things that were automatic when using Matplotlib, for example: placing ticks and tick labels in the right positions. This is the result so far:

![png]({{ "/assets/images/first_bokeh_circ_hist.png" | absolute_url}})

In the near future I would like to improve on this function's flexibility and tidy up the code. You can take a look at it in this [PR](https://github.com/arviz-devs/arviz/pull/1266).

Have a great weekend! :)
