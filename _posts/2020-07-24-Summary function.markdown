---
yout: post
title:  "ArviZ Summary function"
date:   2020-07-24 16:00 +0530
categories: jekyll update
---

`az.summary` is one of the most commonly used functions in ArviZ. It displays a dataframe with summary statistics and/or diagnostics. Given that summary statistics are computed differently for circular variables, this function had a boolean argument for handling them: `include_circ`. When `True`, `az.summary` computed circular statistics for every variable in the model and displayed them on extra columns along the non-circular statistics dataframe. 
In the past few days, I changed this argument to `circ_var_names`, and now takes an iterable object. With this argument, the user can indicate which of the variables in the model are circular. Consequently, `az.summary` computes circular statistics only for said variables with no need to append extra columns.
You can take a look at the code in this [PR](https://github.com/arviz-devs/arviz/pull/1313).
See you in the next post! :)