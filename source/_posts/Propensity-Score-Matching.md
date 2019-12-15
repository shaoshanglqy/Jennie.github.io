---
title: Propensity Score Matching
date: 2019-12-15 21:38:33
tags: ['Data Science','Python']
categories: ['Data Science']
---

## Description

Propensity Score Matching is a Sample Matching Method, it can effectively eliminate the effecting facotors between different groups and avoid the selecttion bias between two sample groups when we can not conduct random sampling.

## Algorithm

Reuce X to one dimension through dimension reduction method, and get the Propensity Socre of every sample through Logistic Regression, then we match the samples, the most used way is Newares Neighbor matching, NNM.