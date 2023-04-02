---
layout: page
title: Pix2Pix GAN distillation
description: Implementation of "Teachers Do More Than Teach. Compressing Image-to-Image Models" paper
img: assets/img/project_1.jpeg
importance: 1
category: work
---

Teacher generator is being pruned using binary search over BatchNorm scaling factors to meet certain computational budget requirements and trainer using Kernel Alignment distillation loss and GAN losses.

Computational budget is computed through the calculation of MAC operation and dividing it by the desired fraction. 

As a result student model learns to imitate the outputs of the teacher model but with 10x smaller compute.

<a href="https://github.com/bestasoff/pix2pix_distill">Link</a> to github for more details.