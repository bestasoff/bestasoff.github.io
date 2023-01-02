---
layout: post
title:  How to fine tune VERY large model if it doesn’t fit on your GPU
date: 2022-04-11
description: Memory-efficient techniques to defeat the problem of “CUDA memory error..” during training
tags: formatting images
categories: Medium
---

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/post_1_cover.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Source: <a href="https://github.com/mryab/efficient-dl-systems/tree/main/week04_large_models">here</a>
</div>

The article is inspired by “Efficient Deep Learning Systems” [course](https://github.com/mryab/efficient-dl-systems) taught at Yandex School of Data Analysis.

**Prerequisites**: I suppose you know how neural network’s forward and backward passes work. It is exceptionally important to get the point of the article. As framework I’ll use [PyTorch](https://pytorch.org/).

So, it begins…
==============

You’ve probably been wondering why when you’re trying to implement some paper which uses some large model (aka _gpt-2-xl_) with >500 mln parameters you can’t even fit it on your GPU or use the same batch size as in the paper during training. Then maybe you gave up and started using a lighter version of the model or trained it on the smaller batch size, which did not allow you to get comparable with paper results.

**But,** there are some techniques that will help you to cope with described problem.

Let’s discuss some of the approaches and see how to use them to fine-tune 1.5 billion parameters _GPT-2-XL_ model in the end of the article.

Core of the problem
===================

Let’s understand the essence of the problem of lack of GPU memory needed to load the model onto GPU.

Suppose you have a model with _1e9_ FP32 (floating point 32 bit) parameters. You want to train that model on your lovely GPU using, for example, _Adam_ optimizer.

So, **let’s count**. I guess you’ll be shocked.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/post_1_memory_c.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Imagine you have _NVIDIA GeForce RTX 3060_ with 12 GB of memory. Firstly, _1e9_ FP32 parameters is about **4 GB** of your GPU memory. Also the same amount of memory will be reserved for gradients. So, we have already **8 GB** in total reserved not having started training yet and not having loaded the optimizer, because it is also not free in terms of memory. Adam optimizer needs to store **first** and **second** moments for each parameter, i.e. **8 GB** of additional memory.

Eventually, we have to have about **16 GB** of free GPU memory just to correctly load the model onto the GPU which, in our case, has only 12 GB of free memory. Looks terrible, yes?

But there are some approaches which we can use to try to solve the problem. Here are some of them below:

*   gradient accumulation / micro-batching;
*   gradient checkpointing;
*   model-parallel training;
*   pipelining;
*   tensor-parallelism;
*   mixed precision training;
*   memory offloading;
*   optimizer 8-bit quantization.

Today we will learn about them.

Let’s go!
---------

Gradient checkpointing
======================

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/post_1_2.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Source: <a href="https://github.com/mryab/efficient-dl-systems/tree/main/week04_large_models">here</a>
</div>

Overview
--------

What if a model is larger than GPU, i.e. we cannot fit batch size 1? There is a solution — gradient checkpointing. Let’s have a look at that concept.

For a simple feed-forward neural network with _n_ layers, the computation graph for obtaining gradients looks as follows:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/post_1_3.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Source: <a href="https://github.com/cybertronai/gradient-checkpointing">here</a>
</div>

The activations of the neural network layers correspond to the nodes marked with an _f_. During the forward pass all these nodes are evaluated in order. The gradient of the loss with respect to the activations and parameters of these layers is indicated by the nodes marked with _b_. During the backward pass, all these nodes are evaluated in the reversed order. The results obtained for the _f_ nodes are needed to compute the _b_ nodes, and hence all _f_ nodes are kept in memory after the forward pass. Only when backpropagation has progressed far enough to have computed all dependencies of an _f_ node, can it be erased from memory. This means that the memory required by simple backprop grows linearly with the number of neural net layers _n_.

Below there is the order in which these nodes are computed. The purple shaded circles indicate which of the nodes need to be held in memory at any given time.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img src="assets/img/post_1.gif">
    </div>
</div>
<div class="caption">
    Source: <a href="https://github.com/cybertronai/gradient-checkpointing">here</a>
</div>


Gradient checkpointing
----------------------

Simple backpropagation as described above is optimal in terms of computation: it only computes each node once. However, if we are willing to recompute nodes we can potentially save a lot of memory. We might for instance simply recompute every node from the forward pass each time we need it. The order of execution, and the memory used, then look as follows:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img src="assets/img/post_2.gif">
    </div>
</div>
<div class="caption">
    Source: <a href="https://github.com/cybertronai/gradient-checkpointing">here</a>
</div>

This strategy is optimal in terms of memory. However, note that the number of node evaluations now scales with _n²_, whereas it previously scaled as _n_: each of the _n_ nodes is recomputed on the order of _n_ times. The slowness of computation makes this method impractical for use in deep learning.

To strike a balance between memory and computation we need to come up with a strategy that allows nodes to be recomputed, but not too often. The strategy we use here is to mark a subset of the neural net activations as _checkpoint nodes_.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img src="assets/img/post_3.gif">
    </div>
</div>
<div class="caption">
    Source: <a href="https://github.com/cybertronai/gradient-checkpointing">here</a>
</div>

In this example the optimal choice is to mark every _sqrt(n)_\-th node as a checkpoint. This way, both the number of checkpoint nodes and the number of nodes between checkpoints are on the order of _sqrt(n)_, which means that the required memory now also scales with the order of _n_. So, the additional computation required by this strategy is equivalent to a single forward pass through the network.

Example:
--------

{% highlight python linenos %}

import torch
import torch.nn as nn
import torch.utils.checkpoint

class Layer(nn.Sequential):
  def __init__(self, *args, **kwargs):
      super(self, Layer).__init__(*args)
  
  def forward(self, *args):
    return super().forward(*args)
  
class CheckPointed(nn.Sequential):
  def forward(self, *args):
    return torch.utils.checkpoint.checkpoint(super().forward, *args)

model = nn.Sequential(
  CheckPointed(
    Layer(nn.Linear(128, 128), nn.ReLU()),
    Layer(nn.Linear(128, 128), nn.ReLU()),
    Layer(nn.Linear(128, 128), nn.ReLU())
  ),
  CheckPointed(
    Layer(nn.Linear(128, 128), nn.ReLU()),
    Layer(nn.Linear(128, 128), nn.ReLU()),
    Layer(nn.Linear(128, 128), nn.ReLU())
  )
)

{% endhighlight %}

After learning the details of gradient checkpointing let’s have a look at how easy is to use that concept in PyTorch:

Model-parallel training, pipelining, tensor-parallelism, memory offloading
==========================================================================

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/post_1_4.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Source: <a href="https://github.com/mryab/efficient-dl-systems/tree/main/week04_large_models">here</a>
</div>

It’s a very big and difficult topic, we will discuss it in the next posts.

Gradient accumulation / micro-batching
======================================

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/post_1_5.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Source: <a href="https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html">here</a>
</div>

Overview
--------

Deep learning models are getting bigger and bigger. It becomes difficult to fit such networks in the GPU memory. As a result, we are sometimes forced to use small batches during training, which may lead to a slower convergence and lower accuracy.

What is gradient accumulation?
------------------------------

When training a neural network, we usually divide our data into mini-batches. The network predicts batch labels, which are used to compute the loss with respect to the actual targets. Next, we perform backward pass to compute gradients and update model weights.

Gradient accumulation modifies the last step of the training process: instead of updating the network weights on every mini-batch, we can save gradient values, proceed to the next mini-batch and add up the new gradients to previously saved. The weight update is then done only after several mini-batches have been processed by the model.

Gradient accumulation helps to imitate a larger batch size. Imagine you want to use 64 images in one mini-batch, but “CUDA memory error…” once you go beyond the size of 8. In that case, you can use batches of 8 images and update weights once after 64 / 8 = 8 batches being processed by the model. If you accumulate gradients from each of these 8 batches, the results will be (almost) the same and you will be able to perform training! Yoah!

Example:
--------

{% highlight python linenos %}

# loop through batches
for inputs, targets in data_loader:

    inputs = inputs.to(device)
    targets = targets.to(device)

    # forward pass
    preds = model(inputs)
    loss  = criterion(preds, targets)

    # backward pass
    loss.backward()

    # weights update
    optimizer.step()
    optimizer.zero_grad()

{% endhighlight %}

The standard training loop without gradient accumulation usually looks like this:

In PyTorch gradient accumulation can be done very easily. You should do the step of your optimizer once `accumulation_steps` mini-batches have been processed by your model. Also you can divide the running loss by `accumulation_steps` depending on the nature of your loss function:

{% highlight python linenos %}

# batch accumulation parameter
accumulation_steps = 8 # we want to do model's update only after 64 images being processed

# loop through enumaretad batches
for batch_idx, (inputs, targets) in enumerate(data_loader):

    # extract inputs and labels
    inputs = inputs.to(device)
    targets = targets.to(device)

    # forward pass
    preds = model(inputs)
    loss  = criterion(preds, targets)

    # normalize loss to account for batch accumulation
    loss = loss / accumulation_steps

    # backward pass
    loss.backward()

    # weights update
    if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(data_loader)):
        optimizer.step()
        optimizer.zero_grad()

{% endhighlight %}

Beautiful, yes? The gradients are computed when we call `loss.backward()` and are accumulated by PyTorch until we call `optimizer.zero_grad()`.

Important
---------

You should note that some network architectures use batch-specific operations, i. e. BatchNorm and therefore it may yield slightly different results when using the same batch size with and without gradient accumulation.

Mixed-precision training
========================

Overview
--------

Mixed-precision training means converting some or all parameters which are _FP32_ numbers to smaller formats such as FP16, TF16 (tensor float), BF16 (bfloat).

Key benefits
------------

Key benefits of mixed-precision training are:

*   Reduced memory usage;
*   Faster performance (due to higher arithmetic intensity or smaller communication footprint);
*   Can use specialized hardware for even faster computation;

But now we are interested only in the first advantage — reducing memory usage. Let’s see how to do it with PyTorch models.

Example:
--------

{% highlight python linenos %}

import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, *args):
    super(self, Model).__init__(*args)
    self.model = nn.Sequential(
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 20)
    )
    
  def forward(self, x):
    return self.model(x)
  
fp32_model = Model() # FP32 model
fp16_model = Model().half() # just call .half() method to obtain FP16 version of the defined model

{% endhighlight %}

As the result, after doing `.half()` , the model becomes 2 times smaller.

Different formats the model can be converted (i.e. BF16, TF16) and loss scaling we will discuss in future posts.

But you should remember that there are some operations that cannot be done in FP16, i.e. `Softmax`. PyTorch has `torch.autocast` which helps to process these situations.

Optimizer 8-bit quantization
============================

Increasing model size is an effective way to achieve better performance. However, training such large models requires storing the model, gradient, and state of the optimizer (e.g., exponentially smoothed sum and squared sum of previous gradients for Adam), all in a fixed amount of available memory.

Going from 32-bit optimizers to 8-bit optimizers reduces the range of possible values from **2³²** values to just **2⁸= 256**. It makes a huge impact on the amount of memory to be reserved by the optimizer.

[Research](https://arxiv.org/pdf/2110.02861.pdf) presents new 8-bit Adam optimizer which “_maintains 32-bit performance at a fraction of the original memory footprint_”. That’s what the authors say in their paper:

> Our 8-bit optimizers have three components: (1) block-wise quantization that isolates outliers and distributes the error more equally over all bits; (2) dynamic quantization, which quantizes both small and large values with high precision; and (3) a stable embedding layer to improve stability during optimization for models with word embeddings.
> 
> With these components, performing an optimizer update with 8-bit states is straightforward. We dequantize the 8-bit optimizer states to 32-bit, perform the update, and then quantize the states back to 8-bit for storage. We do this 8-bit to 32-bit conversion element-by-element in registers, which means no slow copies to GPU memory or additional temporary memory are needed to perform quantization and dequantization. For GPUs, this makes 8-bit optimizers faster than regular 32-bit optimizers…

Let’s glance at the inspiring results of the usage of 8-bit Adam:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/post_1_6.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Source: <a href="https://arxiv.org/pdf/2110.02861.pdf">here</a>
</div>

As we can see, utilising quantized Adam saves **about 8.5 GB** of GPU memory. **Looks fantastic!**

Now that we have understood the usefulness of using it, let’s take a look at how to use it from python.

[Bitsandbytes](https://github.com/facebookresearch/bitsandbytes) package by Facebook is a lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers and quantization functions. It allows us to use _8-bit Adam_.

Example:
--------

{% highlight python linenos %}

import bitsandbytes as bnb
import torch

class Model(torch.nn.Module):
  def __init__(self):
    super(self, Model).__init__()
    self.model = torch.nn.Linear(128, 1)
    
  def forward(self, inp):
    return self.model(inp)

model = Model()

# adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # comment out old optimizer
adam = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # add bnb optimizer
adam = bnb.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995), optim_bits=8) # equivalent

# ... learning loop

{% endhighlight %}

As you can see above, the usage of quantized optimizer is pretty simple, but the result of it is gianormous.

Combining all above approaches to fine-tune _GPT-2-XL_ on GPU
=============================================================

Eventually, as we learned all above methods, let’s utilise them to solve real problem. We have to fine-tune GPT-2-XL model with > 1.5 billion parameters. Obviously, it can’t be loaded on the _NVIDIA GeForce RTX 3060_ GPU with 12 GB of memory.

Let’s list all methods we can use:

*   Gradient checkpointing;
*   Mixed-precision training (I do a trick: I use two samples of the same model. First is `.half`\-ed and loaded onto GPU, let’s name it `gpu_model`. Second is just on the CPU, let’s name it `cpu_model`. We evaluate GPU model, then load the gradients from `gpu_model`on the `cpu_model`, then do `optimizer.step()`, load updated parameters onto the `gpu_model`);
*   Gradient accumulation with batch\_size=64, minibatch\_size=4. Don’t forget to scale loss by `accumulation_steps`;
*   8-bit Adam optimizer.

Let’s utilise them all. Look at the code:

<code will be available soon>

As the result, utilisation of all above methods allowed us to fine-tune 16GB GPT-2-XL model on our GPU. I think it’s tremendous!

Conclusion
==========

In this post you learned key concepts of efficient memory usage which can be used in various hard tasks such as presented above.

We will discuss other concepts in the future posts.

Thank you for reading the article!

References:
===========

*   8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION, [paper](https://arxiv.org/pdf/2110.02861.pdf);
*   bitsandbytes, [git](https://github.com/facebookresearch/bitsandbytes);
*   gradient-checkpointing, [git](https://github.com/cybertronai/gradient-checkpointing);
*   mixed-precision training, [git](https://github.com/mryab/efficient-dl-systems/tree/main/week05_fast_pipelines);
