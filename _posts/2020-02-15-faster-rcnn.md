---
title: "Object Detection - Diving into Faster RCNN"
tags: ["object detection", "rcnn", "faster-rcnn"]
toc: true
toc_sticky: true
---

Last year, I had a chance to be involved in an Advanced Computer Vision class held by a non-profit organization. During the class, object detection is one of the fields that I found myself interested. This motivated me to write a series of blogs in order to understand better some famous approaches that has been applied in the field. Though, the idea has been postponed until now :v. The first part of this series is about Faster RCN, one of the state-of-the-art methods used for object detection. In this blog post, I will walk you through the details of Faster RCNN. Hopefully, at the end of this blog, you would figure out the way Faster RCNN works.

## A little warm-up
In object detection, we receive an image as input and localize bounding boxes, indicating various types of objects, as output.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/example.png" width="100%">
<p align="center">
source: <a href="https://highstonecorp.com/post/object-detection.html" target="_blank">https://highstonecorp.com/post/object-detection.html</a>
</p>

So what are bounding boxes? A bounding box is just a rectangle in the image. Its coordinates is defined as a tuple of $(x, y, w, h)$, where $(x, y)$ is the coordinate of the bounding box's center and $w, h$ is its width, height, respectively. A bounding box is said to be best-fit an object if it is the smallest rectangle that fully encloses the object like the figure below.
<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/bb.png" width="40%">

Hence, we should label best-fit bounding box for the best quality of supervised data. In the next part, we will discuss Faster RCNN in detail.

## Faster RCNN architecture
Faster RCNN, published in 2015, is the last of the RCNN trilogy (RCNN - Fast RCNN - Faster RCNN), which relies on proposed regions to detect objects.
Though, unlike its predecessors  which use `selective search` to find out the best regions, Faster RCNN makes use of neural network and "learn" to propose regions directly.
These proposed regions is then fed into another neural network to be refined once again.

Let us take a look at the overall architecture of Faster RCNN. The model comprises of $2$ modules

* The `region proposal module` takes feature map from a feature network and proposes regions.
* The `Fast RCNN detector module` takes those regions to predict the classes that the object belongs to.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/frcnn.png" width="95%">

Note that the feature network, which is VGG in the context of this blog, is shared between both modules. Also, to easily keep track of the story, let us follow a specific case in which we are given an image of shape $320\times400\times3$.
### Feature Shared Network

The original paper uses ZF-net and VGG-net as feature network.
Though, we only discuss VGG in the scope of this blog.
The VGG receives an input image and produce a feature map with reduced spatial size.
This size is determined by the net structure.
In the case of VGG, the image spatial size is reduced $16$ times at the output layer.
Hence, in our example, the feature map's shape  is $320/16 \times400/16\times512$, or $20 \times 25 \times 512$.
The number $512$ is due to the number of filters in the last layer of VGG.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/feature_map.png" width="55%">

The feature map is then used for both `region proposal network` and `region-based convolutional neural network`, which will be discussed later.
### Region Proposal Network (RPN)
The goal of RPN is to propose regions that highly contain object. In order to do that, given the feature map, RPN does
* generate a predefined number of fixed-size anchors based on the feature map
* predict the objectness of each of these anchors
* refine their coordinates

#### Predefined anchors
<!-- RPN accepts VGG feature map as input. -->

Specifically, for each pixel spatial location on the VGG feature map, we generate a predefined number of fixed size anchors.
The shape of these anchor boxes are determined by a combination of predefined scales and edge ratios.
In our example, if we use $3$ scales $64$, $128$, $256$ and $3$ edge ratios $1:1$, $1:2$, $2:1$, there will be $3*3=9$ type of anchors at each pixel location.
A total of $20 * 25 * 9 = 4500$ anchors will be generated as a result.
<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/anchor_boxes.png" width="100%">

It is important to note that even though anchor boxes are created based on the feature map's spatial location, they reference to the original input image,
in which anchor boxes generated from the same feature map pixel location are centered at the same point on the original input, as illustrated in this figure below.
<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/anchor.png" width="45%">

#### RPN architecture
The RPN is then designed to predict objectness of each anchor (classification) and refine its coordinates (regression).
It consists of $3$ layers: one convolutional layer with $512$ filters of size $3 \times 3$ followed by two sibling $1 \times 1$ convolutional layers.
These two sibling layers - one with $K$ filters and the other with $4K$ filters - allow for classification and regression, respectively.
$K$ is determined as the number of generated anchors at each feature map location.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/rpn.png" width="70%">

In our example when $K = 9$, after passing the VGG feature map through RPN, it produces a classification output with shape of $20 \times 25 \times 9$ and a regression output with shape of $20 \times 25 \times 36$. The total predictions of RPN will then have the shape of $20\times25\times45$, like the figure above.

#### Create labeled data for training RPN
##### Label for classification

Now, we need labeled data to train the RPN. For training classification task, each anchor box is labeled as
* positive if it contains object
* negative if it is background
* ignored if we want it to be ignored when training

based on the overlap with its nearest ground-truth bounding box.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/anchor_label.png" width="95%">
To be more specific, we use the famous IoU (Intersection over Union) to measure the overlap. Let $p$ denotes the IoU between current anchor box and its nearest ground-truth bounding box. Then, we need to decide two thresholds $p_{neg}$, $p_{pos}$ for labelling. The labelling rule is then detailed as follows
* If $p \geq p_{pos}$, label the bounding box as positive
* If $p \leq p_{neg}$, label it as negative
* If $p_{neg} < p < p_{pos}$, ignore it when training

##### Label for regression

The anchor box refinement is modeled as a regression problem, in which we predict the delta $({\color{red}{t_x, t_y, t_w, t_h}})$ for each anchor box.
This delta denotes the change needed to refine our predefined anchor boxes, as illustrated in this figure below

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/anchor_reg.png" width="50%">

Formally, we have

<div>
$$\begin{align}
\color{blue}{x} & = x_a + {\color{red}{t_x}}*w_a \\
\color{blue}{y} & = y_a + {\color{red}{t_y}}*h_a \\
\color{blue}{w} & = w_a * e^{\color{red}{t_w}} \\
\color{blue}{h} & = h_a * e^{\color{red}{t_h}}
\end{align}
$$
</div>

or
<div>
$$\begin{align}
\color{red}{t_x} & = ({\color{blue}{x}} - x_a) / w_a \\
\color{red}{t_y} & = ({\color{blue}{y}} - y_a) / h_a \\
\color{red}{t_w} & = log({\color{blue}{w}}/w_a) \\
\color{red}{t_h} & = log({\color{blue}{h}}/h_a)
\end{align}
$$
</div>

where $(x_a, y_a, w_a, h_a)$ denotes the anchor box's coordinates and $({\color{blue}{x, y, w, h}})$ denotes the refined box's coordinates.

To create data for anchor regression training, we calculate the "ground-truth" delta $({\color{red}{t_x^\*, t_y^\*, t_w^\*, t_h^\*}})$ based on each anchor box's coordinates $(x_a, y_a, w_a, h_a)$ and its nearest ground-truth bounding box's coordinates $({\color{blue}{x^\*, y^\*, w^\*, h^\*}})$.

<div>
$$
\begin{align}
\color{red}{t_x^*} & = ({\color{blue}{x^*}} - x_a) / w_a \\
\color{red}{t_y^*} & = ({\color{blue}{y^*}} - y_a) / h_a \\
\color{red}{t_w^*} & = log({\color{blue}{w^*}}/w_a) \\
\color{red}{t_h^*} & = log({\color{blue}{h^*}}/h_a)
\end{align}
$$
</div>

Among those generated anchor boxes, the positive anchors are probably outnumbered by the negative ones. Thus, to avoid imbalanced classification, only a specific number of anchor boxes is used for training. In our example, among $4500$ anchor boxes generated, assume that we have $500$ "positive" anchor boxes, $2000$ "negative" anchor boxes and $2000$ "ignored" anchor boxes. Then, we only chose $256$ anchor boxes for training the RPN, including $128$ boxes of each type ("positive" and "negative").

#### RPN losses
##### Regression Loss

The smooth L1 loss is used for regression training. Its formulation is as below
<div>
$$smooth_{L1}(x) =
\begin{cases}
0.5x^2 & \mbox{if} \;  \lvert x \rvert < 1, \\
\lvert x \rvert - 0.5 & \mbox{otherwise}.
\end{cases}
$$
</div>
where $x$ denotes the difference between prediction and ground truth $t  - {\color{blue}{t^*}}$.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/loss.png" width="60%">

The reason smooth L1 loss is preferred to L1 and L2 loss is because it can handle the problem of these two losses. Being quadratic for small values ($\lvert x \rvert < 1$) and linear for large values ($\lvert x \rvert \geq 1$), smooth L1 loss is now less sensitive to outliers than L2 loss and also does not suffer from the problem of L1 loss, which is not differentiable around zero.

##### Classification Loss
For RPN binary classification, the binary cross-entropy loss is used.

#### Use RPN to propose regions
After training, we use RPN to predict the bounding box coordinates at each feature map location.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/rpn_inf.png" width="60%">

<!-- $$\begin{align}
\color{blue}{x} & = x_a + \color{red}{t_x}*w_a \\
\color{blue}{y} & = y_a + \color{red}{t_y}*h_a \\
\color{blue}{w} & = w_a * e^{\color{red}{t_w}} \\
\color{blue}{h} & = h_a * e^\color{red}{t_h}
\end{align}$$ -->

Assume that the RPN predict $1000$ positive bounding boxes out of $4500$ anchor boxes. There are a lot of proposals. Hence, it is very likely that there are many bounding boxes referring to the same object, among those are predicted by RPN. This leads to redundant proposals, which can be eliminated by an algorithm known as `non max suppression`.

##### Non-max suppression

The idea of non max suppression is to filter out all but the box with highest confidence score
for each highly-overlapped bounding box cluster (like the figure below),
making sure that a particular object is identified only once.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/hepburn.png" width="50%">

Given a list of proposals along with their confidence score and a predefined overlap threshold,
the algorithm can be summarized as follows 
* Initialize a list $L$ to contain bounding boxes.
* Sort the list, denoted by $S$,  by confidence score in descending order
* Iterate through $S$, at each iteration
    * Compute the overlap between the current bounding box and the remain bounding boxes in $S$
    * Suppress all bounding boxes that have the computed overlap above the predefined threshold hold from $S$
    * Discard the current box from $S$, then move it to $L$
* Return $L$

After non max suppression, we obtain some "good" bounding boxes in the input image.
These boxes correspond with scaled regions in the VGG feature map.
Then, these feature map patches are extracted as proposed regions, as shown in the figure below

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/nms.png" width="65%">

### Region-based Convolutional Neural Network
Now we have proposed regions ready for the next phase. One notable problem arises here is that these proposed regions are not in the same shape, which make it difficult for neural network training. This is where we need RoI pooling layer to help construct fixed-size feature maps from these arbitrary-size regions.

#### RoI Pooling
To understand RoI pooling, let begin with a 2D example. No matter what the shape of the input slice is, a $2 \times 2$ RoI pooling layer always transform the input to the output of size $2 \times 2$ by
* Split the input into a $2 \times 2$ matrix of roughly equal regions
* Do max pooling on each region

like this figure below (given input of shape $4 \times 4$ or $5 \times 5$).
<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/pooling.png" width="75%">

##### RoI used in Faster RCNN
In Faster RCNN, we apply RoI pooling to a 3D proposed regions to obtain fixed-size regions.
In our example, if $7\times7$ RoI pooling is used, those fixed-size regions have the shape of $7\times7\times512$.
<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/roi.png" width="55%">

#### Detection Network
Those fixed-size feature maps from RoI pooling are then flattened and subsequently fed into a fully connected network for final detection.
The net consists of $2$ fully connected layers of $4096$ neurons, followed by other $2$ sibling fully connected layers - one
has $N$ neurons for classifying proposals and the other has $4\*(N - 1)$ neurons for bounding box regression,
where $N$ denotes the number of classes, including the background.
Note that when a bounding box is classified as background, regression is unneeded.
Hence, it makes sense that we only need $4*(N - 1)$ neurons for regression in total.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/dnet.png" width="70%">

In our example, each $7\times7\times512$ feature map is fed to the detection net to produce the classification output has size of $4$, and the regression output has size of $12$.

#### Labeled data for RCNN
##### Label for classification

Similar to the RPN, we make use of IoU metric to label data. Let $r$ now denotes the overlap between a refined anchor box produced by RPN and its nearest ground-truth anchor box. For each anchor box we label as follows
* if $r \leq r_{min}$, label the proposed anchor box as background.
* if $r_{min} < r < r_{max}$, ignore it when training.
* if $r \geq r_{max}$, label it as the class to which its nearest ground-truth box belongs.

where $r_{min}$ and $r_{max}$ are the two predefined thresholds.

##### Label for bounding box regression

For regression, we also calculate the "ground-truth" deltas $({\color{red}{t_x^\*, t_y^\*, t_w^\*, t_h^\*}})$ in the same fashion as those in RPN,
but now based on each refined anchor box's coordinates from the RPN $(x_r, y_r, w_r, h_r)$ and its nearest ground-truth
bounding box's coordinates $({\color{blue}{x^\*, y^\*, w^\*, h^\*}})$.

#### RCNN losses
RCNN also uses smooth  L1 loss for regression and categorical cross-entropy loss for classification.

Now, we are done walking through Faster RCNN. Its entire architecture can be pictured as follows

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/rcnn/full_net.png" width="90%">

__References__
1. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, 2015. ([arxiv](https://arxiv.org/abs/1506.01497))
