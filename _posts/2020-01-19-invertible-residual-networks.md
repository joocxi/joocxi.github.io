---
title: "Invertible Residual Networks for Generative Modeling"
tags: ["generative model", "invertible residual networks", "normalizing flows", "computer vision"]
toc: true
toc_sticky: true
---

If you have been involving in machine learning for a while, you should have known about residual networks,
which are proved to be powerful for image classification.
Yet, apart from classification, they can be made invertible by some simple tricks to be used in other machine learning tasks as well.
This family of residual networks called **Invertible Residual Networks** has been proposed  recently by J Behrmann, 2018.
In this blog post, I will walk you through the invention of invertible residual networks.

## The motivation
Classification only tells difference between data points. This is clearly not fulfilling for greedy human beings. We want a better understanding of data or, to be more specific, the data distribution itself. This underlying distribution of data is literally what the task of density estimation tries to estimate.

### Density estimation
Density estimation can be achieved through maximum likelihood estimation, in which we try to maximize the expected log-likelihood

<div>
$$
\mathbf{E}_{x \sim p_{data}(x)}{p(x; \theta)}
$$
</div>

where $p_{data}(x)$ denotes the empirical distribution of observed data
and $p(x; \theta)$ denotes our assumed parametric distribution (simply known as model distribution).

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/iresnet/dist.png" width="75%">

Yet, when data is complex and high dimensional, a problem arises.
It is hard to construct a parametric distribution which is not only expressive enough to capture the complexity of data but also tractable for maximum likelihood estimation.
This hurdle can be overcome with the help of normalizing flows, in which we rely on them to construct our more complex model distribution from a simple latent prior.
As a parametric bridge between these two distributions, normalizing flows allow for tractable data likelihood computation, which makes maximum likelihood estimation now possible.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/iresnet/norm_flows.png" width="95%">

### Normalizing Flows
Normalizing flows were first introduced to solve the problem of density estimation.
Though, it later became popular when introduced to deal with variational inference by [1].
The idea of normalizing flows is very simple that it transforms one distribution
to another arbitrarily complex distribution, through a sequence of invertible mapping functions.

#### A change of variables
But, let's first take a look at a change of variables rule, which forms the basis for normalizing flows.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/iresnet/var_change.png" width="60%">

Given a random variable $x$ with its density function known as $p(x)$, if we map this variable using
an invertible mapping $f$, so that $z = f(x)$ and $x = f^{-1}(z) \; \forall x, z,$ then $z$ is still a random variable.
Its normalized density function is then defined as follows

$$
p(z) = p(x) \left\lvert \det\frac{\partial f^{-1}}{\partial z} \right\rvert = p(x) {\left\lvert \det\frac{\partial f}{\partial x} \right\rvert}^{-1}, \tag{1}
$$

where the first equality is due to preservation of total probability in both domain; and the second equality follows from the inverse function theorem.

Taking logarithm of each side, we can rewrite $(1)$  as
<!-- $$\ln p(x) = \ln p(z) + \ln \left\lvert \det J_f(x) \right\rvert$$ -->

$$
\ln p(z) = \ln p(x) - \ln \left\lvert \det J_f(x) \right\rvert
$$

where $J_f$ denotes the Jacobian matrix of function $f$ evaluated at point $x$.

#### A normalizing flow
We can now form a normalizing flow by chaining together a finite sequence of these variable changes just described above.
As an example, let us consider a flow in figure 2, in which we have
$$z \equiv z_K = f_K \circ ... \circ f_2  \circ f_1 (x) \equiv F(x)$$

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/iresnet/flows.png" width="80%">

By consecutively applying variables change formula $(2)$, we get

<div>
$$
\begin{align}
\ln p(z) = \ln p(z_K) & = \ln p(z_{K-1}) - \ln \left\lvert \det J_{f_K}(z_{K-1})\right\rvert \\
& = \ln p(z_{K-2}) - \sum_{k=K-1}^{K}\ln \left\lvert \det J_{f_k}(z_{k-1})\right\rvert \\
& = \;... \\
& = \ln p(x) - \sum_{k=1}^{K} \ln \left\lvert \det J_{f_k}(z_{k-1})\right\rvert \\
\end{align}
$$
</div>

Continuing the derivation we get

<div>
$$
\begin{align}
\ln p(z) & = \ln p(x) - \ln \left\lvert \prod_{k=1}^{K} \det J_{f_k}(z_{k-1})\right\rvert \\
& = \ln p(x) - \ln \left\lvert \det \prod_{k=1}^{K} J_{f_k}(z_{k-1})\right\rvert \ (\because \det(AB) = \det(A)\det(B)) \\
& = \ln p(x) - \ln \left\lvert \det J_F(x)\right\rvert \ (\textrm{because of derivative chain rule})\\
\end{align}
$$
</div>

It is easy to realize that the last equation is literally a variables change formula with transformation $F$. This does make sense because a normalizing flow can also be viewed as a change of variables but with a much more complex invertible transformation. Here, $F$ is clearly invertible as it is a composition of an arbitrary number of invertible functions.

By designing an appropriate $F$, we can obtain an arbitrarily complex normalized density function at the completion of a normalizing flow. Hence, normalizing flows can be intuitively interpreted as a systematic way to distort the input density function, making it more complex (like in variational inference setting) or simpler (like in density estimation setting). However, in order for normalizing flows to be useful in practice, we need to have two conditions satisfied:
* Those transformation functions $f_k$ obviously need to be `invertible`.
* The determinant of their Jacobian matrices $J_{f_k}$ need to be `easy to compute`, in order to obtain a tractable likelihood.

In fact, many approaches have been proposed to construct those easy-to-use transformation functions lately. Inspired by normalizing flows, the authors of the paper has also managed to exploit residual networks as transformation functions used for normalizing flows. Thus, before diving into the details, let take a look back at the architecture of residual networks.

### The awesome residual block
Residual network is composed of a sequence of residual blocks, with each block can be simplified as this figure below

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/iresnet/residual.png" width="23%">

As we can see, each residual block consists of a residual part denoted by $g$ and an identity part.
From mathematical viewpoint, each block can be clearly counted as a function which takes input $x$ and produces $z$ as output.
Formally, we have
$$
z = f(x) = g(x) + x \tag{1}
$$

Back to our main story, it is obvious that the goal is to make use of residual network as a transformation function for normalizing flows. Since residual network can be interpreted as a composition function of multiple residual blocks, making each individual block invertible is **a sufficient condition** for the invertibility of the whole net.

## Making the nets invertible
In the inverse phase, each block takes $z$ as input and produces $x$ as output. Thus, in order for a block to be invertible, we need to enforce the existence and uniqueness of output $x$ for each input $z$.

From $(1)$, we have

$$
x = z - g(x)
$$

Let define $h(x) = z - g(x)$ to be a function of $x$, where z acts as a constant. The requirement can now be formulated as follows: The equation $x = h(x)$ must have only one root or, to put it in a formal way, **$h(x)$ has a unique fixed point**.

> **Fixed point:**
> Let X be a metric space and let T: X &rightarrow; X be a mapping in X. A fixed point of T is a point in X such that T(x) = x.

Fortunately, this requirement can be obtained according to the famous Banach fixed-point theorem.

### Fixed point theorem
The Banach fixed point theorem, also called contraction mapping theorem, states that every contraction mapping in a complete metric space admits a unique fixed point.

> **Contraction mapping:**
> Let $(M, d)$ be a complete metric space. A function $T$: $M$ &rightarrow; $M$ is a contraction mapping if there exists a real number $0 \leq k < 1$ such that:
>
> $$\quad d(T(x), T(y)) \leq k d(x, y) , \quad \quad \forall x, y \in M$$
>
>The smallest $k$ for which the above inequality holds is called the **Lipschitz constant** of $f$, denoted by $Lip(T)$

**Banach theorem:** Let $(M, d)$ be a complete metric space and $T$: $M$ &rightarrow; $M$ be a contraction mapping. Then T has a unique fixed point $x \in M$. Furthermore, if $y \in M$ is arbitrary chosen, then the iterates $\{ {x_n}\}_{n=0}^\infty$, given by

<div>
$$
\begin{align}
& x_0 = y \\
& x_n = T(x_{n-1}), n \geq 1,
\end{align}
$$
</div>

have the property that $\lim_{n\to\infty} x_n = x$ as illustrated in the figure below.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/iresnet/fixed_point.png" width="90%">
<p align="center">Fixed point convergence illustration in Banach theorem</p>


### Enforcing Lipschitz constraint
Based on the Banach theorem above, our enforcing condition then becomes

$$
Lip(h) < 1 \;\textrm{or}\; Lip(g) < 1
$$

Hence $g$ can be implemented as a composition of contractive linear or nonlinear mappings like the figure below.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/iresnet/iresidual.png" width="30%">
<p align="center">Contractive residual mapping</p>

* For nonlinear mappings, `ReLU`, `ELU` and `tanh` are the possible choices for contraction constraint.
* For linear mappings, implemented as convolutional layers $W_i$, they can be made contractive by satisfying the condition $$\lVert W_i \rVert_2 < 1 \quad \forall W_i$$ where $\lVert a \rVert_2$ denotes the spectral norm of matrix a.

> **Spectral norm of a matrix:**
> The largest singular value of a matrix.

The spectral norm of non-square matrix $W_i$ can be directly estimated using the power iteration algorithm (by Gouk et el. 2018), which yields an underestimate $\tilde \sigma_i \leq \lVert W_i \rVert_2$. The algorithm can be summarized as follows:

<div>
$$
\begin{align}
& \textrm{Initialize} \; x_0 \\
x_k & = W_i^T W_i x_{k - 1}, \; \forall k, 1 \leq k \leq n \\
\tilde \sigma_i & = \frac{\lVert W_i x_n\rVert_2}{\lVert x_n \rVert_2} \\
\end{align}
$$
</div>

We then normalize the parameter $W_i$ by

<div>
$$
\tilde{W_i} = \begin{cases}
\frac{cW_i}{\tilde \sigma_i}, & \mbox{if} \; \frac{c}{\tilde \sigma_i} < 1 \\
W_i, &\mbox{else}
\end{cases}
$$
</div>
where $c$ is the hyperparameter ($c < 1$).

### How to yield the inverse output
Though the constraint above guarantees invertibility of the residual network,
it does not provide any analytical form for the inverse.
Fortunately, inverse output of each residual block can be yielded through a simple fixed-point iteration, as described below
* Initialize value $x = x_0$
* For each iteration $i$, $x_{i+1} = h(x_i) = z - g(x_i)$

## Computing the likelihood
So far, we already managed to construct invertible residual networks. We can now make use of them as a transformation for density estimation.
<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/iresnet/iresnet.png" width="100%">
<p align="center">Invertible residual network as transformation function in normalizing flows</p>

But there is still one problem need to be dealt with. In order for the likelihood to be tractable,
we need to compute the determinant of the Jacobian matrix of the residual network $F$ or, instead,
the determinant of the Jacobian matrix of each residual block $f$.

$$
\ln p(x) = \ln p(z) + \ln \left\lvert \det J_F(x)\right\rvert = \ln p(z) + \sum_{k=1}^{K} \ln \left\lvert \det J_{f_k}(z_{k-1})\right\rvert
$$

The computation of the determinant of full Jacobian matrix requires $O(d^3)$ time, which makes it prohibitive for high-dimensional data like image. Fortunately, we can approximate the term in a certain way.

### The log determinant term

For each residual block $f$, we have

<div>
$$
\begin{align}
\ln \left\lvert \det J_f(x)\right\rvert & = \ln (\det J_f(x)) \textrm{( $\det J_f$ is always positive)} \\
& = tr(\ln J_f(x)) \textrm{($\ln \det A = tr(\ln(A))$)} \\
& = tr(ln\frac{\partial f}{\partial x}) \\
& = tr(ln\frac{\partial (x + g(x))}{\partial x})\\
& = tr(ln\ (I + J_g(x))) \textrm{($I$ denotes identity matrix)} \\
& = tr(\sum_{k=1}^\infty(-1)^{k + 1}\frac{J_g^k}{k}) \textrm{(power series expression of matrix logarithm)} \\
& = \sum_{k=1}^\infty(-1)^{k + 1}\frac{tr(J_g^k)}{k} \textrm{($tr(A + B) = tr(A) + tr(B)$)} \\
\end{align}
$$
</div>

> **Matrix logarithm and its power series expression:**
> A logarithm of matrix $M$ is any matrix $X$ such that $e^X = M$. It can be expressed as a power series $$ln(M) = \sum_{k=1}^\infty(-1)^{k + 1}\frac{(M - I)^k}{k}$$ whenever the series converges.

Now the log-determinant term has been rewritten as an infinite sum of traces of matrix powers, which makes it easier to approximate.
Even though, there is still a bunch of drawbacks if we want to approximate the term:
* Computing $tr(J_g)$ costs $O(d^2)$
* Computing matrix powers $J_g^k$ requires knowledge of full Jacobian
* The series is infinite

#### Hutchinson trace estimator
Evaluating the trace of matrix powers $J_g^k$ is expensive due to the full knowledge of Jacobian matrix
and also matrix-matrix multiplications, hence comes the Hutchinson method for trace approximation.

Hutchinson trace estimator is a Monte Carlo approach to approximate the trace of matrix powers, for example $J_g^k$ in our case, without fully evaluating them.
Specifically, a random vector $v$ is introduced to estimate the trace

$$tr(A) = \mathrm{E}_{v \sim p(v)}v^{T}Av $$

with the constraint that $v$ is drawn from a fixed distribution $p(v)$, satisfying $\mathrm{E}[v] = 0$ and $\mathrm{Var}[v] = I$. Thus it is obvious that the Gaussian $N(0, I)$ is a good choice for $p(v)$. Applying the trace estimator, we have

$$
tr(J_g^k) = \mathrm{E}_{v \sim N(0, I)} v^T J_g^k v
$$

The matrix power computation can be circumvented by evaluating $v^T J_g^k v$  in a recursive fashion

<div>
$$
\begin{align}
w_0 & = v \\
w_k & = J_g w_{k - 1}, \forall k \geq 1 \\
v^T J_g^k v & = v^T w_k \\
\end{align}
$$
</div>
which requires now only matrix-vector multiplication. Also, the term $w_k$ can be evaluated roughly as
the same cost as evaluating $g$ using reverse-mode automatic differentiation,
alleviating the heavy computation of evaluating $J_g$ explicitly.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/iresnet/diff.png" width="45%">

The only problem remains is the computation of infinite series, which can be addressed by truncating the series at a finite index $n$

$$
\ln |\det J_f(x)| \approx \sum_{k=1}^{n}(-1)^{k + 1}\frac{tr(J_g^k)}{k}
$$

**References**
1. [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)
2. [Invertible Residual Networks](https://arxiv.org/abs/1811.00995)
3. [Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/abs/1908.09257)
4. [High-Dimensional Probability Estimation with Deep Density Models](https://arxiv.org/abs/1302.5125)
5. [Hutchinson's Trick](http://blog.shakirm.com/2015/09/machine-learning-trick-of-the-day-3-hutchinsons-trick/)
6. [Spectral Normalization Explained](https://christiancosgrove.com/blog/2018/01/04/spectral-normalization-explained.html)
7. [Regularisation of Neural Networks by Enforcing Lipschitz Continuity](https://arxiv.org/abs/1804.04368)
8. [Large-scale Log-determinant Computation through Stochastic Chebyshev Expansions](http://proceedings.mlr.press/v37/hana15.pdf)
