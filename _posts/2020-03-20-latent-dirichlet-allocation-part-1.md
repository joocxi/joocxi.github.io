---
title: "Variational Expectation Maximization for Latent Dirichlet Allocation - Understanding the theory"
tags: ["latent dirichlet allocation", "lda", "topic modeling", "em", "variational inference", "variational em"]
toc: true
toc_sticky: true
---

Text data is everywhere. When having massive amounts of them, a need naturally arises is that we want them to be organized efficiently. A naive way is to organize them based on topics, meaning that text covering the same topics should be put into the same groups. The problem is that we do not know which topics a text document belongs to and manually labeling topics for all of them is very expensive. Hence, topic modeling comes as an efficient way to automatically discover abstract topics contained in these text documents.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/lda_doc.png" width="400" alt="">

One of the most common topic models is Latent Dirichlet Allocation (LDA), was introduced long time ago (D. Blei et al, 2003) but is still powerful now. LDA is a complex, hierarchical latent variable model with some probabilistic assumptions over it. Thus, before diving into detail of LDA, let us review some knowledges about `latent variable model` and how to handle some problems associated with it.

>**Note:** My blog on LDA contains two parts. This is the first part about theoretical understanding of LDA. The  second part involves a basic implementation of LDA, which you can check out [here]({{< ref "post/lda-part-2.md" >}}).

## Latent variable model
A latent variable model assumes that data, which we can observe, is controlled by some underlying unknown factor. This dependency is often parameterized by a known distribution along with its associated parameters, known as model parameter. A simple latent variable model consists of three parts: observed data $x$, latent variable $z$ that controls $x$ and model parameter $\theta$ like the picture below
<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/lda_latent.png" width="300">
Latent variables increases our model's expressiveness (meaning our model can capture more complex data) but there's no such thing as a free lunch. Typically, there are two main problems associated with latent variable models that need to be solved
* The first one is **learning** in which we try to find the "optimal" parameters ${\theta^*}$ based on some criterion.
One powerful technique for learning is `maximum likelihood estimation` preferring to chose the parameters that maximize
the likelihood $p(x;\theta)$. Though, maximum likelihood estimation in latent variable models is hard.
A maximum likelihood method named `Expectation Maximization` can solve this difficulty for some kind of models.
It is also helpful for LDA so we will discuss the method in the next section.

* In many cases, latent variables can capture meaningful pattern in the data.
Hence, given new data, we are often interested in the value of latent variables.
This raises the problem of **inference** where we want to deduce the posterior $p(x|z;\theta)$.
Though, in many cases the posterior is hard to compute.
For example, when $z$ is continuous, we have

$$
p(z|x;\theta) = \frac{p(x, z ;\theta)}{p(x;\theta)} = \frac{p(x, z ;\theta)}{\int_z p(x, z;\theta)}
$$

  The integral in the denominator often makes the posterior intractable. 
A method to solve this problem, named `Variational Inference`, will also be discussed later.

### Expectation Maximization (EM)
Introducing latent variables to a statistical model makes its likelihood function non-convex. Thus, it becomes hard to find a maximum likelihood solution. The EM algorithm was introduced to solve the maximum likelihood estimation problem in these kind of statistical models. The algorithm iteratively alternates between building an expected log-likelihood (`E step`), which is a convex lower bound to the non-convex log-likelihood, and maximizing it over parameters (`M step`).

But how does EM construct the expected log-likelihood?
We have
<div>
\begin{align}
\log p(x; \theta) & \geq \log p(x;\theta) - KL({{q(z)}}||p(z|x;\theta)) \\
& = \log p(x;\theta) - (\mathrm{E}_{z\sim q(z)}\log q(z) - \mathrm{E}_{z \sim q(z)}\log p(z|x; \theta)) \\
& = \mathrm{E}_{z\sim {q(z)}}(\log p(x;\theta) + \log p(z|x;\theta)) - \mathrm{E}_{z\sim q{(z)}}\log {q(z)} \\
& = \mathrm{E}_{z\sim q(z)}\log p(x, z;\theta) - \mathrm{E}_{z\sim q(z)}\log {q(z)} = L(q, \theta) \tag{1} \\
\end{align}
</div>

for any choice of $q(z)$. It is obvious that $L(q, \theta)$ is a lower bound of $\log p(x;\theta)$ and the equality holds if and only if $q(z) = p(z\mid x;\theta)$.
EM aims to construct a lower bound that is easy to maximize. By initializing parameter $\theta_{old}$ and choosing $q(z) = p(z\mid x;\theta_{old})$ at each `E-step`, the lower bound becomes

<div>
$$
L(\theta) = \mathrm{E}_{z\sim p(z|x;\theta_{old})} \log p(x,z;\theta) - \mathrm{E}_{z\sim {p(z|x;\theta_{old})}}\log {p(z|x;\theta_{old})}
$$
</div>

EM then maximizes $L(\theta)$ at each `M-step`

<div>
\begin{align}
\mathop{max}_{\theta} L(\theta) & = \mathop{max}_{\theta} \mathrm{E}_{z\sim p(z|x;\theta_{old})} \log p(x,z;\theta) - \mathrm{E}_{z\sim {p(z|x;\theta_{old})}}\log {p(z|x;\theta_{old})} \\
& = \mathop{max}_{\theta} \mathrm{E}_{z\sim p(z|x;\theta_{old})} \log p(x,z;\theta) \\
\end{align}
</div>

<!-- $L(q, \theta)$ is still involved two unknown components which is hard to optimize. EM deals with this problem by initializing parameter $\theta_{old}$ and construct the lower bound by choosing $\color{blue}{q(z)} = p(z|x;\theta_{old})$. The lower bound becomes
$$
L(\theta) = \mathrm{E}_{z\sim p(z|x;\theta_{old})} \log p(x,z;\theta) - \mathrm{E}_{z\sim {p(z|x;\theta_{old})}}\log {p(z|x;\theta_{old})}
$$ -->

The EM algorithm can be summarized as follows
* Initialize parameter $\theta = \theta^{(0)}$
* For each loop $t$ start from $0$
    <img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/em_loop.png" width="600">
    * Estimate the posterior $p(z\mid x; {\theta^{(t)}})$
    * Maximize the expected log-likelihood
  
        $$\mathop{max}_{\theta^{(t+1)}} \mathrm{E}_{z\sim p(z|x ;{\theta^{(t)}})} {p(x, z ;{\theta^{(t+1)}}})$$
  
    * If the convergence standard is satisfied, stop

>**Note**: It is easy to notice that the EM algorithm can only be applied if we can compute (or approximate) the posterior distribution analytically, given the current parameter ${\theta^{(t)}}$.

If you want to go into the details of EM, **Gaussian Mixture** (when $z$ is discrete) and **Probabilistic Principal Component Analysis** (or Probabilistic PCA in short, when $z$ is continuous) are the two perfect cases to study.

### Variational Inference

In many of the cases, the posterior distribution $p(z|x;\theta)$ we are interested in can not be inferred analytically, or in other words, it is intractable. This leads naturally to the field of `approximate inference`, in which we try to approximate the intractable posterior. Variational inference is such a technique in approximate inference which is fast and effective enough for a good approximation of $p(z|x;\theta)$. The process can be pictured as follows
<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/vi.png" width="400">

As we can see, the idea of variational inference is simple that we reformulate the problem of inference as an optimization problem by
* First, posit a variational family ${q(z;v)}$ controlled by variational parameter $v$
* Then, find the optimal ${q(z;v^*)}$ in this family, which is as "close" to $p(z\mid x;\theta)$ as possible

Specifically, the goal of variational inference is then to minimize the $KL$ divergence between the variational family and the true posterior: $\mathop{min}_{q, v}KL({{q(z;v)}}||p(z|x;\theta))$. But how can we minimize such an intractable term?
<!-- But how can we minimize a term that can not be evaluated analytically? -->

Recall from $(1)$ (with the variational distribution ${q(z;v)}$  now being chosen as ${q(z)}$), we have the $ELBO$

<div>
\begin{align}
& \log p(x;\theta) - KL({{q(z;v)}}||p(z|x;\theta))\\
& = \mathrm{E}_{z\sim {q(z;v)}}\log p(x, z;\theta) - \mathrm{E}_{z\sim {q(z;v)}}\log {q(z;v)}\\
\end{align}
</div>

Since $\log p(x;\theta)$ is considered as constant, minimizing the KL divergence is equivalent to maximizing the $ELBO$. The optimization problem becomes

<div>
$$
\mathop{max}_{q}\mathrm{E}_{z\sim {q(z;v)}}\log p(x, z;\theta) - \mathrm{E}_{z\sim {q(z;v)}}\log {{q(z;v)}} \tag{2}
$$
</div>

which now can be optimized with a suitable choice of ${q(z;v)}$.

>**Note**: EM and variational inference both involve maximizing the ELBO of the log-likelihood. However, EM produces a point estimate of the optimal model parameter, meanwhile variational inference results in an approximation of the posterior distribution.

#### Mean field approximation
Several forms of variational inference has been proposed to design a tractable variational family. The simplest out of them is `mean-field approximation`, which makes a strong assumption that all latent variables are mutually independent. The variational distribution can then be factorized as
$$
q(z;v) = \prod_{k=1}^{K}q(z_k;v_k) \tag{3}
$$

where $z$ consists of $K$ latent variables $(z_1, z_2, ..., z_K)$ and each latent variable $z_k$ is controlled by its own variational parameter $v_k$.

We will not go into detail here, but substituting $(3)$ into $(2)$, taking the derivative with respect to each $q(z_k;v_k)$, then setting the derivative to zero we obtain the coordinate ascent update

$$
q^*(z_k;v_k^*) \propto \mathrm{E}_{z_{-k} \sim q_{-k}(z_{-k};{v_{-k}})} \log p(z_k, z_{-k}, x;\theta) \tag{4}
$$

where $(\cdot)_{-k}$ denotes all but the $k$th element.

Note that until now we didn't specify the functional form for each variational factor $q(z_k;v_k)$ yet. Fortunately, the optimal form of each $q(z_k;v_k)$ can be derived from the $RHS$ expression of the coordinate update $(4)$, which is often easy to work with for many models.

<!-- Note that in $(3)$ we didn't specify the functional form for each optimal variational factor $q(z_k;v_k)$. These optimal function forms can be derived from the $RHS$ of the coordinate update, which is easy for many models. -->

#### Coordinate ascent update

We can then use the coordinate ascent algorithm to find the optimal mean-field distribution. The algorithm can be summarized as follows
* Initialize $v = v^{(0)}$
* For each loop $t$ start from $0$
    * For each loop  $k$ from $1$ to $K$
        * Estimate $q^\ast(z_k;v_k^\ast) \propto \mathrm{E}\_{z_{-k} \sim q_{-k}(z_{-k};{v_{-k}}^{(t)})} \log p(z_k, z_{-k}, x;\theta)$
        * Set $q(z_k; v_k^{(t+1)}) = q^\ast(z_k;v_k^\ast)$
    * Compute the $ELBO$ to check convergence

By now we are familiar with the concept of latent variable model. Let us move on to discuss LDA in the next section.

## Latent Dirichlet Allocation

LDA is a latent variable model on observed text data or, to be more specific, a collection of `words` in each `document`. The model is built based on the assumptions that each `document` is a distribution over a predefined number of `topics`; meanwhile, each `topic` is considered as a distribution over `words` in a fixed vocabulary. For example, suppose that we have $4$ `topics` *<economics, animal, science, art>* and a total of $6$ `words` *<money, astronomy, investment, rabbit, painting, chemical>*. Then our assumptions can be illustrated like this figure below.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/lda_dist.png" width="700">
<p align="center">Two probabilistic assumptions in LDA</p>

Also, according to this figure, we can say that the document is $40\%$ of *economics*, $20\%$ of *animal*, $30\%$ of *science* and $10\%$ of *art*. Seem familiar? This is basically the categorical distribution.

### Categorical distribution
Formally, categorical distribution is a `discrete` probability distribution, describing the possibility that one discrete random variable belongs to one of $K$ categories. The distribution is parameterized by a $K$-dimensional vector $\theta$ denoting probabilities assigned to each category. It probability mass function is defined as
$$
p(x=i) = \theta_i
$$
, where $x$ is the random variable and $i$ ranges from $1$ to $K$ (representing the $K$ categories).

In our example above, the document is a categorical distribution over $K = 4$ `topics`, 
with its parameter $\theta = [0.4, 0.2, 0.3, 0.1]$.
Similarly, each `topic` is also a categorical distribution over $K = 6$ `words`.

### Dirichlet distribution
Another distribution which plays an important role in LDA is the Dirichlet distribution (hence the name LDA). Dirichlet distribution is a `continuous` multivariate probability distribution over a $(K-1)$-simplex, which can be seen as a set of $K$-dimensional vectors $x=[x_1, x_2, ..., x_K]$ such that each $x_k \geq 0$ and $\sum_{k=1}^Kx_k = 1$. For example, the 2-simplex is a triangle in $3D$ space (see figure below).

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/simplex.png" width="350">


The distribution is parameterized by a positive $K$-dimensional vector $\alpha$, with its probability density function defined as

$$
p(x;\alpha) = \frac{1}{B(\alpha)} \prod_{k=1}^{K}x_k^{\alpha_k - 1}
$$

where $B(\cdot)$ is the famous [beta function](https://en.wikipedia.org/wiki/Beta_function). The parameter $\alpha$ governs how the density is distributed on the simplex space. For example, the picture below shows how the distribution is concentrated with different $\alpha$ in the case of 2-simplex (brighter color denoting more dense areas).

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/lda_dirichlet.png" width="700">
<p align="center">Based on <a href="https://towardsdatascience.com/dirichlet-distribution-a82ab942a879?gi=686effc9deea" target="_blank">Dirichlet distribution blog by Sue Liu</a>
</p>

It is noticeable that sample from a Dirichlet distribution is parameter of a categorical distribution.
Thus, Dirichlet distribution is also seen as a distribution over categorical distribution.
But why we need the Dirichlet distribution?
It is because, in the context of Bayesian statistics, we want to control the uncertainty over some parameters rather than just a point estimate of them.
To be more specific, given data $x$ with its likelihood function $f(x;\theta)$, we want to infer the full distribution of $\theta$ given $x$, but not an optimal point $\theta^*$.
Given a prior $p(\theta)$, the posterior is proportional to the likelihood time the prior
<div>
$$
p(\theta|x) \propto p(x|\theta)p(\theta)
$$
</div>
If the posterior has the same functional form with the prior,
the prior is said to be a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) to the likelihood.
For example, the Dirichlet distribution is the conjugate prior to the categorical likelihood function.
It means that, when the likelihood function is categorical and Dirichlet is chosen as a prior, then the posterior
<div>
$$
p(\theta|x;\alpha) \propto Cat(x|\theta)Dir(\theta;\alpha)
$$
</div>
will have the same form as the prior, which is a Dirichlet distribution. Conjugate prior makes it easy to calculate the posterior over parameter of interest $\theta$. Thus, in the case of LDA where the categorical distribution is used to represent the topic distribution of each document and the word distribution of each topic, there is no better choice than Dirichlet distribution as a conjugate prior to control these categorical distributions.

LDA is also a generative model. Hence, to understand its architecture clearly, we better see how it generates documents.

### Generative process
Suppose that we have $T$ topics and a vocabulary of $V$ words. Model LDA has 2 parameters $(\alpha, \beta)$ where
* $\alpha$ denotes the Dirichlet prior that controls topic distribution of each document.
* $\beta$ is a $2D$ matrix of size $T \times V$ denotes word distribution of  all topics ($\beta_i$ is a word distribution of the `i + 1`th topic).

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/lda_gen.png" width="722">
<p align="center">Document generative process in LDA</p>

The generative process can be pictured as above. Specifically,
* For each document $d$ with $N_d$ words
    * Sample document's topic distribution $\theta \sim Dir(\alpha)$
    * For each word positions $j$ from $1$ to $N_d$
        * Sample the topic of the current word $t_j \sim Cat(\theta)$
    * Sample the current word based on the topic $t_j$ and the word distribution parameters $\beta$, $w_j \sim Cat(\beta_{t_j})$

>**Warning**: $\theta$ is now a latent variable, not model parameter. I keep the notation the same as the original paper for your ease of reference.

### The two problems of LDA
<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/lda_graph.png" width="500">

LDA is a latent variable model, consisting of: observed data $w$;  model parameters $\alpha, \beta$; and latent variables $z, \theta$; as shown in the figure above. Hence, just like any typical latent variable model, LDA also have two problems needed to be solved.

<!--
$$
p(w;\alpha, \beta) = \int p(\theta;\alpha) \prod_{i=1}^{N_d} \sum_{t=0}^{T - 1} p(z_i = t|\theta) p(w_i | \beta, z_i=t) d\theta
$$ -->

#### Inference
Given a document $d$ has $N$ words $\{w_1^{(d)}, ..., w_N^{(d)}\}$ and model parameters $\alpha$, $\beta$;
infer the posterior distribution $p(z, \theta| w^{(d)}; \alpha, \beta)$.
We can then use mean-field approximation to approximate $p(z, \theta| w^{(d)}; \alpha, \beta)$,
by introducing the mean-field variational distribution $q(z, \theta; \gamma, \phi) = q(\theta;\gamma)\prod_{i=1}^{N}q(z_i;\phi_i)$.

<img class="img-center" src="{{ site.url }}{{ site.baseurl }}/assets/images/lda/lda_mf.png" width="249">

Deriving the ELBO to yield coordinate ascent update for each variational parameter is mathematically heavy so I will not put the mathematical stuff here. For reference, the derivation could be found in the Appendix of the original paper. Based on the coordinate ascent update, we obtain the optimal form for $q(\theta;\gamma)$ which is a Dirichlet distribution and each $q(z_i;\phi_i)$ which is a categorical distribution. The coordinate ascent algorithm then return the optimal parameters $\gamma^*, \phi^*$.

#### Parameter estimation
In LDA, the problem of parameter estimation is: find $\alpha, \beta$ that maximizes the likelihood function

$$
p(w;\alpha, \beta) = \int p(\theta;\alpha) \prod_{i=1}^{N_d} \sum_{t=0}^{T - 1} p(z_i = t|\theta) p(w_i | \beta, z_i=t) d\theta
$$

Since the posterior $p(z, \theta| w^{(d)}; \alpha, \beta)$ can not be computed exactly but can only
be approximated (for instance, via variational inference in the previous section),
we can not apply the EM algorithm directly to solve the estimation problem.
To handle this, an algorithm named **variational EM algorithm**, which combines EM and mean-field inference,
was introduced. Variational inference is now used in the E-step to compute the posterior, approximately.
The algorithm used for LDA can be summarized as follows
* Initialize parameters $\alpha, \beta$ to $\alpha^{(0)}, \beta^{(0)}$
* For each loop $t$ start from $0$
    * **E step**:
        * For each document $d$
            * Use coordinate ascent update algorithm to yield optimal $\gamma^{(d)}, \phi^{(d)}$
    * **M step**: Maximize the expected log-likelihood (up to some constant) with respect to $\alpha, \beta$

    * If the convergence standard is satisfied, stop

>**Note**: Actually, there are many techniques to solve the two problems of LDA. Though, we only discuss about Variational EM in the scope of this blog :v

__References__

1. Latent Dirichlet Allocation ([pdf](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf))
2. Mean-field variational inference([pdf](https://www.cs.cmu.edu/~epxing/Class/10708-17/notes-17/10708-scribe-lecture13.pdf))
