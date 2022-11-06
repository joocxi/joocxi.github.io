---
title: "Variational Expectation Maximization for Latent Dirichlet Allocation - A simple demo"
tags: ["latent dirichlet allocation", "lda", "topic modeling", "python", "variational inference", "variational em"]
toc: true
toc_sticky: true
---

After discussing LDA in the previous part, we now get our hands dirty by implementing the Variational EM algorithm. Recall that the goal is to estimate $\alpha, \beta$ that maximizes the ELBO

<div>
$$
\mathrm{E}_{q}\log p(w, \theta, z;\alpha, \beta) - \mathrm{E}_{q}\log {q(z, \theta;\gamma, \phi)}
$$
</div>

where the joint likelihood and the variational distribution are factorized as follows

$$
p(w, \theta, z; \alpha, \beta) = Dir(\theta;\alpha)\prod_{n=1}^{N}Cat(z_n;\theta) Cat(w_n;z_n, \beta)
$$

$$
q(z, \theta; \gamma, \phi) = Dir(\theta;\gamma)\prod_{n=1}^{N}Cat(z_n;\phi_n)
$$

## Working with the ELBO
But before getting into code, we need to derive the ELBO. Substituting these factorizations into the ELBO, we obtain

<div>
\begin{align}
L & = \mathrm{E}_{q}\log p(w, \theta, z;\alpha, \beta) - \mathrm{E}_{q}\log {q(z, \theta;\gamma, \phi)} \\
& = \mathrm{E}_{q}\log Dir(\theta; \alpha) + \sum_{n=1}^{N} \Big[ \mathrm{E}_{q} \log Cat(z_n; \theta) + \mathrm{E}_{q} \log Cat(w_n; z_n, \beta) \Big] \\
& \quad - \mathrm{E}_{q} \log Dir(\theta; \gamma) - \sum_{n=1}^{N}\mathrm{E}_{q} \log Cat (z_n;\phi_n) \tag{1}
\end{align}
</div>

### Dealing with expected values

To handle the expectations in the ELBO, we need to rewrite the Dirichlet distribution in exponential form as follows

<div>
\begin{align}
Dir(x;\alpha) & = \frac{1}{B(\alpha)} \prod_{k=1}^{K}x_k^{\alpha_k - 1} \\
& = \frac{\Gamma(\sum_{k=1}^{K}\alpha_k)}{\sum_{k=1}^{K}\Gamma(\alpha_k)} \prod_{k=1}^{K}x_k^{\alpha_k - 1} \\
& = \exp\Big[ \sum_{k=1}^{K} (\alpha_k - 1) \log x_k + \log \Gamma(\sum_{k=1}^{K}\alpha_k) - \sum_{k=1}^{K}\log\Gamma(\alpha_k) \Big]
\end{align}
</div>

>**Exponential family distribution**
> $$
> p(x|\theta) = h(x) exp(\eta \cdot T(x) - A(\eta))
> $$
>
> where $h(x)$ is known as **base measure**, $\eta(\theta)$ is **natural parameter**, $T(x)$ is **sufficient statistic** and $A(\theta)$ is **log normalizer**. One important property of the exponential family is that the mean of the sufficient statistic $T(x)$ can be derived by differentiating the natural parameter $A(\eta)$
> $$
> E[T_j]= \frac{\partial A(\eta)}{\partial \eta_j} \tag{2}
> $$

Applying property $(2)$ for the case of the Dirichlet distribution, we have
<div>
\begin{align}
\mathrm{E}_{x \sim Dir(x;\alpha)}\log x_k & = \frac{\partial(\sum_{j=1}^{K}\log\Gamma(\alpha_j) - \log \Gamma(\sum_{j=1}^{K}\alpha_j))  }{\partial (\alpha_k - 1)} \\
& = \frac{\partial(\sum_{j=1}^{K}\log\Gamma(\alpha_j) - \log \Gamma(\sum_{j=1}^{K}\alpha_j))  }{\partial \alpha_k} \\
& = \Psi(\alpha_k) - \Psi(\sum_{j=1}^{K}\alpha_j)
\end{align}
</div>

where $\Psi(\cdot)$ is the derivative of the logarithm of Gamma function (also known as Digamma).

Also, the categorical distribution can be represented using [Iverson bracket](https://en.wikipedia.org/wiki/Iverson_bracket) $[\cdot]$

$$
Cat(x;\theta) = \prod_{i=1}^K \theta_i^{[x=i]}
$$

where $[x=i]$ evaluates to  $1$ if $x = i$, $0$ otherwise (with the assumption that values of $x$ fall into the range $\{1, 2, ..., K\}$).

<!-- $$
\log Cat(x;\theta) = \sum_{i=1}^K [x=i] \log \theta_i
$$ -->

Expectation of a function $f(x)$ with respect to the categorical distribution is computed as
<div>
$$
\mathrm{E}_{x\sim Cat(x;\theta)} f(x) = \sum_{i=1}^{K} \theta_i f(i)
$$
</div>

### Deriving the ELBO
Using these results above, we have
<div>
\begin{align}
& \mathrm{E}_{q}\log Dir(\theta; \alpha) \\
& = \mathrm{E}_{q} \Big[\sum_{k=1}^{K} (\alpha_k - 1) \log \theta_k + \log \Gamma(\sum_{k=1}^{K}\alpha_k) - \sum_{k=1}^{K}\log\Gamma(\alpha_k)\Big] \\
& = \sum_{k=1}^{K}(\alpha_k - 1) \mathrm{E}_{\theta \sim Dir(\theta;\gamma)} \log\theta_k + \log \Gamma(\sum_{k=1}^{K}\alpha_k) - \sum_{k=1}^{K}\log\Gamma(\alpha_k) \\
& = \sum_{k=1}^{K}(\alpha_k - 1) (\Psi(\gamma_k) - \Psi(\sum_{j=1}^{K}\gamma_j)) + \log \Gamma(\sum_{k=1}^{K}\alpha_k) - \sum_{k=1}^{K}\log\Gamma(\alpha_k) \tag{3} \\
& \\
& \mathrm{E}_{q} \log Cat(z_n; \theta) \\
& = E_{z_n \sim Cat(z_n;\phi_n), \theta \sim Dir(\theta;\gamma)} \log Cat(z_n;\theta) \\
& = E_{z_n \sim Cat(z_n;\phi_n), \theta \sim Dir(\theta;\gamma)} \sum_{j=1}^{K}[z_n=j]\log \theta_j \\
& = \sum_{i=1}^{K} \phi_{ni} \mathrm{E}_{\theta \sim Dir(\theta; \gamma)} \sum_{j=1}^{K}[i=j]\log \theta_j \\
& = \sum_{i=1}^{K} \phi_{ni} \mathrm{E}_{\theta \sim Dir(\theta; \gamma)} \log \theta_i \\
& = \sum_{i=1}^{K} \phi_{ni}(\Psi(\gamma_i) - \Psi(\sum_{j=1}^{K}\gamma_j)) \tag{4} \\
& \\
& \mathrm{E}_{q} \log Cat(w_n;z_n, \beta) \\
& = \mathrm{E}_{z_n \sim Cat(z_n;\phi_n)} \log Cat(w_n;\beta_{z_n}) \\
& = \mathrm{E}_{z_n \sim Cat(z_n;\phi_n)} \sum_{j=1}^{V} [w_n=j] \log \beta_{z_n j} \\
&\quad \textrm{($w_n$ represents word index in the vocabulary)} \\
& = \sum_{i=1}^{K} \phi_{ni} \sum_{j=1}^{V} [w_n=j] \log \beta_{ij}\\
& = \sum_{i=1}^{K} \sum_{j=1}^{V} \phi_{ni} [w_n=j] \log \beta_{ij} \tag{5} \\
& \\
& \mathrm{E}_{q} \log Dir(\theta;\gamma) \\
& = \mathrm{E}_{q} \Big[\sum_{k=1}^{K} (\gamma_k - 1) \log \theta_k + \log \Gamma(\sum_{k=1}^{K}\gamma_k) - \sum_{k=1}^{K}\log\Gamma(\gamma_k)\Big] \\
& = \sum_{k=1}^{K}(\gamma_k - 1) \mathrm{E}_{\theta \sim Dir(\theta;\gamma)} \log\theta_k + \log \Gamma(\sum_{k=1}^{K}\gamma_k) - \sum_{k=1}^{K}\log\Gamma(\alpha_k) \\
& = \sum_{k=1}^{K}(\gamma_k - 1) (\Psi(\gamma_k) - \Psi(\sum_{j=1}^{K}\gamma_j)) + \log \Gamma(\sum_{k=1}^{K}\gamma_k) - \sum_{k=1}^{K}\log\Gamma(\gamma_k) \tag{6} \\
& \\
& \mathrm{E}_{q} \log q(z_n; \phi_n) \\
& = \mathrm{E}_{z_n \sim Cat(z_n;\phi_n)} \log Cat(z_n;\phi_n) \\
& = \mathrm{E}_{z_n \sim Cat(z_n;\phi_n)} \sum_{j=1}^{K}[z_n = j]\log \phi_{nj} \\
& = \sum_{i=1}^{K} \phi_{ni} \sum_{j=1}^{K} [i=j] \log \phi_{nj} \\
& = \sum_{i=1}^{K} \phi_{ni} \log \phi_{ni} \tag{7}
\end{align}
</div>

Substituting $(3), (4), (5), (6), (7)$ into $(1)$, the ELBO becomes

<div>
\begin{align}
L&(\gamma, \phi;\alpha, \beta) \\
= &\sum_{k=1}^{K}(\alpha_k - 1) (\Psi(\gamma_k) - \Psi(\sum_{j=1}^{K}\gamma_j)) + \log \Gamma(\sum_{k=1}^{K}\alpha_k) - \sum_{k=1}^{K}\log\Gamma(\alpha_k) \\
& + \sum_{n=1}^{N} \sum_{i=1}^{K} \phi_{ni}(\Psi(\gamma_i) - \Psi(\sum_{j=1}^{K}\gamma_j)) \\
& + \sum_{n=1}^{N} \sum_{i=1}^{K} \sum_{j=1}^{V} \phi_{ni} [w_n = j]\log \beta_{ij}\\
& - \sum_{k=1}^{K}(\gamma_k - 1) (\Psi(\gamma_k) - \Psi(\sum_{j=1}^{K}\gamma_j)) - \log \Gamma(\sum_{k=1}^{K}\gamma_k) + \sum_{k=1}^{K}\log\Gamma(\gamma_k) \\
& - \sum_{n=1}^{N} \sum_{i=1}^{K} \phi_{ni} \log \phi_{ni} \tag{8}
\end{align}
</div>
which is now much easier to deal with.

## Preparing data
Now we dive into the code. For illustration purpose, we use a public dataset from Kaggle. The dataset contains news headlines crawled from ABC News. Here is some code to load the data
``` python
news_data_path = "abcnews-date-text.csv"

gdown.download("https://drive.google.com/uc?id=1BGaMi0XURByE0WM4omDwskoq83WnTXyx",
               news_data_path,
               quiet=False)

data_df = pd.read_csv(news_data_path, error_bad_lines=False);
data_df.head()
```

A small piece of the data will look like this

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

    td, th, tr {
        border: 1px solid #ddd;
    }
</style>
<table border="0" class="dataframe" style="display:table;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>publish_date</th>
      <th>headline_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20030219</td>
      <td>aba decides against community broadcasting lic...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20030219</td>
      <td>act fire witnesses must be aware of defamation</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20030219</td>
      <td>a g calls for infrastructure protection summit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20030219</td>
      <td>air nz staff in aust strike for pay rise</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20030219</td>
      <td>air nz strike to affect australian travellers</td>
    </tr>
  </tbody>
</table>
</div>

## Preprocessing data
There is a total of 1186018 headlines in the original dataset but for a quick experiment, we extract the very first 10000 headlines only
``` python
data = data_df["headline_text"][:10000]
```
Then, we need to do some preprocessing stuff
* Remove the stop words using `stopwords` from `nltk` package
* Build the vocabulary with `word2idx` and `idx2word`
* Create the `corpus` containing all documents

``` python
corpus = []
word2idx = {}
idx2word = {}

for line in data:
    doc = [w for w in line.split(' ') if w not in stopwords.words()]
    for word in doc:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            idx2word[len(idx2word)] = word

    corpus.append(doc)
```

## Global configuration
Next, we set up some global configuration before implementing LDA model.

``` python
max_doc_length = 0
for doc in corpus:
    if max_doc_length < len(doc):
        max_doc_length = len(doc)

class Config:
    corpus = corpus
    word2idx = word2idx
    idx2word = idx2word
    num_vocabs = len(word2idx) # V
    max_doc_length = max_doc_length # N
    va_threshold = 1e-6 # threshold for variational infrence
    em_threshold = 1e-4 # threshold for variational EM
```

## LDA model definition
We then define an `LDA` class to handle the main logic of Variational EM
``` python
class LDA(object):
    def __init__(self,
                 corpus,
                 num_topics,
                 num_words,
                 num_vocabs,
                 word2idx,
                 idx2word):

        self.corpus = corpus # collection of documents
        self.K = num_topics # number of topics in total
        self.V = num_vocabs # number of vocabulary

        self.word2idx = word2idx
        self.idx2word = idx2word

        # model parameters
        self.alpha = None
        self.beta = None

        # sufficient statistics
        self.beta_ss = None
```

## Compute the log-likelihood
Evaluating $(8)$ requires the computation of Gamma and Digamma functions. Fortunately, we can make use of the `scipy` package to handle the heavy work.
``` python
from scipy.special import digamma, loggamma
```
We can then implement the `log_likelihood` function for current document and variational parameters $\gamma$, $\phi$; given model parameters $\alpha, \beta$

```python
def log_likelihood(doc,
                   gamma,
                   phi):
    """
    Compute the (approximate) log-likelihood
    """
    # (K,)
    digamma_derivative = digamma(gamma) - digamma(np.sum(gamma))

    l1 = loggamma(np.sum(self.alpha)) - np.sum(loggamma(self.alpha)) \
        + np.sum((self.alpha - 1) * digamma_derivative) \
        - loggamma(np.sum(gamma)) + np.sum(loggamma(gamma)) \
        - np.sum((gamma - 1) * digamma_derivative)

    l2 = 0
    for i in range(self.K):
        for n in range(len(doc)):
            if phi[n, i] > 0:
                l2 += phi[n, i] * (digamma_derivative[i] \
                    + np.log(self.beta[i, self.word2idx[doc[n]]]) \
                    - np.log(phi[n, i]))

    return l1 + l2
```

## Variational Inference
The goal of variational inference is to find the optimal $\phi^\ast, \gamma^\ast$ for the mean-field distribution $q(z, \theta; \gamma, \phi)$. By taking derivatives of $(8)$ with respect to $\phi, \gamma$ and set it to zero, we obtain coordinate updates
$$
\phi_{ni} \propto \beta_{iv} \exp(\Psi(\gamma_i) - \Psi(\sum_{j=1}^{K}\gamma_j))
$$

$$
\gamma_i = \alpha_i + \sum_{n=1}^{N} \phi_{ni}
$$

where $v$ denotes the unique index of the word $w_n$ in the vocabulary. For more detailed derivation of these updates, we refer to Appendix A (section 3.1, 3.2) of the original LDA paper. Then, we can implement the coordinate ascent algorithm for variational inference as follows
``` python
def variational_inference(self, doc):
    """
    Do the variational inference for each document
    """

    N = len(doc)
    # init variational parameters
    # (N, K)
    phi = np.full(N, self.K), 1.0 / self.K)
    # (K,)
    gamma = self.alpha + N * 1.0 / self.K

    old_likelihood = -math.inf

    # coordinate ascent
    while True:
        # update phi
        for n in range(N):
            for i in range(self.K):
                phi[n, i] = self.beta[i, self.word2idx[doc[n]]] \
                * np.exp(digamma(gamma[i]))

        # normalize phi   
        phi = phi / np.sum(phi, axis=1, keepdims=True)

        # update gamma
        gamma = self.alpha + np.sum(phi, axis=0)

        likelihood = self.log_likelihood(doc, gamma, phi)

        converged = (old_likelihood - likelihood) / likelihood

        old_likelihood = likelihood

        if converged < cfg.va_threshold:
            break

    return phi, gamma, likelihood
```

## Variational EM

Recall that the Variational EM algorithm consisting of
* Initialize parameters $\alpha^{(0)}, \beta^{(0)}$
* For each loop $t$ start from $0$
    * **E step**: For each document $d$, obtain the optimal $\gamma^{(d)}, \phi^{(d)}$ of the variational distribution $q(z, \theta; \gamma, \phi) = q(\theta;\gamma)\prod_{n=1}^{N}q(z_n;\phi_n)$
    * **M step**: Maximize the expected log-likelihood (up to some constant)
    <div>$$
    \mathop{max}_{\alpha^{(t+1)}, \beta^{(t+1)}} \sum_{d=1}^{M} \mathrm{E}_{z, \theta \sim q(z, \theta; \gamma^{(d)}, \phi^{(d)})} {\log p(w, z, \theta ;{\alpha^{(t+1)}, \beta^{(t+1)}}})
    $$</div>
    * If the convergence standard is satisfied, stop

Hence, the Variational EM algorithm can be implemented as function `variational_em` below
``` python
def variational_em(self):
    """
    Fit LDA model using variational EM
    """

    self.init_param()

    old_llhood = -math.inf

    ite = 0

    while True:
        ite += 1
        llhood = self.variational_e_step(self.corpus)
        self.m_step()

        converged = (old_llhood - llhood) / llhood
        old_llhood = llhood

        print("STEP EM: {} - Likelihood: {} - Converged rate: {}".\
            format(ite, llhood, converged))

        if converged < cfg.em_threshold:
            break
```

The function `init_param` is to initialize parameter $\alpha, \beta$
``` python
 def init_param(self):
    """
    Init parameters
    """
    self.alpha = np.full(self.K, 1.0)
    self.beta = np.random.randint(1, 50, (self.K, self.V))
    self.beta = self.beta / np.sum(self.beta, axis=1, keepdims=True)
```
Then, two functions `variational_e_step` and `m_step` are corresponding to `E-step` and `M-step` of the algorithm, respectively.

### E Step
In the `E-step`, we perform variational inference for each document $d$ to obtain $\phi^{(d)}, \gamma^{(d)}$
``` python
def variational_e_step(self, corpus):
    """
    Approximate the posterior distribution

    : corpus - list of documents

    """

    total_likelihood = 0

    self.beta_ss = np.zeros((self.K, self.V)) + 1e-20

    for i, doc in enumerate(corpus):
        phi, gamma, doc_likelihood = self.variational_inference(doc)

        # add to total likelihood
        total_likelihood += doc_likelihood

        # update statistics
        for n in range(len(doc)):
            for k in range(self.K):
                self.beta_ss[k, self.word2idx[doc[n]]] += phi[n, k]

    return total_likelihood
```

### M Step
In `M-step`, we obtain optimal $\alpha, \beta$. Though, the optimal update for $\alpha$ is kind of complex (Appendix A, section 4.2). Thus, for a simple illustration, we consider `alpha` as fixed in the scope of this blog. Setting the derivate of the ELBO $(8)$ with respect to $\beta$ to zero, we yield

$$
\beta_{ij} \propto \sum_{d=1}^{M} \sum_{n=1}^{N} \phi_{ni}^{(d)} w_{n}^{j} \quad \textrm{(Appendix A, section 4.1)}
$$

For coding convenience, we implement these updates right in the function `variational_e_step` and store these unnormalized results in the variable `self.beta_ss`. Hence, in the `M-step`, we just normalize and assign it to `self.beta`
``` python
def m_step(self):
    """
    Maximum likelihood estimation
    """

    # alpha is considered fixed known constant, hence skip here
    # self.alpha

    # (K, V)
    self.beta = self.beta_ss / np.sum(self.beta_ss, axis=1, keepdims=True)

```

## Training

Now everything is setup. We then run the following code for training LDA
```python
model = LDA(corpus=cfg.corpus,
            num_topics=10,
            num_words=cfg.max_doc_length,
            num_vocabs=cfg.num_vocabs,
            word2idx=cfg.word2idx,
            idx2word=cfg.idx2word)

model.variational_em()
```
The output will be like
```
STEP EM: 1 - Likelihood: -357373.4968604174 - Converged rate: 1.798192951590305
STEP EM: 2 - Likelihood: -303489.3001228882 - Converged rate: 0.1775489175918577
STEP EM: 3 - Likelihood: -299860.97107707005 - Converged rate: 0.012100037670076039
STEP EM: 4 - Likelihood: -294189.9329730006 - Converged rate: 0.019276791856062188
STEP EM: 5 - Likelihood: -287787.73198189674 - Converged rate: 0.02224626097510863
STEP EM: 6 - Likelihood: -282157.87102739484 - Converged rate: 0.0199528757925569
STEP EM: 7 - Likelihood: -277893.16594093136 - Converged rate: 0.015346563388931922
...........................................
STEP EM: 32 - Likelihood: -264753.70609608945 - Converged rate: 0.00017462683995490507
STEP EM: 33 - Likelihood: -264713.55673188687 - Converged rate: 0.00015167097861647994
STEP EM: 34 - Likelihood: -264676.9793795508 - Converged rate: 0.0001381961983313688
STEP EM: 35 - Likelihood: -264643.03067223117 - Converged rate: 0.00012828113112741676
STEP EM: 36 - Likelihood: -264611.45086573914 - Converged rate: 0.000119344066134337
STEP EM: 37 - Likelihood: -264582.51780543657 - Converged rate: 0.0001093536358432039
STEP EM: 38 - Likelihood: -264556.4361364331 - Converged rate: 9.85864089506385e-05
```

## Result
After training, we can extract the top words of the 10 "abstract" topics.
``` python
topk = 10

indices = np.argpartition(model.beta, -topk, axis=1)[:, -topk:]

topic_top_words_dict = {}
for k in range(model.K):
    topic_top_words_dict["Topic {}".format(k + 1)] = \
        [model.idx2word[idx] for idx in indices[k]]

topic_df = pd.DataFrame(topic_top_words_dict)
topic_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic 1</th>
      <th>Topic 2</th>
      <th>Topic 3</th>
      <th>Topic 4</th>
      <th>Topic 5</th>
      <th>Topic 6</th>
      <th>Topic 7</th>
      <th>Topic 8</th>
      <th>Topic 9</th>
      <th>Topic 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>air</td>
      <td>murder</td>
      <td>gets</td>
      <td>continue</td>
      <td>denies</td>
      <td>tas</td>
      <td>community</td>
      <td>high</td>
      <td>act</td>
      <td>lead</td>
    </tr>
    <tr>
      <th>1</th>
      <td>takes</td>
      <td>probe</td>
      <td>season</td>
      <td>protesters</td>
      <td>farmers</td>
      <td>south</td>
      <td>public</td>
      <td>ban</td>
      <td>three</td>
      <td>australian</td>
    </tr>
    <tr>
      <th>2</th>
      <td>baghdad</td>
      <td>charged</td>
      <td>trial</td>
      <td>water</td>
      <td>vic</td>
      <td>final</td>
      <td>go</td>
      <td>work</td>
      <td>forces</td>
      <td>union</td>
    </tr>
    <tr>
      <th>3</th>
      <td>election</td>
      <td>aust</td>
      <td>saddam</td>
      <td>found</td>
      <td>korea</td>
      <td>clash</td>
      <td>water</td>
      <td>continues</td>
      <td>claims</td>
      <td>four</td>
    </tr>
    <tr>
      <th>4</th>
      <td>first</td>
      <td>pm</td>
      <td>home</td>
      <td>back</td>
      <td>boost</td>
      <td>get</td>
      <td>missing</td>
      <td>wins</td>
      <td>dead</td>
      <td>minister</td>
    </tr>
    <tr>
      <th>5</th>
      <td>coast</td>
      <td>crash</td>
      <td>top</td>
      <td>plan</td>
      <td>urged</td>
      <td>world</td>
      <td>new</td>
      <td>set</td>
      <td>council</td>
      <td>mp</td>
    </tr>
    <tr>
      <th>6</th>
      <td>oil</td>
      <td>may</td>
      <td>win</td>
      <td>anti</td>
      <td>govt</td>
      <td>call</td>
      <td>killed</td>
      <td>calls</td>
      <td>support</td>
      <td>qld</td>
    </tr>
    <tr>
      <th>7</th>
      <td>coalition</td>
      <td>iraqi</td>
      <td>still</td>
      <td>death</td>
      <td>iraq</td>
      <td>cup</td>
      <td>fire</td>
      <td>funds</td>
      <td>us</td>
      <td>british</td>
    </tr>
    <tr>
      <th>8</th>
      <td>howard</td>
      <td>court</td>
      <td>two</td>
      <td>sars</td>
      <td>north</td>
      <td>australia</td>
      <td>rain</td>
      <td>security</td>
      <td>troops</td>
      <td>wa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>nsw</td>
      <td>police</td>
      <td>woman</td>
      <td>protest</td>
      <td>health</td>
      <td>report</td>
      <td>hospital</td>
      <td>drought</td>
      <td>says</td>
      <td>group</td>
    </tr>
  </tbody>
</table>
</div>

Full code available [here](https://colab.research.google.com/drive/15nqnmXiA3RnfiYPHDrRVMjJVTdszaGJ-?usp=sharing).

__References__

1. Latent Dirichlet Allocation. David M. Blei, Andrew Y. Ng, Michael I. Jordan. 2003 ([pdf](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf))
2. Dataset ([link](https://www.kaggle.com/therohk/million-headlines))
