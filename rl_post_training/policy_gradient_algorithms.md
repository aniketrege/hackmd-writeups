# RL for LLM Post-Training: Foundations of Policy Gradients

> *A ground-up learning resource on the math and mechanics of RL for LLMs. Heavily plagiarized from Nathan Lambert's [rlhfbook](https://rlhfbook.com/) with a few extra info from LLMs and other sources, NOT my own original work.*

---

## 1. Introduction: Why RL Post-Training?
> TODO: rewrite this whole section

Supervised Fine-Tuning (SFT) is highly effective for establishing formatting, tone, and factual foundations in Large Language Models (LLMs). However, SFT inherently suffers from **exposure bias**—where errors accumulate autoregressively at inference time because the model was only trained on ground-truth prefix tokens. Furthermore, standard cross-entropy loss cannot directly optimize for non-differentiable, holistic, or downstream goals, such as human preference alignment, unit test pass rates in code generation, or multi-step reasoning accuracy.

**Reinforcement Learning (RL) post-training** addresses these fundamental limitations by directly optimizing the policy’s parameters to maximize expected performance under a given reward metric or learned reward model.


## 2. Math & Intuition of Policy Gradients

### Notation: Language Models as Episodic MDPs
In the context of LLMs, a prompt is denoted as $x$, and the generated sequence of tokens is denoted as a trajectory $y = (y_1, y_2, \dots, y_T)$, collectively represented as trajectory $\tau = (x, y)$.

To bridge classical Reinforcement Learning (RL) notation with Language Model (LM) post-training, the generation process is modeled as an episodic Markov Decision Process (MDP):

| MDP Concept | Language Model Mapping |
| :--- | :--- |
| **State ($s_t$)** | Prompt and tokens generated so far: $(x, y_{<t})$ |
| **Action ($a_t$)** | Next token $y_t$ |
| **Transition $P(s_{t+1} \mid s_t, a_t)$** | Deterministic: append token $y_t$ to sequence $(x, y_{<t})$ |
| **Policy ($\pi_\theta(a_t \mid s_t)$)** | Next-token probability distribution of the language model |
| **Episode** | One prompt $\rightarrow$ completion sequence |
| **Terminal Reward ($R(\tau)$)** | Reward model (RM) score or program verifier output |
| **Discount Factor ($\gamma$)** | Typically $1.0$ (no discounting) |

*Note: This MDP can be modeled in two ways: **token-level** (where each token is a separate action, typically used in [PPO with GAE](#gae) for dense credit assignment) or **sequence-level** (where the entire completion is treated as one action, used in [REINFORCE](#reinforce) and [GRPO](#grpo).


Distinction between token-level and sequence-level

* Agent ($\pi_\theta$) takes action $a_t$ in state $s_t$, i.e., completion $y_t$ from prompt $x$. Usually $a$ would be a single token but LLM literature often calls the entire completion $y$ (a sequence of tokens) a single action
* $r_t \to$ per-step reward, $\mathcal{R}(\tau)\to$ reward of entire trajectory
![image](https://hackmd.io/_uploads/SyYjT28dzg.png)
-- Figure from rlhfbook [\[Lambert, 2026\]](https://rlhfbook.com/c/06-policy-gradients#deriving-the-policy-gradient)

### Policy Gradient Equation
---
:::success
Goal: make actions more likely when they lead to better outcomes
:::


We define the primary objective function $J(\theta)$ as the **expected return** across trajectories sampled from the parameterized policy $\pi_\theta$:

$$J(\theta) \triangleq \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$$

Where:
* $\theta \rightarrow$ trainable network parameters (weights).
* $\tau \rightarrow$ trajectory $(s_0, a_0, s_1, a_1, ~...)$
* $p_\theta(\tau) \rightarrow$ the pdf or distribution of generating trajectory $\tau$ under policy $\pi_\theta$.
* $R(\tau)\to$ the scalar reward assigned to the entire trajectory, i.e. $\Sigma_{t=0}^\infty \gamma ^t r_t$. 

> Note: LLMs always have finite horizon ($t=0\to T$), where a discount factor $\gamma$ is not essential and typically = 1. For infinite horizon ($t=0\to\infty$), $\gamma$ is necessary because without it the policy can get infinite reward, which cannot be used meaningfully to distinguish between good and bad trajectories and improve $\pi_\theta$ during training.

Now, from the definition of expectation, we can also write $J(\theta)$ as an integral over all possible trajectories $\tau$:

$$J(\theta)= \int_\tau p_\theta(\tau) R(\tau) d\tau$$

To maximize expected reward using gradient ascent, we compute the gradient of $J(\theta)$ with respect to $\theta$:

\begin{align}
g \triangleq \nabla_\theta J(\theta) &= \nabla_\theta \int_\tau p_\theta(\tau) R(\tau) d\tau \\ &= \int_\tau \nabla_\theta p_\theta(\tau) R(\tau) d\tau
\end{align}

:::danger
There are two major issues that make actually computing policy gradient $g$ practically impossible (intractable):
:::
 

**1. Trajectory Space:** In language models with vocabulary size $|V| = 32,000$ and sequence length $T = 512$, the discrete space of all possible completions contains $|V|^T = 32,000^{512}$ sequences. Exhaustive summation over the domain is mathematically impossible.
* Shannon Number: the number of possible Chess games $\sim10^{120}$. Reasonable games (no stupid moves) $\sim10^{40}$
* Eddington Number: protons in the observable universe $\sim10^{80}$

**2. Invalid Probability Distribution:** The term $\nabla_\theta p_\theta(\tau)$ is a vector of partial derivatives, *not* a probability distribution, and so we cannot draw samples from it:
   * **Total Mass is Zero:** Since total probability sums to 1 ($\int p_\theta(\tau) d\tau = 1$), taking the gradient of both sides shows that $\int \nabla_\theta p_\theta(\tau) d\tau = \nabla_\theta (1) = 0$.
   * **Negative Components:** Individual entries in $\nabla_\theta p_\theta(\tau)$ can be negative, violating non-negativity ($P \geq 0$).

#### ❓How do we Compute The Policy Gradient?

Goal: express the gradient as an expectation under a distribution from which we can sample ($p_\theta$)
Idea: apply the **log-derivative identity**:

$$\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
$$

Substituting this identity into our gradient equation yields:


\begin{align}
g \triangleq \nabla_\theta J(\theta) &= \int_\tau \nabla_\theta p_\theta(\tau) R(\tau) d\tau \\ 
&= \int_\tau p_\theta(\tau) \left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right] d\tau
\end{align}

Because $p_\theta(\tau)$ is a valid probability distribution, this integral transforms into a true expectation:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right]$$

<a id="mc_sampling"></a>
### Approximating $\nabla J$ via [Monte Carlo Integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration)
---
We can now approximate the policy gradient $g= \nabla_\theta J(\theta)$ with a Monte Carlo method, i.e., empirically drawing $N$ i.i.d. rollout trajectories $\tau^{(1)}, \tau^{(2)}, \dots, \tau^{(N)} \sim \pi_\theta$:

\begin{align}\nabla_\theta J(\theta) \approx \hat{g}_{\text{MC}} &= \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(\tau^{(i)}) R(\tau^{(i)}) \\ &= \frac{1}{N} \sum_{i=1}^N f(\tau^{(i)}), \text{ where } f(x)\triangleq\nabla_\theta \log p_\theta(x)R(x)
\end{align}


:::info
❓Why is MC desirable?
:::
* **MC is an Unbiased Estimator:** g is the true analytical gradient. The bias of the empirical N-sample MC estimator is defined as:
$$\text{Bias}(\hat{g}_{MC}) \triangleq \mathbb{E}[\hat{g}_{MC}] - g$$
Let's compute the bias:

\begin{align}
\mathbb{E}[\hat{g}_{MC}] &= \mathbb{E}\left[ \frac{1}{N} \sum_{i=1}^N f(\tau^{(i)}) \right] \\
&= \frac{1}{N} \sum_{i=1}^N \mathbb{E}_{\tau^{(i)} \sim p_\theta}[f(\tau^{(i)})] \hspace{1cm} \text{... linearity of }\mathbb{E} \\
 &= \frac{1}{N} \sum_{i=1}^N  \int_\Omega p_\theta(\tau) f(\tau) d\tau \\
&= \frac{1}{N} \sum_{i=1}^N g \hspace{4cm} \text{... }\tau^{(i)} \text{ drawn i.i.d. from }p_\theta(\tau)\\&= \frac{1}{N} (N \cdot g) = g \\
\therefore \text{Bias}(\hat{g}_{MC}) &= 0
\end{align}

* **Strong Law of Large Numbers (LLN):** Assuming samples are drawn i.i.d. and have finite mean ([Lebesgue Integrable](https://en.wikipedia.org/wiki/Lebesgue_integral)), as batch sample size $N \to \infty$, the sample average [converges almost surely](https://en.wikipedia.org/wiki/Convergence_of_random_variables#Almost_sure_convergence) (i.e. with probability 1) to the true expected gradient.

$$\hat{g}_{\text{MC}} = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(\tau^{(i)}) R(\tau^{(i)}) \xrightarrow{\text{a.s.}} \mathbb{E}_{\tau \sim p_\theta} \left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right] = g$$

* **Central Limit Theorem (CLT):** Assuming samples have finite mean and finite positive variance, the standard error of the estimator $\hat{g}_{MC}$ shrinks at the rate of $\mathcal{O}(1 / \sqrt{N})$, independent of the dimensionality of $\theta$ (e.g. 4 billion for a "small" LLM. This avoids the **curse of dimensionality** of numerical integration methods, whose required computations grow exponentially with $d$). Thus means that increasing samples by 4x halves error by 2x, regardless of $d$ (important for RLOO / GRPO rollout batch sizes!)

#### 🚨 Insight: The Off-Policy Dilemma

:::info
The policy gradient expectation requires trajectories to be drawn from the **current** policy $\pi_\theta$. Once an SGD update is applied to yield $\theta'$, old trajectories represent samples from $\pi_{\theta_{\text{old}}}$, not $\pi_{\theta'}$. Direct reuse introduces off-policy bias. Modern algorithms like PPO resolve this via **Importance Sampling** corrections (see [PPO Section](#proximal-policy-optimization-ppo-schulman-et-al-2017))
:::

---

### Back to Policy Gradient

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim p_\theta}[\nabla_\theta\log p_\theta(\tau) R(\tau)]
$$

A trajectory $\tau=(s_0,a_0,s_1,a_1,...a_T,s_T)$'s joint probability decomposes into initial prompt distribution $d_0(s_0)$, policy actions $\pi_\theta(a_t \mid s_t)$, and environment transition dynamics $p(s_{t+1} \mid s_t, a_t)$:

$$p_\theta(\tau) = d_0(s_0) \prod_{t=0}^T \pi_\theta(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t)$$

Taking the logarithm:

$$\log p_\theta(\tau) = \log d_0(s_0) + \sum_{t=0}^T \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^T \log p(s_{t+1} \mid s_t, a_t)$$

Taking the gradient with respect to $\theta$:
* $\nabla_\theta \log d_0(s_0) = 0$ (initial state distribution is independent of policy parameters).
* $\nabla_\theta \log p(s_{t+1} \mid s_t, a_t) = 0$ (environment dynamics / token shifts are independent of policy parameters).

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

::: success
Thus the gradient of the trajectory distribution $p_\theta(\tau)$ reduces to a sum of gradients at each step (token) of the trajectory from the  language model policy probabilities we are learning, $\pi_\theta$.
:::




**🚨 Insight: Why Environment Dynamics Disappear (Model-Free RL)** Policy gradients are strictly **model-free**, i.e. the environment dynamics do not need to be known, modeled, or differentiated through.

Let's now substitute this back to policy gradient equation:

\begin{align}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau\sim p_\theta}\left[R(\tau)\cdot\nabla_\theta\log p_\theta(\tau)\right] \\
&=\mathbb{E}_{\tau\sim p_\theta}\left[ R(\tau)\cdot\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]
\end{align}

Often we write the more general form (**Policy Gradient Theorem**):

$$
g=\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^T ~\Psi_t\cdot\nabla_\theta\log \pi_\theta(a_t \mid s_t)\right]
$$

:::info
🚨 All algorithms discussed in this blog (REINFORCE, RLOO, PPO, GRPO, etc.) are different choices of $\Psi_t$ plus regularization
:::


For Language Models, this sum is over each generated token in the completion. Intuitively,
*  $\nabla_\theta\log \pi_\theta(a_t \mid s_t)\to$ which direction in parameter space makes action $a_t$ more likely, i.e. how does each parameter $\theta$ influence the probability of taking action $a_t$ in state $s_t$ (choosing the "best" next token)
*  $\Psi_t\to$ how good was the action (scalar)
    i. $\Psi_t>0 \to$ update $\theta$ to increase likelihood of $a_t\mid s_t$
    ii. $\Psi_t<0 \to$ update $\theta$ to decrease likelihood of $a_t\mid s_t$


Our choice of $\Psi_t$ determines bias, variance, and compute requirements of our learning algorithm.

---
<a id="psi_t_taxonomy"></a>
### $\Psi_t$ : Expected Future Return from picking $a_t$ in $s_t$

In the simplest case (from our derivation above), 

$$
\Psi_t = R(\tau) = \sum_{t=0}^T~r_t
$$

This has an intuitive issue: gradients should evaluate actions only on future states that they can actually influence i.e. on $t' \geq t$. We can easily replace return $R$ with discounted "Return-to-go", i.e.

$$
\Psi_t = G_t \triangleq \sum_{k=0}^{T-t} \gamma^k~ r_{t+k}
$$

🚨 Very soon we will discuss some other forms of $\Psi_t$ \[[Schulman et al. 2015](https://arxiv.org/abs/1506.02438)\]
<details>
<summary>
    Taxonomy of Other Forms
    </summary>
    
1. **Baselined Return** $= G_t - b(s_t) \to$ subtract empirical estimate of baseline return of state $s_t$ to center estimate. See usage as advantage in [REINFORCE](#reinforce) and [RLOO](#rloo) sections below.
    
2. **Value Function** $V^\pi(s_t) = \mathbb{E}[G_t\mid s_t] \to$ expected future return from state $s_t$. See usage as a learned baseline in PPO [below](#ppo_value_fn). 
    
3. **Q-Function** $Q^\pi(s_t,a_t) = \mathbb{E}[G_t\mid s_t,~ a_t] \to$ expected future return from taking action $a_t$ in state $s_t$.
    
4. **Advantage** $A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)\to$ how much better was taking action $a_t$ than the average from state $s_t$? See usage in PPO [below](#ppo_value_fn).
    
5. **Temporal Difference (TD) Residual $\to$** see usage in [PPO via GAE below](#gae).
</details>



### Baselining: Trying to Reduce Noise
---
:::danger
Raw returns $R(\tau)$ and $G_t$ are noisy!
:::

* The same action $a_t$ can appear in different trajectories $\tau$ with vastly different return because of future sampled actions
* Impossible to differentiate between taking good actions in low return states, and bad actions in high return states. Both will give same $R(\tau)$ and $G_t$.
* If rewards are strictly positive (e.g., $R \in [10, 20]$), every sampled trajectory receives a positive gradient update. Suboptimal responses are actively reinforced, leading to severe sample inefficiency and high variance.

#### 🛠️ Reducing Noise: Baselined Return

Subtracting a baseline $b$ centers the noisy signal by asking "was return $G_t$ better or worse than expected from $s_t$?"

$$
\Psi_t = G_t - b(s_t)
$$

So the policy gradient equation now becomes

\begin{align}
\nabla_\theta J(\theta) &= \mathbb{E}_{a \sim \pi_\theta}\left[ 
\sum_{t=0}^T\nabla_\theta \log\pi_\theta(a_t\mid s_t) \cdot\left(G_t - b(s_t)\right)\right] \\
&= \mathbb{E}\left[ 
\sum_{t=0}^T\nabla_\theta \log\pi_\theta(a_t\mid s_t) \cdot G_t\right] - \mathbb{E}\left[ 
\sum_{t=0}^T\nabla_\theta \log\pi_\theta(a_t\mid s_t) \cdot b(s_t)\right] 
\end{align}

The second term vanishes:
\begin{align}
&- \mathbb{E}\left[ 
\sum_{t=0}^T\nabla_\theta \log\pi_\theta(a_t\mid s_t) \cdot b(s_t)\right]\\
&- 
\sum_{t=0}^T b(s_t)\cdot \mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]\hspace{3cm}\text{... linearity of }\mathbb{E}  \\ 
&- \sum_{t=0}^T b(s_t)\cdot \sum_a \pi_\theta(a \mid s) \cdot  \nabla_\theta \log \pi_\theta(a \mid s)\hspace{9mm}\text{... sum over all possible actions }a \\
&- \sum_{t=0}^T b(s_t)\cdot \sum_a  \nabla_\theta \pi_\theta(a \mid s)\hspace{4cm}\text{... log-derivative identity}
\\
&- \sum_{t=0}^T b(s_t)\cdot \nabla_\theta\sum_a   \pi_\theta(a \mid s)\hspace{39mm}\text{... linearity of }\nabla\\
&- \sum_{t=0}^T b(s_t)\cdot \nabla_\theta(1) = 0\hspace{51mm}\text{... summing over valid prob. dist}^n
\end{align}

⚠️ The baseline term introduces 0 bias as it doesn't shift expected value of the gradient. We are thus free to subtract any function $b$ of $s_t$ we want to reduce variance without affecting bias!

What are common baselines?

* Average reward over batch (simplest, empirical estimate)
* Moving average of recent rewards
* Learned Value Function $V_\phi(s)\to$ bridge to "Actor-Critic" methods like PPO ([see below!](#proximal-policy-optimization-(PPO)-[Schulman-et-al.-2017]))

## 3. Policy Gradient Learning Algorithms

All algorithms I discuss here aim to learn $\pi_\theta$ to maximize expected return and fit into the overall RLHF Framework below:
![image](https://hackmd.io/_uploads/HJrF0elFMx.png)
-- Figure 1, from rlhfbook [\[Lambert, 2026\]](https://rlhfbook.com/c/06-policy-gradients#deriving-the-policy-gradient)

Here,
* $\pi_{ref}$ or $\pi_{base} \to$ anchor the model being trained ($\pi_\theta$ or $\pi_{PPO}$ in Figure 1) so it doesn't drift too far from something reasonable or [reward hack](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/).
* KL Penalty at each token $KL_t$ = $\log\pi_\theta(a_t \mid s_t) - \log\pi_{ref}(a_t \mid s_t)$. Practically:
i. Generate tokens from $\pi_\theta$
ii. Compute log probabilities of these tokens under both $\pi_\theta$ and $\pi_{ref}$
iii. Subtract for per-token $KL_t$
* Final "KL-shaped" reward $\tilde{r}_t = -\beta~KL_t$ for intermediate tokens and $\tilde{r}_t = R(\tau)-\beta~KL_t$ for the last token

🚨 KL regularization is crucial with learned rewards in RLHF as the policy being trained will typically exploit them. With more recent "verifiable" rewards from deterministic engines (RLVR) e.g. code or math, the KL penalty either is typically reduced or unused (e.g. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) drops $\pi_{ref}$ entirely)

<a id="reinforce"></a>
### REINFORCE \[[Williams, 1992](https://dl.acm.org/doi/10.1007/BF00992696)\]
---
The simplest policy gradient algorithm based on Monte Carlo!

<span style="color:red">RE</span>ward <span style="color:red">I</span>ncrement = <span style="color:red">N</span>onnegative <span style="color:red">F</span>actor X <span style="color:red">O</span>ffset <span style="color:red">R</span>einforcement X <span style="color:red">C</span>haracteristic <span style="color:red">E</span>ligibility

* Reward Increment $\to$ Policy Gradient step
* Nonnegative Factor $\alpha\to$ learning rate
* Offset Reinforcement $G_t - b(s_t) \to$ baselined return
* Characteristic Eligibility $\nabla\log\pi_\theta\to$ attribution for which parameters produced the action for that specific scalar return (Offset Reinforcement)

$$\Delta \theta = \alpha\cdot(G_t - b(s_t)) \cdot \nabla \log \pi_\theta (a_t \mid s_t) $$

Circling back to [other forms of $\Psi_t$](#psi_t_taxonomy), $G_t - b(s_t)$ is a form of "**Advantage**" $A_t$ i.e. how much better the realized return is than the average return expected from that state $s_t$.

Using this terminology, we have our REINFORCE policy gradient equation:
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^T ~A_t\cdot\nabla_\theta\log \pi_\theta(a_t \mid s_t)\right]
$$

The simplest REINFORCE baseline calculates a single global scalar average across all $N prompts in the training batch: $$b_\text{batch} = \frac{1}{N} \sum_{i=1}^{N} R(s_i, a_{i})$$

Advantage for prompt $s$: $\hat{A}(s, a) = R(s, a) - b_\text{batch}$.

![image](https://hackmd.io/_uploads/HJ1g3Wgtzg.png)
-- Figure from rlhfbook [\[Lambert, 2026\]](https://rlhfbook.com/c/06-policy-gradients#deriving-the-policy-gradient)

<a id="rloo"></a>
### RLOO \[[Kool et al. 2019](https://openreview.net/forum?id=r1lgTGL5DE)\]
---
> Uses: [MR-Search (03/26)](https://arxiv.org/abs/2603.11327), 

REINFORCE Leave-One-Out: Sample $K\geq2$ completions $a_1, ... a_K$ per prompt $s$. Use $K-1$ of them to estimate baseline.

$$
b(s,a_k) = \dfrac{1}{K-1} \sum_{i=1.i\neq k}^K R(s, a_i)
$$

This results in the advantage applied separately to each prompt:

$$
\hat{A}(s, a_k) = G(s, a_k) - b(s, a_k)
$$

This naturally handles prompt difficulty: hard prompts get low rewards across all completions and thus baseline value is low.

RLOO Example:

| Completion | Reward | Baseline (Avg. of Others) | Advantage |
| -------- | -------- | -------- | -------- |
| 1 | 0.8 | $\frac{0.3 + 0.6 + 0.5}{3} = 0.467$ | +0.333 |
| 2 | 0.3 | $\frac{0.8 + 0.6 + 0.5}{3} = 0.633$ | -0.333 | 
| 3 | 0.6 | $\frac{0.8 + 0.3 + 0.5}{3} = 0.533$ | +0.067 | 
| 4 | 0.5 | $\frac{0.8 + 0.3 + 0.6}{3} = 0.567$ | -0.067 | 

* Completion 1 (highest reward) gets reinforced (positive advantage), Completion 2 $\rightarrow$ suppressed. Completions 3 & 4 get small updates (small advantages)

:::info
⚠️ Just like REINFORCE without a critic (i.e. a Value function), the same sequence-level advantage is broadcast to every token in the sequence.
:::

Both REINFORCE and RLOO lay foundations for algorithms to follow!
* Intuitive links to GRPO and other grouped-completion algorithms
* Don't need a separate Value function (saves memory and compute)
* Simple to implement/debug
* Good baseline in practice ([Ahmadian et al. 2024](https://arxiv.org/abs/2402.14740), [Wang et al. 2024](https://arxiv.org/abs/2410.01257))

![image](https://hackmd.io/_uploads/BkKJKzetMe.png)

<a id="vanilla_pg_issues"></a>
#### ❓ So what's the problem with these critic-free, simple methods? 
---
:::danger
**Variance**! Why?
:::

1. **Sensitivity to step size** $\alpha$ we take to update $\pi_\theta$
i.  $\alpha$ too large $\to$ policy collapse
ii. $\alpha$ too small $\to$ very slow learning
2. **On-Policy Nature**: vanilla policy gradient is on-policy, i.e. $\pi_\theta$ is simultaneously the model being optimized and the data generator for the next training iteration. If we take too large a step, the next training iteration will generate bad data and we can never recover (degenerate)
3. **Baselining**: reduces variance of raw return $G_t$ but critic-free $b(s)$ still depends on either batch size $N$ in REINFORCE (over which we average to compute it; $b$ will be noisy if N is small) or the number of completions per prompt in RLOO ($K-1$)

<details>
<summary>
    Sneak Peek: How do PPO and GRPO further reduce variance?
    </summary>
    
1. **GAE & Value Function Critics (PPO)**: PPO trains a dedicated learned Value Model $V_\phi(s_t)$ that predicts expected future returns at every individual token position. It uses Temporal Difference (TD) residuals $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ and Generalized Advantage Estimation (GAE) to provide token-specific credit assignment, trading a small amount of bias for significantly lower variance
    
2. **Group Normalization (GRPO)**: GRPO normalizes rewards across $G$ completions per prompt using standard deviation ($z$-score: $\hat{A}_i = \frac{R_i - \text{mean}}{\text{std}}$), scaling the variance directly across the prompt's completion group
</details>

<a id="ppo"></a>
### PPO \[[Schulman et al. 2017](https://arxiv.org/abs/1707.06347)\]
---
> Uses: [InstructGPT (03/22)](https://arxiv.org/abs/2203.02155), Honestly too many to count 

:::success
**Idea: Constrained Updates**! Taking larger steps isn't a problem if we **trust** the local **region** we take steps in.
:::
Trust Region Policy Optimization [\[Schulman et al. 2015\]](https://arxiv.org/abs/1502.05477) adds a hard TR constraint but requires an expensive second order optimization.

<details>
<summary>
    A few more details on TRPO and its Issues
    </summary>
    
**Goal**: mathematically guarantee that the new policy $\pi_\theta$ cannot drift further than $\delta$ away from the old policy $\pi_{\theta_{\text{old}}}$ in probability space:
    
$$\mathbb{E}_{\text{states}}\left[ D_{\text{KL}}\left(\pi_{\theta_{\text{old}}}(\cdot \mid s) \;\|\; \pi_\theta(\cdot \mid s)\right) \right] \le \delta$$

**Expensive**: To strictly enforce this hard non-linear constraint at every optimization step, TRPO could not use standard first-order gradient ascent (like SGD or Adam). Instead, it required second-order optimization:
* **Curvature / Hessian Calculation**: TRPO needed to measure the curvature of the policy space, requiring the Fisher Information Matrix (the matrix of second-order partial derivatives of the KL divergence with respect to parameters $\theta$).
* **Conjugate Gradient Solvers**: Because calculating and inverting a full Hessian matrix is impossible for large networks, TRPO relied on iterative conjugate gradient methods and Hessian-vector products to approximate the constrained update direction, followed by a line search.
* Why this fails for LLMs: For an LLM with billions of parameters ($\theta \in \mathbb{R}^{7\text{B}+}$), performing conjugate gradient iterations on curvature matrices across billions of dimensions at every step is computationally prohibitive, memory-intensive, and complex to parallelize across distributed GPUs.
    
**PPO's Fix**: replacing TRPO's hard KL constraint with a simple clipped objective ($\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t$).
    
</details>   <br>


>❓ So how do we constrain our trained policy to be **proximal** to where it started at each step without having to do a second-order optimization? How do we prevent sampled data shifting off-policy?

**Importance Sampling ratio** (from the [Monte-Carlo section](#off-policy-dilemma) above), which asks "how much has $\pi_\theta$ changed at a token level from when it was last updated ($\pi_{\theta_{\text{old}}}$) in the previous RL training batch?"

$$\rho_t(\theta) \triangleq \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

* $\pi_\theta \to$ current policy being optimized
* $\pi_{\theta_{\text{old}}} \to$ policy from the previous training iteration that supplied data used to compute advantage to update $\pi_\theta$
* If $\rho_t = 1 \to$ policy hasn't changed much for given $a_t$
* If $\rho_t > 1 \to$, action $a_t$ is more likely under new policy $\pi_\theta$
* If $\rho_t < 1 \to$, action $a_t$ is less likely under new policy $\pi_\theta$

Recall that by definition, the policy gradient was

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^T ~A^{\pi_\theta}_t\cdot\nabla_\theta\log \pi_\theta(a_t \mid s_t)\right]
$$

:::warning
⚠️ The expectation is over trajectories from the probability distribution $p_\theta$ induced by policy $\pi_\theta$, but in practice we need to compute advantages and gradient updates from a batch of data sampled from $\pi_{\theta_\text{old}}$.
:::

To correct for this distribution mismatch, we can reweight the advantage by $\rho_t(\theta)$ to account for how much more or less likely they are under the policy being optimized ($\pi_\theta$) vs the policy they were sampled from ($\pi_{\theta_\text{old}}$). 

We can construct a **surrogate objective** to optimize this:

$$
J(\theta) = \mathbb{E}[\rho_t(\theta) \cdot \hat{A}_t]
$$


:::danger 
🚨 There are two main issues with optimizing this directly, similar to [those with vanilla policy gradient](#so-what's-the-problem-with-these-critic-free-simple-methods):
:::

1. **Arbitrarily Large Updates:** For any action where $\hat{A}_t > 0$, increasing $\pi_\theta(a_t \mid s_t)$ increases the objective linearly with ratio $\rho_t(\theta)$. An unconstrained optimizer can aggressively push $\pi_\theta(a_t \mid s_t) \to 1$ and drive $\rho_t(\theta) \to \infty$ for those sampled actions
2. **On-Policy Variance**: If $\rho_t(\theta)$ diverges far from 1, the current policy $\pi_\theta$ has shifted into regions of the trajectory space that were rarely sampled during trajectory sampling under $\pi_{\theta_\text{old}}$, making the empirical batch expectation a bad estimator of the true objective

<a id="ppo_clip_loss"></a>
#### PPO's Trust Region: Clipping the Objective!
---
What if we just clip the ratio to a trust region around 1 to prevent policy drift issues, i.e. $1-\epsilon \leq \rho_t \leq 1 + \epsilon$

$$J(\theta) = L^{CLIP} \triangleq \mathbb{E}_t \left[\min\left( \rho_t \hat{A}_t, ~\text{clip} \left( \rho_t, 1-\epsilon, 1+\epsilon \right) \hat{A}_t \right) \right]$$

:::info
🚨 With PPO and its peers, we will stick to working with the objective $J(\theta)$ and not the gradient $\nabla J_\theta$, as the gradient is not as interpretable once we incorporate the $\min$ and $\text{clip}$ operations
:::

![image](https://hackmd.io/_uploads/SJEQHUUYfe.png)

Let's look at all three regions of this space:

:::success
$\hat{A}_t >0$
> We want to increase likelihood of actions under $\pi_\theta$ that gave positive advantage $\hat{A}_t$
:::
1.  $\rho_t < 1 - \epsilon \to$ action $a_t$ is less likely under new policy $\pi_\theta$
 $J = \rho_t \hat{A}_t \to$ normal gradient, push prob up $\to$ **reinforce**
2. $1 - \epsilon < \rho_t < 1 + \epsilon \to$ action equally likely under new policy
 $J = \rho_t \hat{A}_t \to$ normal gradient, push prob up
4. $\rho_t > 1 + \epsilon \to$ action already more likely under new policy 
  $J = (1+\epsilon)\hat{A}_t \to$ zero gradient $\to$ **clipped**
  
:::danger
$\hat{A}_t <0$
> We want to decrease likelihood of actions under $\pi_\theta$ that gave negative advantage $\hat{A}_t$
:::
1.  $\rho_t < 1 - \epsilon \to$ action $a_t$ is already less likely under new policy $\pi_\theta$
   $J = (1-\epsilon)\hat{A}_t \to$ zero gradient $\to$ **clipped**
2. $1 - \epsilon < \rho_t < 1 + \epsilon \to$ action equally likely under new policy
 $J = \rho_t \hat{A}_t \to$ normal gradient, push prob down
4.  $\rho_t > 1+ \epsilon \to$ action $a_t$ is more likely under new policy $\pi_\theta$
 $J = \rho_t \hat{A}_t \to$ normal gradient, push prob down $\to$ **suppress** 

:::warning
$\hat{A}_t =0$
>Zero advantage $\to$ action was exactly as good as average (expected) $\to$ zero gradient, no learning
:::

:::info
**Clipping $\rho$**: caps policy drift ($\pi_{\theta_{\text{old}}} \to \pi_\theta$) when $\pi_\theta$ has already moved in a good direction, but never blocks correction from a bad direction.
:::

![image](https://hackmd.io/_uploads/SyPEwvUFMx.png)

<a id="ppo_value_fn"></a>
#### Learned Baseline: Value Functions
---

PPO is an **actor-critic method**, and learns a value model (critic) alongside the policy (actor) to predict future return from state $s_t$ i.e., per token. This is another form of $\Psi_t$ which we mentioned in the [taxonomy of $\Psi_t$ above](#psi_t_taxonomy).

\begin{align}
V^\pi(s_t) &\triangleq \mathbb{E}[G_t \mid s_t] \\
\hat{A_t} &= G_t - V_\phi(s_t) \\ 
\end{align}


* Vanilla policy gradient (REINFORCE, RLOO) $\to$ estimate $V$ by taking samples from the environment (Monte Carlo). Unbiased but can be high variance!
* PPO $\to$ learns a separate model $V_\phi$ trained on samples from the environment
* $V_\phi$ predicts per-token reward after KL shaping, i.e. $\tilde{r}_t = -\beta KL_t$
* Loss is MSE with actual post-KL return measured from environment i.e. $\mathcal{L}_{V} = \dfrac{1}{2} \left(V_\phi(s_t) - G_t \right)^2, \text{where } G_t=\sum_{k=t}^T \tilde{r}_k$

See [PPO training loop](#ppo_train_loop) below for more details.

<a id="td_residuals"></a>
#### Temporal Difference (TD) Residuals

The TD residual measures how much the value function differs from the actual measured reward in one step (token).

# $$\delta_t^V = \tilde{r}_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

::: info
🚨 GAE \[[Schulman et al. 2016](https://arxiv.org/abs/1506.02438)\] uses the TD residual for estimating advantage for **control over the bias-variance trade-off** of the estimator.
:::

<a id="gae"></a>
#### Generalized Advantage Estimation (GAE): k-step TD

\begin{align}
\hat{A}_t^{(1)} &= \delta_t^V~\hspace{3cm}...\text{ 1-step : low v, high b} \\
\hat{A}_t^{(2)} &= \delta_t^V + \gamma \cdot\delta_{t+1}^V ~\hspace{8mm}...\text{ 2-step}\\
&.\\
&.\\
\hat{A}_t^{(k)} &= \sum_{l=0}^{k-1} \gamma^{l} \cdot\delta_{t+l}^V ~\hspace{11mm}...\text{ k-step: more v, less b}~ ✅ \\
&.\\
&.\\
\hat{A}_t^{(\infty)} &= \sum_{l=0}^\infty \gamma^{l} \cdot r_{t+l} - V_\phi(s_t) \\
 &= \left(r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} ~+~ ... \right) - V_\phi(s_t)  \\ \\
&= G_t - V_\phi(s_t)~\hspace{10mm}...\text{ MC advantage: highest v, zero b} 
\end{align}

TD with fewer steps has lower variance as noise has fewer stochastic future steps to compound over and higher bias as it relies more heavily on the critic's estimate of the future $V_\phi(s_{t+1})$. As the number of steps increases, variance grows while bias drops as we incorporate actual environment rewards rather than critic predictions. This continues until reaching the full MC rollout horizon (theoretically $\infty$ but in practice terminal step $T$ or EOS token of the LLM), which yields zero bias and highest variance. See appendix for a deeper dive into the [bias-variance tradeoff](#bias_var_tradeoff).

:::    success
GAE uses an exponentially weighted average across all k-step estimates 
:::

\begin{align}
\hat{A}_t^{GAE} &\triangleq (1-\lambda) \left(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2\hat{A}_t^{(3)} + ... \right) \\ 
&= \sum_{l=0}^\infty (\lambda\gamma)^l\delta_{t+l}^V, ~\lambda \in [0, 1]
\end{align}

| $\lambda$ | Behavior | Bias | Variance |
| -------- | -------- | -------- | -------- |
| 0 | 1-step TD $A_t^{(1)}$ | high | low|
| 0.95 | GAE default| balanced | balanced |
| 1 | Full MC advantage $A_t^{(\infty)}$ | zero | high|



<a id="ppo_objective"></a>
#### Final PPO Objective
----
The final PPO objective (we are trying to maximize) is:

$$J^{PPO}(\theta) \approx L^{CLIP} - \beta \cdot D_{KL}[\pi_\theta \mid\mid \pi_{ref}]$$

This has two levels of regularization:
1. The [clipping objective from above](ppo_clip_loss) which minimizes per-batch policy drift (trust region)
2. A KL regularization penalty for drift from the original reference policy (typically $\pi_{SFT}$)

These are complementary and non-redundant.

<a id="ppo_train_loop"></a>
#### PPO Training Loop
----
1. **Generate**: sample prompts, generate completions $a$ with current policy $\pi_\theta$
2. **Score**: compute trajectory score $R(\tau)$ with reward model $r_\psi$, and apply per-token KL shaping: \begin{align}\tilde{r}_t &= \begin{cases} -\beta \cdot \text{KL}_t & \text{for } t < T \\ R(\tau) - \beta \cdot \text{KL}_T & \text{for } t = T \end{cases} \\ \\ \text{KL}_T &\triangleq D_{KL}[\pi_\theta \mid\mid \pi_{ref}]\end{align}
3. **Estimate advantages**:  Use learned value function $V_\phi$ and shaped rewards $\tilde{r}_t$ to:
i. **GAE**: Compute 1-step TD residuals $\delta_t^V = \tilde{r}_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$, then calculate $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}^V$.
ii. **Pure MC**: Compute returns $G_t = \sum_{k=t}^T \tilde{r}_k$ and set $A_t^{\text{MC}} = G_t - V_\phi(s_t)$.
4. **Update (K epochs)**: Perform $K$ SGD steps on the rollout batch using the joint loss on the same batch: clipped policy gradient (with ratio $\rho_t = \frac{\pi_\theta}{\pi_{\theta_{old}}}$) + value function MSE loss $\mathcal{L}_V = 1/2 (G_t - V_\phi(s_t))^2$
5. **Repeat**: new batch, sync $\pi_{\theta_{old}} \gets \pi_\theta$

Typical: $K = 2–4$ gradient steps per batch before re-generating completions.

PPO needs 4 models in memory!
| Model| Purpose | Updates? |
| -------- | -------- | -------- | 
| Policy $\pi_\theta$ |  Generates completions $a$ | Yes |
| Value Function $V_\phi$ | Estimate future returns $\mathbb{E}[G_t \mid s_t]$ | Yes |
| Reference Policy $\pi_{ref}$ | KL Penalty for policy drift | Frozen|
| Reward model $r_\psi$ | Scores completions | Frozen|

:::info
Value function training uses on-policy rollouts to compute targets. The model $V_\phi$ predicts $V_t =V_\phi(s_t)$ at each token, which is trained via MSE against the target return $\hat{V}_t = G_t = \sum_t^T r_t$. The advantage $A_t = \hat{V}_t - V_t$ then weights the policy gradient update.

> (Note: In this conceptual figure, intermediate KL penalties are omitted ($\beta = 0$), making intermediate rewards $r_t = 0$ for $t < T$ and keeping $G_t$ constant at $0.73$ across all tokens. In practical RLHF ($\beta > 0$), intermediate tokens receive $-\beta \text{KL}_t$ penalties, causing $G_t$ to shift token-by-token as rewards accumulate.)

![image](https://hackmd.io/_uploads/rkz4QcvKfe.png)
:::

<a id="grpo"></a>
### GRPO \[[Shao et al. 2024](https://arxiv.org/abs/2402.03300)\]
---
> [DeepSeekMath (02/24)](https://arxiv.org/abs/2402.03300), [DeepSeek-R1 (01/25)](https://arxiv.org/abs/2501.12948), 

:::    success 
**Idea**: We can drop $V_\phi$ entirely by using group statistics to approximate advantage.

Sounds familiar? Similar to k completion sampling of [RLOO](#rloo)!
:::

Concretely, instead of using a value network $V_\phi(s_t)$ to predict expected returns at every state, GRPO samples a group of $G$ completions $\{a_1, a_2, \dots, a_G\}$ for the same prompt $s$ from the rollout policy $\pi_{\theta_{\text{old}}}$.

For each completion $g \in \{1, \dots, G\}$, the reward model $r_\psi$ (or rule-based verifier) assigns a sequence score $R_i = R(s, a_i)$. GRPO computes a normalized group $z$-score advantage:

$$\hat{A}_i = \frac{R_i - \text{mean}(R_1, \dots, R_G)}{\text{std}(R_1, \dots, R_G) + \epsilon}$$

Some features:
* Above average completions get $\hat{A}_i > 0$ and below average get negative advantage
*  Like REINFORCE and RLOO, this single scalar advantage $\hat{A}_t$ is computed for the full completion and broadcast identically to every token $t$ in completion $g$
* Simple to implement and debug compared to PPO
* Saves memory (No $V_\phi$ needed)
* Good fit for RLVR where rewards are both trustworthy and higher variance (0 or 1 for correctness)

#### GRPO Objective

GRPO combines this group-relative advantage with PPO's per-token importance-sampling ratio and trust-region clipping:

$$
J(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \left( \min\left( \rho_{i,t} \hat{A}_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i \right) - \beta \, D_{\text{KL}}\left(\pi_\theta \parallel \pi_{\text{ref}}\right)_{i,t} \right)
$$

where:

* **Per-Token Importance Ratio**: $\rho_{i,t} = \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})}$, where $s_{i,t}$ is the state at step $t$ (prompt $s$ + preceding token actions $a_{i,<t}$) and $a_{i,t}$ is the token action at step $t$.
* **Sequence-Level (Broadcasted) Advantage:** while $\rho$ is per-token, the same advantage $\hat{A}_i$ is broadcast to all tokens in the completion $i$
* **KL Penalty Directly in Objective**: Unlike PPO (which shapes intermediate token rewards $\tilde{r}_t$ with KL), GRPO penalizes policy drift $D_{\text{KL}}$ directly inside the loss objective.

![image](https://hackmd.io/_uploads/Hyu1WBttze.png)

:::    success
Overview of Policy Gradient Algorithms
::: 
| Algorithm | Advantage Calculation | Advantage Granularity | Update Style | Models in Memory | KL Penalty |
|-----| -----| -----|----- | -----| -----|
| [REINFORCE](#reinforce) | $A_k = G_k - \text{mean}(R_\text{batch})$ | Sequence-level ($k$)| Unconstrained |3 ($\pi_\theta, \pi_{\text{ref}}, r_\psi$)| Optional (in Reward) | 
| [RLOO](#rloo) |   $A_k = R_k - \frac{1}{K-1}\sum_{i \neq k}^K R_i$ |Sequence-level ($k$) | Unconstrained |3 ($\pi_\theta, \pi_{\text{ref}}, r_\psi$)| Optional (in Reward) |
|[PPO](#ppo) | \begin{align}A_t= G_t - V_\phi(s_t) \hspace{5mm }&\text{ ... Pure MC} \\ A_t=\sum_{l=0}^T(\gamma\lambda)^l \delta_{t+l}^V\hspace{5mm }&\text{ ... GAE}\end{align}| Token-level ($t$)| Importance Sampling ($\rho$) Clipping |4 ($\pi_\theta, V_\phi, \pi_{\text{ref}}, r_\psi$)| In Reward |
| [GRPO](#grpo) | $A_k = \dfrac{R_k - \mu_G}{\sigma_G}$ | Sequence-level ($k$) | Importance Sampling ($\rho$) Clipping | 3 ($\pi_\theta, \pi_{\text{ref}}, r_\psi$) <br /> 2 no KL (drop $\pi_\text{ref}$) <br /> 1 RLVR (drop $r_\psi$) | In Loss (default, optional) |

> Note: [Dr. GRPO](https://arxiv.org/abs/2503.20783) proposes removing the std dev ($\sigma_G$) term from GRPO - this fixes the problem of the model being encouraged to be consistent / not diverse (low $\sigma_G$ $\to$ high $A$), but this downweights "needle-in-a-haystack" correct answers (e.g. 1 of G are correct) which are likely very valuable learning signal for the model. Dr. GRPO's advantage is the same as RLOO up to a scaling constant

<details>
<summary>
    Wait What? Dr. GRPO and RLOO have the same advantage?
    </summary>
Well, up to a scaling constant.
  
\begin{align}
    \hat{A}_\text{drgrpo} &= r_i - \text{mean}(r_1, ... r_G) = r_i - \dfrac{1}{G}\sum_{j=1}^G r_j    \\
    \hat{A}_\text{rloo} &= r_i  - \dfrac{1}{G-1}\sum_{j=1, j\neq i}^G r_j   
\end{align}
    
The proof is left as an exercise to the reader (I've always wanted to write that!) 
    
Hint: multiply $\hat{A}_\text{drgrpo}$ by $\dfrac{G}{G-1}$ and see where it goes.
    
</details>


#### Other Group-Relative Advantage Variants: GSPO and CISPO

:::danger
Problem: aggregating per-token importance ratios $\rho_{i,t}$ across long sequences (large $t$) and/or large, sparse models (e.g. modern mixture-of-experts (MoE) models) is numerically unstable. e.g., a single token with a large ratio can dominate the update.
:::

**GSPO**: have a single, length-normalized importance weight $\rho_i$ per response $i$ using the geometric mean across tokens $t$, which stays in a numerically stable range for long sequences:

$$\rho_i(\theta) = \exp\left( \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \log \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})} \right)$$

The objective is exactly GRPO with sequence-level importance sampling ratio $\rho_{i} \gets \rho_{i,t}$

$$J_{\text{GSPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \min\left( \rho_i(\theta) \hat{A}_i, \; \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right)$$

**CISPO**:  rather than clipping the objective (which zeros gradient outside trust region $\to$ no learning for those tokens), CISPO clips the importance weights themselves using a stop-gradient operator $\text{sg}(\cdot)$ and (often asymmetric) bounds $[1-\epsilon^-, 1+\epsilon^+]$ e.g.  $[0.8, 1.2]$

\begin{align}
\hat{\rho}_{i,t} &= \text{clip}\left( \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})}, \; 1-\epsilon^-, \; 1+\epsilon^+ \right) \\\\
J_{\text{CISPO}}(\theta) &= \frac{1}{G} \sum_{i=1}^G \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \text{sg}\left(\hat{\rho}_{i,t}\right) \hat{A}_{i,t} \log \pi_\theta(a_{i,t} \mid s_{i,t})
\end{align}

The stop-gradient acts as an identity function during the forward pass ($\text{sg}(x) = x$), but treats its input as a fixed constant during backpropagation ($\nabla_\theta ~\text{sg}(x) = 0$). Why?

When we write a policy gradient objective of the form: $$J(\theta) = w(\theta) \cdot \log \pi_\theta(a_t \mid s_t)$$

Then by product rule: $$\nabla_\theta J(\theta) = \underbrace{\left(\nabla_\theta w(\theta)\right) \cdot \log \pi_\theta(a_t \mid s_t)}_{\text{unwanted gradient through weight}} + w(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

We do not want backpropagation to differentiate through the importance weight calculation itself. We want the clipped ratio $w = \hat{\rho}_{i,t}$ to act purely as a fixed scalar coefficient multiplying the policy's log-likelihood gradient. By applying the stop-gradient operator $\text{sg}\left(\hat{\rho}_{i,t}\right)$ (or ``.detach()`` in PyTorch), autodiff simplifies directly to: $$\nabla_\theta J_{\text{CISPO}}(\theta) = \hat{\rho}_{i,t} \cdot \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})$$This ensures that the weight caps the magnitude of the step without distorting the direction of the log-likelihood gradient.

In CISPO, every token receives a gradient update, the weight simply caps maximum amplification. Asymmetric bounds ($\epsilon^+ > \epsilon^-$) allow more aggressive updates for positive advantages, encouraging exploration

### Summarizing Policy Gradient

:::    info
In most RLHF setups, data quality and reward signal quality dominate; algorithm choice mostly determines stability, efficiency, and engineering burden. 
:::

All methods optimize the same policy gradient objective:

$$
\nabla_\theta J(\theta) =  \mathbb{E}_{\tau\sim\pi_\theta} \left[ \sum_t \Psi_t \cdot \nabla_\theta \log\pi_\theta(a_t \mid s_t) \right] 
$$

They differ primarily in:

1. **Choice of** $\Psi_t\to$ baselined return (REINFORCE, RLOO), Advantage (PPO), Grouped Relative Advantage (GRPO, GSPO, CISPO)
2. **How updates are bounded** $\to$ Unconstrained (REINFORCE, RLOO), clipping objective (PPO, GRPO), clipping weights (CISPO)
3. **Memory and Compute** $\to$ 2-4 models (potentially 1 for RLVR e.g. DeepSeek R1 Zero), training separate $V_\phi$ or not

<a id="opd"></a>
## 4. On-Policy Distillation
### SDPO [\[Hübotter et al., Jan '26\]](https://arxiv.org/abs/2601.20802)
---




### MOPD [\[Ma et al., June \'26\]](https://arxiv.org/abs/2606.30406)
---

* Online Imitation Learning - Online because KL for all trajectories is generated by $\pi_{student}$ itself

* SDFT: Sequential Learning Without Catastrophic Forgetting

Other things I want to cover:
* Flash REINFORCE
* Finetuning with sampling (Aayush Karan, Yilun Du)

## Other Resources

1. https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/
2. https://louieworth.github.io/blog/opd_reflection/
3. https://www.open-thoughts.ai/blog/launch
4. https://cameronrwolfe.substack.com/p/agentic-rl?utm_source=profile&utm_medium=reader2
5. SELF-DISTILLATION ENABLES CONTINUAL LEARNING, Shenfeld et al.

## Appendix

<a id="bias_var_tradeoff"></a>
### Deep Dive: Bias-Variance Tradeoff

Let $g = \nabla_\theta J(\theta)$ denote the true, analytical gradient of expected return that we wish to compute. Let $\hat{g}$ denote our empirically estimated gradient (constructed from rollout samples via a baseline or learned critic predictions).

The **Bias** of an estimator captures the systematic error between its expected value $\hat{g}$ (avg over all possible random rollout batches) and the true analytical value $g$.

$$
\text{Bias}(\hat{g}) \triangleq \mathbb{E}[\hat{g}] - g $$

The **Variance** of an estimator measures how much individual empirical gradient estimates scatter or fluctuate from batch to batch around their own mean

$$
\text{Var}(\hat{g}) \triangleq \mathbb{E}\left[ \|\hat{g} - \mathbb{E}[\hat{g}]\|^2 \right]
$$

The tradeoff comes from how we learn the estimator via mean squared error:

$$\text{MSE}(\hat{g}) \triangleq \mathbb{E}\left[ \|\hat{g} - g\|^2 \right] = \|\text{Bias}(\hat{g})\|^2 + \text{Var}(\hat{g})$$

* **High Bias (Steering problem):** SGD steps are stable and smooth but steer the policy in the wrong direction because of imperfect estimates or flawed reward proxy. e.g. 1-step TD residuals using an inaccurate critic $V_\phi$ will have high bias as $V_\phi$ will point the gradient in an inaccurate direction

* **High Variance (Noise problem):** SGD steps jump erratically from batch to batch due to stochastic sampling. e.g. MC advantages are unbiased (which we [proved above](mc_sampling)) but high variance, i.e. on average the sampled gradient steps $\hat{g}$ point in the exact direction of the true expected gradient $g$ but are noisy in each batch