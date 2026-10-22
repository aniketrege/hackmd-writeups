# RL for LLM Post-Training: Foundations of Policy Gradients

> *A ground-up learning resource on the math and mechanics of RL for LLMs. Heavily adapted from Nathan Lambert's RLHF book \[[Lambert, 2025](https://rlhfbook.com/)\], with additional material from the primary papers and other sources cited throughout.*

> Slide Deck I made as an intro to this writeup: [link](https://docs.google.com/presentation/d/1xmfaAemFJ4_Nnw-6rDdo4ZaEC5bOIOgZk40ziHhjXEU/edit?usp=sharing).

---

## 1. Introduction: Why RL Post-Training?

Supervised Fine-Tuning (SFT) or Instruction Tuning trains a large language model (LLM) to assign high likelihood to demonstration responses for given prompts (instructions). This is effective when high-quality targets are available, but it optimizes token-level imitation rather than the downstream property we actually care about. Many post-training goals like human preferences (safe, helpful, harmless, non-sycophantic), coding, math, or tool use are evaluated on complete sampled responses and may be non-differentiable (e.g. for retrieval, how would you formalize a "high recall" loss token-wise? If you do, please get in touch with me!).

**Reinforcement Learning (RL) post-training** provides a way to optimize the expected value of such a reward directly. The model generates responses under its current policy, those responses are scored, and policy-gradient estimators increase the probability of actions associated with above-baseline outcomes.

Because RL trains on model-generated responses, it can also reduce the teacher-forcing mismatch sometimes called **exposure bias**. This is a secondary motivation rather than a guarantee: language models can sometimes recover from imperfect prefixes \[[He et al., 2021](https://aclanthology.org/2021.emnlp-main.415/)\], and RL introduces its own problems, including noisy gradients, reward hacking, entropy collapse, and policy drift.

The biggest challenge with RL is typically NOT writing down the expected reward objective. It is estimating and optimizing its gradient with tolerable variance while keeping policy updates stable.

## 2. Math & Intuition of Policy Gradients

### Notation: Language Models as Episodic MDPs
In the context of LLMs, a prompt is denoted $x$ and the generated completion $y = (y_1, y_2, \dots, y_T)$; together they form a trajectory $\tau = (x, y)$.

To bridge classical RL notation with LLM post-training, the generation process is modeled as an episodic Markov Decision Process (MDP):

| MDP Concept | Language Model Mapping |
| :--- | :--- |
| **State ($s_t$)** | Prompt and tokens generated so far: $(x, y_{<t})$ |
| **Action ($a_t$)** | Next token $y_t$ |
| **Transition $p(s_{t+1} \mid s_t, a_t)$** | Deterministic: append token $y_t$ to sequence $(x, y_{<t})$ |
| **Policy ($\pi_\theta(a_t \mid s_t)$)** | Next-token probability distribution of the language model |
| **Episode** | One prompt $\rightarrow$ completion sequence |
| **Terminal Reward ($R(\tau)$)** | Reward model (RM) score or program verifier output |
| **Discount Factor ($\gamma$)** | Typically $1.0$ (no discounting) |

There are two useful abstractions:
1. The **token-level MDP view**: each token $a_t = y_t$ is an action taken in state $s_t = (x, y_{<t})$, with per-step reward $r_t$ (used by [PPO with GAE](#gae) for per-token credit assignment).
2. The **sequence-level contextual-bandit view** (common in LLM papers): the prompt is the state $s = x$, the whole completion is a single action $a = y$, and it receives a single reward $R(\tau)$ for its overall quality.

> [REINFORCE](#reinforce), [RLOO](#rloo), and [GRPO](#grpo) assign rewards and advantages at the sequence level, but the update still differentiates through every token's log-probability, since $\log \pi_\theta(y \mid x) = \sum_t \log \pi_\theta(y_t \mid x, y_{<t})$.

![image](https://hackmd.io/_uploads/SyYjT28dzg.png)
-- The RL loop for LLMs (sequence-level view), from \[[Lambert, 2025](https://rlhfbook.com/c/06-policy-gradients#fig:rlhf-overview)\]

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
* $p_\theta(\tau) \rightarrow$ the probability mass assigned to discrete trajectory $\tau$ under policy $\pi_\theta$.
* $R(\tau)\to$ the scalar return of the trajectory, i.e. $\sum_{t=0}^{T-1}\gamma^t r_t$.

> Note: LLM generation has a finite horizon $T$, so no discounting is needed and typically $\gamma=1$.

Because token trajectories are discrete, the expectation is a sum over all possible trajectories (general RL texts often write an integral):

$$J(\theta)= \sum_\tau p_\theta(\tau) R(\tau)$$

To maximize expected reward using gradient ascent, we compute the gradient of $J(\theta)$ with respect to $\theta$:

\begin{align}
g \triangleq \nabla_\theta J(\theta)
&= \nabla_\theta \sum_\tau p_\theta(\tau)R(\tau)\\
&= \sum_\tau \nabla_\theta p_\theta(\tau)R(\tau).
\end{align}

:::danger
Two issues make computing the policy gradient $g$ directly intractable:
:::

**1. Trajectory Space:** In language models with vocabulary size $|V| = 32,000$ and sequence length $T = 512$, the discrete space of all possible completions contains $|V|^T = 32,000^{512} \approx 10^{2306}$ sequences. Exhaustive summation is computationally impossible (for scale, there are $\sim10^{80}$ atoms in the observable universe (Eddington's number estimated $\approx 1.6\times10^{79}$ protons), and at least $\sim10^{120}$ possible chess games (the Shannon number)).

**2. Invalid Probability Distribution:** The term $\nabla_\theta p_\theta(\tau)$ is a vector of partial derivatives, *not* a probability distribution, and so we cannot draw samples from it:
   * **Total Mass is Zero:** Since $\sum_\tau p_\theta(\tau)=1$, taking the gradient gives $\sum_\tau\nabla_\theta p_\theta(\tau)=\nabla_\theta 1=0$.
   * **Negative Components:** Individual entries in $\nabla_\theta p_\theta(\tau)$ can be negative, violating non-negativity ($P \geq 0$).

#### ❓How do we Compute The Policy Gradient?

Goal: express the gradient as an expectation under a distribution from which we can sample ($p_\theta$)
Idea: apply the **log-derivative identity**:

$$\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
$$

Substituting this identity into our gradient equation yields:


\begin{align}
g \triangleq \nabla_\theta J(\theta)
&= \sum_\tau \nabla_\theta p_\theta(\tau)R(\tau)\\
&= \sum_\tau p_\theta(\tau)\left[\nabla_\theta\log p_\theta(\tau)R(\tau)\right].
\end{align}

Because $p_\theta(\tau)$ is a valid probability distribution, this sum is a true expectation:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right]$$

<a id="mc_sampling"></a>
### Approximating $\nabla J$ via [Monte Carlo Integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration)
---
We can now approximate the policy gradient $g= \nabla_\theta J(\theta)$ with a Monte Carlo method, i.e., empirically drawing $N$ i.i.d. rollout trajectories $\tau^{(1)}, \tau^{(2)}, \dots, \tau^{(N)} \sim p_\theta$:

\begin{align}\nabla_\theta J(\theta) \approx \hat{g}_{\text{MC}} &= \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(\tau^{(i)}) R(\tau^{(i)}) \\ &= \frac{1}{N} \sum_{i=1}^N f(\tau^{(i)}), \text{ where } f(\tau)\triangleq\nabla_\theta \log p_\theta(\tau)R(\tau)
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
 &= \frac{1}{N} \sum_{i=1}^N \sum_{\tau} p_\theta(\tau) f(\tau) \hspace{2cm} \text{... each }\tau^{(i)} \sim p_\theta\\
&= \frac{1}{N} \sum_{i=1}^N g \hspace{4cm} \text{... log-derivative identity above}\\&= \frac{1}{N} (N \cdot g) = g \\
\therefore \text{Bias}(\hat{g}_{MC}) &= 0
\end{align}

* **Strong Law of Large Numbers (LLN):** Assuming samples are drawn i.i.d. and have finite mean ([Lebesgue Integrable](https://en.wikipedia.org/wiki/Lebesgue_integral)), as batch sample size $N \to \infty$, the sample average [converges almost surely](https://en.wikipedia.org/wiki/Convergence_of_random_variables#Almost_sure_convergence) (i.e. with probability 1) to the true expected gradient.

$$\hat{g}_{\text{MC}} = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(\tau^{(i)}) R(\tau^{(i)}) \xrightarrow{\text{a.s.}} \mathbb{E}_{\tau \sim p_\theta} \left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right] = g$$

* **Central Limit Theorem (CLT):** Assuming finite mean and finite positive variance, the standard error of the estimator $\hat{g}_{MC}$ shrinks at the rate $\mathcal{O}(1 / \sqrt{N})$: 4x more rollouts halves the error. Crucially, unlike exhaustive enumeration or grid-based integration, whose cost grows exponentially with trajectory length (**the curse of dimensionality**), this *rate* does not depend on the dimensionality of trajectory space. The variance itself can still grow with sequence length, which is why the variance-reduction tricks below matter. This also explains why more completions per prompt in [RLOO](#rloo) or [GRPO](#grpo) give more stable baselines, albeit with diminishing returns.
<details>
<summary>
    What is dimensionality here?
    </summary>

For a fixed prompt and completion length $T$, a completion is a vector of possible token choices:

$$
\tau = (y_1, \ldots, y_T), \qquad y_t\in\{1,\ldots,|V|\}
$$

There are $|V|$ choices along each of $T$ dimensions, giving $|V|^T$ possible trajectories. This exponential growth of the discrete trajectory space, and hence the cost of exhaustive summation over it, is the curse of dimensionality.
</details>

<a id="off-policy-dilemma"></a>
#### 🚨 Insight: The Off-Policy Dilemma

:::info
The policy gradient expectation requires trajectories to be drawn from the **current** policy $\pi_\theta$. In LM RL, collecting fresh rollouts can be slow and expensive: the model must generate completions, score them with a reward model or verifier, and sometimes evaluate them under a reference model. It is thus much more sample efficient to take multiple optimizer steps with a single rollout batch. While the first update is on-policy, once an SGD update changes the policy, old trajectories are samples from $\pi_{\theta_{\text{old}}}$ rather than the updated (current) policy $\pi_\theta$. Naively reusing the batch no longer gives the original on-policy estimator. PPO uses likelihood (importance sampling) ratios and a clipped surrogate objective (soft trust region) to make limited reuse more stable, but does not exactly correct for arbitrary policy drift or guarantee a hard trust region (see [PPO Section](#ppo)). The same mismatch arises in asynchronous RL, when the rollout workers lag behind the policy being trained \[[Noukhovitch et al., 2025](https://arxiv.org/abs/2410.18252)\].
:::

---

### Back to Policy Gradient

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim p_\theta}[\nabla_\theta\log p_\theta(\tau) R(\tau)]
$$

A trajectory $\tau=(s_0,a_0,s_1,a_1,\dots,a_{T-1},s_T)$'s joint probability decomposes into initial prompt distribution $d_0(s_0)$, policy actions $\pi_\theta(a_t \mid s_t)$, and environment transition dynamics $p(s_{t+1} \mid s_t, a_t)$:

$$p_\theta(\tau) = d_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) p(s_{t+1} \mid s_t, a_t)$$

Taking the logarithm:

$$\log p_\theta(\tau) = \log d_0(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log p(s_{t+1} \mid s_t, a_t)$$

Taking the gradient with respect to $\theta$:
* $\nabla_\theta \log d_0(s_0) = 0$ (initial state distribution is independent of policy parameters).
* $\nabla_\theta \log p(s_{t+1} \mid s_t, a_t) = 0$ (environment dynamics / token shifts are independent of policy parameters).

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

::: success
Thus the gradient of the trajectory log-probability $\log p_\theta(\tau)$ reduces to a sum of per-step (per-token) gradients of the language model policy $\pi_\theta$ we are learning.
:::

**🚨 Insight: Why Environment Dynamics Disappear** This policy-gradient estimator is **model-free** in the sense that the environment dynamics do not need to be known or differentiated through. This does not forbid combining policy gradients with learned environment models.

Let's now substitute this back to policy gradient equation:

\begin{align}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau\sim p_\theta}\left[R(\tau)\cdot\nabla_\theta\log p_\theta(\tau)\right] \\
&=\mathbb{E}_{\tau\sim p_\theta}\left[ R(\tau)\cdot\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]
\end{align}

More generally, $R(\tau)$ can be replaced by a per-step weight $\Psi_t$ \[[Schulman et al., 2016](https://arxiv.org/abs/1506.02438)\]:

$$
g=\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^{T-1} ~\Psi_t\cdot\nabla_\theta\log \pi_\theta(a_t \mid s_t)\right]
$$

The classic **Policy Gradient Theorem** \[[Sutton et al., 2000](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)\] is the case $\Psi_t = Q^{\pi}(s_t, a_t)$.

:::info
🚨 This template is a useful organizing principle: REINFORCE and RLOO use Monte Carlo advantage estimators, while PPO, GRPO, and related methods additionally optimize clipped or normalized surrogate objectives. Not every arbitrary choice of $\Psi_t$ preserves the exact policy gradient.
:::


For Language Models, this sum is over each generated token in the completion. Intuitively,
*  $\nabla_\theta\log \pi_\theta(a_t \mid s_t)\to$ which direction in parameter space makes action $a_t$ more likely, i.e. how does each parameter $\theta$ influence the probability of taking (sampling) action $a_t$ in state $s_t$ (e.g. the next token in a LM)
*  $\Psi_t\to$ how good was the action (scalar)
    i. $\Psi_t>0 \to$ update $\theta$ to increase likelihood of $a_t\mid s_t$
    ii. $\Psi_t<0 \to$ update $\theta$ to decrease likelihood of $a_t\mid s_t$


Our choice of $\Psi_t$ determines bias, variance, and compute requirements of our learning algorithm.

---
<a id="psi_t_taxonomy"></a>
### Choosing $\Psi_t$

In the simplest case (from our derivation above), 

$$
\Psi_t = R(\tau) = \sum_{t'=0}^{T-1} r_{t'}
$$

This is noisier than necessary: action $a_t$ cannot influence rewards received before step $t$. Replacing $R(\tau)$ with the (discounted) **return-to-go** drops those terms without biasing the gradient \[[Achiam, 2018](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)\]:

$$
\Psi_t = G_t \triangleq \sum_{k=0}^{T-t-1} \gamma^k~ r_{t+k}
$$

> With a terminal-only reward (the common LLM case: $r_t=0$ for $t<T-1$), $G_t = R(\tau)$ at every token. Return-to-go only starts to matter once per-token rewards such as the [KL penalty](#kl_shaping) are added.

🚨 Other choices of $\Psi_t$ \[[Schulman et al., 2016](https://arxiv.org/abs/1506.02438)\]:
<details>
<summary>
    Taxonomy of Other Forms
    </summary>
    
1. **Baselined Return** $= G_t - b(s_t) \to$ subtract empirical estimate of baseline return of state $s_t$ to center estimate. See usage as advantage in [REINFORCE](#reinforce) and [RLOO](#rloo) sections below.
    
2. **Value Function** $V^\pi(s_t) = \mathbb{E}[G_t\mid s_t] \to$ expected future return from state $s_t$. See usage as a learned baseline in PPO [below](#ppo_value_fn). 
    
3. **Q-Function** $Q^\pi(s_t,a_t) = \mathbb{E}[G_t\mid s_t,~ a_t] \to$ expected future return from taking action $a_t$ in state $s_t$.
    
4. **Advantage** $A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)\to$ how much better was taking action $a_t$ than the average from state $s_t$? See usage in PPO [below](#ppo_value_fn).
    
5. **Temporal Difference (TD) Residual** $\delta_t^V = r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t) \to$ see usage in [PPO via GAE below](#gae).
</details>



### Baselining: Trying to Reduce Noise
---
:::danger
Raw returns $R(\tau)$ and $G_t$ are noisy!
:::

* The same action $a_t$ can appear in different trajectories $\tau$ with vastly different return because of future sampled actions
* Raw returns conflate how good the *state* was with how good the *action* was: a good action taken from a bad state can receive a lower return than a bad action taken from a good state.
* If rewards are strictly positive (e.g., $R \in [10, 20]$), every sampled trajectory receives a positive gradient update. Suboptimal responses are actively reinforced, leading to severe sample inefficiency and high variance.

#### 🛠️ Reducing Noise: Baselined Return

Subtracting a baseline $b$ centers the noisy signal by asking "was return $G_t$ better or worse than expected from $s_t$?"

$$
\Psi_t = G_t - b(s_t)
$$

So the policy gradient equation now becomes

\begin{align}
\nabla_\theta J(\theta) &= \mathbb{E}_{\tau \sim p_\theta}\left[ 
\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t\mid s_t) \cdot\left(G_t - b(s_t)\right)\right] \\
&= \mathbb{E}\left[ 
\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t\mid s_t) \cdot G_t\right] - \mathbb{E}\left[
\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t\mid s_t) \cdot b(s_t)\right]
\end{align}

The baseline term vanishes for every $t$. Condition on the state $s_t$ (law of total expectation); since $b(s_t)$ does not depend on $a_t$, it factors out of the inner expectation over $a_t \sim \pi_\theta(\cdot\mid s_t)$:
\begin{align}
&\mathbb{E}\left[ 
\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t\mid s_t) \cdot b(s_t)\right]\\
&= \sum_{t=0}^{T-1} \mathbb{E}_{s_t}\Big[ b(s_t)\cdot \mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]\Big]\hspace{1cm}\text{... linearity of }\mathbb{E}\text{, then condition on }s_t  \\ 
&= \sum_{t=0}^{T-1} \mathbb{E}_{s_t}\Big[ b(s_t)\cdot \sum_a \pi_\theta(a \mid s_t) \cdot  \nabla_\theta \log \pi_\theta(a \mid s_t)\Big]\hspace{9mm}\text{... sum over all possible actions }a \\
&= \sum_{t=0}^{T-1} \mathbb{E}_{s_t}\Big[ b(s_t)\cdot \sum_a  \nabla_\theta \pi_\theta(a \mid s_t)\Big]\hspace{4cm}\text{... log-derivative identity}
\\
&= \sum_{t=0}^{T-1} \mathbb{E}_{s_t}\Big[ b(s_t)\cdot \nabla_\theta\sum_a   \pi_\theta(a \mid s_t)\Big]\hspace{39mm}\text{... linearity of }\nabla\\
&= \sum_{t=0}^{T-1} \mathbb{E}_{s_t}\big[ b(s_t)\cdot \nabla_\theta(1)\big] = 0\hspace{4cm}\text{... summing over valid prob. dist}^n
\end{align}

⚠️ The baseline adds no bias: it doesn't shift the expected gradient. We are thus free to subtract any function $b(s_t)$ that does not depend on the action $a_t$ to reduce variance without affecting bias!

What are common baselines?

* Average reward over other samples in a batch (a leave-one-out version avoids using a sample in its own baseline)
* Moving average of recent rewards
* Learned Value Function $V_\phi(s)\to$ bridge to actor-critic methods like [PPO](#ppo)

## 3. Policy Gradient Learning Algorithms

All algorithms I discuss here aim to learn $\pi_\theta$ to maximize expected return and fit into the overall RLHF Framework below:
![image](https://hackmd.io/_uploads/HJrF0elFMx.png)
-- Figure 1, adapted from \[[Lambert et al., 2022](https://huggingface.co/blog/rlhf)\]

<a id="kl_shaping"></a>
Here (Figure 1's $\lambda_{KL}$ and $r_\theta$ are our $\beta$ and reward model $r_\psi$):
* $\pi_{ref}$ or $\pi_{base} \to$ anchor the model being trained ($\pi_\theta$ or $\pi_{PPO}$ in Figure 1) so it doesn't drift too far from something reasonable or reward hack \[[Weng, 2024](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)\].
* Sampled log-ratio at each token $KL_t = \log\pi_\theta(a_t \mid s_t) - \log\pi_{ref}(a_t \mid s_t)$, computed on the rollout tokens (so $\pi_\theta = \pi_{\theta_{\text{old}}}$ here). It is a single-sample, unbiased estimate of $D_{\mathrm{KL}}(\pi_\theta(\cdot\mid s_t)\|\pi_{ref}(\cdot\mid s_t))$, not the exact KL at that state.
* Final "KL-shaped" reward $\tilde{r}_t = -\beta KL_t$ for intermediate tokens and $\tilde{r}_t = R(\tau)-\beta KL_t$ for the last token

🚨 KL regularization is especially useful with learned rewards because the trained policy can exploit reward-model errors. With verifiable rewards from deterministic code or math checkers, some recent systems reduce or omit the KL penalty (e.g. DAPO \[[Yu et al., 2025](https://arxiv.org/abs/2503.14476)\]).

<a id="reinforce"></a>
### REINFORCE \[[Williams, 1992](https://doi.org/10.1007/BF00992696)\]
---
The simplest policy gradient algorithm based on Monte Carlo!

<span style="color:red">RE</span>ward <span style="color:red">I</span>ncrement = <span style="color:red">N</span>onnegative <span style="color:red">F</span>actor X <span style="color:red">O</span>ffset <span style="color:red">R</span>einforcement X <span style="color:red">C</span>haracteristic <span style="color:red">E</span>ligibility

* Reward Increment $\to$ Policy Gradient step
* Nonnegative Factor $\alpha\to$ learning rate
* Offset Reinforcement $R(\tau) - b \to$ baselined return
* Characteristic Eligibility $\nabla_\theta\log\pi_\theta\to$ attribution for which parameters produced the action for that specific scalar return (Offset Reinforcement)

$$\Delta \theta = \alpha\cdot(R(\tau) - b) \cdot \sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta (a_t \mid s_t) $$

Williams' episodic REINFORCE uses the total episode return $R(\tau)$; the modern form uses the return-to-go and a state-dependent baseline, $G_t - b(s_t)$ ; recall $G_t = R(\tau)$ for terminal-only LLM rewards.

Circling back to [other forms of $\Psi_t$](#psi_t_taxonomy), $G_t-b(s_t)$ is an **advantage estimator**: it estimates how much better the realized return was than the baseline expected from state $s_t$. Using this terminology, we have our REINFORCE policy gradient equation:
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^{T-1} ~\hat A_t\cdot\nabla_\theta\log \pi_\theta(a_t \mid s_t)\right]
$$

One simple REINFORCE baseline is a moving average of rewards from previous batches. A same-batch mean is also commonly used:
$$b_\text{batch} = \frac{1}{N} \sum_{i=1}^{N} R(s_i, a_i).$$

Advantage for prompt $s$: $\hat{A}(s, a) = R(s, a) - b_\text{batch}$.

> Subtlety: because this mean includes the current sample, it is not strictly action-independent and scales the expected gradient by $(N-1)/N$. A leave-one-out mean or an explicit $N/(N-1)$ correction removes this finite-batch bias. A single global mean also does not account for prompt difficulty.

![image](https://hackmd.io/_uploads/HJ1g3Wgtzg.png)
-- REINFORCE, from \[[Lambert, 2025](https://rlhfbook.com/c/06-policy-gradients#fig:reinforce-arch)\]

<a id="rloo"></a>
### RLOO \[[Kool et al., 2019](https://openreview.net/forum?id=r1lgTGL5DE)\]
---
> Adapted to LLMs by [Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740). Uses: MR-Search \[[Xiao et al., 2026](https://arxiv.org/abs/2603.11327)\]

REINFORCE Leave-One-Out: Sample $K\geq2$ completions $a_1, \dots, a_K$ per prompt $s$. Each completion's baseline is the mean reward of the other $K-1$:

$$
b(s,a_k) = \dfrac{1}{K-1} \sum_{i=1, i\neq k}^K R(s, a_i)
$$

giving a per-completion advantage applied separately to each prompt:

$$
\hat{A}(s, a_k) = R(s, a_k) - b(s, a_k)
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
* Good baseline in practice, often matching or even beating PPO for RLHF \[[Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740); [Wang et al., 2025](https://arxiv.org/abs/2410.01257)\]

![image](https://hackmd.io/_uploads/BkKJKzetMe.png)
-- RLOO, from \[[Lambert, 2025](https://rlhfbook.com/c/06-policy-gradients#fig:rloo-arch)\]

<a id="vanilla_pg_issues"></a>
#### ❓ So what's the problem with these simple methods? 
---
:::danger
**The learning bottleneck is expensive LM rollouts; vanilla PG uses each batch for a single update:** sample, take one policy step, discard the batch, resample. This keeps the estimator on-policy and unbiased, but is wasteful.
:::

1. **No data reuse (the reason PPO exists):** $\pi_\theta$ is both the model being optimized *and* the data generator, so after one update the batch is stale. Classically, REINFORCE/RLOO therefore take **one gradient step, throw the batch away, and resample** \[[Schulman et al., 2017](https://arxiv.org/abs/1707.06347)\]. LM rollouts (generate $\to$ score $\to$ optionally evaluate $\pi_{\text{ref}}$) are the expensive part, so we *want* several optimizer steps per batch, but those later steps are off-policy and no longer estimate $\nabla J$ under the current $\pi_\theta$ (see [the off-policy dilemma](#off-policy-dilemma)). PPO's importance ratios and clipping exist to make this limited reuse safer.
2. **Step size:** even a single on-policy step can be destructive if the learning rate $\alpha$ is too large (policy collapse), while too small an $\alpha$ learns very slowly. PPO's clipping does not help with the *first* step on a batch (where $\rho_t = 1$); learning rate and gradient-norm clipping control that.
3. **Noisy baselines:** baselining reduces variance of raw return $G_t$ but critic-free baselines are still sample *estimates*: the batch mean in REINFORCE (noisy if $N$ is small) or the other $K-1$ completions in RLOO.

<details>
<summary>
    Sneak Peek: How do PPO and GRPO further reduce variance?
    </summary>
    
1. **GAE & Value Function Critics (PPO)**: PPO trains a dedicated learned Value Model $V_\phi(s_t)$ that predicts expected future returns at every individual token position. It uses Temporal Difference (TD) residuals $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ and Generalized Advantage Estimation (GAE) to provide token-specific credit assignment, trading a small amount of bias for significantly lower variance
    
2. **Group Normalization (GRPO)**: GRPO centers and scales rewards across $G$ completions per prompt ($\hat{A}_i = \frac{R_i-\text{mean}}{\text{std}}$). This standardizes update scale within each prompt, but it also changes how prompts are weighted and can introduce a question-difficulty bias.
</details>

<a id="ppo"></a>
### PPO \[[Schulman et al., 2017](https://arxiv.org/abs/1707.06347)\]
---
> Uses: Fine-Tuning LMs from Human Preferences \[[Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)\], Learning to Summarize \[[Stiennon et al., 2020](https://arxiv.org/abs/2009.01325)\], InstructGPT \[[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)\], Anthropic HH \[[Bai et al., 2022](https://arxiv.org/abs/2204.05862)\], Llama 2 \[[Touvron et al., 2023](https://arxiv.org/abs/2307.09288)\]; honestly too many to count.

:::success
**Idea: Constrained Updates**! The expensive part is generating the batch! We want to reuse each rollout batch for several optimizer steps. These extra steps are safer if we stay **proximal** to the policy that produced the data (i.e. nearly on-policy).
:::
Trust Region Policy Optimization (TRPO) \[[Schulman et al., 2015](https://arxiv.org/abs/1502.05477)\] adds a hard trust-region constraint but requires an expensive second-order optimization.

<details>
<summary>
    A few more details on TRPO and its Issues
    </summary>
    
**Goal**: constrain the new policy $\pi_\theta$ to stay within $\delta$ (in average KL) of the old policy $\pi_{\theta_{\text{old}}}$:
    
$$\mathbb{E}_{\text{states}}\left[ D_{\text{KL}}\left(\pi_{\theta_{\text{old}}}(\cdot \mid s) \;\|\; \pi_\theta(\cdot \mid s)\right) \right] \le \delta$$

**Expensive**: To strictly enforce this hard non-linear constraint at every optimization step, TRPO could not use standard first-order gradient ascent (like SGD or Adam). Instead, it required second-order optimization:
* **Curvature / Hessian Calculation**: TRPO needed to measure the curvature of the policy space, requiring the Fisher Information Matrix (the matrix of second-order partial derivatives of the KL divergence with respect to parameters $\theta$).
* **Conjugate Gradient Solvers**: Because calculating and inverting a full Hessian matrix is impossible for large networks, TRPO relied on iterative conjugate gradient methods and Hessian-vector products to approximate the constrained update direction, followed by a line search.
* Why this fails for LLMs: For an LLM with billions of parameters ($\theta \in \mathbb{R}^{7\text{B}+}$), performing conjugate gradient iterations on curvature matrices across billions of dimensions at every step is computationally prohibitive, memory-intensive, and complex to parallelize across distributed GPUs.
    
**PPO's Fix**: replacing TRPO's hard KL constraint with a simple clipped objective ($\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t$) optimized with first-order methods. PPO gives no hard trust region or monotonic-improvement guarantee (though practical TRPO's average-KL version is also only a heuristic approximation of its theory). The PPO paper also tests an adaptive KL-penalty variant, which performed worse than clipping.
    
</details>   <br>


>❓ So how do we constrain our trained policy to be **proximal** to where it started at each step without having to do a second-order optimization? How do we prevent sampled data shifting off-policy?

**Importance Sampling ratio** (see [the off-policy dilemma](#off-policy-dilemma) above), which asks "how much more or less likely is token $a_t$ under the current policy $\pi_\theta$ than under the rollout policy ($\pi_{\theta_{\text{old}}}$) that generated the batch?"

$$\rho_t(\theta) \triangleq \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

* $\pi_\theta \to$ current policy being optimized
* $\pi_{\theta_{\text{old}}} \to$ frozen snapshot of the policy that generated the current rollout batch (and its advantages)
* If $\rho_t = 1 \to$ policy hasn't changed for this $a_t$
* If $\rho_t > 1 \to$, action $a_t$ is more likely under new policy $\pi_\theta$
* If $\rho_t < 1 \to$, action $a_t$ is less likely under new policy $\pi_\theta$

Recall that by definition, the policy gradient was

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^{T-1} ~A^{\pi_\theta}_t\cdot\nabla_\theta\log \pi_\theta(a_t \mid s_t)\right]
$$

:::warning
⚠️ The expectation is over trajectories from the probability distribution $p_\theta$ induced by policy $\pi_\theta$, but in practice we need to compute advantages and gradient updates from a batch of data sampled from $\pi_{\theta_\text{old}}$.
:::

At each sampled state, $\rho_t(\theta)$ is used to scale (reweight) the estimated advantage $\hat{A}_t$, reflecting how much more or less probable the observed action $a_t$ is under the current policy $\pi_\theta$ versus the old policy $\pi_{\theta_{\text{old}}}$ that generated the data. This forms the basis for PPO's local surrogate objective. However, this approach does **not** perform full importance weighting across entire trajectories, and thus does not fully correct for changes in the state distribution that arise after policy updates.

This gives the (unclipped) **surrogate objective** from conservative policy iteration \[[Kakade & Langford, 2002](https://dl.acm.org/doi/10.5555/645531.656005); [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)\]:

$$
L^{CPI}(\theta) = \mathbb{E}_t[\rho_t(\theta) \cdot \hat{A}_t]
$$

Since $\nabla_\theta \rho_t = \rho_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ and $\rho_t = 1$ at $\theta = \theta_{\text{old}}$, its gradient at the start of each batch is exactly the policy gradient above.

:::danger 
🚨 There are two main issues with optimizing this directly, similar to [those with vanilla policy gradient](#vanilla_pg_issues):
:::

1. **Very Large Updates:** For any action where $\hat{A}_t > 0$, the objective grows linearly in $\rho_t(\theta)$, and $\rho_t$ can grow up to $1/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$, which is huge for low-probability tokens.
2. **Off-Policy Estimation Error**: If $\rho_t(\theta)$ drifts far from 1, $\pi_\theta$ has shifted toward trajectories that were rarely sampled under $\pi_{\theta_\text{old}}$, so the batch average becomes a poor (high-variance) estimate of the true objective.

<a id="ppo_clip_loss"></a>
#### PPO's Trust Region: Clipped Surrogate Objective!
---
What if we just clip the ratio to a trust region around 1 to prevent policy drift issues, i.e. $1-\epsilon \leq \rho_t \leq 1 + \epsilon$

$$J_{\text{PPO}}(\theta) = L^{CLIP}(\theta) \triangleq \mathbb{E}_t \left[\min\left( \rho_t \hat{A}_t, ~\text{clip} \left( \rho_t, 1-\epsilon, 1+\epsilon \right) \hat{A}_t \right) \right]$$

:::info
🚨 With PPO and its peers, we will stick to working with the surrogate objective (e.g. $L^{CLIP}$, which is *maximized*; code minimizes the loss $-L^{CLIP}$) and not the gradient, as the gradient is not as interpretable once we incorporate the $\min$ and $\text{clip}$ operations
:::

![image](https://hackmd.io/_uploads/SJEQHUUYfe.png)
-- PPO's clipped objective vs. ratio $\rho_t$ (plotted as $R(\theta)$), from \[[Lambert, 2025](https://rlhfbook.com/c/06-policy-gradients#fig:ppo-obj)\]

Let's look at all three regions of this space (writing $J$ for $J_{\text{PPO}}$):

:::success
$\hat{A}_t >0$
> We want to increase likelihood of actions under $\pi_\theta$ that gave positive advantage $\hat{A}_t$
:::
1.  $\rho_t < 1 - \epsilon \to$ action $a_t$ is less likely under new policy $\pi_\theta$
 $J = \rho_t \hat{A}_t \to$ normal gradient, push prob up $\to$ **reinforce**
2. $1 - \epsilon < \rho_t < 1 + \epsilon \to$ action roughly as likely under new policy
 $J = \rho_t \hat{A}_t \to$ normal gradient, push prob up
3. $\rho_t > 1 + \epsilon \to$ action already more likely under new policy 
  $J = (1+\epsilon)\hat{A}_t \to$ zero gradient $\to$ **clipped**
  
:::danger
$\hat{A}_t <0$
> We want to decrease likelihood of actions under $\pi_\theta$ that gave negative advantage $\hat{A}_t$
:::
1.  $\rho_t < 1 - \epsilon \to$ action $a_t$ is already less likely under new policy $\pi_\theta$
   $J = (1-\epsilon)\hat{A}_t \to$ zero gradient $\to$ **clipped**
2. $1 - \epsilon < \rho_t < 1 + \epsilon \to$ action roughly as likely under new policy
 $J = \rho_t \hat{A}_t \to$ normal gradient, push prob down
3.  $\rho_t > 1+ \epsilon \to$ action $a_t$ is more likely under new policy $\pi_\theta$
 $J = \rho_t \hat{A}_t \to$ normal gradient, push prob down $\to$ **suppress** 

:::warning
$\hat{A}_t =0$
>Zero advantage $\to$ action was exactly as good as average (expected) $\to$ zero gradient, no learning
:::

:::info
**Clipping $\rho$**: caps policy drift ($\pi_{\theta_{\text{old}}} \to \pi_\theta$) when $\pi_\theta$ has already moved in a good direction, but never blocks correction from a bad direction. It does not literally clip every ratio or guarantee a bound on total KL policy drift.
:::

![image](https://hackmd.io/_uploads/SyPEwvUFMx.png)
-- PPO, from \[[Lambert, 2025](https://rlhfbook.com/c/06-policy-gradients#fig:ppo-arch)\]

<a id="ppo_value_fn"></a>
#### Learned Baseline: Value Functions
---

PPO is an **actor-critic method**: it learns a value model (critic) alongside the policy (actor) to predict expected future return from every prefix state $s_t$. This supplies a per-token baseline (see the [taxonomy of $\Psi_t$ above](#psi_t_taxonomy)).

\begin{align}
V^\pi(s_t) &\triangleq \mathbb{E}[G_t \mid s_t] \\
\hat{A}_t &= G_t - V_\phi(s_t)
\end{align}

* Vanilla policy gradient (REINFORCE, RLOO) $\to$ estimates its baseline from sampled returns (Monte Carlo). Unbiased but can be high variance!
* PPO $\to$ learns a separate model $V_\phi$ trained on samples from the environment
* $V_\phi(s_t)$ predicts the cumulative future KL-shaped return from prefix $s_t$, i.e. the sum of future $\tilde{r}_k$ including the final reward $R(\tau)$, NOT the immediate per-token reward (one step)
* Loss is MSE against a return target $\hat V_t$: $\mathcal{L}_{V} = \dfrac{1}{2}\left(V_\phi(s_t)-\hat V_t\right)^2$. The simplest target is the Monte Carlo return $G_t=\sum_{k=t}^{T-1}\tilde r_k$ ($\gamma=1$). Standard implementations use the GAE $\lambda$-return $\hat V_t = \hat A_t^{\mathrm{GAE}} + V_{\phi_{\mathrm{old}}}(s_t)$ (see [below](#gae)), which equals $G_t$ when $\lambda=1$ \[[Huang et al., 2024](https://arxiv.org/abs/2403.17031)\]. Many also **clip** $V_\phi$ to within $\pm\epsilon$ of $V_{\phi_{\mathrm{old}}}$ and take the max of the clipped and unclipped losses, analogous to clipping $\rho_t$.

See [PPO training loop](#ppo_train_loop) below for more details.

<a id="td_residuals"></a>
#### Temporal Difference (TD) Residuals

The TD residual is the critic's one-step **Bellman error**. The true value function satisfies the Bellman equation \[[Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html), §3.5\]:

$$V^\pi(s_t) = \mathbb{E}\left[\tilde r_t + \gamma V^\pi(s_{t+1}) \mid s_t\right]$$

The TD residual measures how far the critic is from satisfying it on a sampled step. It compares the prediction $V_\phi(s_t)$ with the bootstrapped **TD target**: the reward actually received plus the critic's estimate of everything after it.

$$\delta_t^V = \underbrace{\tilde{r}_t + \gamma V_\phi(s_{t+1})}_{\text{TD target}} - \underbrace{V_\phi(s_t)}_{\text{current prediction}}$$

* **TD error in classical RL:** this is the error signal of TD learning \[[Sutton, 1988](https://doi.org/10.1007/BF00115009); [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html), §6.1\], where TD(0) trains the critic via $V(s_t) \leftarrow V(s_t) + \alpha\,\delta_t$. (Sutton & Barto write $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$; their $R_{t+1}$ is our $\tilde r_t$.)
* **Advantage estimate:** if $V_\phi = V^\pi$, then $\mathbb{E}[\delta_t^V \mid s_t, a_t] = Q^\pi(s_t,a_t) - V^\pi(s_t) = A^\pi(s_t,a_t)$, so $\delta_t^V$ is a one-step advantage estimate \[[Schulman et al., 2016](https://arxiv.org/abs/1506.02438)\]. For LLMs, where transitions are deterministic, this holds exactly rather than only in expectation.

::: info
🚨 GAE \[[Schulman et al., 2016](https://arxiv.org/abs/1506.02438)\] uses the TD residual for estimating advantage for **control over the bias-variance trade-off** of the estimator.
:::

<a id="gae"></a>
#### Generalized Advantage Estimation (GAE): k-step TD

\begin{align}
\hat{A}_t^{(1)} &= \delta_t^V~\hspace{3cm}...\text{ 1-step: low variance, high bias} \\
\hat{A}_t^{(2)} &= \delta_t^V + \gamma \cdot\delta_{t+1}^V ~\hspace{8mm}...\text{ 2-step}\\
&.\\
&.\\
\hat{A}_t^{(k)} &= \sum_{l=0}^{k-1} \gamma^{l} \cdot\delta_{t+l}^V ~\hspace{11mm}...\text{ k-step: more variance, less bias} \\
&.\\
&.\\
\hat{A}_t^{(\infty)} &= \sum_{l=0}^\infty \gamma^{l} \cdot \tilde{r}_{t+l} - V_\phi(s_t) \\
 &= \left(\tilde r_t + \gamma \tilde r_{t+1} + \gamma^2 \tilde r_{t+2} ~+~ ... \right) - V_\phi(s_t)  \\ \\
&= G_t - V_\phi(s_t)~\hspace{10mm}...\text{ MC advantage: highest variance, no critic-induced bias}
\end{align}

TD with fewer steps usually has lower variance because fewer stochastic future rewards enter the estimate, but it relies more heavily on the critic and is biased when the critic is inaccurate in estimating the future return $V_\phi(s_{t+1})$. As the number of steps increases, variance tends to grow while critic-induced bias drops as we incorporate actual environment rewards rather than critic predictions. At the complete MC rollout horizon (terminal step $T$ or EOS token of the LLM), the Monte Carlo advantage has no critic-induced bias, but the highest variance. See the appendix for a deeper dive into the [bias-variance tradeoff](#bias_var_tradeoff).

:::    success
GAE uses an exponentially weighted average across all k-step estimates 
:::

\begin{align}
\hat{A}_t^{GAE} &\triangleq (1-\lambda) \left(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2\hat{A}_t^{(3)} + ... \right) \\ 
&= \sum_{l=0}^\infty (\lambda\gamma)^l\delta_{t+l}^V, ~\lambda \in [0, 1]
\end{align}

(The mixture form degenerates at $\lambda=1$, but the TD-residual sum remains well-defined.) For LLMs the sum is truncated at the end of the completion, $l \leq T-t-1$.

| $\lambda$ | Behavior | Bias | Variance |
| -------- | -------- | -------- | -------- |
| 0 | 1-step TD $\hat A_t^{(1)}$ | high if critic is inaccurate | low|
| 0.95 | common default (original PPO; the GAE paper found $\lambda \in [0.9, 0.99]$ works best) | balanced | balanced |
| 1 | Full MC advantage | zero critic-induced bias | high|



<a id="ppo_objective"></a>
#### PPO Loss Components
----
PPO maximizes $L^{CLIP}$ (code minimizes $-L^{CLIP}$) while jointly training the critic:

1. **Actor:** the [clipped objective](#ppo_clip_loss) $L^{CLIP}(\theta)$, with $\hat A_t$ computed by [GAE](#gae) from the [KL-shaped rewards](#kl_shaping) $\tilde r_t$. The KL to $\pi_{\mathrm{ref}}$ therefore enters the actor update only through $\hat A_t$, not as a separate loss term.
2. **Critic:** the [value loss](#ppo_value_fn) $\mathcal{L}_V$ against the same $\tilde r$-based return targets, optionally with value clipping.

The PPO paper's full objective, $L^{CLIP} - c_1\mathcal{L}_V + c_2 S[\pi_\theta]$, also includes an entropy bonus $S$, which common LLM implementations leave off by default. Note the two different anchors: clipping keeps $\pi_\theta$ close to the **rollout policy** $\pi_{\theta_{\text{old}}}$ (refreshed every batch), while the KL penalty keeps it close to the **fixed reference** $\pi_{\mathrm{ref}}$ (typically $\pi_{\mathrm{SFT}}$). Some methods instead put an explicit differentiable KL term in the loss (e.g. [GRPO](#grpo_objective)) or drop it entirely (common in RLVR).

<a id="ppo_train_loop"></a>
#### PPO Training Loop
----
1. **Generate**: sample prompts, generate completions $a$ with current policy $\pi_\theta$
2. **Score**: compute trajectory score $R(\tau)$ with reward model $r_\psi$, and apply per-token KL shaping using the sampled log-ratio $KL_t=\log\pi_{\theta_{\text{old}}}(a_t\mid s_t)-\log\pi_{ref}(a_t\mid s_t)$ (the RM score is added at the final token):
\begin{align}
\tilde{r}_t &=
\begin{cases}
-\beta KL_t & \text{for } t < T-1 \\
R(\tau)-\beta KL_t & \text{for } t=T-1.
\end{cases}
\end{align}
3. **Estimate advantages**:  Use learned value function $V_\phi$ and KL-shaped rewards $\tilde{r}_t$ to:

    i. **GAE**: Compute 1-step TD residuals $\delta_t^V = \tilde{r}_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$, then calculate $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}^V$ and value targets $\hat V_t = \hat A_t + V_\phi(s_t)$.

    ii. **Pure MC**: Compute returns $G_t = \sum_{k=t}^{T-1} \tilde{r}_k$ and set $\hat{A}_t^{\text{MC}} = G_t - V_\phi(s_t)$ and value targets $\hat V_t = \hat A_t + V_\phi(s_t)$.

4. **Update ($\mu$ epochs)**: Perform $\mu$ epochs of minibatch updates on the rollout batch using the clipped policy objective (with ratio $\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$) + value function loss.
5. **Repeat**: new batch, sync $\pi_{\theta_{old}} \gets \pi_\theta$

Typical: $\mu = 2$–$4$ epochs over each batch (e.g. 4 in \[[Huang et al., 2024](https://arxiv.org/abs/2403.17031)\]) before re-generating completions.

RLHF with PPO uses four models:
| Model| Purpose | Updates? |
| -------- | -------- | -------- | 
| Policy $\pi_\theta$ |  Generates completions $a$ | Yes |
| Value Function $V_\phi$ | Estimate future returns $\mathbb{E}[G_t \mid s_t]$ | Yes |
| Reference Policy $\pi_{ref}$ | KL Penalty for policy drift | Frozen|
| Reward model $r_\psi$ | Scores completions | Frozen|

They can but don't need to be four full model weights in memory (expensive): the policy and value function may share a backbone plus a value head, offload frozen networks, or replace the learned reward model with a programmatic verifier (RLVR).

:::info
Value function training uses on-policy rollouts to compute targets. The model $V_\phi$ predicts $V_t =V_\phi(s_t)$ at each token, which is trained via MSE against the target return $\hat{V}_t$ (here the MC return $G_t = \sum_{k=t}^{T-1} \tilde r_k$; see note below). The advantage $A_t = \hat{V}_t - V_t$ then weights the policy gradient update.

> (Note: this conceptual figure omits intermediate KL penalties ($\beta=0$), so intermediate rewards $\tilde r_t=0$ for $t<T-1$, and uses $\lambda=1$, so $\hat V_t = G_t = 0.73$ at every token. With per-token KL shaping, intermediate tokens receive $-\beta KL_t$, so $\hat V_t$ changes token-by-token.)

![image](https://hackmd.io/_uploads/rkz4QcvKfe.png)
-- Value function training, from \[[Lambert, 2025](https://rlhfbook.com/c/06-policy-gradients#fig:value_fn_training)\]
:::

<a id="grpo"></a>
### GRPO \[[Shao et al., 2024](https://arxiv.org/abs/2402.03300)\]
---
> Introduced in DeepSeekMath. Uses: DeepSeek-R1 \[[DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)\], and most open RLVR recipes since

:::    success 
**Idea**: We can drop $V_\phi$ entirely by using group statistics to approximate advantage.

Sounds familiar? Similar to the $K$-completion sampling of [RLOO](#rloo)!
:::

Concretely, instead of using a value network $V_\phi(s_t)$ to predict expected returns at every state, GRPO samples a group of $G$ completions $\{a_1, a_2, \dots, a_G\}$ for the same prompt $s$ from the rollout policy $\pi_{\theta_{\text{old}}}$.

For each completion $i \in \{1, \dots, G\}$, the reward model $r_\psi$ (or rule-based verifier) assigns a sequence score $R_i = R(s, a_i)$. GRPO computes a normalized group $z$-score advantage:

$$\hat{A}_i = \frac{R_i - \text{mean}(R_1, \dots, R_G)}{\text{std}(R_1, \dots, R_G)}$$

(Implementations add a small constant to the std for groups where every completion gets the same reward; those groups get $\hat A_i = 0$ and contribute no learning signal.)

Some features:
* Above average completions get $\hat{A}_i > 0$ and below average get negative advantage
*  Like REINFORCE and RLOO, this single scalar advantage $\hat{A}_i$ is computed for the full completion and broadcast identically to every token $t$ in completion $i$
* Simple to implement and debug compared to PPO
* Saves memory (No $V_\phi$ needed)
* Good fit for RLVR, where a verifier supplies a trustworthy but often sparse outcome reward such as binary correctness (1 for correct, 0 for wrong)

<a id="grpo_objective"></a>
#### GRPO Objective

GRPO combines this group-relative advantage with PPO's per-token importance-sampling ratios and trust-region clipping:

$$
J_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \left( \min\left( \rho_{i,t} \hat{A}_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i \right) - \beta \, \widehat D_{\mathrm{KL},i,t} \right)
$$

where:

* **Per-Token Importance Ratio**: $\rho_{i,t} = \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})}$, where $s_{i,t}$ is the state at step $t$ (prompt $s$ + preceding token actions $a_{i,<t}$) and $a_{i,t}$ is the token action at step $t$.
* **Sequence-Level (Broadcasted) Advantage:** while $\rho$ is per-token, the same advantage $\hat{A}_i$ is broadcast to every token $t$ in completion $i$
* **Reuse:** DeepSeekMath takes a single policy update per rollout batch \[[Shao et al., 2024](https://arxiv.org/abs/2402.03300), §4.2\], so there $\rho_{i,t}=1$ and clipping is inactive. Later recipes reuse each batch for several minibatch steps (e.g. 16 for DeepSeek-R1), which is where the ratio and clipping matter.
* **KL penalty in the objective:** unlike typical RLHF PPO (KL in the reward), original GRPO subtracts a differentiable per-token KL estimate inside the loss \[[Shao et al., 2024](https://arxiv.org/abs/2402.03300), eqs. 3–4\], using Schulman's nonnegative, unbiased $k_3$ estimator \[[Schulman, 2020](http://joschu.net/blog/kl-approx.html)\]:
$$
\widehat D_{\mathrm{KL},i,t}
=\frac{\pi_{\mathrm{ref}}(a_{i,t}\mid s_{i,t})}{\pi_\theta(a_{i,t}\mid s_{i,t})}
-\log\frac{\pi_{\mathrm{ref}}(a_{i,t}\mid s_{i,t})}{\pi_\theta(a_{i,t}\mid s_{i,t})}-1.
$$
Many later GRPO or RLVR implementations set $\beta=0$ and omit $\pi_{\mathrm{ref}}$ \[[Yu et al., 2025](https://arxiv.org/abs/2503.14476); [Liu et al., 2025](https://arxiv.org/abs/2503.20783)\]; Hugging Face TRL's `GRPOTrainer` now defaults to `beta=0.0` \[[TRL docs](https://huggingface.co/docs/trl/en/grpo_trainer)\]. DeepSeek-R1-Zero kept a small penalty ($\beta=0.001$), periodically resetting $\pi_{\mathrm{ref}}$ to the current policy \[[DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)\].

![image](https://hackmd.io/_uploads/Hyu1WBttze.png)
-- GRPO, from \[[Lambert, 2025](https://rlhfbook.com/c/06-policy-gradients#fig:grpo-arch)\]

:::    success
Overview of Policy Gradient Algorithms
::: 
| Algorithm | Advantage Calculation | Advantage Granularity | Update Style | Models in Memory | KL Penalty |
|-----| -----| -----|----- | -----| -----|
| [REINFORCE](#reinforce) | $A_k = R_k - \text{mean}(R_\text{batch})$ | Sequence-level ($k$)| Unconstrained |3 ($\pi_\theta, \pi_{\text{ref}}, r_\psi$)| Optional (in Reward) | 
| [RLOO](#rloo) |   $A_k = R_k - \frac{1}{K-1}\sum_{i \neq k}^K R_i$ |Sequence-level ($k$) | Unconstrained |3 ($\pi_\theta, \pi_{\text{ref}}, r_\psi$)| Optional (in Reward) |
|[PPO](#ppo) | \begin{align}A_t= G_t - V_\phi(s_t) \hspace{5mm }&\text{ ... Pure MC} \\ A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l \delta_{t+l}^V\hspace{5mm }&\text{ ... GAE}\end{align}| Token-level ($t$)| Clipped IS ratio ($\rho_t$) |4 ($\pi_\theta, V_\phi, \pi_{\text{ref}}, r_\psi$)| In Reward |
| [GRPO](#grpo) | $A_k = \dfrac{R_k - \mu_G}{\sigma_G}$ | Sequence-level ($k$) | Clipped IS ratio ($\rho_{i,t}$) | 3 ($\pi_\theta, \pi_{\text{ref}}, r_\psi$) <br /> 2 with $\beta=0$ (drop $\pi_\text{ref}$) or a verifier (drop $r_\psi$, e.g. R1-Zero) <br /> 1 with both | In Loss (original), often dropped |

> Note: Dr. GRPO \[[Liu et al., 2025](https://arxiv.org/abs/2503.20783)\] identifies two biases in original GRPO:
> 1. **Question-level difficulty bias:** dividing each prompt's centered rewards by its own $\sigma_G$ up-weights low-variance prompts (those that are too easy or too hard). Dr. GRPO removes this denominator, which removes the extra amplification given to low-variance groups, including groups containing rare correct answers.
> 2. **Response-level length bias:** averaging each response's token loss by $1/|a_i|$ gives larger per-token updates to short correct responses and smaller penalties to long incorrect ones, nudging incorrect responses to grow longer. Dr. GRPO instead sums token losses and divides by a fixed constant (e.g. the generation budget).
>
> As the authors note (App. A), its centered advantage equals RLOO's leave-one-out advantage up to a constant factor, though the complete algorithms can still differ in loss aggregation, clipping, and KL treatment.

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


### Beyond Per-Token Ratios: GSPO and CISPO
---

Both keep GRPO's group-relative advantage and change how importance ratios are used.

:::danger
Problem: GRPO weights and clips each token with its own local importance ratio $\rho_{i,t}$, even though the reward and advantage are sequence-level. Each $\rho_{i,t}$ is a single-sample estimate, so its noise accumulates over long responses and can lead to irreversible collapse \[[Zheng et al., 2025](https://arxiv.org/abs/2507.18071)\]. In mixture-of-experts (MoE) models things get worse: the experts that a token is routed to can change after each update, so token-level ratios can fluctuate wildly.
:::

**GSPO** \[[Zheng et al., 2025](https://arxiv.org/abs/2507.18071)\] starts from the principle that *the unit of optimization should match the unit of reward*. The reward (and advantage) is assigned to the whole response, and what is actually sampled from $\pi_{\theta_{\text{old}}}$ is the whole response $a_i$, so the importance ratio should be over the whole response too, not over individual tokens. GSPO therefore uses a single length-normalized sequence likelihood ratio $\rho_i$ per response $i$, equal to the geometric mean of token ratios:

$$\rho_i(\theta) = \exp\left( \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \log \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})} \right)$$

Clipping and optimization are then performed at sequence level:

$$J_{\text{GSPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \min\left( \rho_i(\theta) \hat{A}_i, \; \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right)$$

Because $\rho_i$ is a geometric mean, it stays much closer to 1 than individual token ratios, so GSPO uses a far tighter clip range than PPO's $\epsilon = 0.2$ (e.g. $3\times10^{-4}$ / $4\times10^{-4}$ for the lower / upper bounds).

<details>
<summary>
    GSPO's Bonus Benefit: MoE models and Routing Replay
    </summary>

After a single gradient update, roughly 10% of the experts a token is routed to can change, so $\pi_\theta$ and $\pi_{\theta_{\text{old}}}$ score the same token with different sub-networks and $\rho_{i,t}$ becomes very noisy. To keep GRPO training on MoE models from failing to converge, the Qwen team used *Routing Replay*: cache the experts each token was routed to under $\pi_{\theta_{\text{old}}}$, and force $\pi_\theta$ to reuse them when computing $\rho_{i,t}$. This works, but adds memory and communication overhead and limits the MoE's effective capacity. GSPO only depends on the sequence likelihood, which stays stable even when individual routing decisions change, so it trains MoE models without Routing Replay \[[Zheng et al., 2025](https://arxiv.org/abs/2507.18071), §5.3; [Qwen Team, 2025](https://qwenlm.github.io/blog/gspo/)\].

</details>
</br>

**CISPO**, from MiniMax-M1 \[[MiniMax, 2025](https://arxiv.org/abs/2506.13585)\]: in PPO/GRPO, rare but important low-probability "fork" tokens (e.g. *However*, *Wait*, *Recheck*) quickly exceed $1+\epsilon$ and are clipped out of the gradient after the first on-policy update (no learning). Rather than clipping the objective (which zeros the gradient outside the trust region), CISPO clips the importance weights themselves, using a stop-gradient operator $\text{sg}(\cdot)$ and bounds $[1-\epsilon^-, 1+\epsilon^+]$. MiniMax-M1 effectively imposes no lower bound (large $\epsilon^-$) and tunes only $\epsilon^+$. Like [DAPO](https://arxiv.org/abs/2503.14476), it averages over all tokens in the group ($1/\sum_i |a_i|$) rather than per response:

\begin{align}
\hat{\rho}_{i,t} &= \text{clip}\left( \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\theta_{\text{old}}}(a_{i,t} \mid s_{i,t})}, \; 1-\epsilon^-, \; 1+\epsilon^+ \right) \\\\
J_{\text{CISPO}}(\theta) &= \frac{1}{\sum_{i=1}^G |a_i|} \sum_{i=1}^G \sum_{t=1}^{|a_i|} \text{sg}\left(\hat{\rho}_{i,t}\right) \hat{A}_{i} \log \pi_\theta(a_{i,t} \mid s_{i,t})
\end{align}

The stop-gradient (`.detach()` in PyTorch) acts as an identity function during the forward pass ($\text{sg}(x) = x$), but blocks gradients in backward pass by treating its input as a fixed constant during backpropagation ($\nabla_\theta ~\text{sg}(x) = 0$). Why?

When we write a policy gradient objective of the form: $$J(\theta) = w(\theta) \cdot \log \pi_\theta(a_t \mid s_t)$$

Then by product rule: $$\nabla_\theta J(\theta) = \underbrace{\left(\nabla_\theta w(\theta)\right) \cdot \log \pi_\theta(a_t \mid s_t)}_{\text{unwanted gradient through weight}} + w(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

We do not want backpropagation to differentiate through the importance weight itself. We want the clipped ratio $w=\hat{\rho}_{i,t}$ to act as a fixed scalar multiplying the policy's log-likelihood gradient. By applying the stop-gradient operator $\text{sg}\left(\hat{\rho}_{i,t}\right)$, autodiff gives:
$$
\nabla_\theta J_{\text{CISPO}}(\theta)
=\hat{\rho}_{i,t}\hat A_{i}\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t}).
$$
The clipped ratio bounds the coefficient's amplification while allowing every token to retain a gradient contribution. Because clipping changes the importance weights, the resulting estimator is biased.

### Summarizing Policy Gradient

:::    info
In most RLHF setups, data quality and reward signal quality dominate; algorithm choice mostly determines stability, efficiency, and engineering burden. 
:::

All methods estimate (approximations of) the same policy gradient:

$$
\nabla_\theta J(\theta) =  \mathbb{E}_{\tau\sim p_\theta} \left[ \sum_t \Psi_t \cdot \nabla_\theta \log\pi_\theta(a_t \mid s_t) \right] 
$$

They differ primarily in:

1. **Choice of** $\Psi_t\to$ baselined Monte Carlo return (REINFORCE, RLOO), critic-based GAE advantage (PPO), group-normalized advantage (GRPO, GSPO, CISPO)
2. **How updates are bounded** $\to$ unconstrained (REINFORCE, RLOO), clipped objective with token ratios (PPO, GRPO) or a sequence ratio (GSPO), clipped IS weights (CISPO)
3. **Prompt and token weighting** $\to$ group $z$-score (GRPO, GSPO, CISPO), per-response averaging $1/|a_i|$ (GRPO) vs. per-token averaging $1/\sum_i|a_i|$ (DAPO, CISPO) vs. a constant (Dr. GRPO). Vanilla REINFORCE/RLOO usually have no IS ratio.
4. **Memory and compute** $\to$ whether training requires a critic, frozen reference model, and learned reward model, i.e. 1-4 neural models (e.g. DeepSeek-R1-Zero needs 2: policy + reference, with a rule-based reward).

<a id="opd"></a>
## 4. Coming Next: On-Policy Distillation & Other Topics 🚧
:::warning
**Work in progress.** This will be expanded here or in a follow-up post. Here's what I plan to cover and the papers I'm reading:
:::

On-policy distillation is a form of online imitation learning: the student $\pi_{student}$ generates its own trajectories, and the teacher's feedback (e.g. a per-token KL) is computed on those samples rather than on fixed teacher demonstrations.

* **SDPO** \[[Hübotter et al., 2026](https://arxiv.org/abs/2601.20802)\]: reinforcement learning via self-distillation
* **MOPD** \[[Ma et al., 2026](https://arxiv.org/abs/2606.30406)\]: multi-teacher on-policy distillation for combining capabilities
* **SDFT** \[[Shenfeld et al., 2026](https://arxiv.org/abs/2601.19897)\]: self-distillation for continual learning without catastrophic forgetting
* **FlashREINFORCE** \[[Hu et al., 2026](https://yifanzhang-pro.github.io/FlashREINFORCE/)\]: critic-free, single-rollout asynchronous RL for agentic LMs
* **Finetuning with Sampling** \[[Karan et al., 2026](https://arxiv.org/abs/2610.02140)\]: keeps the plain SFT objective but uses MCMC sampling to reshape off-policy expert traces toward the model's own distribution, asking whether SFT can get RL's on-policy benefits (better generalization, less forgetting) without policy gradients
* **Pedagogical RL** \[[Chakraborty et al., 2026](https://noahziems.com/pedagogical-rl)\]: a privileged teacher (a copy of the model that sees the answer) is trained with [GRPO](#grpo) to produce trajectories that are both correct and learnable for the student, which then imitates them; an alternative to the "blind" on-policy sampling of GRPO and on-policy distillation

For a preview, see [this OPD reflection](https://louieworth.github.io/blog/opd_reflection/) in Additional Reading.

## Other Resources

### Technical References

1. \[Achiam, 2018\] [Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) — policy-gradient derivation, reward-to-go, and baselines
2. \[Sutton & Barto, 2018\] [Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) — Bellman equations, TD learning, and policy gradients
3. \[Lambert, 2025\] [Reinforcement Learning from Human Feedback](https://rlhfbook.com/) — the book this writeup is adapted from
4. \[Schulman et al., 2016\] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — $\Psi_t$ taxonomy, GAE, and its bias-variance trade-off
5. \[Schulman et al., 2017\] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — original PPO objective
6. \[Ahmadian et al., 2024\] [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://aclanthology.org/2024.acl-long.662/) — modern RLOO for language models
7. \[Shao et al., 2024\] [DeepSeekMath](https://arxiv.org/abs/2402.03300) — original GRPO formulation
8. \[Liu et al., 2025\] [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) — Dr. GRPO and GRPO's length/difficulty biases
9. \[Zheng et al., 2025\] [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) — GSPO
10. \[MiniMax, 2025\] [MiniMax-M1](https://arxiv.org/abs/2506.13585) — CISPO
11. \[DeepSeek-AI, 2025\] [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — large-scale GRPO/RLVR recipe

### Additional Reading

1. [Training frontier knowledge work agents: A 397B RL training guide with SkyRL](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/) (Mercor)
2. [On-Policy Distillation: Promise, Pitfalls, and Prospects](https://louieworth.github.io/blog/opd_reflection/) (Li Jiang)
3. [Launching the OpenThoughts Project](https://www.open-thoughts.ai/blog/launch)
4. [Agentic RL: Frameworks and Best Practices](https://cameronrwolfe.substack.com/p/agentic-rl) (Cameron R. Wolfe)
5. [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) (Shenfeld et al., 2026)

## Appendix

<a id="bias_var_tradeoff"></a>
### Deep Dive: Bias-Variance Tradeoff

Let $g = \nabla_\theta J(\theta)$ denote the true, analytical gradient of expected return that we wish to compute. Let $\hat{g}$ denote our empirically estimated gradient (constructed from rollout samples via a baseline or learned critic predictions).

The **Bias** of an estimator captures the systematic error between its expected value $\mathbb{E}[\hat{g}]$ (avg over all possible random rollout batches) and the true analytical value $g$.

$$
\text{Bias}(\hat{g}) \triangleq \mathbb{E}[\hat{g}] - g $$

The **Variance** of an estimator measures how much individual empirical gradient estimates scatter or fluctuate from batch to batch around their own mean

$$
\text{Var}(\hat{g}) \triangleq \mathbb{E}\left[ \|\hat{g} - \mathbb{E}[\hat{g}]\|^2 \right]
$$

The estimator's mean-squared error decomposes into squared bias and variance:

$$\text{MSE}(\hat{g}) \triangleq \mathbb{E}\left[ \|\hat{g} - g\|^2 \right] = \|\text{Bias}(\hat{g})\|^2 + \text{Var}(\hat{g})$$

* **High Bias (Steering problem):** SGD steps are stable and smooth but steer the policy in the wrong direction because of imperfect estimates of $V_\phi$. e.g. 1-step TD residuals using an inaccurate critic $V_\phi$ will have high bias as $V_\phi$ will point the gradient in an inaccurate direction

* **High Variance (Noise problem):** SGD steps jump erratically from batch to batch due to stochastic sampling. Complete Monte Carlo returns give an unbiased on-policy estimator for the specified objective (i.e. on average the sampled gradient steps $\hat{g}$ point in the exact direction of the true expected gradient $g$, which we proved [above](#mc_sampling)), but individual batches can still be very noisy (high variance).