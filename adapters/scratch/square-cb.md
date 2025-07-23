Excellent! Let's now do a **deep dive into Inverse Gap Weighting (IGW)** — the core exploration strategy behind **SquareCB**.

---

# 🎯 What Is Inverse Gap Weighting (IGW)?

IGW is a **probability assignment rule** for exploration in contextual bandits.
It **increases the probability** of selecting actions with predicted reward **close to the best action** — i.e., *small prediction gaps*.

It was introduced in **[SquareCB: Regret Optimal Algorithms for Contextual Bandits](https://arxiv.org/abs/2002.04986)** (Foster & Rakhlin, 2020).

---

## 🧠 Core Idea

* You have a regressor $f_t(x, a)$ that predicts reward for each action $a$ in context $x$
* At time $t$, you want to sample one action $a_t$ based on those predictions
* Instead of greedy, IGW gives **higher probability to actions that are close to the best**

---

## 🔢 Step-by-Step Breakdown of IGW

### Step 1: Predict Rewards

For each action $a$, compute predicted reward:

$$
\hat{y}_a = f_t(x, a)
$$

### Step 2: Compute Gaps

Let:

* $a^* = \arg\max_a \hat{y}_a$: the best predicted action
* Gap for each action:

$$
g_a = \hat{y}_{a^*} - \hat{y}_a
$$

This gap represents how far an action is from the best.

---

### Step 3: Compute Inverse Gap Scores

For exploration, IGW sets score:

$$
s_a = \frac{1}{\max(g_a, \delta)^2}
$$

* $\delta > 0$: small constant to avoid division by zero (usually 0.01 or 0.001)
* The **closer** an action is to the best (i.e., small $g_a$), the **higher** the score

---

### Step 4: Normalize to Get Probabilities

Convert scores to probabilities:

$$
p_a = \frac{s_a}{\sum_{a'} s_{a'}}
$$

This ensures all $p_a$ sum to 1.

---

## ✅ Final Sampling Distribution

$$
p(a) = \frac{1}{Z} \cdot \frac{1}{\max(\hat{y}_{a^*} - \hat{y}_a, \delta)^2}
\quad \text{where} \quad
Z = \sum_{a'} \frac{1}{\max(\hat{y}_{a^*} - \hat{y}_{a'}, \delta)^2}
$$

This is the **IGW formula** used in SquareCB.

---

## 📈 Intuition Behind IGW

* **Sharp exploitation**: Actions close to the best get much higher probability
* **Efficient exploration**: Even suboptimal actions get nonzero probability
* **Faster learning**: Focuses on resolving ambiguity between top candidates

Compared to ε-greedy:

|                             | ε-Greedy | IGW (SquareCB)    |
| --------------------------- | -------- | ----------------- |
| Uniform random on ε         | ✅ Yes    | ❌ No              |
| Rewards used in exploration | ❌ No     | ✅ Yes (gap-based) |
| Adaptive to confidence      | ❌ No     | ✅ Yes             |

---

## 🔬 Theory: Why It Works

In SquareCB, **regret is bounded** by the **regression oracle’s squared loss**, thanks to this selection rule:

$$
R_T = O\left( \sqrt{K T \log T \cdot \text{Regret}_{\text{reg}} } \right)
$$

* $K$: number of actions
* $T$: time steps
* $\text{Regret}_{\text{reg}}$: regression error over time

This makes SquareCB **statistically optimal** for contextual bandits with model-based predictions.

---

## 🧪 Numerical Recap (Tiny Example)

| Action | Prediction | Gap from Best | $1 / \text{gap}^2$  |
| ------ | ---------- | ------------- | ------------------- |
| A1     | 0.9        | 0.0           | 10000 (if δ = 0.01) |
| A2     | 0.7        | 0.2           | 25                  |
| A3     | 0.3        | 0.6           | 2.78                |

So:

* A1 gets **almost all** the probability
* A2 gets **a little**
* A3 gets **almost none**, because it’s far from the top

---

## ⚙️ Code Snippet (Pure Python)

```python
import numpy as np

def inverse_gap_weighting(preds, delta=0.01):
    y_best = np.max(preds)
    gaps = np.maximum(y_best - preds, delta)
    scores = 1.0 / (gaps ** 2)
    probs = scores / np.sum(scores)
    return probs

# Example predicted rewards
preds = np.array([0.9, 0.7, 0.3])
probs = inverse_gap_weighting(preds)
print(probs)  # → [0.997, 0.0025, 0.0003]
```

---

## 🧩 Where IGW is used

* ✅ **SquareCB** (PEARL, VW, academic papers)
* ✅ **Exploration module** in PEARL’s `contextual` package
* ✅ **UCB-style** sampling in deep contextual bandits (NeuralUCB, etc.)

---

## 🧠 Summary

| Property                   | IGW / SquareCB                       |
| -------------------------- | ------------------------------------ |
| Uses model predictions?    | ✅ Yes                                |
| Adapts to prediction gaps? | ✅ Yes                                |
| Smooth exploration?        | ✅ Yes (via gaps)                     |
| Guarantees?                | ✅ Regret bounds via regression error |
| Works with neural nets?    | ✅ Yes                                |

---

## ⏭️ Want to go deeper?

* [ ] Implement SquareCB + IGW in Pearl with neural regressor
* [ ] Log and visualize action probabilities across time
* [ ] Integrate with replay buffer and learn step

Tell me what you want next and I’ll guide you!
