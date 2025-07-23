Great — let’s now fully dive into:

# 🔷 RegCB-Opt (Regularized Confidence Bound – Optimistic)

From:

> 📄 *“A Contextual Bandit Bake-Off”*, Bietti et al., 2018
> Implemented in VW as: `--cb_explore_adf --regcbopt`

---

## ✅ 1. High-Level Intuition

> RegCB-Opt chooses the action whose predicted reward + confidence is highest.

That is, it acts **optimistically** with respect to **uncertainty**:

* Explores when uncertain
* Exploits when confident

It's a principled alternative to epsilon-greedy, based on confidence intervals.

---

## 🎯 2. What It Solves

| Problem                               | RegCB-Opt Solution                                |
| ------------------------------------- | ------------------------------------------------- |
| Need structured exploration           | Uses uncertainty in model to explore purposefully |
| Can't assume constant exploration (ε) | Learns where to explore adaptively                |
| Need theory-backed algorithm          | Based on optimism under uncertainty (O-U-U)       |

---

## 🧮 3. Mathematical Formulation

Assume:

* Each action `a` has its own feature vector `x_a`
* You maintain a linear model `f(x) = θ^T x`

---

### ✴️ Step 1: Estimate reward

```python
r_hat_a = θ^T x_a
```

---

### ✴️ Step 2: Compute confidence width (uncertainty)

```python
delta_a = α * sqrt( x_a^T A_inv x_a )
```

Where:

* `A` is the design matrix (like `X^T X`)
* `A_inv` is its inverse
* `α` is the confidence multiplier (exploration strength)

---

### ✴️ Step 3: Compute upper confidence bound (UCB)

```python
ucb_a = r_hat_a + delta_a
```

---

### ✴️ Step 4: Choose action

```python
a_t = argmax(ucb_a)
```

---

## 🔁 4. Algorithm Step-by-Step

### At each round:

1. For each action `a`:

   * Form `x_a = concat(context, action)`
   * Predict `r_hat = θ^T x_a`
   * Compute `delta = α * sqrt( x_a^T A_inv x_a )`
   * Compute `ucb = r_hat + delta`
2. Choose the action with highest `ucb`
3. Observe reward `r`
4. Update:

   * `A += x_a x_a^T`
   * `b += r x_a`
   * `θ = A_inv * b`

---

## 🔢 5. Numerical Example (2D context + 2D action)

Suppose:

* Context: `c = [0.2, 0.1]`
* Actions:

  * `a1 = [1.0, 0.5]`
  * `a2 = [0.5, 1.0]`
* Features:

  * `x1 = [0.2, 0.1, 1.0, 0.5]`
  * `x2 = [0.2, 0.1, 0.5, 1.0]`

Initial state:

* `A = I` (4x4 identity)
* `b = [0, 0, 0, 0]`
* `θ = A_inv @ b = [0, 0, 0, 0]`

So:

* `r_hat = 0` for both
* `delta = α * sqrt(x^T x)`

You pick the action with higher `delta` (equal if norms are same).

After receiving reward, say `r = 1.0`, you update `A`, `b`, and `θ`.

---

## 🧠 6. Summary of Parameters

| Symbol  | Meaning                                    |
| ------- | ------------------------------------------ |
| `x_a`   | Combined feature vector (context + action) |
| `A`     | Covariance matrix `= sum(x x^T)`           |
| `b`     | Reward vector `= sum(r x)`                 |
| `θ`     | Model weights: `A_inv @ b`                 |
| `delta` | Confidence interval width                  |
| `α`     | Confidence scale (hyperparameter)          |

---

## ✅ 7. Ready to Implement?

Shall I now generate the **`regcb_opt.py` file** in the same format as `lin_reg_iwr.py` and then the training script like `train_linreg_iwr.py`?

You’ll get:

* Model class `RegCBOpt`
* Methods: `predict`, `choose`, `update`, `batch_update`, `save`, `load`
* Trainer script with same 3 example data

Say “yes” and I’ll generate both files.
