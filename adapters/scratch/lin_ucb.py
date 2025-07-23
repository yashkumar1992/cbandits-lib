import numpy as np


#exploration factor
alpha = 0.5

# context features dimension
ctx_dim = 2
# action features dimension
act_dim = 2


# INIT Variables
# Context features
A0 = np.identity(ctx_dim)
b0 = np.zeros(ctx_dim)
beta = np.linalg.inv(A0) @ b0

# Action features
A = np.identity(act_dim)
b = np.zeros(act_dim)
theta = np.linalg.inv(A) @ b


# PREDICTION
# sample input
# context features
c = np.array([0.5, 0.5])
# action features
a = np.array([1.0, 0.0])

pred = c @ beta + a @ theta
exploration = alpha * np.sqrt(c @ np.linalg.inv(A0) @ c + a @ np.linalg.inv(A) @ a)
score = pred + alpha*exploration

print(f"Predicted score: {pred}")


# reward
reward = 1.0

# UPDATE
A0 += np.outer(c, c)
b0 += reward * c
beta = np.linalg.inv(A0) @ b0

A += np.outer(a, a)
b += reward * a
theta = np.linalg.inv(A) @ b

pred = c @ beta + a @ theta
exploration = alpha * np.sqrt(c @ np.linalg.inv(A0) @ c + a @ np.linalg.inv(A) @ a)
score = pred + alpha*exploration

print(f"Predicted score: {pred}")





