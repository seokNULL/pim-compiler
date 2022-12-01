
"""
Utility tools for operations
"""

def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / x_exp.sum()