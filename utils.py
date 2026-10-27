import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0",
             s=80, loc='best'):
    """Plots logistic regression data with two axes."""

    # Find positive and negative examples
    pos = (y == 1).reshape(-1,)
    neg = (y == 0).reshape(-1,)

    # Plot positive examples
    ax.scatter(
        X[pos, 0], X[pos, 1],
        marker='x', s=s, c='green', label=pos_label
    )

    # Plot negative examples
    ax.scatter(
        X[neg, 0], X[neg, 1],
        marker='.', s=s, label=neg_label,
        facecolors='none', edgecolors='#0096ff', lw=3
    )

    ax.legend(loc=loc)

# Plot the dataset and decision boundary
x0 = np.arange(0, 6)
x1 = 3 - x0

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(x0, x1, c="b")
ax.axis([0, 4, 0, 3.5])
ax.fill_between(x0, x1, alpha=0.2)

plot_data(X, y, ax)

ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
ax.set_title("Logistic Regression Decision Boundary")

plt.show()