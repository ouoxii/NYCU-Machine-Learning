#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab
import matplotlib.pyplot as plt
import os
import imageio.v2
from matplotlib.ticker import ScalarFormatter, MaxNLocator


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, symmetric_sne=False):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 500
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    np.random.seed(42)
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    Y_frames = []  # for visualization

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)

        if symmetric_sne:
            num = np.exp(-np.add(np.add(num, sum_Y).T, sum_Y))  # modified from tsne to symmetric sne
        else:
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))

        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            if symmetric_sne:
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)  # modified from tsne to symmetric sne
            else:
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            Y_frames.append(Y.copy())

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    method = 'symmetric_sne' if symmetric_sne else 'tsne'
    plot_pairwise_similarity_distribution(P, Q, method, perplexity)

    # Return solution
    return Y, Y_frames

def create_animation(Y_frames, labels, filename, symmetric_sne):
    # Create the directory if it doesn't exist
    output_dir = "./tsne_visualization"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames = []
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    unique_labels = np.unique(labels)
    distinct_colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                      '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990']
    color_map = dict(zip(unique_labels, distinct_colors))

    for i, Y in enumerate(Y_frames):
        ax.clear()
        for label in unique_labels:
            mask = labels == label
            ax.scatter(Y[mask, 0], Y[mask, 1], c=color_map[label],
                      label=f'Class {int(label)}', alpha=0.6, s=20)

        ax.set_title(f'Iteration {(i + 1) * 10}')

        if symmetric_sne:
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)

        else:
            ax.set_xlim(-80, 80)
            ax.set_ylim(-80, 80)

        fig.canvas.draw()

        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(frame[:,:,:3])

    gif_path = os.path.join(output_dir, filename)
    imageio.v2.mimsave(gif_path, frames, loop=1, fps=5)
    print("Optimization procedure shows in", gif_path)

    # Save the final .png image
    png_path = gif_path.replace('.gif', '.png')
    plt.savefig(png_path, dpi=100, bbox_inches='tight')
    plt.close()
    print("The final result shows in", png_path)



def plot_pairwise_similarity_distribution(P, Q, method, perplexity):
    output_dir = "./tsne_visualization"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))

    # Plot P (high-dimensional space)
    plt.subplot(1, 2, 1)
    plt.hist(P.flatten(), bins=35, log=True, density=True)
    plt.title(f"{method} High-dimensional space (P) with perplexity ", fontsize=12)
    plt.xlabel('Pairwise Similarity')
    plt.ylabel('Frequency (log scale in proportion)')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # Plot Q (low-dimensional space)
    plt.subplot(1, 2, 2)
    plt.hist(Q.flatten(), bins=35, log=True, density=True)
    plt.title(f"{method} Low-dimensional space (Q)", fontsize=12)
    plt.xlabel('Pairwise Similarity')
    plt.ylabel('Frequency (log scale in proportion)')
    ax = plt.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/similarity_distribution_{method}_perplexity_{int(perplexity)}.png")
    # plt.show()



def save_gif(Y_frames, labels, filename, output_dir="./tsne_visualization"):
    import matplotlib.pyplot as plt
    import imageio
    os.makedirs(output_dir, exist_ok=True)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    frames = []

    for i, Y in enumerate(Y_frames):
        fig, ax = plt.subplots(figsize=(6, 6))
        for j, label in enumerate(unique_labels):
            idx = labels == label
            ax.scatter(Y[idx, 0], Y[idx, 1], s=10, c=colors[j].reshape(1, -1), label=f"Class {int(label)}")
        ax.set_title(f"Iteration {(i + 1) * 10}")
        ax.axis('off')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    gif_path = os.path.join(output_dir, filename)
    imageio.mimsave(gif_path, frames, fps=5)
    print(f"Saved animation: {gif_path}")


def combine_similarity_distributions(perplexities):
    output_dir = "./tsne_visualization"
    os.makedirs(output_dir, exist_ok=True)

    fig, axs = plt.subplots(len(perplexities), 2, figsize=(12, 4 * len(perplexities)))

    for i, perp in enumerate(perplexities):
        for method, sym_flag in [("t-SNE", False), ("Symmetric SNE", True)]:
            Y, _ = tsne(X, 2, 50, perp, symmetric_sne=sym_flag)

            P = x2p(X, perplexity=perp)
            P = (P + P.T) / np.sum(P + P.T)

            sum_Y = np.sum(np.square(Y), axis=1)
            if sym_flag:
                num = np.exp(-np.add(np.add(-2. * np.dot(Y, Y.T), sum_Y).T, sum_Y))
            else:
                num = 1. / (1. + np.add(np.add(-2. * np.dot(Y, Y.T), sum_Y).T, sum_Y))
            np.fill_diagonal(num, 0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            col = 0 if not sym_flag else 1
            ax = axs[i, col] if len(perplexities) > 1 else axs[col]
            ax.hist(P.flatten(), bins=35, alpha=0.5, label='P (High-Dim)', log=True, density=True)
            ax.hist(Q.flatten(), bins=35, alpha=0.5, label='Q (Low-Dim)', log=True, density=True)
            ax.set_title(f"{method} (Perplexity={perp})")
            ax.set_xlabel('Pairwise Similarity')
            ax.set_ylabel('Density (log)')
            ax.legend()

    plt.tight_layout()
    combined_path = os.path.join(output_dir, "combined_similarity_distributions.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"Saved combined similarity distribution plot: {combined_path}")



if __name__ == "__main__":
    print("Running t-SNE and Symmetric SNE on 2,500 MNIST digits...")
    X = np.loadtxt("./tsne_python/mnist2500_X.txt")
    labels = np.loadtxt("./tsne_python/mnist2500_labels.txt")

    perplexities = [10, 20, 50, 100]
    output_dir = "./tsne_visualization"
    os.makedirs(output_dir, exist_ok=True)

    for perplexity in perplexities:
        for method, symmetric_sne in [("tsne", False), ("symmetric_sne", True)]:
            print(f"\n=== {method.upper()} with perplexity={perplexity} ===")

            # Step 1: Run t-SNE or Symmetric SNE
            Y, Y_frames = tsne(X, no_dims=2, initial_dims=50, perplexity=perplexity, symmetric_sne=symmetric_sne)

            # Step 2: Save the 500-iteration result PNG
            plt.figure(figsize=(6, 6))
            for label in np.unique(labels):
                idx = labels == label
                plt.scatter(Y[idx, 0], Y[idx, 1], s=10, label=str(int(label)))
            plt.title(f"{method.upper()} projection\nperplexity={perplexity}, iter=500")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{method}_perplexity_{perplexity}.png"))
            plt.close()

            # Step 3: Save embedding animation as GIF
            gif_filename = f"{method}_perplexity_{perplexity}.gif"
            create_animation(Y_frames, labels, gif_filename, symmetric_sne)

            # Step 4: Save P vs. Q pairwise similarity distribution
            P = x2p(X, perplexity=perplexity)
            P = (P + P.T) / np.sum(P + P.T)

            sum_Y = np.sum(np.square(Y), axis=1)
            if symmetric_sne:
                num = np.exp(-np.add(np.add(-2. * np.dot(Y, Y.T), sum_Y).T, sum_Y))
            else:
                num = 1. / (1. + np.add(np.add(-2. * np.dot(Y, Y.T), sum_Y).T, sum_Y))
            np.fill_diagonal(num, 0)
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            plot_pairwise_similarity_distribution(P, Q, method, perplexity)

    print("\nAll visualizations completed.")
