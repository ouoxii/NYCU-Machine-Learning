import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import eigh


def load_image(image_path):
    img = np.array(Image.open(image_path).convert('RGB'))
    h, w, _ = img.shape
    return img, h, w


def compute_kernel_matrix(img, gamma_s, gamma_c):
    h, w, c = img.shape
    N = h * w
    spatial = np.array([[i // w, i % w] for i in range(N)])
    rgb = img.reshape((-1, 3))

    spatial_dists = squareform(pdist(spatial, 'sqeuclidean'))
    rgb_dists = squareform(pdist(rgb, 'sqeuclidean'))

    K = np.exp(-gamma_s * spatial_dists) * np.exp(-gamma_c * rgb_dists)
    return K


def construct_laplacian(W, method='normalized'):
    D = np.diag(np.sum(W, axis=1))
    if method == 'ratio':
        return D - W
    elif method == 'normalized':
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        return np.identity(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        raise ValueError("method must be 'normalized' or 'ratio'")


def spectral_embedding(L, n_clusters, skip_first=True):
    eigvals, eigvecs = eigh(L)
    start_idx = 1 if skip_first else 0
    U = eigvecs[:, start_idx:start_idx + n_clusters]
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def initialize_kmeans(X, k, method='kmeans++'):
    N = X.shape[0]
    if method == 'random':
        np.random.seed(42)
        centers = X[np.random.choice(N, k, replace=False)]
    elif method == 'kmeans++':
        np.random.seed(42)
        centers = [X[np.random.choice(N)]]
        for _ in range(k - 1):
            dist_sq = np.min([np.sum((X - c)**2, axis=1) for c in centers], axis=0)
            probs = dist_sq / np.sum(dist_sq)
            next_center = X[np.random.choice(N, p=probs)]
            centers.append(next_center)
        centers = np.array(centers)
    else:
        raise ValueError("method must be 'random' or 'kmeans++'")
    return centers


def kmeans(X, k, max_iter=100, tol=1e-4, init_method='kmeans++'):
    N = X.shape[0]
    centers = initialize_kmeans(X, k, init_method)
    labels = np.zeros(N, dtype=int)
    snapshots = []

    for it in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        snapshots.append(new_labels.copy())

        if np.sum(new_labels != labels) < tol * N:
            print(f"Converged at iteration {it + 1}")
            break

        labels = new_labels
        for i in range(k):
            if np.any(labels == i):
                centers[i] = X[labels == i].mean(axis=0)

    return labels, snapshots


def kernel_kmeans(K, k, max_iter=100, tol=1e-3, init_method='kmeans++'):
    N = K.shape[0]
    if init_method == 'random':
        np.random.seed(42)
        labels = np.random.randint(0, k, size=N)
    elif init_method == 'kmeans++':
        np.random.seed(42)
        centers = [np.random.randint(0, N)]
        for _ in range(1, k):
            D = np.array([1 - np.max(K[i, centers]) for i in range(N)])
            probs = D ** 2 / np.sum(D ** 2)
            next_center = np.random.choice(N, p=probs)
            centers.append(next_center)
        labels = np.zeros(N, dtype=int)
        for i in range(N):
            labels[i] = np.argmax([K[i, c] for c in centers])
    else:
        raise ValueError("init_method must be 'random' or 'kmeans++'")

    snapshots = []

    for it in range(max_iter):
        cluster_indices = [np.where(labels == c)[0] for c in range(k)]
        intra_K = np.zeros(k)
        for c in range(k):
            idx = cluster_indices[c]
            if len(idx) > 0:
                intra_K[c] = np.sum(K[np.ix_(idx, idx)]) / (len(idx) ** 2)

        new_labels = np.zeros(N, dtype=int)
        for i in range(N):
            best_c = 0
            min_dist = float('inf')
            for c in range(k):
                idx = cluster_indices[c]
                if len(idx) == 0:
                    continue
                term1 = K[i, i]
                term2 = -2 * np.sum(K[i, idx]) / len(idx)
                term3 = intra_K[c]
                dist = term1 + term2 + term3
                if dist < min_dist:
                    min_dist = dist
                    best_c = c
            new_labels[i] = best_c

        snapshots.append(new_labels.copy())
        changed = np.sum(new_labels != labels)
        print(f"Iteration {it + 1}: {changed} points changed.")

        if changed < tol * N:
            print("Converged.")
            break

        labels = new_labels

    return labels, snapshots


def labels_to_rgb(labels, n_clusters):
    colormap = plt.get_cmap("tab10", n_clusters)
    colors = (colormap(np.arange(n_clusters))[:, :3] * 255).astype(np.uint8)
    return colors[labels]


def save_gif_and_png(snapshots, h, w, n_clusters, file_prefix, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    gif_frames = []
    for labels in snapshots:
        rgb_img = labels_to_rgb(labels, n_clusters).reshape(h, w, 3)
        gif_frames.append(rgb_img)

    gif_path = os.path.join(output_dir, f"{file_prefix}.gif")
    imageio.mimsave(gif_path, gif_frames, format='GIF', duration=0.5)
    print(f"Saved GIF to {gif_path}")

    png_path = os.path.join(output_dir, f"{file_prefix}_final.png")
    final_frame = gif_frames[-1]
    Image.fromarray(final_frame).save(png_path)
    print(f"Saved final PNG to {png_path}")

def visualize_eigenspace(U, labels, file_prefix, output_dir='./eigenspace_vis'):
    os.makedirs(output_dir, exist_ok=True)
    k = U.shape[1]
    fig_path = os.path.join(output_dir, f"{file_prefix}_eigenspace.png")

    if k == 2:
        plt.figure(figsize=(8, 6))
        for c in np.unique(labels):
            plt.scatter(U[labels == c, 0], U[labels == c, 1], label=f"Cluster {c}", s=5)
        plt.xlabel("Eigenvector 1")
        plt.ylabel("Eigenvector 2")
        plt.title("Eigenspace Visualization (2D)")
        plt.legend()
        plt.grid(True)
        plt.savefig(fig_path)
        plt.close()
        print(f"Eigenspace plot saved to: {fig_path}")

    elif k == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for c in np.unique(labels):
            ax.scatter(U[labels == c, 0], U[labels == c, 1], U[labels == c, 2], label=f"Cluster {c}", s=5)
        ax.set_title("Eigenspace Visualization (3D)")
        ax.set_xlabel("Eigenvector 1")
        ax.set_ylabel("Eigenvector 2")
        ax.set_zlabel("Eigenvector 3")
        ax.legend()
        plt.savefig(fig_path)
        plt.close()
        print(f"Eigenspace 3D plot saved to: {fig_path}")

    else:
        print(f"[Warning] Eigenspace visualization only supported for k=2 or 3 (got k={k}). Skipped.")


def spectral_clustering_pipeline(image_path, n_clusters, gamma_s, gamma_c, method='normalized', init_method='kmeans++'):
    img, h, w = load_image(image_path)
    K = compute_kernel_matrix(img, gamma_s, gamma_c)
    L = construct_laplacian(K, method)
    U = spectral_embedding(L, n_clusters)
    labels, snapshots = kmeans(U, n_clusters, init_method=init_method)
    file_prefix = f"{os.path.splitext(os.path.basename(image_path))[0]}_spectral_{method}_{init_method}_{n_clusters}clusters"
    save_gif_and_png(snapshots, h, w, n_clusters, file_prefix)
    visualize_eigenspace(U, labels, file_prefix)


def kernel_kmeans_pipeline(image_path, n_clusters, gamma_s, gamma_c, init_method='kmeans++'):
    img, h, w = load_image(image_path)
    K = compute_kernel_matrix(img, gamma_s, gamma_c)
    labels, snapshots = kernel_kmeans(K, n_clusters, init_method=init_method)
    file_prefix = f"{os.path.splitext(os.path.basename(image_path))[0]}_kernel_{init_method}_{n_clusters}clusters"
    save_gif_and_png(snapshots, h, w, n_clusters, file_prefix)

def run_all_configs(image_path, gamma_s, gamma_c, n_clusters_list, init_methods, modes, laplacian_methods):
    for mode in modes:
        for init_method in init_methods:
            for n_clusters in n_clusters_list:
                if mode == 'kernel':
                    print(f"\n[Kernel K-Means] clusters={n_clusters}, init={init_method}")
                    try:
                        kernel_kmeans_pipeline(
                            image_path=image_path,
                            n_clusters=n_clusters,
                            gamma_s=gamma_s,
                            gamma_c=gamma_c,
                            init_method=init_method
                        )
                    except Exception as e:
                        print(f"Error in kernel_kmeans_pipeline: {e}")

                elif mode == 'spectral':
                    for lap_method in laplacian_methods:
                        print(f"\n[Spectral Clustering] clusters={n_clusters}, init={init_method}, laplacian={lap_method}")
                        try:
                            spectral_clustering_pipeline(
                                image_path=image_path,
                                n_clusters=n_clusters,
                                gamma_s=gamma_s,
                                gamma_c=gamma_c,
                                method=lap_method,
                                init_method=init_method
                            )
                        except Exception as e:
                            print(f"Error in spectral_clustering_pipeline: {e}")

                else:
                    print(f"Unknown mode: {mode}")



if __name__ == "__main__":
    gamma_s = 0.001
    gamma_c = 0.001
    n_clusters_list = [2, 3, 4]
    init_methods = ['random', 'kmeans++']
    modes = ['kernel', 'spectral']
    laplacian_methods = ['ratio', 'normalized']

    for i in range(1, 3):
        image_path = f"./image{i}.png"
        run_all_configs(
            image_path=image_path,
            gamma_s=gamma_s,
            gamma_c=gamma_c,
            n_clusters_list=n_clusters_list,
            init_methods=init_methods,
            modes=modes,
            laplacian_methods=laplacian_methods
        )
