import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# === [1] 讀取 pgm 圖像資料 ===
def read_pgm(folder, size=(64, 64)):
    images, labels, filenames = [], [], []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".pgm"):
            path = os.path.join(folder, fname)
            img = Image.open(path).convert('L').resize(size)
            images.append(np.array(img).flatten())
            labels.append(fname.split('.')[0])
            filenames.append(fname)
    return np.array(images), labels, filenames, size[0], size[1]

# === [2] PCA 實作 ===
def pca(X, num_components):
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    cov = np.dot(X_centered, X_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    U = np.dot(X_centered.T, eigvecs[:, -num_components:])
    U = U / np.linalg.norm(U, axis=0)
    return U, X_mean

# === [3] LDA 實作（需要 PCA 降維後再做）===
def lda(X, y, num_components):
    class_labels = list(set(y))
    mean_total = np.mean(X, axis=0)
    Sw, Sb = np.zeros((X.shape[1], X.shape[1])), np.zeros((X.shape[1], X.shape[1]))

    for c in class_labels:
        X_c = X[np.array(y) == c]
        mean_c = np.mean(X_c, axis=0)
        Sw += (X_c - mean_c).T @ (X_c - mean_c)
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_total).reshape(-1, 1)
        Sb += n_c * mean_diff @ mean_diff.T

    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    idx = np.argsort(-np.abs(eigvals))[:num_components]
    W = eigvecs[:, idx].real
    return W

# === [4] 顯示 Eigenfaces / Fisherfaces ===
def show_eigenfaces(W, h, w, title="Eigenfaces", save_folder="img"):
    plt.figure(figsize=(10, 10))
    for i in range(min(25, W.shape[1])):
        plt.subplot(5, 5, i+1)
        plt.imshow(W[:, i].reshape((h, w)), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, f"{title.replace(' ', '_')}.png"))
    plt.show()

# === [5] 重建圖像顯示 ===

def show_reconstruction(X, X_rec, h, w, filenames, save_folder="img"):
    plt.figure(figsize=(20, 4))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(X[i].reshape(h, w), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, 10, i+11)
        plt.imshow(X_rec[i].reshape(h, w), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.suptitle("Reconstruction using PCA")
    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    plt.savefig(os.path.join(save_folder, "reconstruction.png"))
    plt.show()

# === [6] k-NN classifier ===
def knn_classifier(Z_train, y_train, Z_test, y_test, k=1):
    distances = cdist(Z_test, Z_train)
    knn_idx = np.argsort(distances, axis=1)[:, :k]
    preds = []

    for i in range(Z_test.shape[0]):
        votes = [y_train[j] for j in knn_idx[i]]
        preds.append(max(set(votes), key=votes.count))

    acc = np.mean([p == t for p, t in zip(preds, y_test)])
    return acc

if __name__ == "__main__":
    # === 路徑設定 ===

    train_folder = "./Yale_Face_Database/Training"
    test_folder = "./Yale_Face_Database/Testing"

    # === 讀取資料 ===
    X_train, y_train, fnames_train, h, w = read_pgm(train_folder)
    X_test, y_test, _, _, _ = read_pgm(test_folder)

    # === Part 1: PCA 特徵臉與重建 ===
    n_components = 25
    W_pca, X_mean = pca(X_train, n_components)
    show_eigenfaces(W_pca, h, w, title="PCA Eigenfaces")

    # PCA 重建圖像
    Z_train_pca = (X_train - X_mean) @ W_pca
    X_train_rec = Z_train_pca @ W_pca.T + X_mean
    show_reconstruction(X_train, X_train_rec, h, w, fnames_train)

    # === Part 1: LDA 特徵臉（需先 PCA 降維） ===
    X_train_pca_50 = (X_train - X_mean) @ W_pca[:, :50]
    W_lda = lda(X_train_pca_50, y_train, n_components)
    fisherfaces = W_pca[:, :50] @ W_lda
    show_eigenfaces(fisherfaces, h, w, title="LDA Fisherfaces")

    # === Part 2: 分類 using PCA + kNN ===
    Z_test_pca = (X_test - X_mean) @ W_pca
    acc_pca = knn_classifier(Z_train_pca, y_train, Z_test_pca, y_test, k=3)

    # === Part 2: 分類 using LDA + kNN ===
    X_test_pca_50 = (X_test - X_mean) @ W_pca[:, :50]
    Z_train_lda = X_train_pca_50 @ W_lda
    Z_test_lda = X_test_pca_50 @ W_lda
    acc_lda = knn_classifier(Z_train_lda, y_train, Z_test_lda, y_test, k=3)
