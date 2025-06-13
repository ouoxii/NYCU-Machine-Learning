import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist

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
def pca_feature_space(X, n_components):
    X_mean = np.mean(X, axis=0, keepdims=True)  # shape: (1, D)
    X_centered = X - X_mean  # shape: (N, D)

    # 計算 D x D 共變異矩陣
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 取出前 n_components 個主成分（按 eigenvalue 降序排序）
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx[:n_components]]

    # 投影矩陣：每列為一個 eigenface（D x n_components）
    projection_matrix = eigvecs
    return projection_matrix, X_mean

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
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(10, 10))
    for i in range(min(25, W.shape[1])):
        plt.subplot(5, 5, i + 1)
        plt.imshow(W[:, i].reshape((h, w)), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(save_folder, filename))
    plt.show()

# === [5] 重建圖像顯示 ===

def show_reconstruction(images, images_reconstructed, height, width, file_names_prefix, n=10, output_dir='img', title="Reconstruction"):
    os.makedirs(output_dir, exist_ok=True)

    subject_ids = [name.split('.')[0] for name in file_names_prefix]
    subject_to_index = {}
    for i, subject in enumerate(subject_ids):
        if subject not in subject_to_index:
            subject_to_index[subject] = i
        if len(subject_to_index) >= n:
            break

    selected_indices = list(subject_to_index.values())

    plt.figure(figsize=(2 * n, 4))
    for i, idx in enumerate(selected_indices):
        plt.subplot(2, n, i + 1)
        plt.imshow(images[idx].reshape(height, width), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(images_reconstructed[idx].reshape(height, width), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

        image_filename = f"{output_dir}/{file_names_prefix[idx]}.png"
        # plt.imsave(image_filename, images_reconstructed[idx].reshape(height, width), cmap='gray')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(output_dir, filename))
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


# Read PGM function
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

# Kernel PCA
def kernel_pca(X, n_components, kernel='rbf', gamma=0.001, degree=3):
    N = X.shape[0]
    if kernel == 'rbf':
        pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(-gamma * pairwise_sq_dists)
    elif kernel == 'poly':
        K = (X @ X.T + 1) ** degree
    else:
        raise ValueError("Unsupported kernel")

    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    eigvals, eigvecs = np.linalg.eigh(K_centered)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    alphas = eigvecs[:, :n_components]
    lambdas = eigvals[:n_components]

    alphas = alphas / np.sqrt(lambdas + 1e-10)
    return K_centered, alphas

# Kernel LDA (KFDA) - implementation

def kernel_lda(X, y, n_components, kernel='rbf', gamma=0.001, degree=3):
    N = X.shape[0]
    classes = list(set(y))
    y = np.array(y)

    # === Step 1: Compute Gram matrix K ===
    if kernel == 'rbf':
        pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
        K = np.exp(-gamma * pairwise_sq_dists)
    elif kernel == 'poly':
        K = (X @ X.T + 1) ** degree
    else:
        raise ValueError("Unsupported kernel")

    # === Step 2: Center K ===
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # === Step 3: Compute class means in feature space ===
    N_c = []
    M = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        N_c.append(len(idx))
        K_c = K_centered[:, idx]
        M.append(np.mean(K_c, axis=1, keepdims=True))  # mean vector in RKHS

    M = np.hstack(M)
    total_mean = np.mean(K_centered, axis=1, keepdims=True)

    # === Step 4: Between-class scatter matrix S_b and within-class scatter S_w ===
    Sb = np.zeros((N, N))
    Sw = np.zeros((N, N))
    for i in range(len(classes)):
        mean_diff = M[:, [i]] - total_mean
        Sb += N_c[i] * (mean_diff @ mean_diff.T)

        idx = np.where(y == classes[i])[0]
        for j in idx:
            diff = K_centered[:, [j]] - M[:, [i]]
            Sw += diff @ diff.T

    # === Step 5: Solve generalized eigenvalue problem ===
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw + 1e-6 * np.eye(N)) @ Sb)
    idx = np.argsort(-np.abs(eigvals))
    eigvecs = eigvecs[:, idx[:n_components]]

    # projection matrix in dual space
    return (K_centered @ eigvecs).real  # Z_train_klda



# Kernel projection for test data
def kernel_project(X_train, X_test, alphas, kernel='rbf', gamma=0.001, degree=3):
    if kernel == 'rbf':
        pairwise_sq = cdist(X_test, X_train, 'sqeuclidean')
        K_test = np.exp(-gamma * pairwise_sq)
    elif kernel == 'poly':
        K_test = (X_test @ X_train.T + 1) ** degree
    else:
        raise ValueError("Unsupported kernel")

    return K_test @ alphas


def cross_validate_knn(X_train_proj, y_train, X_test_proj, y_test, k_list):
    best_k = None
    best_acc = 0
    acc_dict = {}

    for k in k_list:
        acc = knn_classifier(X_train_proj, y_train, X_test_proj, y_test, k=k)
        acc_dict[k] = acc
        print(f"k = {k}: Accuracy = {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"\nBest k = {best_k} with accuracy = {best_acc * 100:.2f}%")
    return best_k, acc_dict

def run_part2_with_cv(Z_train_pca, Z_test_pca, Z_train_lda, Z_test_lda, y_train, y_test):
    k_list = [1, 3, 5, 7, 9]

    print("Cross-validation on PCA space:")
    best_k_pca, accs_pca = cross_validate_knn(Z_train_pca, y_train, Z_test_pca, y_test, k_list)

    print("\nCross-validation on LDA space:")
    best_k_lda, accs_lda = cross_validate_knn(Z_train_lda, y_train, Z_test_lda, y_test, k_list)

    # 整理成表格
    df_result = pd.DataFrame({
        'k': k_list,
        'PCA Accuracy': [accs_pca[k] for k in k_list],
        'LDA Accuracy': [accs_lda[k] for k in k_list],
    })

    df_result['PCA Accuracy'] = df_result['PCA Accuracy'].apply(lambda x: f"{x * 100:.2f}%")
    df_result['LDA Accuracy'] = df_result['LDA Accuracy'].apply(lambda x: f"{x * 100:.2f}%")

    from IPython.display import display
    display(df_result)

    return best_k_pca, best_k_lda, df_result

def save_accuracy_table_as_image(df_result, filename='img/part2_accuracy_table.png'):
    os.makedirs('img', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    table = ax.table(cellText=df_result.values,
                     colLabels=df_result.columns,
                     cellLoc='center',
                     loc='center')
    table.scale(1, 2.2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# kNN classifier
def knn_classifier(Z_train, y_train, Z_test, y_test, k=1):
    distances = cdist(Z_test, Z_train)
    knn_idx = np.argsort(distances, axis=1)[:, :k]
    preds = []

    for i in range(Z_test.shape[0]):
        votes = [y_train[j] for j in knn_idx[i]]
        preds.append(max(set(votes), key=votes.count))

    acc = np.mean([p == t for p, t in zip(preds, y_test)])
    return acc

def evaluate_kernel_methods(X_train, X_test, y_train, y_test, n_components=25, k_list=[1, 3, 5, 7, 9]):
    kernels = ['linear', 'poly', 'rbf']
    results = []

    for kernel in kernels:
        print(f"\n[Kernel PCA] Kernel = {kernel}")
        if kernel == 'linear':
            # Linear kernel PCA ≈ standard PCA
            W_pca, X_mean = pca_feature_space(X_train, n_components)
            Z_train_kpca = (X_train - X_mean) @ W_pca
            Z_test_kpca = (X_test - X_mean) @ W_pca
        else:
            _, alpha_kpca = kernel_pca(X_train, n_components=n_components, kernel=kernel, gamma=0.001, degree=3)
            Z_train_kpca = alpha_kpca
            Z_test_kpca = kernel_project(X_train, X_test, alpha_kpca, kernel=kernel, gamma=0.001, degree=3)

        for k in k_list:
            acc = knn_classifier(Z_train_kpca, y_train, Z_test_kpca, y_test, k=k)
            results.append(['PCA', kernel, k, acc])

    for kernel in kernels:
        print(f"\n[Kernel LDA] Kernel = {kernel}")
        if kernel == 'linear':
            # Linear LDA = standard LDA
            W_pca, X_mean = pca_feature_space(X_train, 50)
            X_train_pca = (X_train - X_mean) @ W_pca
            X_test_pca = (X_test - X_mean) @ W_pca
            W_lda = lda(X_train_pca, y_train, n_components)
            Z_train_klda = X_train_pca @ W_lda
            Z_test_klda = X_test_pca @ W_lda
        else:
            Z_train_klda = kernel_lda(X_train, y_train, n_components=n_components, kernel=kernel, gamma=0.001, degree=3)
            Z_test_klda = kernel_project(X_train, X_test, Z_train_klda, kernel=kernel, gamma=0.001, degree=3)

        for k in k_list:
            acc = knn_classifier(Z_train_klda, y_train, Z_test_klda, y_test, k=k)
            results.append(['LDA', kernel, k, acc])

    df_kernel_result = pd.DataFrame(results, columns=['Method', 'Kernel', 'k', 'Accuracy'])
    df_kernel_result['Accuracy'] = df_kernel_result['Accuracy'].apply(lambda x: f"{x * 100:.2f}%")
    return df_kernel_result

def save_kernel_accuracy_tables(df_kernel_result):
    os.makedirs('img', exist_ok=True)

    # 整理成 pivot table
    df_pca = df_kernel_result[df_kernel_result['Method'] == 'PCA'].pivot(index='Kernel', columns='k', values='Accuracy')
    df_lda = df_kernel_result[df_kernel_result['Method'] == 'LDA'].pivot(index='Kernel', columns='k', values='Accuracy')

    # 儲存 PCA 結果圖
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('off')
    table = ax.table(cellText=df_pca.values,
                     rowLabels=df_pca.index,
                     colLabels=["k=" + str(c) for c in df_pca.columns],
                     cellLoc='center', rowLoc='center',
                     loc='center')
    table.scale(1.2, 1.8)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.tight_layout()
    plt.savefig("img/part3_pca_kernel_accuracy.png")
    plt.close()

    # 儲存 LDA 結果圖
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('off')
    table = ax.table(cellText=df_lda.values,
                     rowLabels=df_lda.index,
                     colLabels=["k=" + str(c) for c in df_lda.columns],
                     cellLoc='center', rowLoc='center',
                     loc='center')
    table.scale(1.2, 1.8)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.tight_layout()
    plt.savefig("img/part3_lda_kernel_accuracy.png")
    plt.close()




if __name__ == "__main__":
    # === 路徑設定 ===

    train_folder = "./Yale_Face_Database/Training"
    test_folder = "./Yale_Face_Database/Testing"

    # === 讀取資料 ===
    X_train, y_train, fnames_train, h, w = read_pgm(train_folder)
    X_test, y_test, _, _, _ = read_pgm(test_folder)

    # === Part 1: PCA 特徵臉與重建 ===
    n_components = 25
    W_pca, X_mean = pca_feature_space(X_train, n_components)
    show_eigenfaces(W_pca, h, w, title="First 25 eigenfaces")


    # PCA 重建圖像
    Z_train_pca = (X_train - X_mean) @ W_pca
    X_train_rec = Z_train_pca @ W_pca.T + X_mean
    show_reconstruction(X_train, X_train_rec, h, w, fnames_train, title="10 reconstructed images from PCA")

    # LDA 特徵臉（需先 PCA 降維）
    X_train_pca_50 = (X_train - X_mean) @ W_pca[:, :50]
    W_lda = lda(X_train_pca_50, y_train, n_components)
    fisherfaces = W_pca[:, :50] @ W_lda
    show_eigenfaces(fisherfaces, h, w, title="First 25 fisherfaces")

    # LDA 重建圖像
    Z_train_lda = X_train_pca_50 @ W_lda
    X_train_rec_lda = Z_train_lda @ W_lda.T @ W_pca[:, :50].T + X_mean
    show_reconstruction(X_train, X_train_rec_lda, h, w, fnames_train, title="10 reconstructed images from LDA")

    # === Part 2: 分類 using PCA + LDA + kNN + Cross Validation ===

    # 前處理
    Z_test_pca = (X_test - X_mean) @ W_pca
    X_test_pca_50 = (X_test - X_mean) @ W_pca[:, :50]
    Z_train_lda = X_train_pca_50 @ W_lda
    Z_test_lda = X_test_pca_50 @ W_lda

    # 交叉驗證找最佳 k 並輸出結果表格
    best_k_pca, best_k_lda, df_result = run_part2_with_cv(
        Z_train_pca, Z_test_pca, Z_train_lda, Z_test_lda, y_train, y_test
    )
    save_accuracy_table_as_image(df_result)

    # === Part 3: Kernel PCA ===
    # kernel PCA 訓練與分類（以 RBF 為例）
    _, alpha_kpca = kernel_pca(X_train, n_components=25, kernel='rbf', gamma=0.001)
    Z_train_kpca = alpha_kpca
    Z_test_kpca = kernel_project(X_train, X_test, alpha_kpca, kernel='rbf', gamma=0.001)
    acc_kpca = knn_classifier(Z_train_kpca, y_train, Z_test_kpca, y_test, k=3)
    print(f"KPCA accuracy: {acc_kpca * 100:.2f}%")
    Z_train_klda = kernel_lda(X_train, y_train, n_components=25, kernel='rbf', gamma=0.001)
    Z_test_klda = kernel_project(X_train, X_test, Z_train_klda, kernel='rbf', gamma=0.001)
    acc_klda = knn_classifier(Z_train_klda, y_train, Z_test_klda, y_test, k=3)
    print(f"KLDA accuracy: {acc_klda * 100:.2f}%")

    df_kernel_result = evaluate_kernel_methods(
    X_train, X_test, y_train, y_test, n_components=25, k_list=[1, 3, 5, 7, 9]
)
    save_kernel_accuracy_tables(df_kernel_result)



