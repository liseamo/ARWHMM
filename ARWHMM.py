# =======================================
# IMPORTS
# =======================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from random import randrange
import re
import traceback
import warnings

# =======================================
# PARAMÈTRES GÉNÉRAUX
# =======================================
base_path = Path(r"S:\Philippe\Partage_Lise\data\imu_files")
ir_base_folder = Path(r"C:\Users\lamodeo\Desktop\gpio_binaire")

target_fs = 5      # Fréquence cible pour downsampling
ar_order = 1       # Ordre AR pour le modèle
num_states = 3     # Nombre d'états latents
eps = 0.0025       # Petite valeur pour pondérations W

# =======================================
# CLASSE MODELE ARWHMM
# =======================================
class SimpleARWHMM:
    def __init__(self, num_states=2, ar_order=1, max_iter=20, tol=1e-4):
        self.K = num_states
        self.p = ar_order
        self.max_iter = max_iter
        self.tol = tol

    def _init_parameters(self, X):
        T, D = X.shape
        self.D = D
        self.pi = np.full(self.K, 1 / self.K)
        self.A = np.full((self.K, self.K), 1 / self.K)
        self.coefs = np.random.randn(self.K, self.p, self.D, self.D)
        self.covs = np.array([np.eye(D) for _ in range(self.K)])

    def _compute_log_likelihoods(self, X):
        T = X.shape[0]
        log_lik = np.zeros((T, self.K))
        for t in range(self.p, T):
            X_past = X[t - self.p:t].flatten()
            x_t = X[t]
            for k in range(self.K):
                coefs_k = self.coefs[k].reshape(self.p * self.D, self.D)
                mu = X_past @ coefs_k
                log_lik[t, k] = multivariate_normal.logpdf(x_t, mean=mu, cov=self.covs[k])
        log_lik[:self.p, :] = -np.inf
        return log_lik

    def _forward_backward(self, log_lik, W):
        T = log_lik.shape[0]
        log_alpha = np.zeros((T, self.K))
        log_beta = np.zeros((T, self.K))

        # Correction : Protection contre log(0) et valeurs nulles
        W_safe = np.maximum(W, eps)

        log_alpha[self.p] = np.log(np.maximum(self.pi, eps)) + np.log(W_safe[self.p]) + log_lik[self.p]
        for t in range(self.p + 1, T):
            for j in range(self.K):
                log_alpha[t, j] = np.log(W_safe[t, j]) + log_lik[t, j] + np.logaddexp.reduce(
                    log_alpha[t - 1] + np.log(self.A[:, j])
                )

        log_beta[-1] = 0
        for t in reversed(range(self.p, T - 1)):
            for i in range(self.K):
                log_beta[t, i] = np.logaddexp.reduce(
                    np.log(self.A[i]) + np.log(W_safe[t + 1]) + log_lik[t + 1] + log_beta[t + 1]
                )

        log_gamma = log_alpha + log_beta
        log_gamma -= np.max(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        xi = np.zeros((T - 1, self.K, self.K))
        for t in range(self.p, T - 1):
            for i in range(self.K):
                for j in range(self.K):
                    xi[t, i, j] = (
                        log_alpha[t, i]
                        + np.log(self.A[i, j])
                        + np.log(W_safe[t + 1, j])
                        + log_lik[t + 1, j]
                        + log_beta[t + 1, j]
                    )
            xi[t] -= np.max(xi[t])
            xi[t] = np.exp(xi[t])
            xi[t] /= np.sum(xi[t])
        return gamma, xi

    def _m_step(self, X, gamma, xi):
        T = X.shape[0]
        self.pi = gamma[self.p] / np.sum(gamma[self.p])
        # Vérification et renormalisation de self.pi
        if not np.isclose(np.sum(self.pi), 1):
            self.pi = self.pi / np.sum(self.pi)
            warnings.warn("self.pi renormalisé car la somme ≠ 1", RuntimeWarning)

        self.A = np.sum(xi[self.p:], axis=0)
        self.A /= np.sum(self.A, axis=1, keepdims=True)
                # Vérification et renormalisation de self.A
        row_sums = np.sum(self.A, axis=1)
        if not np.allclose(row_sums, 1):
            self.A = self.A / row_sums[:, None]
            warnings.warn("self.A renormalisée car certaines lignes ne somment pas à 1", RuntimeWarning)

        for k in range(self.K):
            weights = gamma[self.p:, k]
            sw = np.sqrt(weights)
            X_past = np.array([X[t - self.p:t].flatten() for t in range(self.p, T)])
            Y = X[self.p:]
            Xw = X_past * sw[:, None]
            B = np.zeros((self.p * self.D, self.D))
            for d in range(self.D):
                y_d = Y[:, d] * sw
                B[:, d] = np.linalg.pinv(Xw) @ y_d
            self.coefs[k] = B
            residuals = Y - X_past @ B
            self.covs[k] = (residuals.T * weights) @ residuals / np.sum(weights) + 1e-6 * np.eye(self.D)

    def fit(self, X, W):
        X = np.asarray(X)
        W = np.asarray(W)
        print(f"🔄 Début de l'entraînement avec {X.shape[0]} échantillons et {self.K} états.")
        self._init_parameters(X)
        prev_ll = -np.inf
        i=0
        max_iter_safe=1000
        while i < max_iter_safe:
                log_lik = self._compute_log_likelihoods(X)
                gamma, xi = self._forward_backward(log_lik, W)
                self._m_step(X, gamma, xi)
                ll = np.sum(log_lik[self.p:] * gamma[self.p:])
                i += 1
                print(f"🔁 Iteration {i}, Log-Likelihood: {ll:.2f}")
                if np.abs(ll - prev_ll) < self.tol:
                    print(f"🎉 Convergence atteinte à l'itération {i}.")
                    break
                prev_ll = ll
        else:
                print(f"⚠️ Arrêt forcé après {max_iter_safe} itérations sans convergence.")
        self.gamma_ = gamma

    def predict_states(self):
        return np.argmax(self.gamma_, axis=1)

# =======================================
# PRÉPARATION DES DONNÉES (IMU / IR)
# =======================================
X_all, W_all, timestamps_all, ir_all = [], [], [], []

all_csv_files = list(base_path.glob("*/session*/*.imu_relative.csv"))
print(f"🔍 {len(all_csv_files)} fichiers IMU trouvés")

for csv_file in all_csv_files:
    try:
        # -------- CHARGEMENT IMU --------
        df = pd.read_csv(csv_file)
        df = df.select_dtypes(include=[np.number]).dropna() #garde juste les colonnes numériaues (floqt ou int), dropna enleve les lignes avec valeurs manquantes 
        df = df[~df.isin([np.inf, -np.inf]).any(axis=1)] #enléve les lignes avec des valeurs infinies

        # Timestamps IMU
        time_col = "Time" if "Time" in df.columns else ("IMU Time (s)" if "IMU Time (s)" in df.columns else None)
        if time_col is None:
            warnings.warn(f"Pas de colonne de temps dans {csv_file.name}, utilisation d'un index temporel.", UserWarning)
            timestamps = np.arange(len(df))
            fs = 30  # Fréquence par défaut
        else:
            timestamps = df[time_col].values
            fs = 1 / np.median(np.diff(timestamps)) if len(np.diff(timestamps)) > 0 else 30

        # -------- CHARGEMENT IR --------
        mouse = csv_file.parts[-3].replace("m", "")
        session_num = int(re.search(r'\d+', csv_file.parts[-2]).group())
        ir_folder = ir_base_folder / mouse
        ir_files = list(ir_folder.glob(f"IR_{mouse}_session{session_num}_binaire*.csv"))
        if not ir_files:
            warnings.warn(f"Pas de fichier IR pour {mouse} session {session_num}", UserWarning)
            continue

        # Correction : Lecture des fichiers IR avec gestion des types
        ir_binary = pd.read_csv(ir_files[0], header=None, low_memory=False).iloc[:, 0].values

        # Alignement des longueurs IMU et IR AVANT downsampling
        min_len = min(len(df), len(ir_binary))
        df = df.iloc[:min_len]
        ir_binary = ir_binary[:min_len]
        timestamps = timestamps[:min_len]

        # Downsampling
        factor = max(1, int(fs / target_fs))
        df_ds = df.iloc[::factor].copy()
        ir_binary_ds = ir_binary[::factor]
        timestamps_ds = timestamps[::factor]

        # Vérification finale des longueurs
        if len(df_ds) != len(ir_binary_ds):
            min_len = min(len(df_ds), len(ir_binary_ds))
            df_ds = df_ds.iloc[:min_len]
            ir_binary_ds = ir_binary_ds[:min_len]
            timestamps_ds = timestamps_ds[:min_len]

        # Extraction des features
        acc_cols = [col for col in df_ds.columns if 'acc' in col.lower()]
        if len(acc_cols) < 3:
            warnings.warn(f"Pas assez de colonnes d'accélération dans {csv_file.name}", UserWarning)
            continue

        acc_df = df_ds[acc_cols[:3]].copy()
        acc_df.columns = ['accX', 'accY', 'accZ']
        acc_df["acc_norm"] = np.linalg.norm(acc_df[["accX", "accY", "accZ"]].values, axis=1)
        acc_df["acc_norm_smooth"] = acc_df["acc_norm"].rolling(window=5, min_periods=1).mean()
        acc_df["acc_norm_std"] = acc_df["acc_norm"].rolling(window=5, min_periods=1).std()
        acc_df["d_accX"] = acc_df["accX"].diff().fillna(0)
        acc_df["d_acc_norm"] = acc_df["acc_norm"].diff().fillna(0)

        # Ajout des données d'orientation si disponibles
        ori_cols = [col for col in df_ds.columns if 'ori' in col.lower()]
        if ori_cols:
            ori_df = df_ds[ori_cols[:3]].copy()
            ori_df.columns = ["yaw", "pitch", "roll"]
            ori_df["ori_norm"] = np.linalg.norm(ori_df[["yaw", "pitch", "roll"]].values, axis=1)
            ori_df["ori_norm_smooth"] = ori_df["ori_norm"].rolling(window=5, min_periods=1).mean()
            if len(acc_df) != len(ori_df):
                print(f"❌ Erreur : acc_df et ori_df n'ont pas le même nombre de lignes pour {csv_file.name} ({len(acc_df)} vs {len(ori_df)})")
                continue
            features_all = pd.concat([acc_df, ori_df], axis=1) 
        else:
            features_all = acc_df.copy()
            warnings.warn(f"Pas de colonnes d'orientation dans {csv_file.name}, utilisation uniquement des données d'accélération.", UserWarning)

#features_all: dataframes contenant l'ensemble des variables extraites des données IMU (accX, accY, accZ, yaw, pitch, roll) qui serviront à l'entraînement du modèle ARWHMM
        
        # Normalisation
        std = features_all.std() # Calcul des écarts-types de chaques colonnes 
        std[std == 0] = 1e-8  # Remplace les écarts-types nuls par une petite valeur
        features_all_zscored = (features_all - features_all.mean()) / features_all.std() #normalisation z-score (on soustrait la moyenne et on divise par l'écart-type)
        X_ds = features_all_zscored.values # Conversion en numpy array
        if np.isnan(features_all_zscored.values).any() or np.isinf(features_all_zscored.values).any():
            warnings.warn(f"Attention : NaN ou infini détecté dans les features normalisées pour {csv_file.name}", RuntimeWarning)

        # Correction : Création de W avec des valeurs minimales
        W = np.full((len(ir_binary_ds), num_states), eps)
        W[:, 0] = 1 - 2 * eps  # Assurez-vous que la somme des probabilités est 1
        W[ir_binary_ds == 1, 0] = eps
        W[ir_binary_ds == 1, 1] = 1 - 2 * eps

        # Stockage
        if len(X_ds) > ar_order + 10:
            X_all.append(X_ds[ar_order:])
            W_all.append(W[ar_order:])
            timestamps_all.append(timestamps_ds[ar_order:])
            ir_all.append(ir_binary_ds[ar_order:])
            print(f"✅ Session {csv_file.name} ajoutée: {len(X_ds[ar_order:])} échantillons")
        else:
            warnings.warn(f"Session trop courte : {csv_file.name} ignorée.", UserWarning)

    except Exception as e:
        warnings.warn(f"Erreur pour {csv_file.name} : {type(e).__name__} - {e}", RuntimeWarning)
        traceback.print_exc()
# =======================================
# VÉRIFICATION DES TAILLES AVANT ENTRAINEMENT
# =======================================
if len(X_all) == 0:
    raise ValueError("❌ Aucune session valide trouvée.")

# Vérification des tailles
for i in range(len(X_all)):
    min_len = min(len(X_all[i]), len(W_all[i]), len(ir_all[i]))
    X_all[i] = X_all[i][:min_len]
    W_all[i] = W_all[i][:min_len]
    ir_all[i] = ir_all[i][:min_len]
    print(f"✅ Session {i} ajustée à {min_len} échantillons")

# =======================================
# TRAIN / TEST SPLIT
# =======================================
# Sélection aléatoire d'une session pour le test
test_index = randrange(len(X_all))
# Séparation des données d'entraînement et de test
X_test = X_all[test_index]
W_test = W_all[test_index]
#extrire les timestamp et les données IR pour la session de test
ts_test = timestamps_all[test_index] #contient les timestamps de la session de test, sont utilises pour afficher les graphiques
ir_binary_test = ir_all[test_index] #contient la sequence binaire d'IR pour la session de test, utilisée pour l'évaluation du modèle (ground truth)

shapes_X = [x.shape[1] for i, x in enumerate(X_all) if i != test_index]
if len(set(shapes_X)) > 1:
    warnings.warn(f"Erreur : Les matrices X à empiler n'ont pas le même nombre de colonnes : {shapes_X}", RuntimeWarning)
    exit(1)
shapes_W = [w.shape[1] for i, w in enumerate(W_all) if i != test_index]
if len(set(shapes_W)) > 1:
    warnings.warn(f"Erreur : Les matrices W à empiler n'ont pas le même nombre de colonnes : {shapes_W}", RuntimeWarning)
    exit(1)
X_train = np.vstack([x for i, x in enumerate(X_all) if i != test_index])
W_train = np.vstack([w for i, w in enumerate(W_all) if i != test_index])

print(f"✅ Session {test_index+1}/{len(X_all)} choisie pour test")
print(f"✅ Entraînement sur {X_train.shape[0]} points, test sur {X_test.shape[0]} points")

# =======================================
# ENTRAINEMENT DU MODELE
# =======================================
try:
    model = SimpleARWHMM(num_states=num_states, ar_order=ar_order)
    model.fit(X_train, W_train)
except Exception as e:
    print(f"❌ Erreur lors de l'entraînement : {e}")
    traceback.print_exc() #affiche toutes les informations de l'erreur
    exit(1)

# =======================================
# PREDICTION ET EVALUATION
# =======================================
log_lik_test = model._compute_log_likelihoods(X_test)
gamma_test, _ = model._forward_backward(log_lik_test, W_test)
states_test = np.argmax(gamma_test, axis=1)

# Alignement final des tailles
min_len = min(len(states_test), len(ir_binary_test))
states_test = states_test[:min_len]
ir_binary_test = ir_binary_test[:min_len]

# Identification de l'état de nourrissage
feeding_state = np.argmax([
    np.mean(W_test[states_test == k, 1]) if np.any(states_test == k) else 0
    for k in range(num_states)
])

# Correction : Assurez-vous que gt_labels et pred_labels sont des entiers
pred_labels = (states_test == feeding_state).astype(int)
gt_labels = (pd.to_numeric(ir_binary_test, errors='coerce') > 0.5).astype(int)

# Vérification finale des tailles
if len(gt_labels) != len(pred_labels):
    min_len = min(len(gt_labels), len(pred_labels))
    gt_labels = gt_labels[:min_len]
    pred_labels = pred_labels[:min_len]

# Calcul des métriques
f1 = f1_score(gt_labels, pred_labels)
precision = precision_score(gt_labels, pred_labels)
recall = recall_score(gt_labels, pred_labels)

print(f"🎯 F1-score test     : {f1:.3f}")
print(f"🎯 Précision         : {precision:.3f}")
print(f"🎯 Rappel            : {recall:.3f}")
print(f"🔍 État nourrissage détecté : {feeding_state}")

# =======================================
# VISUALISATION
# =======================================
plt.figure(figsize=(12, 3))
plt.plot(ts_test[:min_len], X_test[:min_len, 0], label="accX", alpha=0.5)
plt.plot(ts_test[:min_len], pred_labels, label="Prédiction nourrissage", linewidth=1.5)
plt.plot(ts_test[:min_len], gt_labels, label="IR feeding (binaire)", linestyle='--', linewidth=1.5)
plt.legend()
plt.title("Correspondance prédictions / IR")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(ts_test[:min_len], ir_binary_test[:min_len], label="IR feeding (binaire)", linestyle='--', linewidth=1.5)
plt.legend()
plt.title("Alignement W vs événements IR (session test)")
plt.tight_layout()
plt.show()
