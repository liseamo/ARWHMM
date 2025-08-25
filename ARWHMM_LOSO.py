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
num_states = 5     # Nombre d'états latents
eps = 0.0025       # Petite valeur pour pondérations W
SUP_STRENGTH = 20  # Force de la supervision (à ajuster, ex: 10, 20, 50)

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

        log_alpha[self.p] = np.log(np.maximum(self.pi, eps)) + SUP_STRENGTH * np.log(W_safe[self.p]) + log_lik[self.p]
        for t in range(self.p + 1, T):
            for j in range(self.K):
                log_alpha[t, j] = SUP_STRENGTH * np.log(W_safe[t, j]) + log_lik[t, j] + np.logaddexp.reduce(
                    log_alpha[t - 1] + np.log(self.A[:, j])
                )

        log_beta[-1] = 0
        for t in reversed(range(self.p, T - 1)):
            for i in range(self.K):
                log_beta[t, i] = np.logaddexp.reduce(
                    np.log(self.A[i]) + SUP_STRENGTH * np.log(W_safe[t + 1]) + log_lik[t + 1] + log_beta[t + 1]
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
                        + SUP_STRENGTH * np.log(W_safe[t + 1, j])
                        + log_lik[t + 1, j]
                        + log_beta[t + 1, j]
                    )
            xi[t] -= np.max(xi[t])
            xi[t] = np.exp(xi[t])
            xi[t] /= np.sum(xi[t])
            # Calcul de la vraie log-vraisemblance (log-sum-exp sur log_alpha à la dernière étape)
            log_likelihood = np.logaddexp.reduce(log_alpha[-1])
        return gamma, xi, log_likelihood    

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
       # ...dans la méthode fit...
        while i < max_iter_safe:
            log_lik = self._compute_log_likelihoods(X)
            gamma, xi, ll = self._forward_backward(log_lik, W)  # ll = vraie log-vraisemblance
            self._m_step(X, gamma, xi)
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
print("⏳ Chargement et prétraitement des données IMU et IR...")
X_all, W_all, timestamps_all, ir_all = [], [], [], []

all_csv_files = list(base_path.glob("*/session*/*.imu_relative.csv"))
print(f"🔍 {len(all_csv_files)} fichiers IMU trouvés")

for csv_file in all_csv_files:

    try:
        print(f"➡️ Traitement du fichier : {csv_file.name}")

        # -------- CHARGEMENT IMU --------
        df = pd.read_csv(csv_file)
        df = df.select_dtypes(include=[np.number]).dropna() #garde juste les colonnes numériques (float ou int), dropna enlève les lignes avec valeurs manquantes 
        df = df[~df.isin([np.inf, -np.inf]).any(axis=1)] #enlève les lignes avec des valeurs infinies

        # Timestamps IMU
        time_col = "Time" if "Time" in df.columns else ("IMU Time (s)" if "IMU Time (s)" in df.columns else None)
        if time_col is None:
            warnings.warn(f"Pas de colonne de temps dans {csv_file.name}, utilisation d'un index temporel.", UserWarning)
            timestamps = np.arange(len(df))
            fs = 30  # Fréquence par défaut
        else:
            timestamps = df[time_col].values
            fs = 1 / np.median(np.diff(timestamps)) if len(np.diff(timestamps)) > 0 else 30

        # ...dans la boucle for csv_file in all_csv_files: ...
# -------- CHARGEMENT IR --------
        mouse = csv_file.parts[-3].replace("m", "")
        session_num = int(re.search(r'\d+', csv_file.parts[-2]).group())
        ir_folder = ir_base_folder / mouse
        ir_files = list(ir_folder.glob(f"IR_{mouse}_session{session_num}_binaire*.csv"))
        if not ir_files:
            warnings.warn(f"Pas de fichier IR pour {mouse} session {session_num}", UserWarning)
            continue

        # Lecture du fichier IR avec en-tête
        ir_df = pd.read_csv(ir_files[0], sep=None, engine='python')  # auto-détection du séparateur
        ir_timestamps = ir_df['timestamp'].values
        ir_binary = ir_df['presence_binaire'].values

        # Synchronisation par interpolation sur les timestamps IMU
        from scipy.interpolate import interp1d
        interp_func = interp1d(ir_timestamps, ir_binary, kind='nearest', bounds_error=False, fill_value=(ir_binary[0], ir_binary[-1]))
        ir_binary_aligned = interp_func(timestamps)

        # Correction du décalage temporel IMU/IR par cross-corrélation
        # Utilise la norme d'accélération lissée comme référence
        acc_norm_smooth = None
        if 'acc_norm_smooth' in df.columns:
            acc_norm_smooth = df['acc_norm_smooth'].values
        else:
            acc_cols = [col for col in df.columns if 'acc' in col.lower()]
            if len(acc_cols) >= 3:
                acc = df[acc_cols[:3]].values
                acc_norm = np.linalg.norm(acc, axis=1)
                acc_norm_smooth = pd.Series(acc_norm).rolling(window=5, min_periods=1).mean().values

        if acc_norm_smooth is not None and len(acc_norm_smooth) == len(ir_binary_aligned):
            # Centrage pour la corrélation
            acc_centered = acc_norm_smooth - np.mean(acc_norm_smooth)
            ir_centered = ir_binary_aligned - np.mean(ir_binary_aligned)
            # Cross-corrélation
            corr = np.correlate(acc_centered, ir_centered, mode='full')
            lags = np.arange(-len(acc_centered) + 1, len(acc_centered))
            best_lag = lags[np.argmax(np.abs(corr))]
            print(f"Décalage optimal IMU/IR détecté (en points): {best_lag}")
            # Décalage de la séquence IR
            if best_lag > 0:
                ir_binary_aligned = np.pad(ir_binary_aligned, (best_lag, 0), mode='edge')[:-best_lag]
            elif best_lag < 0:
                ir_binary_aligned = np.pad(ir_binary_aligned, (0, -best_lag), mode='edge')[-best_lag:]
            # Sinon, best_lag == 0, rien à faire
        else:
            print("⚠️ Impossible de corriger le décalage IMU/IR (feature manquante ou taille incohérente)")

        # Alignement des longueurs IMU et IR AVANT downsampling
        min_len = min(len(df), len(ir_binary_aligned))
        df = df.iloc[:min_len]
        ir_binary_aligned = ir_binary_aligned[:min_len]
        timestamps = timestamps[:min_len]
            
        # Downsampling
        factor = max(1, int(fs / target_fs))
        df_ds = df.iloc[::factor].copy()
        ir_binary_ds = ir_binary_aligned[::factor]
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
                warnings.warn(f"Erreur : acc_df et ori_df n'ont pas le même nombre de lignes pour {csv_file.name} ({len(acc_df)} vs {len(ori_df)})", RuntimeWarning)
                continue
            features_all = pd.concat([acc_df, ori_df], axis=1)
            print(f"ℹ️ Colonnes d'orientation détectées dans {csv_file.name}")
        else:
            features_all = acc_df.copy()
            warnings.warn(f"Pas de colonnes d'orientation dans {csv_file.name}, utilisation uniquement des données d'accélération.", UserWarning)

        # Normalisation robuste
        std = features_all.std()
        std[std == 0] = 1e-8  # Remplace les écarts-types nuls par une petite valeur
        features_all_zscored = (features_all - features_all.mean()) / std
        X_ds = features_all_zscored.values
        if np.isnan(features_all_zscored.values).any() or np.isinf(features_all_zscored.values).any():
            warnings.warn(f"Attention : NaN ou infini détecté dans les features normalisées pour {csv_file.name}", RuntimeWarning)

        # Correction : Création de W avec des valeurs minimales
        W = np.full((len(ir_binary_ds), num_states), 0.01)
        W[:, 0] = 0.99
        W[ir_binary_ds == 1, 0] = 0.01
        W[ir_binary_ds == 1, 1] = 0.99

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
print("✅ Chargement et prétraitement terminés.")

print("=== Vérification de la proportion de 1 dans chaque session IR ===")
for i, ir_seq in enumerate(ir_all):
    prop = np.mean(ir_seq)
    print(f"Session {i}: proportion de 1 dans IR = {prop:.3f} (longueur {len(ir_seq)})")
print("===============================================")

# =======================================
# VÉRIFICATION DES TAILLES AVANT ENTRAINEMENT
# =======================================
print("🔎 Vérification des tailles des sessions...")

if len(X_all) == 0:
    raise ValueError("❌ Aucune session valide trouvée.")

# Vérification des tailles
for i in range(len(X_all)):
    min_len = min(len(X_all[i]), len(W_all[i]), len(ir_all[i]))
    X_all[i] = X_all[i][:min_len]
    W_all[i] = W_all[i][:min_len]
    ir_all[i] = ir_all[i][:min_len]
    print(f"✅ Session {i} ajustée à {min_len} échantillons")
print("✅ Vérification des tailles terminée.")

print("=== Vérifications avancées sur les IR binaires ===")
for i, ir_seq in enumerate(ir_all):
    print(f"Session {i}:")
    print(f"  - Longueur IR : {len(ir_seq)}")
    uniques, counts = np.unique(ir_seq, return_counts=True)
    print(f"  - Valeurs uniques IR : {dict(zip(uniques, counts))}")
    if np.isnan(ir_seq).any():
        print("  - ⚠️ Attention : NaN détecté dans IR")
    if np.isinf(ir_seq).any():
        print("  - ⚠️ Attention : Inf détecté dans IR")
    ones_idx = np.where(ir_seq == 1)[0]
    if len(ones_idx) > 0:
        print(f"  - Indices des 1 (extrait) : {ones_idx[:10]} ...")
    else:
        print("  - Aucun événement IR=1 détecté")
print("===============================================")

from sklearn.metrics import f1_score, precision_score, recall_score

f1_scores, precisions, recalls = [], [], []

from sklearn.metrics import f1_score, precision_score, recall_score

print("=== Début Cross-Validation Leave-One-Session-Out ===")

rows = []
f1_scores, precisions, recalls = [], [], []
TP = FP = FN = TN = 0  # pour micro-averaging

for test_index in range(len(X_all)):
    print(f"\n=== Session {test_index+1}/{len(X_all)} en test ===")
    
    # Données test
    X_test = X_all[test_index]
    W_test = W_all[test_index]
    ts_test = timestamps_all[test_index]   # utile si tu veux sauvegarder des courbes plus tard
    ir_test = ir_all[test_index]
    
    # Données train (toutes les autres sessions)
    X_train = np.vstack([x for i, x in enumerate(X_all) if i != test_index])
    W_train = np.vstack([w for i, w in enumerate(W_all) if i != test_index])
    
    # Entraînement
    model = SimpleARWHMM(num_states=num_states, ar_order=ar_order)
    model.fit(X_train, W_train)

    # Décodage sur la session test (sans supervision)
    W_test_eval = np.ones_like(W_test)
    log_lik_test = model._compute_log_likelihoods(X_test)
    gamma_test, _, _ = model._forward_backward(log_lik_test, W_test_eval)
    states_test = np.argmax(gamma_test, axis=1)

    # Identification de l’état "nourrissage" par corrélation
    correlations = []
    for k in range(num_states):
        pred = (states_test == k).astype(float)
        if np.std(pred) > 0 and np.std(ir_test) > 0:
            corr = np.corrcoef(pred, ir_test)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)
    feeding_state = np.argmax(np.abs(correlations))

    # Binaire préd/gt
    pred_labels = (states_test == feeding_state).astype(int)
    gt_labels = (pd.to_numeric(ir_test, errors='coerce') > 0.5).astype(int)

    # Ajustement longueurs (par sécurité)
    m = min(len(gt_labels), len(pred_labels))
    gt_labels = gt_labels[:m]
    pred_labels = pred_labels[:m]

    # Métriques (zero_division=0 pour éviter les warnings si aucun positif)
    f1  = f1_score(gt_labels, pred_labels, zero_division=0)
    prc = precision_score(gt_labels, pred_labels, zero_division=0)
    rcl = recall_score(gt_labels, pred_labels, zero_division=0)

    print(f"  🎯 F1={f1:.3f}, Précision={prc:.3f}, Rappel={rcl:.3f}")

    # Accumule
    f1_scores.append(f1)
    precisions.append(prc)
    recalls.append(rcl)

    # Confusion pour micro-averaging
    tp = int(np.sum((pred_labels == 1) & (gt_labels == 1)))
    fp = int(np.sum((pred_labels == 1) & (gt_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (gt_labels == 1)))
    tn = int(np.sum((pred_labels == 0) & (gt_labels == 0)))
    TP += tp; FP += fp; FN += fn; TN += tn

    # Garde une ligne de résumé par session
    rows.append({
        "session": test_index,
        "n_points": int(m),
        "pos_rate": float(np.mean(gt_labels)),
        "feeding_state": int(feeding_state),
        "corr_max": float(correlations[feeding_state]),
        "F1": float(f1),
        "precision": float(prc),
        "recall": float(rcl),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    })

# Résumé final (macro et micro)
macro_f1  = float(np.mean(f1_scores))
macro_p   = float(np.mean(precisions))
macro_r   = float(np.mean(recalls))
micro_p   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
micro_r   = TP / (TP + FN) if (TP + FN) > 0 else 0.0
micro_f1  = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

print("\n=== Résultats Cross-Validation (Leave-One-Session-Out) ===")
print(f"Macro  — F1: {macro_f1:.3f} | Précision: {macro_p:.3f} | Rappel: {macro_r:.3f}")
print(f"Micro  — F1: {micro_f1:.3f} | Précision: {micro_p:.3f} | Rappel: {micro_r:.3f}")

# Tableau récapitulatif par session (optionnel)
try:
    cv_df = pd.DataFrame(rows)
    # Affichage propre
    with pd.option_context('display.max_rows', None, 'display.width', 120):
        print("\nDétails par session :")
        print(cv_df[["session","n_points","pos_rate","feeding_state","corr_max","F1","precision","recall","TP","FP","FN","TN"]]
              .round({"pos_rate":3,"corr_max":3,"F1":3,"precision":3,"recall":3}))
    # Sauvegarde (optionnelle)
    cv_df.to_csv("loso_results.csv", index=False)
    print("\n💾 Fichier sauvegardé : loso_results.csv")
except Exception as e:
    print("Impossible de créer/sauvegarder le tableau de résumé :", e)
