import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from random import randrange

# --- Fonction utilitaire pour aplatir et nettoyer ---
def safe_1d_floats(x):
    """
    Aplati un contenu MATLAB (cellule, liste, array) en 1D float propre.
    - Gère les listes/cell arrays hétérogènes.
    - Filtre NaN/inf.
    """
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, (list, tuple)):
        parts = [np.asarray(p).ravel() for p in x]
        if len(parts) == 0:
            return np.array([], dtype=float)
        arr = np.concatenate(parts)
    else:
        arr = np.asarray(x).ravel()
    arr = arr.astype(float)
    arr = arr[np.isfinite(arr)]
    return arr

# === PARAMÈTRES ===
base_path = Path("/Users/lise/Library/CloudStorage/GoogleDrive-lamodeo@ensc.fr/Mon Drive/Stage/imu_files")
mat_results_path = Path("/Users/lise/Desktop/Stage/results")
target_fs = 10
ar_order = 1
num_states = 3  # à ajuster
max_iter = 20

# === CLASSE DU MODELE ===
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

        log_alpha[self.p] = np.log(self.pi) + np.log(W[self.p]) + log_lik[self.p]
        for t in range(self.p + 1, T):
            for j in range(self.K):
                log_alpha[t, j] = np.log(W[t, j]) + log_lik[t, j] + np.logaddexp.reduce(
                    log_alpha[t - 1] + np.log(self.A[:, j])
                )

        log_beta[-1] = 0
        for t in reversed(range(self.p, T - 1)):
            for i in range(self.K):
                log_beta[t, i] = np.logaddexp.reduce(
                    np.log(self.A[i]) + np.log(W[t + 1]) + log_lik[t + 1] + log_beta[t + 1]
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
                        + np.log(W[t + 1, j])
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
        self.A = np.sum(xi[self.p:], axis=0)
        self.A /= np.sum(self.A, axis=1, keepdims=True)

        for k in range(self.K):
            weights = gamma[self.p:, k]
            X_past = np.array([X[t - self.p:t].flatten() for t in range(self.p, T)])
            Y = X[self.p:]
            W_diag = np.diag(weights)

            B = np.zeros((self.p * self.D, self.D))
            for d in range(self.D):
                y_d = Y[:, d]
                B_d = np.linalg.pinv(X_past.T @ W_diag @ X_past) @ X_past.T @ W_diag @ y_d
                B[:, d] = B_d.flatten()

            self.coefs[k] = B
            residuals = Y - X_past @ B
            self.covs[k] = (residuals.T @ W_diag @ residuals) / np.sum(weights) + 1e-6 * np.eye(self.D)

    def fit(self, X, W):
        X = np.asarray(X)
        W = np.asarray(W)
        self._init_parameters(X)

        prev_ll = -np.inf
        for i in range(self.max_iter):
            log_lik = self._compute_log_likelihoods(X)
            gamma, xi = self._forward_backward(log_lik, W)
            self._m_step(X, gamma, xi)
            ll = np.sum(log_lik[self.p:] * gamma[self.p:])
            print(f"🔁 Iteration {i+1}, Log-Likelihood: {ll:.2f}")
            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self.gamma_ = gamma

    def predict_states(self):
        return np.argmax(self.gamma_, axis=1)

# === TRAITEMENT DES SESSIONS ===
    
X_all, W_all, timestamps_all, ir_all, feeding_all = [], [], [], [], []
#X_all= les features extraites des signaux IMU
#W_all= supervision, probabilité que chaque frame appartienne à chaque état
# timestamps_all → les timestamps downsamplés.
#ir_all → les événements IR de nourrissage (ir_onsets, ir_offsets).

all_csv_files = list(base_path.glob("*/session*/*.imu_relative.csv"))
print(f"🔍 {len(all_csv_files)} fichiers trouvés") #sensé en avoir 37


#pour chaque csv trouvé: 
for csv_file in all_csv_files:
    try:
    #========RÉCUPÉRATION DES DONNÉES========#
        print("📂", csv_file)
        mouse_id = csv_file.parts[-3].replace("m", "")
        session_name = csv_file.parts[-2]

        #vérification qu'un fichier mat existe 
        mat_file = mat_results_path / mouse_id / f"{session_name}_results.mat"
        if not mat_file.exists():
            print(f"⚠️ Pas de .mat pour {csv_file.name}, ignoré.")
            continue
        
        #chargement et nettoyage des fichiers (garde juste les colonnes numériques et les valeurs finies)
        df = pd.read_csv(csv_file)
        df = df.select_dtypes(include=[np.number]).dropna()
        df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]

        #récupération des timestamps depuis Time ou IMU Times pour obtenir le fréquence d'échantillonage 
        if "Time" in df.columns or "IMU Time (s)" in df.columns:
            time_col = "Time" if "Time" in df.columns else "IMU Time (s)"
            timestamps = df[time_col].values
            fs = 1 / np.median(np.diff(timestamps))
        else:
            fs = 30
            timestamps = np.arange(len(df)) / fs

        #downsampling: On réduit la fréquence à target_fs (par ex. 10 Hz) pour alléger les données.
        #downsampling def: prendre moins d'échantillons pour réduire fs et donc taille mémoire et tps de calcul
        factor = int(fs / target_fs)
        timestamps_ds = timestamps[::factor]


        # ======== EXTRACTION DES FEATURES ======== #

        # --- Données d'accélération ---
        acc_cols = [col for col in df.columns if 'acc' in col.lower()]
        acc_df = df[acc_cols].iloc[::factor].copy()
        acc_df.columns = ['accX', 'accY', 'accZ']

        # Norme totale
        acc_df["acc_norm"] = np.linalg.norm(acc_df[["accX", "accY", "accZ"]].values, axis=1)

        # Lissage (moyenne glissante)
        acc_df["acc_norm_smooth"] = acc_df["acc_norm"].rolling(window=5, min_periods=1).mean()

        # Variabilité locale (écart-type glissant)
        acc_df["acc_norm_std"] = acc_df["acc_norm"].rolling(window=5, min_periods=1).std()

        # Différences (vitesses de changement)
        acc_df["d_accX"] = acc_df["accX"].diff().fillna(0)
        acc_df["d_acc_norm"] = acc_df["acc_norm"].diff().fillna(0)

        # --- Données d'orientation ---
        ori_cols = [col for col in df.columns if 'ori' in col.lower()]
        ori_df = df[ori_cols].iloc[::factor].copy()
        ori_df.columns = ["yaw", "pitch", "roll"]
        ori_df = ori_df.astype(float)

        # Encodage circulaire des angles
        for col in ["yaw", "pitch", "roll"]:
            ori_df[col + "_sin"] = np.sin(np.deg2rad(ori_df[col]))
            ori_df[col + "_cos"] = np.cos(np.deg2rad(ori_df[col]))

        # Norme d'orientation
        ori_df["ori_norm"] = np.linalg.norm(ori_df[["yaw", "pitch", "roll"]].values, axis=1)
        ori_df["ori_norm_smooth"] = ori_df["ori_norm"].rolling(window=5, min_periods=1).mean()

        # --- Fusion des features ---
        features_all = pd.concat([acc_df, ori_df.drop(columns=["yaw", "pitch", "roll"])], axis=1)
        features_all_zscored = (features_all - features_all.mean()) / features_all.std()
        X_ds = features_all_zscored.values


    #========CHARGEMENT DES IR========#
        mat = loadmat(mat_file, simplify_cells=True)

        # -- feeding_bouts (optionnel dans tes .mat) --
        feed_on, feed_off = np.array([]), np.array([])
        if 'feeding_bouts' in mat and mat['feeding_bouts'] is not None:
            fb = np.asarray(mat['feeding_bouts'])
            # Cas: vecteur 1D [on1, off1, on2, off2, ...]
            if fb.ndim == 1 and fb.size % 2 == 0:
                fb = fb.reshape(-1, 2)
            # Cas: 2D déjà en (N,2)
            elif fb.ndim == 2 and fb.shape[1] == 2:
                pass
            else:
                # Format inattendu -> on sécurise en tableau vide
                fb = np.empty((0, 2), dtype=float)

            # Nettoyage: float, finites, start<end, tri par start
            fb = fb.astype(float)
            fb = fb[np.isfinite(fb).all(axis=1)]
            if fb.size > 0:
                fb = fb[fb[:, 0] < fb[:, 1]]      # garde seulement (start<end)
                fb = fb[np.argsort(fb[:, 0])]     # trie par début
                feed_on, feed_off = fb[:, 0], fb[:, 1]

        # Sécurise aussi via la fonction utilitaire
        feed_on  = safe_1d_floats(feed_on)
        feed_off = safe_1d_floats(feed_off)

        # Sauvegarde pour plots finaux
        feeding_all.append((feed_on, feed_off))

        # -- IR on/off pour la supervision faible --
        ir_onsets  = safe_1d_floats(mat.get('ir_onsets', []))
        ir_offsets = safe_1d_floats(mat.get('ir_offsets', []))

        # Assure une taille identique (zip coupe au plus court, mais on nettoie proprement)
        n_pairs = min(len(ir_onsets), len(ir_offsets))
        ir_onsets  = ir_onsets[:n_pairs]
        ir_offsets = ir_offsets[:n_pairs]

        # Enregistre aussi les IR (pour le plot d’alignement W vs IR)
        ir_all.append((ir_onsets, ir_offsets))


    #========CRÉATION DES POIDS========#
        
        W = np.full((len(X_ds), num_states), 1 / num_states)
        eps = 0.0025

        #Pour chaque frame : Si elle tombe dans un intervalle IR → état nourrissage (1) presque certain, Sinon → état non-nourrissage (0) presque certain.
        for i, t in enumerate(timestamps_ds):
            is_feeding = any(start <= t <= end for start, end in zip(ir_onsets, ir_offsets))
            if is_feeding:
                W[i] = np.full(num_states, eps)
                W[i][1] = 1 - eps * (num_states - 1)
            else:
                W[i] = np.full(num_states, eps)
                W[i][0] = 1 - eps * (num_states - 1)

    #========STOCKAGE========#
#Stocker les données préparées de chaque session              
        if len(X_ds) > ar_order + 10:
            X_all.append(X_ds[ar_order:])
            W_all.append(W[ar_order:])
            timestamps_all.append(timestamps_ds[ar_order:])
            ir_all.append((ir_onsets, ir_offsets))
        else:
            print(f"⚠️ Session trop courte : {csv_file.name} ignorée.")

    except Exception as e:
        print(f"❌ Erreur pour {csv_file.name} : {e}")

# === TRAIN / TEST SPLIT ===
print(f"📤 {len(X_all)} sessions disponibles")
#on choisit au hasard une session que sera la ssession test 
test_index = randrange(len(X_all))
X_test = X_all[test_index]
W_test = W_all[test_index]
ts_test = timestamps_all[test_index]
ir_onsets_test, ir_offsets_test = ir_all[test_index]

#concaténation des X_train et W_train sur toutes les sessions sauf celle du test
#@@@@@@@@@@@@@ ça veut dire quoi concaténer
X_train = np.vstack([x for i, x in enumerate(X_all) if i != test_index])
W_train = np.vstack([w for i, w in enumerate(W_all) if i != test_index])
#@@@@@@@@@@@@@ pourquoi len =7 dans le terminal??
print(f"✅ Session {test_index+1}/{len(X_all)} choisie pour test")
print(f"📂 Session test : {all_csv_files[test_index]}")
print(f"✅ Entraînement sur {X_train.shape[0]} points, test sur {X_test.shape[0]} points")
feed_on_test, feed_off_test = feeding_all[test_index]

# === ENTRAÎNEMENT ===
#appel la classe SimpleARWHMM, apprend à segmenter les données en tenant compte des poids W_train
model = SimpleARWHMM(num_states=num_states, ar_order=ar_order, max_iter=max_iter)
model.fit(X_train, W_train)

# === PRÉDICTION ===
#calcul la log-vraisemblance de la session test pour chaque état latent
log_lik_test = model._compute_log_likelihoods(X_test)
#gamma_test est la probabilité a posteriori pour chaque point d’appartenir à chaque état (p(z_t = k | données)), après avoir passé le forward-backward (algorithme de lissage). 
#W_test est le poids de supervision partielle 
gamma_test, _ = model._forward_backward(log_lik_test, W_test)
#On prend pour chaque point temporel, l’état le plus probable.
states_test = np.argmax(gamma_test, axis=1)
#pour savoir quel état correspond à du nourrissage: 
#Pour chaque état k, on calcule la probabilité moyenne d’être en nourrissage (colonne 1 de W_test) pour les points qui ont été assignés à cet état, pui on prend celui ou cette moyenne est la + haute
#=>feeding_state est l’index de l’état "le plus nourrissage".
feeding_state = np.argmax([W_test[states_test == k, 1].mean() for k in range(num_states)])
# On transforme la séquence d’états en labels binaires :1 si l’état courant est feeding_state (nourrissage), 0sinon
pred_labels = (states_test == feeding_state).astype(int)

# === ÉVALUATION ===
#ground truth bisnaire pour la session test: W_test (poids de supervision) la colonne 1 (celle du nourrissage), et on met 1 si > 0.5, sinon 0.
gt_labels = (W_test[:, 1] > 0.5).astype(int)
f1 = f1_score(gt_labels, pred_labels)
precision = precision_score(gt_labels, pred_labels)
recall = recall_score(gt_labels, pred_labels)

#f1-score: mesure de la performance du modèle, de 0 à 1. moyenne harmonique de la précision et du rappel
print(f"🎯 F1-score test     : {f1:.3f}")
#précision: rapport du nb de vrais positifs sur les positifs prédits. mesure la capacité du modèle à ne pas étiqueter comme positif un échantillon qui est en réalité négatif.
print(f"🎯 Précision         : {precision:.3f}")
#Rappel: rapport du nombre de vrais positifs (VP) sur le nombre total de positifs réels (VP + FN). Il mesure la capacité du modèle à trouver tous les échantillons positifs.
print(f"🎯 Rappel            : {recall:.3f}")
print(f"🔍 État nourrissage détecté : {feeding_state}")

# === PLOTS ===
plt.figure(figsize=(12, 3))
plt.plot(ts_test, X_test[:, 0], label="accX", alpha=0.5)
plt.plot(ts_test, pred_labels, label="Prédiction nourrissage", linewidth=1.5)
plt.plot(ts_test, gt_labels, label="IR feeding (binaire)", linestyle='--', linewidth=1.5)
for s, e in zip(feed_on_test, feed_off_test):
    plt.axvspan(s, e, color='green', alpha=0.15, label="Feeding bouts" if 'Feeding bouts' not in plt.gca().get_legend_handles_labels()[1] else None)
plt.legend()
plt.title("Correspondance prédictions / IR")
plt.tight_layout()
plt.show()

# Alignement W vs IR
plt.figure(figsize=(12, 3))
plt.plot(ts_test, W_test[:, 1], label="Poids nourrissage (W)")
for onset, offset in zip(ir_onsets_test, ir_offsets_test):
    plt.axvspan(onset, offset, color='red', alpha=0.2)
plt.legend()
plt.title("Alignement W vs événements IR (session test)")
plt.tight_layout()
plt.show()

