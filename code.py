# ===========================================================
# 🏥 SecureCare HAR — Federated Learning (FedAvg Probabilities)
# Model: Voting Ensemble (Tuned XGBoost + RandomForest)
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import itertools
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1) Load & clean dataset
# -------------------------------
print("Loading dataset...")
df = pd.read_csv("SecureCare_Data.csv") 
df = df.drop(['Time', 'Person'], axis=1)

# Label encode class names
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])
print(f"Dataset shape: {df.shape}")
print(f"Classes: {le.classes_}")

# -------------------------------
# 2) Window-based feature extraction (HAR-style)
# -------------------------------
print("Extracting windowed features...")
WINDOW_SIZE = 128
STEP_SIZE   = 64

def extract_features(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    feats, labels = [], []
    for start in range(0, len(df) - window_size, step_size):
        end = start + window_size
        window = df.iloc[start:end]
        label = window['Class'].mode()[0]
        # Statistical + magnitude features
        ax, ay, az = window['Acc_x'], window['Acc_y'], window['Acc_z']
        amag = np.sqrt(ax**2 + ay**2 + az**2)
        feature_dict = {
            'Acc_x_mean': ax.mean(),   'Acc_y_mean': ay.mean(),   'Acc_z_mean': az.mean(),
            'Acc_x_std':  ax.std(),    'Acc_y_std':  ay.std(),    'Acc_z_std':  az.std(),
            'Acc_x_min':  ax.min(),    'Acc_y_min':  ay.min(),    'Acc_z_min':  az.min(),
            'Acc_x_max':  ax.max(),    'Acc_y_max':  ay.max(),    'Acc_z_max':  az.max(),
            'Acc_mag_mean': amag.mean(),
            'Acc_mag_std':  amag.std(),
            'Acc_range': (pd.DataFrame({'x':ax,'y':ay,'z':az}).max().mean()
                         -pd.DataFrame({'x':ax,'y':ay,'z':az}).min().mean()),
            'Acc_energy':  (ax**2 + ay**2 + az**2).sum()
        }
        feats.append(feature_dict)
        labels.append(label)
    features_df = pd.DataFrame(feats)
    features_df['Class'] = labels
    return features_df

df_feat = extract_features(df)
print(f"Features extracted: {df_feat.shape}")

# -------------------------------
# 3) Train/Test split + normalization
# -------------------------------
print("Splitting and normalizing data...")
X = df_feat.drop('Class', axis=1)
y = df_feat['Class']

scaler = MinMaxScaler()
Xn = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    Xn, y, test_size=0.2, random_state=42, stratify=y
)

n_classes = len(np.unique(y_train))
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Number of classes: {n_classes}")

# -------------------------------
# 4) Tune XGBoost on global train (same as your best run)
# -------------------------------
print("Tuning XGBoost hyperparameters...")
xgb_base = XGBClassifier(eval_metric='mlogloss', random_state=42)

param_dist = {
    'n_estimators':    [200, 400, 600, 800, 1000],
    'max_depth':       [4, 6, 8, 10],
    'learning_rate':   [0.01, 0.05, 0.1],
    'subsample':       [0.7, 0.8, 1.0],
    'colsample_bytree':[0.7, 0.8, 1.0],
    'min_child_weight':[1, 3, 5],
    'gamma':           [0, 0.1, 0.2],
    'reg_lambda':      [1, 1.5, 2],
    'reg_alpha':       [0, 0.1, 0.2]
}

search = RandomizedSearchCV(
    xgb_base, param_distributions=param_dist,
    n_iter=15, cv=3, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
)
search.fit(X_train, y_train)
best_xgb = search.best_estimator_
print(f"\nBest XGBoost Params: {search.best_params_}")

# Fixed RF to pair with tuned XGB in the ensemble
rf_global = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)

def build_best_voter():
    return VotingClassifier(
        estimators=[('xgb', best_xgb), ('rf', rf_global)],
        voting='soft'
    )

# -------------------------------
# 5) Helper: evaluate metrics
# -------------------------------
def evaluate(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec  = recall_score(y_true, y_pred, average='weighted')
    f1   = f1_score(y_true, y_pred, average='weighted')
    return acc, prec, rec, f1

def calculate_client_score(y_true, y_pred_proba):
    """
    Calculate client reputation score using Log Loss.
    Returns score between 0-1 (higher is better).
    """
    try:
        # Calculate log loss
        loss = log_loss(y_true, y_pred_proba)
        # Convert to 0-1 scale (higher is better)
        # Using exponential decay: score = exp(-loss)
        score = np.exp(-loss)
        return min(score, 1.0)  # Cap at 1.0
    except:
        # Fallback to accuracy if log loss fails
        y_pred = np.argmax(y_pred_proba, axis=1)
        return accuracy_score(y_true, y_pred)

# -------------------------------
# 6) Make 5 stratified clients from the TRAIN set
# -------------------------------
print("Creating federated clients...")
def make_clients(X, y, num_clients=5, seed=42):
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
    splits = []
    for _, idx in skf.split(X, y):
        splits.append(idx)
    clients = []
    for idx in splits:
        clients.append((X[idx], y.iloc[idx].values))
    return clients

clients = make_clients(X_train, y_train, num_clients=5, seed=42)
for i, (Xc, yc) in enumerate(clients, 1):
    print(f"Client {i}: {len(yc)} samples")

# -------------------------------
# 7) Federated training (FedAvg on probabilities)
# -------------------------------
print("\nStarting Federated Learning...")
ROUNDS = 5
global_metrics = []
round_client_metrics = []  # list of dicts per round
client_reputation_scores = []  # track reputation scores per round

for r in range(1, ROUNDS+1):
    print(f"\nFederated Round {r}/{ROUNDS}")
    client_probs = []     # predicted probabilities on global test, per client
    client_weights = []   # weight by client size
    client_stats = []     # local metrics on client-held-out validation (optional quick split)

    # Train each client locally on its partition
    round_reputation_scores = []
    for ci, (Xc, yc) in enumerate(clients, 1):
        print(f"  Training Client {ci}...")
        # simple local validation split (10%) to report client-side metrics (no leakage)
        msk = np.random.RandomState(1000 + r + ci).rand(len(yc)) < 0.9
        Xc_tr, yc_tr = Xc[msk], yc[msk]
        Xc_va, yc_va = Xc[~msk], yc[~msk]

        model = build_best_voter()
        model.fit(Xc_tr, yc_tr)

        # metrics on client's local holdout
        if len(yc_va) > 0:
            yva_pred = model.predict(Xc_va)
            cacc, cprec, crec, cf1 = evaluate(yc_va, yva_pred)
        else:
            cacc = cprec = crec = cf1 = np.nan

        # Calculate reputation score using global test set
        p_test = model.predict_proba(X_test)  # shape: [n_test, n_classes]
        reputation_score = calculate_client_score(y_test, p_test)
        round_reputation_scores.append(reputation_score)

        client_stats.append({
            "client": ci,
            "n_train": len(yc_tr),
            "n_val": len(yc_va),
            "acc": cacc, "prec": cprec, "rec": crec, "f1": cf1,
            "reputation_score": reputation_score
        })

        # store probs on the shared global test set for FedAvg
        client_probs.append(p_test)
        client_weights.append(len(yc_tr))  # size-weighted FedAvg

    # FedAvg on probabilities (size-weighted)
    client_weights = np.array(client_weights, dtype=float)
    client_weights = client_weights / client_weights.sum()
    P = np.zeros_like(client_probs[0])
    for w, P_i in zip(client_weights, client_probs):
        P += w * P_i

    y_pred_global = np.argmax(P, axis=1)
    acc, prec, rec, f1 = evaluate(y_test, y_pred_global)
    global_metrics.append((acc, prec, rec, f1))
    round_client_metrics.append(client_stats)
    client_reputation_scores.append(round_reputation_scores)

    print(f"Global after Round {r}: "
          f"Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")
    print(f"Client Reputation Scores: {[f'{score:.3f}' for score in round_reputation_scores]}")

# -------------------------------
# 8) Plot global metrics per round
# -------------------------------
print("\nGenerating visualizations...")
gm = np.array(global_metrics)
r = np.arange(1, ROUNDS+1)

plt.figure(figsize=(10,5))
plt.plot(r, gm[:,0], marker='o', label='Accuracy')
plt.plot(r, gm[:,1], marker='o', label='Precision (weighted)')
plt.plot(r, gm[:,2], marker='o', label='Recall (weighted)')
plt.plot(r, gm[:,3], marker='o', label='F1 (weighted)')
plt.xlabel("Federated Round")
plt.ylabel("Score")
plt.title("Global Performance per Federated Round (Voting Ensemble)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# -------------------------------
# 9) Final confusion matrix (last round)
# -------------------------------
cm = confusion_matrix(y_test, y_pred_global, labels=np.arange(n_classes))

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix (Global Model, Final Round)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, le.inverse_transform(np.arange(n_classes)), rotation=45, ha='right')
plt.yticks(tick_marks, le.inverse_transform(np.arange(n_classes)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

# annotate counts
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# -------------------------------
# 10) Print per-client stats (last round)
# -------------------------------
print("\nPer-Client Metrics (last round local holdout):")
for s in round_client_metrics[-1]:
    print(f"Client {s['client']}: n_train={s['n_train']}, n_val={s['n_val']}, "
          f"acc={s['acc']:.4f}, prec={s['prec']:.4f}, rec={s['rec']:.4f}, f1={s['f1']:.4f}, "
          f"reputation={s['reputation_score']:.4f}")

# -------------------------------
# 11) Print global summary
# -------------------------------
print("\nGlobal Metrics per Round (Acc, Prec, Rec, F1):")
for i, m in enumerate(global_metrics, 1):
    print(f"Round {i}: Acc={m[0]:.4f}, Prec={m[1]:.4f}, Rec={m[2]:.4f}, F1={m[3]:.4f}")

print(f"\nFinal Federated Model Performance:")
print(f"Accuracy: {global_metrics[-1][0]:.4f}")
print(f"Precision: {global_metrics[-1][1]:.4f}")
print(f"Recall: {global_metrics[-1][2]:.4f}")
print(f"F1-Score: {global_metrics[-1][3]:.4f}")

# -------------------------------
# 12) Reputation scores summary
# -------------------------------
print(f"\nClient Reputation Scores Summary:")
print("Round-by-round reputation scores (0-1 scale, higher is better):")
for r, scores in enumerate(client_reputation_scores, 1):
    print(f"Round {r}: {[f'{score:.3f}' for score in scores]}")

# Calculate average reputation per client
avg_reputation = np.mean(client_reputation_scores, axis=0)
print(f"\nAverage Reputation Scores per Client:")
for i, score in enumerate(avg_reputation, 1):
    print(f"Client {i}: {score:.4f}")

# Save reputation scores for future use
np.save('client_reputation_scores.npy', client_reputation_scores)
print(f"\nReputation scores saved to 'client_reputation_scores.npy'")
print("You can load them later with: scores = np.load('client_reputation_scores.npy')")
