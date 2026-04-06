"""
IPL Match Prediction - Flask API with Random Forest
====================================================
Trains a Random Forest model on IPL_2008_2026.csv and
exposes prediction + stats endpoints.

Run:
    pip install flask scikit-learn pandas numpy
    python app.py
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='.', static_url_path='')

# ────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────────────────

CSV_PATH = 'IPL_2008_2026.csv'

def form_to_wins(form_str: str) -> int:
    """Count wins in the last-5-matches form string (e.g. 'W W L W L')."""
    return sum(1 for r in str(form_str).split() if r == 'W')

def is_home_venue(venue: str) -> int:
    return 0 if str(venue).strip().lower() == 'away' else 1

INJURY_SEVERITY = {
    'None': 0,
    'Wicketkeeper Out': 1,
    '2 Players Out': 2,
    'Key Bowler Out': 2,
    'Top Batsman Out': 2,
    'Star AllRounder Out': 3,
}

WEATHER_RISK = {
    'Clear': 0,
    'Partly Cloudy': 0,
    'Cloudy': 1,
    'Hot & Dry': 1,
    'Humid': 1,
    'Drizzle': 2,
}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature-engineered copy of df."""
    d = df.copy()
    # Fill NaN text columns before mapping
    d['injuries'] = d['injuries'].fillna('None').astype(str)
    d['weather']  = d['weather'].fillna('Clear').astype(str)
    d['form']     = d['form'].fillna('').astype(str)
    d['home']     = d['home'].fillna('Away').astype(str)
    d['recent_wins']     = d['form'].apply(form_to_wins)
    d['is_home']         = d['home'].apply(is_home_venue)
    d['injury_severity'] = d['injuries'].map(INJURY_SEVERITY).fillna(1).astype(int)
    d['weather_risk']    = d['weather'].map(WEATHER_RISK).fillna(0).astype(int)
    d['win']             = (d['result'] == 'Win').astype(int)
    return d

# ────────────────────────────────────────────────────────────────────────────
# 2.  MODEL TRAINING
# ────────────────────────────────────────────────────────────────────────────

print("🔄  Loading data …")
raw_df = pd.read_csv(CSV_PATH)
df = build_features(raw_df)

TEAMS     = sorted(raw_df['team'].dropna().unique().tolist())
WEATHERS  = sorted(raw_df['weather'].dropna().unique().tolist())
INJURIES  = sorted(raw_df['injuries'].dropna().unique().tolist())
VENUES    = sorted(raw_df['home'].dropna().unique().tolist())

# Label-encode team & opponent
le_team = LabelEncoder()
le_opp  = LabelEncoder()
df['team_enc']     = le_team.fit_transform(df['team'])
df['opponent_enc'] = le_opp.fit_transform(df['opponent'])

FEATURE_COLS = [
    'team_enc', 'opponent_enc',
    'recent_wins', 'is_home',
    'injury_severity', 'weather_risk'
]
TARGET = 'win'

X = df[FEATURE_COLS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("🌲  Training Random Forest …")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluation ──────────────────────────────────────────────────────────────
y_pred     = model.predict(X_test)
y_prob     = model.predict_proba(X_test)[:, 1]
accuracy   = accuracy_score(y_test, y_pred)
roc_auc    = roc_auc_score(y_test, y_prob)
cv_scores  = cross_val_score(model, X, y, cv=5, scoring='accuracy')
conf_matrix = confusion_matrix(y_test, y_pred).tolist()

feat_imp = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

print(f"✅  Accuracy   : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"✅  ROC-AUC    : {roc_auc:.4f}")
print(f"✅  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"✅  Features   : {feat_imp}")

# ── Team-level statistics cache ──────────────────────────────────────────────
team_stats = {}
for team in TEAMS:
    mask = df['team'] == team
    sub  = df[mask]
    team_stats[team] = {
        'total_matches': int(mask.sum()),
        'wins'         : int(sub['win'].sum()),
        'losses'       : int((sub['win'] == 0).sum()),
        'win_rate'     : round(float(sub['win'].mean() * 100), 1),
        'home_wr'      : round(float(sub[sub['is_home'] == 1]['win'].mean() * 100)
                               if sub[sub['is_home'] == 1].shape[0] else 0, 1),
        'away_wr'      : round(float(sub[sub['is_home'] == 0]['win'].mean() * 100)
                               if sub[sub['is_home'] == 0].shape[0] else 0, 1),
    }

# ────────────────────────────────────────────────────────────────────────────
# 3.  ROUTES
# ────────────────────────────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/api/teams', methods=['GET'])
def get_teams():
    return jsonify({'teams': TEAMS, 'weathers': WEATHERS, 'injuries': INJURIES})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'team_stats': team_stats,
        'model_metrics': {
            'accuracy'   : round(accuracy * 100, 2),
            'roc_auc'    : round(roc_auc, 4),
            'cv_mean'    : round(cv_scores.mean() * 100, 2),
            'cv_std'     : round(cv_scores.std() * 100, 2),
            'train_rows' : len(X_train),
            'test_rows'  : len(X_test),
            'conf_matrix': conf_matrix,
            'feature_importance': feat_imp,
        },
        'total_matches': len(df),
        'teams_count'  : len(TEAMS),
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    required = ['team', 'opponent', 'home', 'form', 'injuries', 'weather']
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    team      = data['team']
    opponent  = data['opponent']
    venue     = data['home']
    form_str  = data['form']
    injuries  = data['injuries']
    weather   = data['weather']

    # Validate team
    if team not in le_team.classes_:
        return jsonify({'error': f'Unknown team: {team}. Valid: {TEAMS}'}), 400
    if opponent not in le_opp.classes_:
        return jsonify({'error': f'Unknown opponent: {opponent}. Valid: {TEAMS}'}), 400

    recent_wins     = form_to_wins(form_str)
    is_home_flag    = is_home_venue(venue)
    inj_severity    = INJURY_SEVERITY.get(injuries, 1)
    wx_risk         = WEATHER_RISK.get(weather, 0)
    team_enc        = le_team.transform([team])[0]
    opponent_enc    = le_opp.transform([opponent])[0]

    X_input = np.array([[team_enc, opponent_enc,
                          recent_wins, is_home_flag,
                          inj_severity, wx_risk]])

    win_prob  = float(model.predict_proba(X_input)[0][1])
    loss_prob = 1.0 - win_prob
    prediction = 'Win' if win_prob >= 0.5 else 'Loss'
    confidence = max(win_prob, loss_prob) * 100

    # H2H historical
    h2h_mask = (raw_df['team'] == team) & (raw_df['opponent'] == opponent)
    h2h_sub  = df[h2h_mask]
    h2h_wins = int(h2h_sub['win'].sum())    if len(h2h_sub) else 0
    h2h_total = len(h2h_sub)

    # Key factors
    factors = []
    if is_home_flag:
        factors.append('🏟️ Home advantage')
    if recent_wins >= 4:
        factors.append('🔥 Excellent recent form')
    elif recent_wins <= 1:
        factors.append('❄️ Poor recent form')
    if inj_severity >= 2:
        factors.append('🤕 Injury concerns')
    if wx_risk == 2:
        factors.append('🌧️ Difficult weather conditions')
    if h2h_total > 0 and h2h_wins / h2h_total > 0.6:
        factors.append('💪 Strong head-to-head record')

    return jsonify({
        'team'       : team,
        'opponent'   : opponent,
        'prediction' : prediction,
        'win_prob'   : round(win_prob * 100, 2),
        'loss_prob'  : round(loss_prob * 100, 2),
        'confidence' : round(confidence, 2),
        'factors'    : factors,
        'input_features': {
            'recent_wins'   : recent_wins,
            'is_home'       : bool(is_home_flag),
            'injury_severity': inj_severity,
            'weather_risk'  : wx_risk,
        },
        'h2h'        : {
            'wins'  : h2h_wins,
            'total' : h2h_total,
            'win_pct': round(h2h_wins / h2h_total * 100, 1) if h2h_total else 0,
        }
    })


@app.route('/api/head2head', methods=['GET'])
def head2head():
    team     = request.args.get('team', '')
    opponent = request.args.get('opponent', '')
    if not team or not opponent:
        return jsonify({'error': 'team and opponent params required'}), 400

    mask = (raw_df['team'] == team) & (raw_df['opponent'] == opponent)
    sub  = df[mask]
    if sub.empty:
        return jsonify({'total': 0, 'wins': 0, 'losses': 0, 'win_pct': 0, 'matches': []})

    matches = []
    for _, row in sub.tail(10).iterrows():
        matches.append({
            'venue'  : row['home'],
            'result' : row['result'],
            'form'   : row['form'],
            'weather': row['weather'],
        })

    return jsonify({
        'team'   : team,
        'opponent': opponent,
        'total'  : len(sub),
        'wins'   : int(sub['win'].sum()),
        'losses' : int((sub['win'] == 0).sum()),
        'win_pct': round(float(sub['win'].mean() * 100), 1),
        'matches': matches,
    })


if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print(f"🏏 IPL Prediction API running on port {port}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=port)
