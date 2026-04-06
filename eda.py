"""
IPL Match Prediction - Exploratory Data Analysis
=================================================
Analyzes IPL_2008_2026.csv dataset with rich visualizations.
Run: python eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#e6edf3',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
})

TEAM_COLORS = {
    'CSK': '#FDB913', 'MI': '#004BA0', 'RCB': '#EC1C24',
    'KKR': '#3A225D', 'RR': '#EA1A85', 'DC': '#0078BC',
    'PBKS': '#ED1B24', 'SRH': '#F26522'
}

# ── Load Data ──────────────────────────────────────────────────────────
print("=" * 65)
print("  IPL 2008–2026  |  Exploratory Data Analysis")
print("=" * 65)

df = pd.read_csv('IPL_2008_2026.csv')

print(f"\n📊 Dataset shape : {df.shape}")
print(f"📋 Columns       : {list(df.columns)}")
print(f"\n{df.head(10).to_string()}")

print("\n\n── Missing Values ──────────────────────────────────────────────")
print(df.isnull().sum())

print("\n── Data Types ──────────────────────────────────────────────────")
print(df.dtypes)

print("\n── Result Distribution ─────────────────────────────────────────")
print(df['result'].value_counts())

print("\n── Unique Values ───────────────────────────────────────────────")
for col in df.columns:
    print(f"  {col:12s}: {df[col].nunique()} unique → {df[col].unique()[:5]}")


# ── Feature Engineering (for EDA) ─────────────────────────────────────
def form_to_wins(form_str):
    """Count recent wins from form string like 'W W L W L'."""
    results = form_str.split()
    return sum(1 for r in results if r == 'W')

def is_home(venue):
    return 0 if str(venue).strip().lower() == 'away' else 1

df['recent_wins'] = df['form'].apply(form_to_wins)
df['is_home'] = df['home'].apply(is_home)
df['win'] = (df['result'] == 'Win').astype(int)
df['has_injury'] = (df['injuries'] != 'None').astype(int)


print("\n\n── Win Rate by Team ────────────────────────────────────────────")
team_stats = df.groupby('team')['win'].agg(['sum', 'count', 'mean']).rename(
    columns={'sum': 'Wins', 'count': 'Games', 'mean': 'WinRate'})
team_stats['WinRate'] = (team_stats['WinRate'] * 100).round(1)
print(team_stats.sort_values('WinRate', ascending=False).to_string())

print("\n── Home vs Away Win Rate ────────────────────────────────────────")
print(df.groupby('is_home')['win'].mean().rename({0: 'Away', 1: 'Home'}))

print("\n── Weather Impact on Wins ───────────────────────────────────────")
print(df.groupby('weather')['win'].mean().sort_values(ascending=False).round(3))

print("\n── Injury Impact on Wins ────────────────────────────────────────")
print(df.groupby('has_injury')['win'].mean().rename({0: 'No Injury', 1: 'Has Injury'}))

print("\n── Recent Form vs Win Rate ──────────────────────────────────────")
print(df.groupby('recent_wins')['win'].mean().round(3))


# ── PLOTS ──────────────────────────────────────────────────────────────
print("\n\n🎨 Generating visualizations...")

fig = plt.figure(figsize=(22, 28))
fig.suptitle('IPL 2008–2026  |  Exploratory Data Analysis',
             fontsize=20, fontweight='bold', color='#58a6ff', y=0.98)

NROW, NCOL = 4, 3
axes = []

# 1. Win Rate by Team
ax1 = fig.add_subplot(NROW, NCOL, 1)
axes.append(ax1)
wr = df.groupby('team')['win'].mean().sort_values(ascending=False)
colors = [TEAM_COLORS.get(t, '#58a6ff') for t in wr.index]
bars = ax1.bar(wr.index, wr.values * 100, color=colors, edgecolor='#30363d', linewidth=0.8)
ax1.set_title('Win Rate by Team (%)', fontsize=12, color='#58a6ff', pad=10)
ax1.set_ylabel('Win Rate (%)')
ax1.axhline(50, color='#f0883e', linewidth=1, linestyle='--', alpha=0.7, label='50%')
ax1.legend(fontsize=8)
for bar, val in zip(bars, wr.values * 100):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8, color='#e6edf3')

# 2. Total Matches Played by Team
ax2 = fig.add_subplot(NROW, NCOL, 2)
ms = df.groupby('team').size().sort_values(ascending=False)
colors2 = [TEAM_COLORS.get(t, '#79c0ff') for t in ms.index]
bars2 = ax2.bar(ms.index, ms.values, color=colors2, edgecolor='#30363d', linewidth=0.8)
ax2.set_title('Total Matches Played by Team', fontsize=12, color='#58a6ff', pad=10)
ax2.set_ylabel('Number of Matches')
for bar, val in zip(bars2, ms.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             str(val), ha='center', va='bottom', fontsize=8)

# 3. Result Distribution (Pie)
ax3 = fig.add_subplot(NROW, NCOL, 3)
result_counts = df['result'].value_counts()
wedge_props = {'edgecolor': '#0d1117', 'linewidth': 2}
pctdist = 0.7
wedges, texts, autotexts = ax3.pie(
    result_counts.values,
    labels=result_counts.index,
    autopct='%1.1f%%',
    colors=['#3fb950', '#f85149'],
    wedgeprops=wedge_props,
    startangle=90
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
ax3.set_title('Overall Win/Loss Distribution', fontsize=12, color='#58a6ff', pad=10)

# 4. Home vs Away Win Rate
ax4 = fig.add_subplot(NROW, NCOL, 4)
ha_wr = df.groupby('is_home')['win'].mean() * 100
labels = ['Away', 'Home']
bar_colors = ['#f85149', '#3fb950']
b4 = ax4.bar(labels, ha_wr.values, color=bar_colors, width=0.5, edgecolor='#30363d')
ax4.set_title('Home vs Away Win Rate (%)', fontsize=12, color='#58a6ff', pad=10)
ax4.set_ylabel('Win Rate (%)')
ax4.set_ylim(0, 100)
for bar, val in zip(b4, ha_wr.values):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5. Recent Form (wins in last 5) vs Win Probability
ax5 = fig.add_subplot(NROW, NCOL, 5)
form_wr = df.groupby('recent_wins')['win'].mean() * 100
ax5.plot(form_wr.index, form_wr.values, 'o-', color='#f0883e', linewidth=2.5,
         markersize=9, markerfacecolor='#ffa657', markeredgecolor='#0d1117', markeredgewidth=1.5)
ax5.fill_between(form_wr.index, form_wr.values, alpha=0.15, color='#f0883e')
ax5.set_title('Recent Form (Wins in last 5) vs Win %', fontsize=12, color='#58a6ff', pad=10)
ax5.set_xlabel('Wins in Last 5 Matches')
ax5.set_ylabel('Win Probability (%)')
ax5.set_xticks(range(6))
for x, y in zip(form_wr.index, form_wr.values):
    ax5.annotate(f'{y:.0f}%', (x, y), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=9)

# 6. Weather Impact on Win Rate
ax6 = fig.add_subplot(NROW, NCOL, 6)
wx_wr = df.groupby('weather')['win'].mean().sort_values(ascending=True) * 100
wx_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(wx_wr)))
bars6 = ax6.barh(wx_wr.index, wx_wr.values, color=wx_colors, edgecolor='#30363d')
ax6.set_title('Weather Conditions vs Win Rate (%)', fontsize=12, color='#58a6ff', pad=10)
ax6.set_xlabel('Win Rate (%)')
ax6.axvline(50, color='#f0883e', linewidth=1.2, linestyle='--', alpha=0.7)
for bar, val in zip(bars6, wx_wr.values):
    ax6.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
             f'{val:.1f}%', va='center', fontsize=9)

# 7. Injury Impact
ax7 = fig.add_subplot(NROW, NCOL, 7)
inj_wr = df.groupby('has_injury')['win'].mean() * 100
b7 = ax7.bar(['No Injury', 'Has Injury'], inj_wr.values,
             color=['#3fb950', '#f85149'], width=0.5, edgecolor='#30363d')
ax7.set_title('Injury Status vs Win Rate (%)', fontsize=12, color='#58a6ff', pad=10)
ax7.set_ylabel('Win Rate (%)')
ax7.set_ylim(0, 100)
for bar, val in zip(b7, inj_wr.values):
    ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 8. Injury Type Distribution
ax8 = fig.add_subplot(NROW, NCOL, 8)
inj_counts = df['injuries'].value_counts()
inj_colors = ['#58a6ff', '#f85149', '#ffa657', '#3fb950', '#bc8cff', '#ff7b72'][:len(inj_counts)]
wedges8, texts8, auto8 = ax8.pie(
    inj_counts.values, labels=inj_counts.index, autopct='%1.1f%%',
    colors=inj_colors, wedgeprops={'edgecolor': '#0d1117', 'linewidth': 1.5},
    startangle=140
)
for t in texts8:
    t.set_fontsize(7.5)
for a in auto8:
    a.set_fontsize(8)
ax8.set_title('Injury Type Distribution', fontsize=12, color='#58a6ff', pad=10)

# 9. Head-to-Head Win Matrix (Heatmap)
ax9 = fig.add_subplot(NROW, NCOL, 9)
teams_order = ['CSK', 'MI', 'RCB', 'KKR', 'RR', 'DC', 'PBKS', 'SRH']
h2h = pd.DataFrame(index=teams_order, columns=teams_order, dtype=float)
for t in teams_order:
    for o in teams_order:
        mask = (df['team'] == t) & (df['opponent'] == o)
        h2h.loc[t, o] = df.loc[mask, 'win'].mean() * 100 if mask.sum() > 0 else np.nan

sns.heatmap(h2h.astype(float), ax=ax9, cmap='RdYlGn', annot=True, fmt='.0f',
            linewidths=0.5, linecolor='#0d1117', cbar=False,
            annot_kws={'size': 7.5, 'weight': 'bold'})
ax9.set_title('Head-to-Head Win Rate (%)\n(Row team vs Column team)', fontsize=11, color='#58a6ff', pad=10)
ax9.set_xlabel('Opponent', fontsize=9)
ax9.set_ylabel('Team', fontsize=9)
ax9.tick_params(axis='both', labelsize=8)

# 10. Wins vs Losses per Team (Stacked Bar)
ax10 = fig.add_subplot(NROW, NCOL, 10)
tw = df.groupby(['team', 'result']).size().unstack(fill_value=0)
if 'Win' not in tw.columns:
    tw['Win'] = 0
if 'Loss' not in tw.columns:
    tw['Loss'] = 0
tw = tw[['Win', 'Loss']]
tw_sorted = tw.loc[tw.sum(axis=1).sort_values(ascending=False).index]
x10 = range(len(tw_sorted))
ax10.bar(x10, tw_sorted['Win'], label='Win', color='#3fb950', edgecolor='#0d1117')
ax10.bar(x10, tw_sorted['Loss'], bottom=tw_sorted['Win'], label='Loss',
         color='#f85149', edgecolor='#0d1117')
ax10.set_xticks(list(x10))
ax10.set_xticklabels(tw_sorted.index, fontsize=9)
ax10.legend(fontsize=9)
ax10.set_title('Wins & Losses per Team', fontsize=12, color='#58a6ff', pad=10)
ax10.set_ylabel('Number of Matches')

# 11. Correlation Heatmap of numeric features
ax11 = fig.add_subplot(NROW, NCOL, 11)
num_df = df[['recent_wins', 'is_home', 'has_injury', 'win']].copy()
corr = num_df.corr()
mask11 = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax11, cmap='coolwarm', annot=True, fmt='.2f',
            linewidths=0.5, linecolor='#0d1117', center=0, vmin=-1, vmax=1,
            annot_kws={'size': 10, 'weight': 'bold'})
ax11.set_title('Feature Correlation Heatmap', fontsize=12, color='#58a6ff', pad=10)
ax11.tick_params(axis='both', labelsize=8)

# 12. Venue Home Advantage
ax12 = fig.add_subplot(NROW, NCOL, 12)
venues = df[df['home'] != 'Away']['home'].value_counts().head(8)
bars12 = ax12.barh(venues.index, venues.values,
                   color=plt.cm.plasma(np.linspace(0.3, 0.9, len(venues))),
                   edgecolor='#30363d')
ax12.set_title('Top Venues (Home Match Count)', fontsize=12, color='#58a6ff', pad=10)
ax12.set_xlabel('Number of Matches')
for bar, val in zip(bars12, venues.values):
    ax12.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
              str(val), va='center', fontsize=9)

# ── Final Layout ─────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.97])
output_path = 'IPL_EDA_Report.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"\n✅ EDA report saved → {output_path}")
plt.show()

print("\n" + "=" * 65)
print("  EDA Complete! Run app.py to start the prediction server.")
print("=" * 65)
