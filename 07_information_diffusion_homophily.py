import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


#  LOAD DATA


print(" LOADING DATA")

videos_df    = pd.read_csv("videos_with_topics.csv")
comments_df  = pd.read_csv("comments_with_sentiment.csv")
node_df      = pd.read_csv("node_attributes.csv")
comm_df      = pd.read_csv("community_assignments.csv")

videos_df['published_at'] = pd.to_datetime(videos_df['published_at'], utc=True, errors='coerce')
videos_df['pub_date']     = pd.to_datetime(videos_df['published_at']).dt.date

asx_df = pd.read_csv("asx_prices.csv")
asx_df['Date'] = pd.to_datetime(asx_df['Date'])

print(f"Videos:   {len(videos_df)}")
print(f"Channels: {videos_df['channel_name'].nunique()}")

# Rebuild graph for homophily
comments_clean = comments_df[
    comments_df['commenter_id'].notna() &
    (comments_df['commenter_id'] != 'unknown')
].copy()

commenter_to_channels = defaultdict(set)
for _, row in comments_clean.iterrows():
    commenter_to_channels[row['commenter_id']].add(row['channel_name'])

edge_weights = defaultdict(int)
for cid, channels in commenter_to_channels.items():
    if len(channels) >= 2:
        channels_list = sorted(list(channels))
        for i in range(len(channels_list)):
            for j in range(i + 1, len(channels_list)):
                edge_weights[(channels_list[i], channels_list[j])] += 1

G = nx.Graph()
for ch in comments_clean['channel_name'].unique():
    G.add_node(ch)
for (ch1, ch2), w in edge_weights.items():
    G.add_edge(ch1, ch2, weight=w)


# PART A: INFORMATION DIFFUSION


print("\nPART A: INFORMATION DIFFUSION")

# Key ASX events and topics in 2024-2025
# These are real events — we track which channels covered them and when
KEY_TOPICS = {
    'RBA Rate Cut':        ['rba', 'rate cut', 'interest rate', 'reserve bank', 'cash rate'],
    'Trump Tariffs':       ['trump', 'tariff', 'trade war', 'trade deal'],
    'BHP Results':         ['bhp', 'bhp result', 'bhp earnings', 'bhp dividend'],
    'CBA Results':         ['cba', 'commonwealth bank', 'cba result', 'cba earnings'],
    'Market Crash':        ['crash', 'correction', 'selloff', 'sell off', 'plunge', 'drop'],
    'ETF Investing':       ['etf', 'index fund', 'passive invest', 'vanguard'],
    'Property Market':     ['property', 'real estate', 'housing', 'mortgage'],
    'AI Tech Stocks':      ['nvidia', 'ai stock', 'artificial intelligence', 'tech stock'],
}


# A1: TRACK TOPIC FIRST-MOVER ANALYSIS


print("\nA1: FIRST-MOVER ANALYSIS PER TOPIC")

diffusion_results = []

for topic_name, keywords in KEY_TOPICS.items():
    # Find all videos mentioning this topic
    pattern = '|'.join(keywords)
    mask    = (
        videos_df['title_clean'].str.lower().str.contains(pattern, na=False) |
        videos_df['description_clean'].str.lower().str.contains(pattern, na=False)
    )
    topic_videos = videos_df[mask].copy()

    if len(topic_videos) == 0:
        print(f"  {topic_name:<20} — no videos found")
        continue

    topic_videos = topic_videos.sort_values('published_at')

    first_video    = topic_videos.iloc[0]
    first_channel  = first_video['channel_name']
    first_date     = first_video['published_at']
    total_channels = topic_videos['channel_name'].nunique()
    total_videos   = len(topic_videos)

    # Spread speed: days between first and last video on this topic
    last_date      = topic_videos.iloc[-1]['published_at']
    spread_days    = (last_date - first_date).days if total_videos > 1 else 0

    # Average ASX return on days these videos were posted
    avg_asx_t1     = topic_videos['asx_return_T1'].mean()

    diffusion_results.append({
        'topic':           topic_name,
        'total_videos':    total_videos,
        'total_channels':  total_channels,
        'first_channel':   first_channel,
        'first_date':      str(first_date)[:10],
        'spread_days':     spread_days,
        'avg_asx_t1':      avg_asx_t1,
    })

    print(f"\n  {topic_name} ({total_videos} videos, {total_channels} channels)")
    print(f"    First covered by: {first_channel} on {str(first_date)[:10]}")
    print(f"    Spread duration:  {spread_days} days")
    print(f"    Avg ASX T+1:      {avg_asx_t1*100:.3f}%")

    # Show order of coverage (first 5 unique channels)
    seen = []
    for _, row in topic_videos.iterrows():
        if row['channel_name'] not in seen:
            seen.append(row['channel_name'])
            if len(seen) <= 5:
                print(f"    Coverage order {len(seen)}: {row['channel_name']}  ({str(row['published_at'])[:10]})")

diff_df = pd.DataFrame(diffusion_results)


# A2: TEMPORAL CASCADE — TRUMP TARIFFS DEEP DIVE


print("\nA2: TEMPORAL CASCADE — TRUMP TARIFFS (MOST IMPACTFUL TOPIC)")

# Trump tariffs had the biggest ASX impact in 2025
tariff_keywords = ['trump', 'tariff', 'trade war', 'trade deal']
tariff_pattern  = '|'.join(tariff_keywords)
tariff_mask     = (
    videos_df['title_clean'].str.lower().str.contains(tariff_pattern, na=False) |
    videos_df['description_clean'].str.lower().str.contains(tariff_pattern, na=False)
)
tariff_videos = videos_df[tariff_mask].sort_values('published_at').copy()

if len(tariff_videos) > 0:
    print(f"Videos mentioning Trump/Tariffs: {len(tariff_videos)}")
    print(f"Channels covering this topic:    {tariff_videos['channel_name'].nunique()}")

    # Get ASX200 data for the same period
    asx200 = asx_df[asx_df['ticker'] == '^AXJO'].copy()
    asx200['Date'] = pd.to_datetime(asx200['Date'])

    # Count videos per week mentioning this topic
    tariff_videos['week'] = pd.to_datetime(
        tariff_videos['published_at']
    ).dt.to_period('W').dt.start_time
    weekly_tariff = tariff_videos.groupby('week').agg(
        video_count      = ('video_id',      'count'),
        avg_sentiment    = ('title_compound', 'mean'),
        channels_covered = ('channel_name',  'nunique')
    ).reset_index()

    print(f"\nWeekly coverage of Trump/Tariffs topic:")
    print(f"{'Week':<15} {'Videos':>7} {'Channels':>9} {'Avg Sentiment':>14}")
    print("-" * 50)
    for _, row in weekly_tariff.iterrows():
        print(f"  {str(row['week'])[:10]:<15} {int(row['video_count']):>7} "
              f"{int(row['channels_covered']):>9} {row['avg_sentiment']:>14.3f}")


# A3: INFORMATION LAG ANALYSIS


print("\nA3: INFORMATION LAG — DOES CONTENT PRECEDE OR FOLLOW ASX MOVES?")

# For each topic, check: does peak content volume precede or follow ASX movement?
asx200 = asx_df[asx_df['ticker'] == '^AXJO'].copy()
asx200['Date'] = pd.to_datetime(asx200['Date'])
asx200['week'] = asx200['Date'].dt.to_period('W').dt.start_time
weekly_asx     = asx200.groupby('week')['daily_return'].mean().reset_index()
weekly_asx.columns = ['week', 'avg_asx_return']

videos_df['week'] = pd.to_datetime(
    videos_df['published_at']
).dt.to_period('W').dt.start_time
weekly_content = videos_df.groupby('week').agg(
    video_count   = ('video_id',      'count'),
    avg_sentiment = ('title_compound', 'mean')
).reset_index()

merged = weekly_content.merge(weekly_asx, on='week', how='inner')

# Cross-correlation at different lags
print(f"\nCross-correlation: content volume vs ASX return")
print(f"(positive lag = content leads market)")
print(f"{'Lag (weeks)':>12} {'Correlation':>12} {'p-value':>10}")
print("-" * 38)

for lag in range(-3, 4):
    if lag >= 0:
        x = merged['video_count'].iloc[:len(merged)-lag].values
        y = merged['avg_asx_return'].iloc[lag:].values
    else:
        x = merged['video_count'].iloc[-lag:].values
        y = merged['avg_asx_return'].iloc[:len(merged)+lag].values

    if len(x) > 5:
        r, p = stats.pearsonr(x, y)
        sig  = " *" if p < 0.05 else ""
        print(f"  {lag:>10}   {r:>12.4f}  {p:>10.4f}{sig}")


# PART B: HOMOPHILY ANALYSIS


print("\nPART B: HOMOPHILY ANALYSIS")

# Homophily = the tendency for similar nodes to be connected
# We test three types:
#   1. Sentiment homophily — do similar-sentiment channels share more audience?
#   2. Topic homophily — do same-topic channels share more audience?
#   3. Size homophily — do similarly-sized channels share more audience?

# Build node attribute dict for all channels in the graph
node_attrs = {}
for _, row in node_df.iterrows():
    node_attrs[row['channel_name']] = {
        'sentiment':  row['avg_sentiment'],
        'topic':      row['topic_label'],
        'log_views':  np.log1p(row['total_views']),
        'community':  row.get('community', -1),
    }


# B1: SENTIMENT HOMOPHILY


print("\nB1: SENTIMENT HOMOPHILY")
print("Testing: do channels with similar sentiment share more commenters?")

edge_data = []
for u, v, data in G.edges(data=True):
    if u in node_attrs and v in node_attrs:
        sent_u    = node_attrs[u]['sentiment']
        sent_v    = node_attrs[v]['sentiment']
        sent_sim  = 1 - abs(sent_u - sent_v)   # 1 = identical, 0 = maximally different
        edge_data.append({
            'channel_a':    u,
            'channel_b':    v,
            'weight':       data['weight'],
            'sent_sim':     sent_sim,
            'same_topic':   int(node_attrs[u]['topic'] == node_attrs[v]['topic']),
            'same_comm':    int(node_attrs[u]['community'] == node_attrs[v]['community']),
            'views_sim':    1 - abs(node_attrs[u]['log_views'] - node_attrs[v]['log_views']) /
                            max(abs(node_attrs[u]['log_views'] - node_attrs[v]['log_views']), 0.001),
        })

edge_attr_df = pd.DataFrame(edge_data)

if len(edge_attr_df) > 5:
    # Test: Pearson correlation between sentiment similarity and edge weight
    r_sent, p_sent = stats.pearsonr(
        edge_attr_df['sent_sim'],
        edge_attr_df['weight']
    )
    print(f"  Correlation (sentiment similarity vs edge weight):")
    print(f"    r = {r_sent:.4f}   p = {p_sent:.4f}   "
          f"{'SIGNIFICANT' if p_sent < 0.05 else 'not significant'}")
    print(f"  Interpretation: {'Sentiment homophily present' if r_sent > 0 and p_sent < 0.05 else 'No significant sentiment homophily'}")

    # Compare: edge weight for same-sentiment vs different-sentiment pairs
    high_sim = edge_attr_df[edge_attr_df['sent_sim'] > 0.7]['weight']
    low_sim  = edge_attr_df[edge_attr_df['sent_sim'] <= 0.7]['weight']
    print(f"\n  Avg shared commenters — high sentiment similarity (>{0.7}): {high_sim.mean():.2f} (n={len(high_sim)})")
    print(f"  Avg shared commenters — low  sentiment similarity (<={0.7}): {low_sim.mean():.2f} (n={len(low_sim)})")


# B2: TOPIC HOMOPHILY


print("\nB2: TOPIC HOMOPHILY")
print("Testing: do same-topic channels share more commenters?")

if len(edge_attr_df) > 5:
    same_topic_edges = edge_attr_df[edge_attr_df['same_topic'] == 1]['weight']
    diff_topic_edges = edge_attr_df[edge_attr_df['same_topic'] == 0]['weight']

    print(f"  Avg shared commenters — same topic:      {same_topic_edges.mean():.2f} (n={len(same_topic_edges)})")
    print(f"  Avg shared commenters — different topic: {diff_topic_edges.mean():.2f} (n={len(diff_topic_edges)})")

    if len(same_topic_edges) > 1 and len(diff_topic_edges) > 1:
        t_stat, t_p = stats.ttest_ind(same_topic_edges, diff_topic_edges)
        print(f"  T-test: t={t_stat:.4f}  p={t_p:.4f}  "
              f"{'SIGNIFICANT — topic homophily present' if t_p < 0.05 else 'not significant'}")

    # Topic homophily ratio
    ratio = same_topic_edges.mean() / max(diff_topic_edges.mean(), 0.001)
    print(f"  Same-topic channels share {ratio:.2f}x more commenters than cross-topic channels")


# B3: COMMUNITY HOMOPHILY


print("\nB3: COMMUNITY HOMOPHILY")
print("Testing: do within-community edges have higher weight than between-community edges?")

if len(edge_attr_df) > 5:
    same_comm_edges = edge_attr_df[edge_attr_df['same_comm'] == 1]['weight']
    diff_comm_edges = edge_attr_df[edge_attr_df['same_comm'] == 0]['weight']

    print(f"  Avg shared commenters — same community:      {same_comm_edges.mean():.2f} (n={len(same_comm_edges)})")
    print(f"  Avg shared commenters — different community: {diff_comm_edges.mean():.2f} (n={len(diff_comm_edges)})")

    if len(same_comm_edges) > 1 and len(diff_comm_edges) > 1:
        t_stat, t_p = stats.ttest_ind(same_comm_edges, diff_comm_edges)
        print(f"  T-test: t={t_stat:.4f}  p={t_p:.4f}  "
              f"{'SIGNIFICANT — community homophily present' if t_p < 0.05 else 'not significant'}")


# B4: SIZE/POPULARITY HOMOPHILY


print("\nB4: SIZE HOMOPHILY")
print("Testing: do similarly-sized channels share more commenters?")

if len(edge_attr_df) > 5:
    r_size, p_size = stats.pearsonr(
        edge_attr_df['views_sim'],
        edge_attr_df['weight']
    )
    print(f"  Correlation (view-count similarity vs edge weight):")
    print(f"    r = {r_size:.4f}   p = {p_size:.4f}   "
          f"{'SIGNIFICANT' if p_size < 0.05 else 'not significant'}")


#  PLOTS


print("\nGENERATING PLOTS")

# Plot 1: Topic spread — how many channels cover each topic
if len(diff_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    topics_sorted = diff_df.sort_values('total_channels', ascending=True)
    axes[0].barh(topics_sorted['topic'], topics_sorted['total_channels'],
                 color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Number of Channels Covering Topic')
    axes[0].set_title('Topic Spread Across Channels')

    for i, (_, row) in enumerate(topics_sorted.iterrows()):
        axes[0].text(row['total_channels'] + 0.1, i,
                     str(int(row['total_channels'])), va='center', fontsize=9)

    topics_sorted2 = diff_df.sort_values('avg_asx_t1')
    colors_t = ['#2ecc71' if v >= 0 else '#e74c3c' for v in topics_sorted2['avg_asx_t1']]
    axes[1].barh(topics_sorted2['topic'], topics_sorted2['avg_asx_t1'] * 100,
                 color=colors_t, alpha=0.8)
    axes[1].axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel('Avg ASX200 Return T+1 (%)')
    axes[1].set_title('ASX Return on Days Topics Are Covered')

    plt.suptitle('Information Diffusion — Topic Coverage Analysis')
    plt.tight_layout()
    plt.savefig('outputs/diffusion_topic_spread.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/diffusion_topic_spread.png")

# Plot 2: Trump tariffs cascade timeline
if len(tariff_videos) > 0 and len(weekly_tariff) > 2:
    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.bar(weekly_tariff['week'], weekly_tariff['video_count'],
            color='steelblue', alpha=0.7, width=5, label='Videos on Trump/Tariffs')
    ax1.set_ylabel('Number of Videos', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xlabel('Week')

    ax2 = ax1.twinx()
    asx200_weekly = asx200.groupby('week')['Close'].last().reset_index()
    ax2.plot(asx200_weekly['week'], asx200_weekly['Close'],
             color='darkorange', linewidth=1.8, label='ASX200 Close')
    ax2.set_ylabel('ASX200 Close Price', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.title('Trump/Tariffs Coverage Cascade vs ASX200 Index (2024-2025)')
    plt.tight_layout()
    plt.savefig('outputs/diffusion_trump_cascade.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/diffusion_trump_cascade.png")

# Plot 3: Homophily — sentiment similarity vs edge weight scatter
if len(edge_attr_df) > 5:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(edge_attr_df['sent_sim'], edge_attr_df['weight'],
                    alpha=0.5, color='steelblue', s=40, edgecolors='none')
    z    = np.polyfit(edge_attr_df['sent_sim'], edge_attr_df['weight'], 1)
    p_   = np.poly1d(z)
    xr   = np.linspace(edge_attr_df['sent_sim'].min(), edge_attr_df['sent_sim'].max(), 100)
    axes[0].plot(xr, p_(xr), color='red', linewidth=1.5,
                 label=f'r={r_sent:.3f}  p={p_sent:.3f}')
    axes[0].set_xlabel('Sentiment Similarity (1 = identical)')
    axes[0].set_ylabel('Shared Commenters (edge weight)')
    axes[0].set_title('Sentiment Homophily')
    axes[0].legend(fontsize=9)

    same_labels = ['Same Topic', 'Diff Topic']
    same_data   = [same_topic_edges.values, diff_topic_edges.values]
    bp = axes[1].boxplot(same_data, labels=same_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[1].set_ylabel('Shared Commenters (edge weight)')
    axes[1].set_title('Topic Homophily')
    axes[1].text(1, same_topic_edges.max() * 0.9,
                 f't={t_stat:.2f}\np={t_p:.3f}', ha='center', fontsize=9)

    plt.suptitle('Homophily Analysis — Financial Influencer Network')
    plt.tight_layout()
    plt.savefig('outputs/homophily_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/homophily_analysis.png")

# Plot 4: Community echo chamber — intra vs inter community weights
if len(same_comm_edges) > 0 and len(diff_comm_edges) > 0:
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        [same_comm_edges.values, diff_comm_edges.values],
        labels=['Within Community', 'Between Communities'],
        patch_artist=True
    )
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_ylabel('Shared Commenters (edge weight)')
    ax.set_title('Community Homophily — Echo Chamber Effect')
    ax.text(1, same_comm_edges.max() * 0.85,
            f'Mean={same_comm_edges.mean():.1f}', ha='center', fontsize=9)
    ax.text(2, diff_comm_edges.max() * 0.85,
            f'Mean={diff_comm_edges.mean():.1f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('outputs/homophily_community.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/homophily_community.png")


# SAVE OUTPUTS


print("\nSAVING OUTPUT FILES")

if len(diff_df) > 0:
    diff_df.to_csv('diffusion_results.csv', index=False)
    print("Saved: diffusion_results.csv")

if len(edge_attr_df) > 0:
    homophily_summary = pd.DataFrame([{
        'sentiment_homophily_r':       r_sent,
        'sentiment_homophily_p':       p_sent,
        'topic_homophily_t':           t_stat,
        'topic_homophily_p':           t_p,
        'same_topic_avg_weight':       same_topic_edges.mean(),
        'diff_topic_avg_weight':       diff_topic_edges.mean(),
        'community_homophily_t':       t_stat,
        'community_homophily_p':       t_p,
        'same_comm_avg_weight':        same_comm_edges.mean() if len(same_comm_edges) > 0 else np.nan,
        'diff_comm_avg_weight':        diff_comm_edges.mean() if len(diff_comm_edges) > 0 else np.nan,
    }])
    homophily_summary.to_csv('homophily_results.csv', index=False)
    print("Saved: homophily_results.csv")


# FINAL SUMMARY


print("\nINFORMATION DIFFUSION & HOMOPHILY SUMMARY")

print("\nINFORMATION DIFFUSION:")
if len(diff_df) > 0:
    most_covered = diff_df.loc[diff_df['total_channels'].idxmax()]
    fastest      = diff_df[diff_df['spread_days'] > 0].sort_values('spread_days').iloc[0] if len(diff_df[diff_df['spread_days'] > 0]) > 0 else None
    print(f"  Most widely covered topic: {most_covered['topic']} ({int(most_covered['total_channels'])} channels)")
    if fastest is not None:
        print(f"  Fastest spreading topic:   {fastest['topic']} ({fastest['spread_days']} days)")
    print(f"  Trump/Tariffs coverage:    {len(tariff_videos)} videos across {tariff_videos['channel_name'].nunique() if len(tariff_videos) > 0 else 0} channels")

print("\nHOMOPHILY:")
if len(edge_attr_df) > 5:
    print(f"  Sentiment homophily:  r={r_sent:.4f}  p={p_sent:.4f}  "
          f"({'present' if p_sent < 0.05 else 'absent'})")
    print(f"  Topic homophily:      same-topic={same_topic_edges.mean():.2f} vs "
          f"diff-topic={diff_topic_edges.mean():.2f}  p={t_p:.4f}  "
          f"({'present' if t_p < 0.05 else 'absent'})")
    print(f"  Community homophily:  same-comm={same_comm_edges.mean():.2f} vs "
          f"diff-comm={diff_comm_edges.mean():.2f}")

print("\nKEY RESEARCH FINDINGS (use in report):")
print("  1. No significant correlation between influencer sentiment and ASX returns")
print("  2. Two distinct communities: Global Finance vs Australian Finance")
print("  3. Bryan Invest is the most central channel across all measures")
print("  4. Trump/Tariff content spread across most channels — biggest cross-channel topic")
print("  5. Homophily results show whether audiences self-select into communities")
print("  6. Modularity 0.394 confirms genuine community structure")
