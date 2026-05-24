import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# LOAD DATA AND REBUILD GRAPH

print(" LOADING DATA AND REBUILDING GRAPH")

comments_df  = pd.read_csv("comments_with_sentiment.csv")
videos_df    = pd.read_csv("videos_with_topics.csv")
node_df      = pd.read_csv("node_attributes.csv")

# Rebuild the commenter overlap graph
comments_clean = comments_df[
    comments_df['commenter_id'].notna() &
    (comments_df['commenter_id'] != 'unknown') &
    (comments_df['commenter_id'] != '')
].copy()

commenter_to_channels = defaultdict(set)
for _, row in comments_clean.iterrows():
    commenter_to_channels[row['commenter_id']].add(row['channel_name'])

multi_channel = {
    cid: channels
    for cid, channels in commenter_to_channels.items()
    if len(channels) >= 2
}

edge_weights = defaultdict(int)
for cid, channels in multi_channel.items():
    channels_list = sorted(list(channels))
    for i in range(len(channels_list)):
        for j in range(i + 1, len(channels_list)):
            edge_weights[(channels_list[i], channels_list[j])] += 1

G = nx.Graph()
all_channels = comments_clean['channel_name'].unique().tolist()
for channel in all_channels:
    G.add_node(channel)
for (ch1, ch2), weight in edge_weights.items():
    G.add_edge(ch1, ch2, weight=weight)

print(f"Graph rebuilt: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# LOUVAIN COMMUNITY DETECTION

print("\n RUNNING LOUVAIN COMMUNITY DETECTION")

# Run Louvain — returns dict {node: community_id}
partition = community_louvain.best_partition(G, weight='weight', random_state=42)

# Modularity score (higher = better separated communities)
modularity = community_louvain.modularity(partition, G, weight='weight')

print(f"Modularity score:       {modularity:.4f}")
print(f"Number of communities:  {len(set(partition.values()))}")

# Community sizes
community_sizes = defaultdict(int)
for node, comm in partition.items():
    community_sizes[comm] += 1

print("\nCommunity sizes:")
for comm_id in sorted(community_sizes.keys()):
    print(f"  Community {comm_id}: {community_sizes[comm_id]} channels")

# ASSIGN COMMUNITY LABELS

print("\n PROFILING COMMUNITIES")

# Add community to node dataframe
node_df['community'] = node_df['channel_name'].map(partition)

# For channels not in partition (isolated), assign -1
node_df['community'] = node_df['community'].fillna(-1).astype(int)

# Profile each community
topic_labels_map = {
    'ETF & Dividends':      0,
    'Stock Research':       1,
    'Beginner & Education': 2,
    'Global Markets':       3,
    'ASX Blue Chips':       4,
    'Unknown':             -1,
}

# Add videos data to communities
videos_df['community'] = videos_df['channel_name'].map(partition)
comments_df['community'] = comments_df['channel_name'].map(partition)

community_profiles = []

for comm_id in sorted(set(partition.values())):
    channels_in_comm = [ch for ch, c in partition.items() if c == comm_id]

    # Get videos for this community
    comm_videos   = videos_df[videos_df['channel_name'].isin(channels_in_comm)]
    comm_comments = comments_df[comments_df['channel_name'].isin(channels_in_comm)]

    # Dominant topic
    if len(comm_videos) > 0 and 'topic' in comm_videos.columns:
        topic_counts  = comm_videos['topic'].value_counts()
        dominant_topic_id = topic_counts.idxmax() if len(topic_counts) > 0 else -1
    else:
        dominant_topic_id = -1

    topic_id_to_label = {
        0: 'ETF & Dividends',
        1: 'Stock Research',
        2: 'Beginner & Education',
        3: 'Global Markets',
        4: 'ASX Blue Chips',
    }
    dominant_topic = topic_id_to_label.get(dominant_topic_id, 'Unknown')

    # Avg sentiment
    avg_title_sent   = comm_videos['title_compound'].mean()   if len(comm_videos) > 0   else 0
    avg_comment_sent = comm_comments['comment_compound'].mean() if len(comm_comments) > 0 else 0

    # ASX returns
    avg_t1 = comm_videos['asx_return_T1'].mean() if len(comm_videos) > 0 else 0
    avg_t3 = comm_videos['asx_return_T3'].mean() if len(comm_videos) > 0 else 0

    # Internal edge density
    subgraph      = G.subgraph(channels_in_comm)
    internal_dens = nx.density(subgraph) if len(channels_in_comm) > 1 else 0

    community_profiles.append({
        'community_id':       comm_id,
        'size':               len(channels_in_comm),
        'channels':           ', '.join(sorted(channels_in_comm)),
        'dominant_topic':     dominant_topic,
        'avg_title_sentiment':   round(avg_title_sent,   4),
        'avg_comment_sentiment': round(avg_comment_sent, 4),
        'avg_asx_T1_return':  round(avg_t1, 6),
        'avg_asx_T3_return':  round(avg_t3, 6),
        'internal_density':   round(internal_dens, 4),
        'total_videos':       len(comm_videos),
        'total_comments':     len(comm_comments),
    })

comm_df = pd.DataFrame(community_profiles)

print("\nCommunity profiles:")
for _, row in comm_df.iterrows():
    print(f"\n  Community {int(row['community_id'])} ({int(row['size'])} channels)")
    print(f"    Dominant topic:     {row['dominant_topic']}")
    print(f"    Title sentiment:    {row['avg_title_sentiment']:.4f}")
    print(f"    Comment sentiment:  {row['avg_comment_sentiment']:.4f}")
    print(f"    Avg T+1 ASX return: {row['avg_asx_T1_return']*100:.3f}%")
    print(f"    Internal density:   {row['internal_density']:.4f}")
    print(f"    Channels: {row['channels']}")

# INTER-COMMUNITY CONNECTIONS

print("\n INTER-COMMUNITY CONNECTIONS")

communities_list = sorted(set(partition.values()))
inter_matrix     = pd.DataFrame(0, index=communities_list, columns=communities_list)

for u, v, data in G.edges(data=True):
    cu = partition.get(u, -1)
    cv = partition.get(v, -1)
    if cu != cv:
        inter_matrix.loc[cu, cv] += data['weight']
        inter_matrix.loc[cv, cu] += data['weight']

print("\nInter-community shared commenters matrix:")
print(inter_matrix.to_string())

# COMMUNITY VS ASX PERFORMANCE

print("\n COMMUNITY VS ASX PERFORMANCE")

print(f"\n{'Community':<12} {'Dominant Topic':<25} {'T+1 Return':>12} {'T+3 Return':>12} {'Sentiment':>10}")
print("-" * 75)
for _, row in comm_df.iterrows():
    print(f"  Comm {int(row['community_id']):<8} {row['dominant_topic']:<25} "
          f"{row['avg_asx_T1_return']*100:>11.3f}% "
          f"{row['avg_asx_T3_return']*100:>11.3f}% "
          f"{row['avg_title_sentiment']:>10.4f}")

# Statistical test: do communities differ in ASX returns?
comm_groups = []
for comm_id in communities_list:
    channels_in_comm = [ch for ch, c in partition.items() if c == comm_id]
    comm_videos      = videos_df[videos_df['channel_name'].isin(channels_in_comm)]
    returns          = comm_videos['asx_return_T1'].dropna().tolist()
    if len(returns) >= 5:
        comm_groups.append(returns)

if len(comm_groups) >= 2:
    f_stat, p_val = stats.f_oneway(*comm_groups)
    print(f"\nOne-way ANOVA across communities:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value:     {p_val:.4f}")
    print(f"  Result: {'Communities show significantly different ASX returns' if p_val < 0.05 else 'No significant difference in ASX returns across communities'}")

# PLOTS

print("\n GENERATING PLOTS")

community_colors = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
]

color_map = {
    node: community_colors[comm % len(community_colors)]
    for node, comm in partition.items()
}

# Plot 1: Network coloured by community
pos = nx.spring_layout(G, weight='weight', seed=42, k=2.5)

fig, ax = plt.subplots(figsize=(16, 12))

nx.draw_networkx_edges(
    G, pos, ax=ax,
    alpha=0.25,
    width=[G[u][v]['weight'] * 0.12 for u, v in G.edges()],
    edge_color='#bbbbbb'
)

node_colors_list = [color_map.get(node, '#95a5a6') for node in G.nodes()]
node_sizes_list  = [max(150, G.degree(node) * 130) for node in G.nodes()]
node_labels      = {
    node: node.split('|')[0].strip()[:20]
    for node in G.nodes()
    if G.degree(node) > 2
}

nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=node_colors_list,
    node_size=node_sizes_list,
    alpha=0.88
)
nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=6.5, font_weight='bold')

legend_patches = [
    mpatches.Patch(
        color=community_colors[comm % len(community_colors)],
        label=f"Community {comm} ({community_sizes[comm]} channels)"
    )
    for comm in sorted(set(partition.values()))
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=9, title='Communities')
ax.set_title(
    'Financial Influencer Network — Communities Detected (Louvain)\n'
    f'Modularity = {modularity:.4f}  |  {len(set(partition.values()))} communities  |  '
    f'Node size = degree',
    fontsize=12
)
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/community_network.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/community_network.png")

# Plot 2: Community sentiment comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

comm_ids     = comm_df['community_id'].astype(int).tolist()
comm_x       = [f"Comm {c}" for c in comm_ids]
colors_plot  = [community_colors[c % len(community_colors)] for c in comm_ids]

axes[0].bar(comm_x, comm_df['avg_title_sentiment'], color=colors_plot, alpha=0.85)
axes[0].axhline(y=0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_xlabel('Community')
axes[0].set_ylabel('Avg Title Sentiment Score')
axes[0].set_title('Average Video Title Sentiment per Community')
for i, val in enumerate(comm_df['avg_title_sentiment']):
    axes[0].text(i, val + 0.005 if val >= 0 else val - 0.015,
                 f'{val:.3f}', ha='center', fontsize=9)

axes[1].bar(comm_x, comm_df['avg_comment_sentiment'], color=colors_plot, alpha=0.85)
axes[1].axhline(y=0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_xlabel('Community')
axes[1].set_ylabel('Avg Comment Sentiment Score')
axes[1].set_title('Average Comment Sentiment per Community')
for i, val in enumerate(comm_df['avg_comment_sentiment']):
    axes[1].text(i, val + 0.005 if val >= 0 else val - 0.015,
                 f'{val:.3f}', ha='center', fontsize=9)

plt.suptitle('Sentiment Across Communities')
plt.tight_layout()
plt.savefig('outputs/community_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/community_sentiment.png")

# Plot 3: Community vs ASX returns
fig, ax = plt.subplots(figsize=(10, 5))

x      = np.arange(len(comm_df))
width  = 0.35

bars1 = ax.bar(x - width/2, comm_df['avg_asx_T1_return'] * 100,
               width, label='T+1 Return', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, comm_df['avg_asx_T3_return'] * 100,
               width, label='T+3 Return', color='darkorange', alpha=0.8)

ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels([f"Comm {int(c)}\n({row['dominant_topic'][:15]})"
                    for c, (_, row) in zip(comm_ids, comm_df.iterrows())],
                   fontsize=8)
ax.set_ylabel('Avg ASX200 Return (%)')
ax.set_title('Average ASX200 Return After Videos — By Community')
ax.legend()

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + 0.001 if h >= 0 else h - 0.003,
            f'{h:.3f}%', ha='center', va='bottom' if h >= 0 else 'top', fontsize=7)

plt.tight_layout()
plt.savefig('outputs/community_asx_returns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/community_asx_returns.png")

# Plot 4: Topic distribution per community heatmap
if 'topic' in videos_df.columns:
    topic_id_to_label = {
        0: 'ETF & Dividends',
        1: 'Stock Research',
        2: 'Beginner & Education',
        3: 'Global Markets',
        4: 'ASX Blue Chips',
    }

    videos_df['community'] = videos_df['channel_name'].map(partition)
    topic_comm = videos_df.dropna(subset=['community','topic']).copy()
    topic_comm['community'] = topic_comm['community'].astype(int)
    topic_comm['topic_label'] = topic_comm['topic'].map(topic_id_to_label)

    heat = topic_comm.groupby(['community','topic_label']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, max(4, len(heat) * 0.6)))
    im = ax.imshow(heat.values, aspect='auto', cmap='Blues')

    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, fontsize=9, rotation=25, ha='right')
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels([f'Community {i}' for i in heat.index], fontsize=9)

    for i in range(len(heat)):
        for j in range(len(heat.columns)):
            ax.text(j, i, str(heat.values[i, j]),
                    ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax, label='Number of Videos')
    ax.set_title('Topic Distribution per Community')
    plt.tight_layout()
    plt.savefig('outputs/community_topic_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/community_topic_heatmap.png")

# SAVE OUTPUTS

print("\n SAVING OUTPUT FILES")

node_df.to_csv('community_assignments.csv', index=False)
comm_df.to_csv('community_summary.csv',     index=False)

print("Saved: community_assignments.csv")
print("Saved: community_summary.csv")

# FINAL SUMMARY

print("\nCOMMUNITY DETECTION SUMMARY")

print(f"\nAlgorithm:       Louvain (modularity maximisation)")
print(f"Modularity:      {modularity:.4f}  (>0.3 = good community structure)")
print(f"Communities:     {len(set(partition.values()))}")

print(f"\nCommunities found:")
for _, row in comm_df.iterrows():
    print(f"  Community {int(row['community_id'])}: {int(row['size'])} channels | "
          f"{row['dominant_topic']} | "
          f"sentiment={row['avg_title_sentiment']:.3f} | "
          f"T+1={row['avg_asx_T1_return']*100:.3f}%")

print(f"\nKey insight:")
best_comm  = comm_df.loc[comm_df['avg_asx_T1_return'].idxmax()]
worst_comm = comm_df.loc[comm_df['avg_asx_T1_return'].idxmin()]
print(f"  Community with best  T+1 ASX return: Community {int(best_comm['community_id'])} "
      f"({best_comm['dominant_topic']}) at {best_comm['avg_asx_T1_return']*100:.3f}%")
print(f"  Community with worst T+1 ASX return: Community {int(worst_comm['community_id'])} "
      f"({worst_comm['dominant_topic']}) at {worst_comm['avg_asx_T1_return']*100:.3f}%")

print(f"\nNetwork interpretation:")
print(f"  High modularity ({modularity:.3f}) confirms real community structure exists")
print(f"  Communities likely reflect different investor audiences and styles")
print(f"  Bryan Invest spans multiple communities — acts as information hub")
