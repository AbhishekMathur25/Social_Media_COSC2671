import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# LOAD DATA

print(" LOADING DATA")

comments_df  = pd.read_csv("comments_with_sentiment.csv")
videos_df    = pd.read_csv("videos_with_topics.csv")
ch_sentiment = pd.read_csv("channel_sentiment_summary.csv")

print(f"Comments:  {len(comments_df)}")
print(f"Videos:    {len(videos_df)}")
print(f"Channels:  {comments_df['channel_name'].nunique()}")
print(f"Unique commenters: {comments_df['commenter_id'].nunique()}")

# BUILD COMMENTER-CHANNEL MAPPING

print("\n BUILDING COMMENTER-CHANNEL MAPPING")

# Remove unknown commenter IDs
comments_clean = comments_df[
    comments_df['commenter_id'].notna() &
    (comments_df['commenter_id'] != 'unknown') &
    (comments_df['commenter_id'] != '')
].copy()

# Map each commenter to all channels they commented on
commenter_to_channels = defaultdict(set)
for _, row in comments_clean.iterrows():
    commenter_to_channels[row['commenter_id']].add(row['channel_name'])

# Only keep commenters who appear on 2+ channels
multi_channel_commenters = {
    cid: channels
    for cid, channels in commenter_to_channels.items()
    if len(channels) >= 2
}

print(f"Total unique commenters:          {len(commenter_to_channels)}")
print(f"Commenters on 2+ channels:        {len(multi_channel_commenters)}")
print(f"Percentage multi-channel:         {len(multi_channel_commenters)/len(commenter_to_channels)*100:.1f}%")

# Show distribution of how many channels commenters appear on
channel_count_dist = defaultdict(int)
for cid, channels in commenter_to_channels.items():
    channel_count_dist[len(channels)] += 1

print("\nCommenter channel spread:")
for n_channels in sorted(channel_count_dist.keys()):
    print(f"  On {n_channels} channel(s): {channel_count_dist[n_channels]} commenters")

# BUILD EDGE LIST

print("\n 3: BUILDING EDGE LIST")

# Count shared commenters between every pair of channels
edge_weights = defaultdict(int)

for cid, channels in multi_channel_commenters.items():
    channels_list = sorted(list(channels))
    for i in range(len(channels_list)):
        for j in range(i + 1, len(channels_list)):
            edge = (channels_list[i], channels_list[j])
            edge_weights[edge] += 1

print(f"Total channel pairs with shared commenters: {len(edge_weights)}")

# Show strongest connections
sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
print("\nTop 15 strongest channel connections (shared commenters):")
for (ch1, ch2), weight in sorted_edges[:15]:
    print(f"  {ch1:<35} <-> {ch2:<35}  {weight} shared")

# CONSTRUCT NETWORKX GRAPH

print("\n 4: CONSTRUCTING NETWORKX GRAPH")

G = nx.Graph()

# Add all channels as nodes (even ones with no overlap)
all_channels = comments_clean['channel_name'].unique().tolist()
for channel in all_channels:
    G.add_node(channel)

# Add weighted edges
MIN_SHARED_COMMENTERS = 1
for (ch1, ch2), weight in edge_weights.items():
    if weight >= MIN_SHARED_COMMENTERS:
        G.add_edge(ch1, ch2, weight=weight)

print(f"Graph nodes:         {G.number_of_nodes()}")
print(f"Graph edges:         {G.number_of_edges()}")
print(f"Graph density:       {nx.density(G):.4f}")
print(f"Connected:           {nx.is_connected(G)}")

# Number of connected components
components = list(nx.connected_components(G))
print(f"Connected components: {len(components)}")
print(f"Largest component:    {max(len(c) for c in components)} nodes")

# COMPUTE CENTRALITY MEASURES

print("\n 5: COMPUTING CENTRALITY MEASURES")

# Use largest connected component for centrality that needs it
largest_cc   = max(components, key=len)
G_main       = G.subgraph(largest_cc).copy()

print(f"Main component: {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges")

# Degree centrality — how many channels does this channel connect to?
degree_cent      = nx.degree_centrality(G)

# Weighted degree (strength) — total shared commenters
strength         = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True))
                    for node in G.nodes()}

# Betweenness centrality — how often is this channel a bridge?
betweenness_cent = nx.betweenness_centrality(G, weight='weight', normalized=True)

# Eigenvector centrality — is this channel connected to important channels?
try:
    eigenvector_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
except nx.PowerIterationFailedConvergence:
    eigenvector_cent = nx.eigenvector_centrality_numpy(G, weight='weight')

# Clustering coefficient — how tightly knit is this channel's neighbourhood?
clustering_coef  = nx.clustering(G, weight='weight')

# PageRank — influence propagation measure
pagerank         = nx.pagerank(G, weight='weight', alpha=0.85)

print("Centrality measures computed.")

# BUILD NODE ATTRIBUTES TABLE

print("\n BUILDING NODE ATTRIBUTES TABLE")

# Merge with channel metadata
channel_meta = videos_df.groupby('channel_name').agg(
    total_views    = ('view_count',       'sum'),
    avg_views      = ('view_count',       'mean'),
    video_count    = ('video_id',         'count'),
    dominant_topic = ('topic',            lambda x: x.mode()[0] if len(x) > 0 else -1),
    avg_sentiment  = ('title_compound',   'mean'),
).reset_index()

topic_labels_map = {
    0: 'ETF & Dividends',
    1: 'Stock Research',
    2: 'Beginner & Education',
    3: 'Global Markets',
    4: 'ASX Blue Chips'
}

channel_meta['topic_label'] = channel_meta['dominant_topic'].map(topic_labels_map)

node_rows = []
for node in G.nodes():
    meta_row = channel_meta[channel_meta['channel_name'] == node]

    node_rows.append({
        'channel_name':      node,
        'degree':            G.degree(node),
        'strength':          strength.get(node, 0),
        'degree_centrality': degree_cent.get(node, 0),
        'betweenness':       betweenness_cent.get(node, 0),
        'eigenvector':       eigenvector_cent.get(node, 0),
        'clustering':        clustering_coef.get(node, 0),
        'pagerank':          pagerank.get(node, 0),
        'total_views':       meta_row['total_views'].values[0]  if len(meta_row) > 0 else 0,
        'avg_views':         meta_row['avg_views'].values[0]    if len(meta_row) > 0 else 0,
        'video_count':       meta_row['video_count'].values[0]  if len(meta_row) > 0 else 0,
        'topic_label':       meta_row['topic_label'].values[0]  if len(meta_row) > 0 else 'Unknown',
        'avg_sentiment':     meta_row['avg_sentiment'].values[0] if len(meta_row) > 0 else 0,
    })

node_df = pd.DataFrame(node_rows).sort_values('degree_centrality', ascending=False)

# PRINT CENTRALITY RANKINGS

print("\n CENTRALITY RANKINGS")

print(f"\n--- DEGREE CENTRALITY (most connected channels) ---")
print(f"{'Channel':<45} {'Degree':>8} {'Strength':>10} {'Cent':>8}")
print("-" * 75)
for _, row in node_df.head(15).iterrows():
    print(f"  {row['channel_name']:<43} {int(row['degree']):>8} {int(row['strength']):>10} {row['degree_centrality']:>8.4f}")

print(f"\n--- BETWEENNESS CENTRALITY (bridge channels) ---")
bet_sorted = node_df.sort_values('betweenness', ascending=False)
print(f"{'Channel':<45} {'Betweenness':>12}")
print("-" * 60)
for _, row in bet_sorted.head(10).iterrows():
    print(f"  {row['channel_name']:<43} {row['betweenness']:>12.6f}")

print(f"\n--- EIGENVECTOR CENTRALITY (connected to important channels) ---")
eig_sorted = node_df.sort_values('eigenvector', ascending=False)
print(f"{'Channel':<45} {'Eigenvector':>12}")
print("-" * 60)
for _, row in eig_sorted.head(10).iterrows():
    print(f"  {row['channel_name']:<43} {row['eigenvector']:>12.6f}")

print(f"\n--- PAGERANK (influence propagation) ---")
pr_sorted = node_df.sort_values('pagerank', ascending=False)
print(f"{'Channel':<45} {'PageRank':>10}")
print("-" * 58)
for _, row in pr_sorted.head(10).iterrows():
    print(f"  {row['channel_name']:<43} {row['pagerank']:>10.6f}")

print(f"\n--- CLUSTERING COEFFICIENT (echo chamber tendency) ---")
cl_sorted = node_df[node_df['degree'] > 1].sort_values('clustering', ascending=False)
print(f"{'Channel':<45} {'Clustering':>10}")
print("-" * 58)
for _, row in cl_sorted.head(10).iterrows():
    print(f"  {row['channel_name']:<43} {row['clustering']:>10.4f}")

# NETWORK SUMMARY STATISTICS

print("\n 8: NETWORK SUMMARY STATISTICS")

degrees = [d for _, d in G.degree()]

print(f"\nBasic statistics:")
print(f"  Nodes:                     {G.number_of_nodes()}")
print(f"  Edges:                     {G.number_of_edges()}")
print(f"  Density:                   {nx.density(G):.4f}")
print(f"  Avg degree:                {np.mean(degrees):.2f}")
print(f"  Max degree:                {max(degrees)}")
print(f"  Avg clustering coeff:      {nx.average_clustering(G):.4f}")

if nx.is_connected(G):
    print(f"  Avg shortest path length:  {nx.average_shortest_path_length(G):.4f}")
    print(f"  Diameter:                  {nx.diameter(G)}")
else:
    print(f"  Avg shortest path (main):  {nx.average_shortest_path_length(G_main):.4f}")
    print(f"  Diameter (main):           {nx.diameter(G_main)}")

print(f"  Degree assortativity:      {nx.degree_assortativity_coefficient(G):.4f}")

# ADD NODE ATTRIBUTES TO GRAPH FOR EXPORT

print("\n ADDING NODE ATTRIBUTES TO GRAPH")

for _, row in node_df.iterrows():
    node = row['channel_name']
    if node in G.nodes():
        G.nodes[node]['degree_centrality'] = float(row['degree_centrality'])
        G.nodes[node]['betweenness']        = float(row['betweenness'])
        G.nodes[node]['eigenvector']        = float(row['eigenvector'])
        G.nodes[node]['pagerank']           = float(row['pagerank'])
        G.nodes[node]['clustering']         = float(row['clustering'])
        G.nodes[node]['total_views']        = float(row['total_views'])
        G.nodes[node]['topic_label']        = str(row['topic_label'])
        G.nodes[node]['avg_sentiment']      = float(row['avg_sentiment'])

nx.write_gexf(G, 'commenter_overlap_network.gexf')
print("Saved: commenter_overlap_network.gexf  (open in Gephi for visualisation)")

# PLOTS

print("\n GENERATING PLOTS")

# Plot 1: Network graph coloured by topic
topic_color_map = {
    'ETF & Dividends':      '#3498db',
    'Stock Research':       '#e74c3c',
    'Beginner & Education': '#2ecc71',
    'Global Markets':       '#f39c12',
    'ASX Blue Chips':       '#9b59b6',
    'Unknown':              '#95a5a6',
}

node_colors  = []
node_sizes   = []
node_labels  = {}

for node in G.nodes():
    row          = node_df[node_df['channel_name'] == node]
    topic        = row['topic_label'].values[0] if len(row) > 0 else 'Unknown'
    node_colors.append(topic_color_map.get(topic, '#95a5a6'))
    deg          = G.degree(node)
    node_sizes.append(max(100, deg * 120))
    # Only label nodes with degree > 2
    if deg > 2:
        node_labels[node] = node.split('|')[0].strip()[:20]

pos = nx.spring_layout(G, weight='weight', seed=42, k=2.5)

fig, ax = plt.subplots(figsize=(16, 12))

nx.draw_networkx_edges(
    G, pos, ax=ax,
    alpha=0.3,
    width=[G[u][v]['weight'] * 0.15 for u, v in G.edges()],
    edge_color='#aaaaaa'
)

nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.85
)

nx.draw_networkx_labels(
    G, pos, node_labels, ax=ax,
    font_size=6.5,
    font_weight='bold'
)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color, markersize=10, label=label)
    for label, color in topic_color_map.items()
    if label != 'Unknown'
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, title='Topic')
ax.set_title('Financial Influencer Network — Commenter Overlap (2024-2025)\n'
             'Node size = degree, Edge width = shared commenters, Colour = dominant topic',
             fontsize=12)
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/network_graph.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/network_graph.png")

# Plot 2: Centrality comparison — top 15 channels
top15       = node_df.head(15).copy()
metrics     = ['degree_centrality', 'betweenness', 'eigenvector', 'pagerank']
metric_labels = ['Degree', 'Betweenness', 'Eigenvector', 'PageRank']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes      = axes.flatten()

for ax, metric, label in zip(axes, metrics, metric_labels):
    sorted_df = node_df.sort_values(metric, ascending=False).head(12)
    colors    = [topic_color_map.get(t, '#95a5a6') for t in sorted_df['topic_label']]
    bars      = ax.barh(sorted_df['channel_name'], sorted_df[metric], color=colors, alpha=0.8)
    ax.set_xlabel(f'{label} Score')
    ax.set_title(f'{label} Centrality — Top 12 Channels')
    ax.tick_params(axis='y', labelsize=7)
    for bar, val in zip(bars, sorted_df[metric]):
        ax.text(val + max(sorted_df[metric]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=6)

plt.suptitle('Network Centrality Measures — Financial Influencer Channels', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/centrality_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/centrality_comparison.png")

# Plot 3: Degree distribution
degrees_list = [d for _, d in G.degree()]
fig, ax      = plt.subplots(figsize=(8, 4))
ax.hist(degrees_list, bins=range(0, max(degrees_list) + 2), color='steelblue',
        edgecolor='white', alpha=0.8)
ax.set_xlabel('Node Degree (number of channels connected to)')
ax.set_ylabel('Frequency')
ax.set_title('Degree Distribution of Influencer Network')
avg_deg = np.mean(degrees_list)
ax.axvline(x=avg_deg, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean degree = {avg_deg:.1f}')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/degree_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/degree_distribution.png")

# Plot 4: Centrality vs Views scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(
    node_df['degree_centrality'],
    np.log1p(node_df['total_views']),
    c=[topic_color_map.get(t, '#95a5a6') for t in node_df['topic_label']],
    alpha=0.7, s=60
)
axes[0].set_xlabel('Degree Centrality')
axes[0].set_ylabel('log(Total Views + 1)')
axes[0].set_title('Degree Centrality vs Channel Popularity')

for _, row in node_df[node_df['degree_centrality'] > 0.15].iterrows():
    axes[0].annotate(
        row['channel_name'][:18],
        (row['degree_centrality'], np.log1p(row['total_views'])),
        fontsize=6, xytext=(4, 2), textcoords='offset points'
    )

axes[1].scatter(
    node_df['betweenness'],
    np.log1p(node_df['total_views']),
    c=[topic_color_map.get(t, '#95a5a6') for t in node_df['topic_label']],
    alpha=0.7, s=60
)
axes[1].set_xlabel('Betweenness Centrality')
axes[1].set_ylabel('log(Total Views + 1)')
axes[1].set_title('Betweenness Centrality vs Channel Popularity')

for _, row in node_df[node_df['betweenness'] > 0.05].iterrows():
    axes[1].annotate(
        row['channel_name'][:18],
        (row['betweenness'], np.log1p(row['total_views'])),
        fontsize=6, xytext=(4, 2), textcoords='offset points'
    )

plt.suptitle('Network Centrality vs Channel Popularity')
plt.tight_layout()
plt.savefig('outputs/centrality_vs_views.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/centrality_vs_views.png")

# SAVE OUTPUTS

print("\n SAVING OUTPUT FILES")

node_df.to_csv('node_attributes.csv', index=False)

network_stats = {
    'nodes':                    G.number_of_nodes(),
    'edges':                    G.number_of_edges(),
    'density':                  nx.density(G),
    'avg_degree':               np.mean(degrees),
    'avg_clustering':           nx.average_clustering(G),
    'connected_components':     len(components),
    'largest_component_size':   max(len(c) for c in components),
    'degree_assortativity':     nx.degree_assortativity_coefficient(G),
    'multi_channel_commenters': len(multi_channel_commenters),
}
pd.DataFrame([network_stats]).to_csv('network_stats.csv', index=False)

print("Saved: node_attributes.csv")
print("Saved: network_stats.csv")

# FINAL SUMMARY

print("\nNETWORK ANALYSIS SUMMARY")

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Density: {nx.density(G):.4f}  ({'sparse' if nx.density(G) < 0.3 else 'dense'} network)")
print(f"Avg clustering: {nx.average_clustering(G):.4f}")

top_degree   = node_df.iloc[0]
top_between  = node_df.sort_values('betweenness', ascending=False).iloc[0]
top_eigen    = node_df.sort_values('eigenvector',  ascending=False).iloc[0]

print(f"\nMost connected channel (degree):     {top_degree['channel_name']}  ({top_degree['degree']} connections)")
print(f"Most bridging channel (betweenness): {top_between['channel_name']}  ({top_between['betweenness']:.4f})")
print(f"Most influential (eigenvector):      {top_eigen['channel_name']}  ({top_eigen['eigenvector']:.4f})")

print(f"\nInterpretation:")
print(f"  High degree = channel whose audience overlaps most with other channels")
print(f"  High betweenness = channel that bridges otherwise disconnected communities")
print(f"  High eigenvector = channel connected to other highly-connected channels")
