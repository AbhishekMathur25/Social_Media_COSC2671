import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords',     quiet=True)
nltk.download('wordnet',       quiet=True)
nltk.download('punkt',         quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# LOAD DATA

print(" LOADING DATA")

videos_df   = pd.read_csv("videos_with_sentiment.csv")
comments_df = pd.read_csv("comments_with_sentiment.csv")

videos_df['published_at'] = pd.to_datetime(videos_df['published_at'], utc=True, errors='coerce')

print(f"Videos:   {len(videos_df)}")
print(f"Comments: {len(comments_df)}")

#TEXT PREPROCESSING

print("\n  TEXT PREPROCESSING")

lemmatizer  = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

# Domain-specific stopwords — common finance words that appear everywhere
# and would dominate topics without adding meaning
finance_stopwords = {
    'stock', 'stocks', 'market', 'markets', 'invest', 'investing',
    'investment', 'investor', 'investors', 'share', 'shares',
    'finance', 'financial', 'money', 'asx', 'australia', 'australian',
    'buy', 'sell', 'get', 'like', 'know', 'think', 'good', 'great',
    'really', 'going', 'make', 'one', 'also', 'much', 'want',
    'would', 'could', 'year', 'time', 'new', 'video', 'watch',
    'thanks', 'thank', 'channel', 'please', 'people', 'us',
    'just', 'even', 'back', 'still', 'well', 'way', 'need',
    'come', 'lot', 'take', 'right', 'look', 'put', 'say',
    'said', 'use', 'used', 'using', 'got', 'go', 'give',
    'see', 'long', 'first', 'ever', 'never', 'always', 'every',
    'best', 'top', 'big', 'small', 'high', 'low', 'next',
    'last', 'many', 'much', 'less', 'more', 'most', 'bit',
    'thing', 'things', 'something', 'nothing', 'everything',
    'today', 'week', 'month', 'years', 'day', 'days',
    'really', 'actually', 'basically', 'literally',
    'show', 'know', 'love', 'comment', 'video', 'watch',
    'subscribe', 'like', 'share', 'follow'
}

all_stopwords = stop_words.union(finance_stopwords)

def preprocess_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    text  = text.lower()
    text  = re.sub(r'[^a-z\s]', ' ', text)
    text  = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in all_stopwords and len(t) > 2]
    return ' '.join(tokens)


# Build corpus from titles + descriptions (no transcripts available)
# We combine both to get more text per document
print("  Preprocessing video titles and descriptions...")

videos_df['text_combined'] = (
    videos_df['title_clean'].fillna('') + ' ' +
    videos_df['title_clean'].fillna('') + ' ' +   # title repeated — gives it more weight
    videos_df['description_clean'].fillna('')
)
videos_df['text_processed'] = videos_df['text_combined'].apply(preprocess_text)

# Remove empty docs
videos_valid = videos_df[videos_df['text_processed'].str.len() > 10].copy()
print(f"  Valid documents for LDA: {len(videos_valid)}")

# Also preprocess comments for comment-level topic modelling
print("  Preprocessing comments...")
comments_df['text_processed'] = comments_df['comment_text_clean'].apply(preprocess_text)
comments_valid = comments_df[comments_df['text_processed'].str.len() > 10].copy()
print(f"  Valid comments for LDA: {len(comments_valid)}")

# BUILD DOCUMENT-TERM MATRIX

print("\n BUILDING DOCUMENT-TERM MATRIX")

# Use CountVectorizer for LDA (LDA works on raw counts not TF-IDF)
vectorizer = CountVectorizer(
    max_features  = 800,
    min_df        = 2,      # word must appear in at least 2 docs
    max_df        = 0.90,   # ignore words in more than 90% of docs
    ngram_range   = (1, 2), # unigrams and bigrams
)

dtm_videos   = vectorizer.fit_transform(videos_valid['text_processed'])
feature_names = vectorizer.get_feature_names_out()

print(f"  Vocabulary size:    {len(feature_names)}")
print(f"  DTM shape:          {dtm_videos.shape}")

# FIND OPTIMAL NUMBER OF TOPICS

print("\n FINDING OPTIMAL NUMBER OF TOPICS")
print("  Testing n_topics from 2 to 12...")

topic_range    = range(2, 13)
log_likelihoods = []
perplexities    = []

for n in topic_range:
    lda_test = LatentDirichletAllocation(
        n_components      = n,
        random_state      = 42,
        max_iter          = 20,
        learning_method   = 'online',
    )
    lda_test.fit(dtm_videos)
    log_likelihoods.append(lda_test.score(dtm_videos))
    perplexities.append(lda_test.perplexity(dtm_videos))
    print(f"  n={n:2d}  log-likelihood={lda_test.score(dtm_videos):>12.2f}  perplexity={lda_test.perplexity(dtm_videos):>8.2f}")

# Plot perplexity and log-likelihood
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(topic_range), log_likelihoods, marker='o', color='steelblue', linewidth=1.5)
ax1.set_xlabel('Number of Topics')
ax1.set_ylabel('Log-Likelihood Score')
ax1.set_title('Log-Likelihood vs Number of Topics')
ax1.set_xticks(list(topic_range))

ax2.plot(list(topic_range), perplexities, marker='o', color='darkorange', linewidth=1.5)
ax2.set_xlabel('Number of Topics')
ax2.set_ylabel('Perplexity Score')
ax2.set_title('Perplexity vs Number of Topics')
ax2.set_xticks(list(topic_range))

plt.suptitle('LDA Topic Model Evaluation')
plt.tight_layout()
plt.savefig('outputs/topic_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/topic_evaluation.png")

# Select optimal: lowest perplexity in reasonable range (2-8)
best_idx    = perplexities.index(min(perplexities[:7]))
optimal_n   = list(topic_range)[best_idx]
print(f"\n  Optimal number of topics (by perplexity): {optimal_n}")
print("  Note: We also evaluate 5 topics for interpretability")

# TRAIN FINAL LDA MODELS

print("\n TRAINING FINAL LDA MODELS")

# Train with optimal n
lda_optimal = LatentDirichletAllocation(
    n_components    = optimal_n,
    random_state    = 42,
    max_iter        = 50,
    learning_method = 'online',
)
lda_optimal.fit(dtm_videos)

# Also train with 5 topics for interpretability comparison
lda_5 = LatentDirichletAllocation(
    n_components    = 5,
    random_state    = 42,
    max_iter        = 50,
    learning_method = 'online',
)
lda_5.fit(dtm_videos)

print(f"  Optimal model ({optimal_n} topics) — log-likelihood: {lda_optimal.score(dtm_videos):.2f}")
print(f"  5-topic model            — log-likelihood: {lda_5.score(dtm_videos):.2f}")

# INTERPRET TOPICS

print("\n INTERPRETING TOPICS")

def get_top_words(model, feature_names, n_words=12):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_words-1:-1]
        top_words   = [feature_names[i] for i in top_indices]
        topics.append(top_words)
    return topics

# Use 5-topic model for final interpretation (more interpretable)
FINAL_N_TOPICS = 5
lda_final      = lda_5
topic_words    = get_top_words(lda_final, feature_names, n_words=12)

# Topic labels — assign manually based on top words
topic_labels = {
    0: "Topic 1",
    1: "Topic 2",
    2: "Topic 3",
    3: "Topic 4",
    4: "Topic 5",
}

print(f"\nTop words per topic (5-topic model):")
for i, words in enumerate(topic_words):
    print(f"  Topic {i+1}: {', '.join(words)}")

print("\nInterpretation guide:")
print("  (Assign labels based on the words above)")

# ASSIGN TOPICS TO VIDEOS

print("\n ASSIGNING TOPICS TO VIDEOS")

doc_topic_matrix           = lda_final.transform(dtm_videos)
videos_valid['topic']      = doc_topic_matrix.argmax(axis=1)
videos_valid['topic_prob'] = doc_topic_matrix.max(axis=1)

# Store all topic probabilities
for i in range(FINAL_N_TOPICS):
    videos_valid[f'topic_{i}_prob'] = doc_topic_matrix[:, i]

print("Topic distribution across videos:")
topic_counts = videos_valid['topic'].value_counts().sort_index()
for topic_id, count in topic_counts.items():
    pct = count / len(videos_valid) * 100
    print(f"  Topic {topic_id+1}: {count:3d} videos ({pct:.1f}%)  |  Top words: {', '.join(topic_words[topic_id][:5])}")

# TOPIC VS ASX PERFORMANCE

print("\n 8: TOPIC VS ASX PERFORMANCE")

videos_valid['asx_return_T1'] = pd.to_numeric(videos_valid['asx_return_T1'], errors='coerce')
videos_valid['asx_return_T3'] = pd.to_numeric(videos_valid['asx_return_T3'], errors='coerce')

print("\nAverage ASX T+1 return by dominant topic:")
topic_asx = videos_valid.groupby('topic').agg(
    count          = ('video_id',        'count'),
    avg_t1_return  = ('asx_return_T1',   'mean'),
    avg_t3_return  = ('asx_return_T3',   'mean'),
    avg_views      = ('view_count',      'mean'),
    avg_sentiment  = ('title_compound',  'mean'),
).reset_index()

for _, row in topic_asx.iterrows():
    print(f"  Topic {int(row['topic'])+1}: n={int(row['count']):3d}  "
          f"T+1={row['avg_t1_return']*100:.3f}%  "
          f"T+3={row['avg_t3_return']*100:.3f}%  "
          f"avg_views={row['avg_views']:,.0f}  "
          f"sentiment={row['avg_sentiment']:.3f}")

# TOPIC DISTRIBUTION PER CHANNEL

print("\n TOPIC DISTRIBUTION PER CHANNEL")

channel_topic = videos_valid.groupby(['channel_name','topic']).size().unstack(fill_value=0)
channel_topic.columns = [f'Topic_{i+1}' for i in channel_topic.columns]
channel_topic['dominant_topic'] = channel_topic.idxmax(axis=1)

print("\nDominant topic per channel:")
for channel, row in channel_topic.iterrows():
    print(f"  {channel:<45} {row['dominant_topic']}")

# PLOTS

print("\n GENERATING PLOTS")

# Plot 1: Top words per topic — bar charts
fig, axes = plt.subplots(1, FINAL_N_TOPICS, figsize=(18, 5))

colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']

for i, (ax, words) in enumerate(zip(axes, topic_words)):
    topic_comp  = lda_final.components_[i]
    word_scores = [(w, topic_comp[list(feature_names).index(w)])
                   for w in words if w in feature_names]
    word_scores.sort(key=lambda x: x[1])

    words_plot  = [w[0] for w in word_scores]
    scores_plot = [w[1] for w in word_scores]

    ax.barh(words_plot, scores_plot, color=colors[i], alpha=0.8)
    ax.set_title(f'Topic {i+1}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Word Weight')
    ax.tick_params(axis='y', labelsize=8)

plt.suptitle('Top Words per Topic — Financial Influencer Videos (2024-2025)', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/topic_words.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/topic_words.png")

# Plot 2: Topic distribution across videos
fig, ax = plt.subplots(figsize=(8, 4))
topic_labels_short = [f'Topic {i+1}' for i in range(FINAL_N_TOPICS)]
ax.bar(topic_labels_short, topic_counts.values, color=colors)
ax.set_xlabel('Topic')
ax.set_ylabel('Number of Videos')
ax.set_title('Video Distribution Across Topics')
for i, val in enumerate(topic_counts.values):
    ax.text(i, val + 0.5, str(val), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/topic_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/topic_distribution.png")

# Plot 3: Topic heatmap — channel vs topic
if len(channel_topic) > 3:
    topic_cols = [c for c in channel_topic.columns if c.startswith('Topic_')]
    heat_data  = channel_topic[topic_cols].copy()

    fig, ax = plt.subplots(figsize=(10, max(6, len(heat_data) * 0.35)))
    im = ax.imshow(heat_data.values, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(len(topic_cols)))
    ax.set_xticklabels(topic_cols, fontsize=9)
    ax.set_yticks(range(len(heat_data)))
    ax.set_yticklabels(heat_data.index, fontsize=7)

    for i in range(len(heat_data)):
        for j in range(len(topic_cols)):
            ax.text(j, i, str(heat_data.values[i, j]),
                    ha='center', va='center', fontsize=6, color='black')

    plt.colorbar(im, ax=ax, label='Number of Videos')
    ax.set_title('Topic Distribution per Channel')
    plt.tight_layout()
    plt.savefig('outputs/topic_channel_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/topic_channel_heatmap.png")

# Plot 4: ASX return by topic
fig, ax = plt.subplots(figsize=(9, 4))
topic_labels_plot = [f'Topic {int(r["topic"])+1}' for _, r in topic_asx.iterrows()]
returns_plot      = [r['avg_t1_return'] * 100 for _, r in topic_asx.iterrows()]
bar_colors        = ['#2ecc71' if v >= 0 else '#e74c3c' for v in returns_plot]

bars = ax.bar(topic_labels_plot, returns_plot, color=bar_colors, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Dominant Topic')
ax.set_ylabel('Avg ASX200 Return Next Day (%)')
ax.set_title('Average Next-Day ASX Return by Video Topic')
for bar, val in zip(bars, returns_plot):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.002 if val >= 0 else val - 0.004,
        f'{val:.3f}%',
        ha='center', va='bottom' if val >= 0 else 'top',
        fontsize=9
    )
plt.tight_layout()
plt.savefig('outputs/topic_asx_returns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/topic_asx_returns.png")

# SAVE OUTPUTS

print("\n SAVING OUTPUT FILES")

videos_valid.to_csv('videos_with_topics.csv', index=False)
channel_topic.reset_index().to_csv('channel_topic_distribution.csv', index=False)
topic_asx.to_csv('topic_asx_summary.csv', index=False)

topic_summary_rows = []
for i, words in enumerate(topic_words):
    topic_summary_rows.append({
        'topic_id':   i,
        'top_words':  ', '.join(words),
        'video_count': topic_counts.get(i, 0),
    })
topic_summary_df = pd.DataFrame(topic_summary_rows)
topic_summary_df.to_csv('topic_summary.csv', index=False)

print("Saved videos_with_topics.csv")
print("Saved channel_topic_distribution.csv")
print("Saved topic_asx_summary.csv")
print("Saved topic_summary.csv")

# FINAL SUMMARY

print("\nTOPIC MODELLING SUMMARY")

print(f"\nModel: LDA with {FINAL_N_TOPICS} topics (sklearn)")
print(f"Corpus: {len(videos_valid)} video documents")
print(f"Vocabulary: {len(feature_names)} terms")
print(f"Log-likelihood: {lda_final.score(dtm_videos):.2f}")
print(f"Perplexity:     {lda_final.perplexity(dtm_videos):.2f}")

print(f"\nTopics found:")
for i, words in enumerate(topic_words):
    count = topic_counts.get(i, 0)
    print(f"  Topic {i+1} ({count} videos): {', '.join(words[:8])}")

print(f"\nASX insight:")
best_topic  = topic_asx.loc[topic_asx['avg_t1_return'].idxmax()]
worst_topic = topic_asx.loc[topic_asx['avg_t1_return'].idxmin()]
print(f"  Topic associated with highest ASX return: Topic {int(best_topic['topic'])+1} ({best_topic['avg_t1_return']*100:.3f}%)")
print(f"  Topic associated with lowest  ASX return: Topic {int(worst_topic['topic'])+1} ({worst_topic['avg_t1_return']*100:.3f}%)")
