import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# LOAD DATA

print("LOADING DATA")

videos_df   = pd.read_csv("videos_filtered.csv")
comments_df = pd.read_csv("comments_filtered.csv")
asx_df      = pd.read_csv("asx_prices.csv")

videos_df['published_at'] = pd.to_datetime(videos_df['published_at'], utc=True, errors='coerce')
videos_df['pub_date']     = pd.to_datetime(videos_df['published_at']).dt.date

asx_df['Date'] = pd.to_datetime(asx_df['Date'])

print(f"Videos loaded:   {len(videos_df)}")
print(f"Comments loaded: {len(comments_df)}")
print(f"Channels:        {videos_df['channel_name'].nunique()}")

# APPLY VADER SENTIMENT

print("\nAPPLYING VADER SENTIMENT")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0, 0.0, 0.0, 0.0
    scores = analyzer.polarity_scores(text)
    return scores['compound'], scores['pos'], scores['neg'], scores['neu']

def label_sentiment(compound):
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Sentiment on video titles
print("  Running VADER on video titles...")

title_scores = videos_df['title_clean'].apply(get_sentiment_scores)
videos_df['title_compound'] = [s[0] for s in title_scores]
videos_df['title_pos']      = [s[1] for s in title_scores]
videos_df['title_neg']      = [s[2] for s in title_scores]
videos_df['title_neu']      = [s[3] for s in title_scores]
videos_df['title_label']    = videos_df['title_compound'].apply(label_sentiment)

# Sentiment on video descriptions
print("  Running VADER on video descriptions...")

desc_scores = videos_df['description_clean'].apply(get_sentiment_scores)
videos_df['desc_compound'] = [s[0] for s in desc_scores]
videos_df['desc_label']    = videos_df['desc_compound'].apply(label_sentiment)

# Combined video sentiment (title weighted more heavily)
videos_df['video_sentiment'] = (
    videos_df['title_compound'] * 0.7 +
    videos_df['desc_compound']  * 0.3
)
videos_df['video_label'] = videos_df['video_sentiment'].apply(label_sentiment)

print(f"  Video sentiment done — {len(videos_df)} videos")
print(f"  Title sentiment distribution:")
print(f"    Positive: {(videos_df['title_label']=='positive').sum()}")
print(f"    Neutral:  {(videos_df['title_label']=='neutral').sum()}")
print(f"    Negative: {(videos_df['title_label']=='negative').sum()}")

# Sentiment on comments
print("  Running VADER on comments...")

comment_scores = comments_df['comment_text_clean'].apply(get_sentiment_scores)
comments_df['comment_compound'] = [s[0] for s in comment_scores]
comments_df['comment_pos']      = [s[1] for s in comment_scores]
comments_df['comment_neg']      = [s[2] for s in comment_scores]
comments_df['comment_label']    = comments_df['comment_compound'].apply(label_sentiment)

print(f"  Comment sentiment done — {len(comments_df)} comments")
print(f"  Comment sentiment distribution:")
print(f"    Positive: {(comments_df['comment_label']=='positive').sum()}")
print(f"    Neutral:  {(comments_df['comment_label']=='neutral').sum()}")
print(f"    Negative: {(comments_df['comment_label']=='negative').sum()}")

# CHANNEL-LEVEL SENTIMENT SUMMARY

print("\n 3: CHANNEL SENTIMENT SUMMARY")

# Aggregate comment sentiment per channel
comment_channel_sentiment = comments_df.groupby('channel_name').agg(
    avg_comment_sentiment = ('comment_compound', 'mean'),
    std_comment_sentiment = ('comment_compound', 'std'),
    total_comments        = ('comment_compound', 'count'),
    pct_positive          = ('comment_label',    lambda x: (x == 'positive').mean() * 100),
    pct_negative          = ('comment_label',    lambda x: (x == 'negative').mean() * 100),
    pct_neutral           = ('comment_label',    lambda x: (x == 'neutral').mean()  * 100),
).reset_index()

# Aggregate title sentiment per channel
title_channel_sentiment = videos_df.groupby('channel_name').agg(
    avg_title_sentiment = ('title_compound',  'mean'),
    total_videos        = ('video_id',        'count'),
    avg_views           = ('view_count',      'mean'),
    total_views         = ('view_count',      'sum'),
).reset_index()

# Merge
channel_summary = title_channel_sentiment.merge(
    comment_channel_sentiment, on='channel_name', how='left'
)

# Sort by avg comment sentiment
channel_summary = channel_summary.sort_values(
    'avg_comment_sentiment', ascending=False
).reset_index(drop=True)

print("\nChannel Sentiment Rankings (by comment sentiment):")
print(f"{'Channel':<45} {'Title Sent':>10} {'Comment Sent':>13} {'% Pos':>7} {'% Neg':>7}")
print("-" * 85)
for _, row in channel_summary.iterrows():
    title_s   = f"{row['avg_title_sentiment']:.3f}"   if pd.notna(row['avg_title_sentiment'])   else "N/A"
    comment_s = f"{row['avg_comment_sentiment']:.3f}" if pd.notna(row['avg_comment_sentiment']) else "N/A"
    pct_pos   = f"{row['pct_positive']:.1f}%"         if pd.notna(row['pct_positive'])          else "N/A"
    pct_neg   = f"{row['pct_negative']:.1f}%"         if pd.notna(row['pct_negative'])          else "N/A"
    print(f"  {row['channel_name']:<43} {title_s:>10} {comment_s:>13} {pct_pos:>7} {pct_neg:>7}")

# SENTIMENT VS ASX CORRELATION

print("\n 4: SENTIMENT VS ASX CORRELATION")

videos_df['asx_return_T1'] = pd.to_numeric(videos_df['asx_return_T1'], errors='coerce')
videos_df['asx_return_T3'] = pd.to_numeric(videos_df['asx_return_T3'], errors='coerce')

# Filter to rows where both sentiment and ASX return exist
corr_df = videos_df[
    videos_df['title_compound'].notna() &
    videos_df['asx_return_T1'].notna()
].copy()

# Correlation: title sentiment vs next-day ASX return
r_t1, p_t1 = stats.pearsonr(
    corr_df['title_compound'],
    corr_df['asx_return_T1']
)

# Correlation: title sentiment vs T+3 ASX return
corr_df_t3 = corr_df[corr_df['asx_return_T3'].notna()]
r_t3, p_t3 = stats.pearsonr(
    corr_df_t3['title_compound'],
    corr_df_t3['asx_return_T3']
)

# Correlation: video combined sentiment vs T+1
r_vid, p_vid = stats.pearsonr(
    corr_df['video_sentiment'],
    corr_df['asx_return_T1']
)

print(f"\nCorrelation Results (n={len(corr_df)} videos):")
print(f"  Title sentiment vs ASX T+1:   r = {r_t1:.4f}  p = {p_t1:.4f}  {'SIGNIFICANT' if p_t1 < 0.05 else 'not significant'}")
print(f"  Title sentiment vs ASX T+3:   r = {r_t3:.4f}  p = {p_t3:.4f}  {'SIGNIFICANT' if p_t3 < 0.05 else 'not significant'}")
print(f"  Video sentiment vs ASX T+1:   r = {r_vid:.4f}  p = {p_vid:.4f}  {'SIGNIFICANT' if p_vid < 0.05 else 'not significant'}")

# Compare: do bullish videos precede positive returns?
bullish_videos = corr_df[corr_df['title_label'] == 'positive']['asx_return_T1']
bearish_videos = corr_df[corr_df['title_label'] == 'negative']['asx_return_T1']
neutral_videos = corr_df[corr_df['title_label'] == 'neutral']['asx_return_T1']

print(f"\nAvg ASX T+1 return after:")
print(f"  Bullish videos (n={len(bullish_videos)}): {bullish_videos.mean()*100:.3f}%")
print(f"  Neutral videos (n={len(neutral_videos)}): {neutral_videos.mean()*100:.3f}%")
print(f"  Bearish videos (n={len(bearish_videos)}): {bearish_videos.mean()*100:.3f}%")

# T-test: bullish vs bearish
if len(bullish_videos) > 1 and len(bearish_videos) > 1:
    t_stat, t_pval = stats.ttest_ind(bullish_videos, bearish_videos)
    print(f"\n  T-test bullish vs bearish: t={t_stat:.3f}  p={t_pval:.4f}  {'SIGNIFICANT' if t_pval < 0.05 else 'not significant'}")

# SENTIMENT OVER TIME

print("\n 5: SENTIMENT OVER TIME")

# Weekly average sentiment
videos_df['week'] = pd.to_datetime(videos_df['published_at']).dt.to_period('W').dt.start_time

weekly_sentiment = videos_df.groupby('week').agg(
    avg_title_sentiment = ('title_compound',  'mean'),
    video_count         = ('video_id',        'count')
).reset_index()

# ASX200 weekly
asx200 = asx_df[asx_df['ticker'] == '^AXJO'].copy()
asx200['Date'] = pd.to_datetime(asx200['Date'])
asx200['week'] = asx200['Date'].dt.to_period('W').dt.start_time
weekly_asx = asx200.groupby('week').agg(
    avg_asx_return = ('daily_return', 'mean'),
    asx_close      = ('Close',        'last')
).reset_index()

weekly_merged = weekly_sentiment.merge(weekly_asx, on='week', how='inner')

print(f"  Weekly data points: {len(weekly_merged)}")

if len(weekly_merged) > 3:
    r_weekly, p_weekly = stats.pearsonr(
        weekly_merged['avg_title_sentiment'],
        weekly_merged['avg_asx_return']
    )
    print(f"  Weekly sentiment vs ASX return: r = {r_weekly:.4f}  p = {p_weekly:.4f}")

# SENTIMENT DISTRIBUTION PER CHANNEL

print("\n 6: GENERATING PLOTS")

# Plot 1: Comment sentiment by channel
top_channels = channel_summary.dropna(subset=['avg_comment_sentiment']).nlargest(
    15, 'total_comments'
)

fig, ax = plt.subplots(figsize=(12, 6))
colors  = [
    '#2ecc71' if v >= 0.05 else '#e74c3c' if v <= -0.05 else '#95a5a6'
    for v in top_channels['avg_comment_sentiment']
]
bars = ax.barh(
    top_channels['channel_name'],
    top_channels['avg_comment_sentiment'],
    color=colors
)
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Average Comment Sentiment (VADER Compound Score)')
ax.set_title('Average Comment Sentiment by Channel')
ax.set_xlim(-0.5, 0.8)
for bar, val in zip(bars, top_channels['avg_comment_sentiment']):
    ax.text(
        val + 0.01 if val >= 0 else val - 0.01,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.3f}',
        va='center',
        ha='left' if val >= 0 else 'right',
        fontsize=8
    )
plt.tight_layout()
plt.savefig('outputs/sentiment_by_channel.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/sentiment_by_channel.png")

# Plot 2: Sentiment over time vs ASX
if len(weekly_merged) > 3:
    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(
        weekly_merged['week'],
        weekly_merged['avg_title_sentiment'],
        color='steelblue',
        linewidth=1.5,
        marker='o',
        markersize=4,
        label='Avg Title Sentiment'
    )
    ax1.axhline(y=0, color='steelblue', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.set_ylabel('Avg Title Sentiment Score', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xlabel('Week')

    ax2 = ax1.twinx()
    ax2.plot(
        weekly_merged['week'],
        weekly_merged['asx_close'],
        color='darkorange',
        linewidth=1.5,
        label='ASX200 Close'
    )
    ax2.set_ylabel('ASX200 Close Price', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.title('Weekly Influencer Sentiment vs ASX200 Index (2024-2025)')
    plt.tight_layout()
    plt.savefig('outputs/sentiment_vs_asx_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: outputs/sentiment_vs_asx_timeline.png")

# Plot 3: Scatter — video sentiment vs T+1 ASX return
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    corr_df['title_compound'],
    corr_df['asx_return_T1'] * 100,
    alpha=0.4,
    color='steelblue',
    edgecolors='none',
    s=30
)

# Trend line
z    = np.polyfit(corr_df['title_compound'], corr_df['asx_return_T1'] * 100, 1)
p    = np.poly1d(z)
xline = np.linspace(corr_df['title_compound'].min(), corr_df['title_compound'].max(), 100)
ax.plot(xline, p(xline), color='red', linewidth=1.5, label=f'Trend  r={r_t1:.3f}  p={p_t1:.3f}')

ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
ax.axvline(x=0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xlabel('Video Title Sentiment (VADER Compound)')
ax.set_ylabel('ASX200 Return Next Day (%)')
ax.set_title('Video Title Sentiment vs Next-Day ASX Return')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/sentiment_vs_asx_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/sentiment_vs_asx_scatter.png")

# Plot 4: Overall sentiment distribution
label_counts = videos_df['title_label'].value_counts()
fig, axes    = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(
    label_counts.index,
    label_counts.values,
    color=['#2ecc71','#95a5a6','#e74c3c']
)
axes[0].set_title('Video Title Sentiment Distribution')
axes[0].set_ylabel('Number of Videos')
for i, (label, val) in enumerate(label_counts.items()):
    axes[0].text(i, val + 0.5, str(val), ha='center', fontsize=10)

comment_label_counts = comments_df['comment_label'].value_counts()
axes[1].bar(
    comment_label_counts.index,
    comment_label_counts.values,
    color=['#2ecc71','#95a5a6','#e74c3c']
)
axes[1].set_title('Comment Sentiment Distribution')
axes[1].set_ylabel('Number of Comments')
for i, (label, val) in enumerate(comment_label_counts.items()):
    axes[1].text(i, val + 5, str(val), ha='center', fontsize=10)

plt.suptitle('Sentiment Distribution — Financial Influencers (2024-2025)')
plt.tight_layout()
plt.savefig('outputs/sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/sentiment_distribution.png")

# SAVE OUTPUTS

print("\nSAVE OUTPUT FILES")

videos_df.to_csv('videos_with_sentiment.csv',     index=False)
comments_df.to_csv('comments_with_sentiment.csv', index=False)
channel_summary.to_csv('channel_sentiment_summary.csv', index=False)
weekly_merged.to_csv('weekly_sentiment_asx.csv',  index=False)

print(f"Saved videos_with_sentiment.csv")
print(f"Saved comments_with_sentiment.csv")
print(f"Saved channel_sentiment_summary.csv")
print(f"Saved weekly_sentiment_asx.csv")

# FINAL SUMMARY

print("\nSENTIMENT ANALYSIS SUMMARY")

print(f"\nVIDEO TITLES:")
print(f"  Avg sentiment:       {videos_df['title_compound'].mean():.4f}")
print(f"  Std deviation:       {videos_df['title_compound'].std():.4f}")
print(f"  Most positive title: {videos_df.loc[videos_df['title_compound'].idxmax(), 'title']}")
print(f"  Most negative title: {videos_df.loc[videos_df['title_compound'].idxmin(), 'title']}")

print(f"\nCOMMENTS:")
print(f"  Avg sentiment:       {comments_df['comment_compound'].mean():.4f}")
print(f"  Std deviation:       {comments_df['comment_compound'].std():.4f}")

print(f"\nASX CORRELATION:")
print(f"  Title sentiment vs T+1 return: r={r_t1:.4f}  p={p_t1:.4f}")
print(f"  Title sentiment vs T+3 return: r={r_t3:.4f}  p={p_t3:.4f}")
print(f"  Interpretation: {'Weak' if abs(r_t1) < 0.2 else 'Moderate' if abs(r_t1) < 0.4 else 'Strong'} {'positive' if r_t1 > 0 else 'negative'} relationship")

print(f"\nMOST BULLISH CHANNEL:  {channel_summary.iloc[0]['channel_name']}  ({channel_summary.iloc[0]['avg_comment_sentiment']:.3f})")
print(f"MOST BEARISH CHANNEL:  {channel_summary.iloc[-1]['channel_name']}  ({channel_summary.iloc[-1]['avg_comment_sentiment']:.3f})")
