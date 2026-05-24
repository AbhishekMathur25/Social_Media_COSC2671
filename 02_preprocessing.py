
# FILE 02: DATA PREPROCESSING
# Financial Influencers & ASX Market Analysis
# COSC 2671 - Assignment 2


import json
import re
import time
import subprocess
import sys
import pandas as pd
import numpy as np

MIN_VIDEOS_PER_CHANNEL = 2
MIN_COMMENT_LENGTH     = 10
START_DATE             = "2024-01-01"
END_DATE               = "2026-01-01"

ASX_TICKERS = [
    "^AXJO",
    "BHP.AX",
    "CBA.AX",
    "WBC.AX",
    "ANZ.AX",
    "NAB.AX",
    "FMG.AX",
    "CSL.AX",
    "WES.AX",
    "RIO.AX",
]

# LOAD RAW DATA

print(" 1: LOADING RAW DATA")

videos_df   = pd.read_csv("videos_clean.csv")
comments_df = pd.read_csv("comments_clean.csv")

print(f"Videos loaded:   {len(videos_df)}")
print(f"Comments loaded: {len(comments_df)}")
print(f"Channels found:  {videos_df['channel_name'].nunique()}")

# FILTER TO DATE RANGE

print("\n 2: FILTERING TO DATE RANGE")

videos_df['published_at'] = pd.to_datetime(
    videos_df['published_at'], utc=True, errors='coerce'
)

before        = len(videos_df)
videos_df     = videos_df[
    (videos_df['published_at'] >= pd.Timestamp(START_DATE, tz='UTC')) &
    (videos_df['published_at'] <= pd.Timestamp(END_DATE,   tz='UTC'))
].copy()

print(f"Videos before date filter: {before}")
print(f"Videos after  date filter: {len(videos_df)}")
print(f"Date range kept: {START_DATE} to {END_DATE}")

valid_video_ids = set(videos_df['video_id'].tolist())
comments_df     = comments_df[comments_df['video_id'].isin(valid_video_ids)].copy()
print(f"Comments after date filter: {len(comments_df)}")

# FILTER CHANNELS

print("\n 3: FILTERING CHANNELS")

channel_video_counts = videos_df['channel_name'].value_counts()
qualified_channels   = channel_video_counts[
    channel_video_counts >= MIN_VIDEOS_PER_CHANNEL
].index.tolist()

print(f"Channels with {MIN_VIDEOS_PER_CHANNEL}+ videos: {len(qualified_channels)}")
print("\nQualified channels:")
for ch in qualified_channels:
    print(f"  {ch:<45} {channel_video_counts[ch]} videos")

videos_df   = videos_df[videos_df['channel_name'].isin(qualified_channels)].copy()
comments_df = comments_df[comments_df['channel_name'].isin(qualified_channels)].copy()

print(f"\nAfter channel filter:")
print(f"  Videos:   {len(videos_df)}")
print(f"  Comments: {len(comments_df)}")

# CLEAN COMMENT TEXT

print("\n 4: CLEANING COMMENT TEXT")

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def is_english(text):
    if not text or len(text.strip()) == 0:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) == 0:
        return False
    ascii_alpha = [c for c in alpha_chars if ord(c) < 128]
    return (len(ascii_alpha) / len(alpha_chars)) >= 0.70


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = remove_emojis(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


print("Removing emojis and non-English comments...")

comments_df['comment_text_clean'] = comments_df['comment_text'].astype(str).apply(clean_text)

before      = len(comments_df)
comments_df = comments_df[
    comments_df['comment_text_clean'].apply(is_english)
].copy()
print(f"  Removed non-English: {before - len(comments_df)}")

before      = len(comments_df)
comments_df = comments_df[
    comments_df['comment_text_clean'].str.len() >= MIN_COMMENT_LENGTH
].copy()
print(f"  Removed too short:   {before - len(comments_df)}")

before      = len(comments_df)
comments_df = comments_df.drop_duplicates(
    subset=['commenter_id', 'comment_text_clean']
).copy()
print(f"  Removed duplicates:  {before - len(comments_df)}")

print(f"  Comments remaining:  {len(comments_df)}")

# CLEAN VIDEO TITLES AND DESCRIPTIONS

print("\n 5: CLEANING VIDEO TITLES AND DESCRIPTIONS")

videos_df['title_clean']       = videos_df['title'].astype(str).apply(clean_text)
videos_df['description_clean'] = videos_df['description'].astype(str).apply(clean_text)
videos_df['pub_date']          = videos_df['published_at'].dt.date

print(f"Titles cleaned: {len(videos_df)}")
print(f"  Earliest: {videos_df['published_at'].min()}")
print(f"  Latest:   {videos_df['published_at'].max()}")

# DOWNLOAD ASX PRICE DATA

print("\n 6: DOWNLOADING ASX PRICE DATA")

print("Upgrading yfinance...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--upgrade", "yfinance", "--quiet"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

import yfinance as yf
print(f"yfinance version: {yf.__version__}")

all_price_rows = []

for ticker in ASX_TICKERS:
    success = False

    # Method 1: standard yf.download
    try:
        raw = yf.download(
            ticker,
            start       = START_DATE,
            end         = END_DATE,
            progress    = False,
            auto_adjust = True,
            threads     = False
        )

        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]

            raw = raw.reset_index()

            if 'Date' not in raw.columns:
                raw.rename(columns={raw.columns[0]: 'Date'}, inplace=True)

            raw['Date']         = pd.to_datetime(raw['Date'])
            raw['ticker']       = ticker
            raw['daily_return'] = raw['Close'].pct_change()

            all_price_rows.append(
                raw[['Date','Open','High','Low','Close','Volume','ticker','daily_return']]
            )
            print(f"  Downloaded: {ticker}  ({len(raw)} trading days)")
            success = True

    except Exception:
        pass

    # Method 2: Ticker.history() fallback
    if not success:
        try:
            tk  = yf.Ticker(ticker)
            raw = tk.history(start=START_DATE, end=END_DATE, auto_adjust=True)

            if not raw.empty:
                raw         = raw.reset_index()
                raw.columns = [str(c) for c in raw.columns]

                if 'Date' not in raw.columns and 'Datetime' in raw.columns:
                    raw.rename(columns={'Datetime': 'Date'}, inplace=True)

                raw['Date']         = pd.to_datetime(raw['Date']).dt.tz_localize(None)
                raw['ticker']       = ticker
                raw['daily_return'] = raw['Close'].pct_change()

                all_price_rows.append(
                    raw[['Date','Open','High','Low','Close','Volume','ticker','daily_return']]
                )
                print(f"  Downloaded (method 2): {ticker}  ({len(raw)} trading days)")
                success = True

        except Exception:
            pass

    if not success:
        print(f"  FAILED both methods: {ticker}")

    time.sleep(0.5)

if all_price_rows:
    asx_df = pd.concat(all_price_rows, ignore_index=True)
    asx_df.to_csv('asx_prices.csv', index=False)
    print(f"\nSaved asx_prices.csv — {len(asx_df)} total rows")

    asx200 = asx_df[asx_df['ticker'] == '^AXJO'].copy()
    if not asx200.empty:
        print(f"\nASX200 summary:")
        print(f"  Trading days:     {len(asx200)}")
        print(f"  Date range:       {asx200['Date'].min().date()} to {asx200['Date'].max().date()}")
        print(f"  Avg daily return: {asx200['daily_return'].mean()*100:.3f}%")
        print(f"  Return std dev:   {asx200['daily_return'].std()*100:.3f}%")
        print(f"  Best day:         {asx200['daily_return'].max()*100:.2f}%")
        print(f"  Worst day:        {asx200['daily_return'].min()*100:.2f}%")
else:
    print("ASX download failed — run: pip install --upgrade yfinance requests")
    asx_df = pd.DataFrame()

# ALIGN VIDEOS TO ASX TRADING DAYS

print("\n 7: ALIGNING VIDEOS TO ASX TRADING DAYS")

if not asx_df.empty and '^AXJO' in asx_df['ticker'].values:
    asx200           = asx_df[asx_df['ticker'] == '^AXJO'][['Date','Close','daily_return']].copy()
    asx200['Date']   = pd.to_datetime(asx200['Date']).dt.normalize()
    asx200           = asx200.sort_values('Date').reset_index(drop=True)

    trading_days = asx200['Date'].tolist()
    returns_map  = dict(zip(asx200['Date'], asx200['daily_return']))
    close_map    = dict(zip(asx200['Date'], asx200['Close']))

    def get_next_trading_day(date, n=1):
        d      = pd.to_datetime(date)
        if d.tzinfo is not None:
            d  = d.tz_localize(None)
        d      = d.normalize()
        future = [x for x in trading_days if x > d]
        return future[n - 1] if len(future) >= n else None

    videos_df['pub_date_norm'] = pd.to_datetime(
        videos_df['published_at']
    ).dt.tz_localize(None).dt.normalize()

    videos_df['asx_return_T0'] = videos_df['pub_date_norm'].map(
        lambda d: returns_map.get(d, np.nan)
    )
    videos_df['asx_return_T1'] = videos_df['pub_date_norm'].apply(
        lambda d: returns_map.get(get_next_trading_day(d, 1), np.nan)
    )
    videos_df['asx_return_T3'] = videos_df['pub_date_norm'].apply(
        lambda d: returns_map.get(get_next_trading_day(d, 3), np.nan)
    )
    videos_df['asx_close_T0']  = videos_df['pub_date_norm'].map(
        lambda d: close_map.get(d, np.nan)
    )

    aligned = videos_df['asx_return_T1'].notna().sum()
    print(f"Videos aligned to T+1 ASX return: {aligned} / {len(videos_df)}")

else:
    print("Skipped — ASX data not available")
    for col in ['asx_return_T0','asx_return_T1','asx_return_T3','asx_close_T0']:
        videos_df[col] = np.nan

#  8: SAVE CLEANED FILES

print("\n 8: SAVING CLEANED FILES")

videos_df.to_csv('videos_filtered.csv',     index=False)
comments_df.to_csv('comments_filtered.csv', index=False)

print(f"Saved videos_filtered.csv   — {len(videos_df)} rows")
print(f"Saved comments_filtered.csv — {len(comments_df)} rows")

# FINAL SUMMARY

print("\nPREPROCESSING SUMMARY")

print(f"\nFINAL DATASET:")
print(f"  Channels:           {videos_df['channel_name'].nunique()}")
print(f"  Videos:             {len(videos_df)}")
print(f"  Comments:           {len(comments_df)}")
print(f"  Unique commenters:  {comments_df['commenter_id'].nunique()}")

print(f"\nNETWORK READINESS:")
commenter_channels = comments_df.groupby('commenter_id')['channel_name'].nunique()
multi              = (commenter_channels > 1).sum()
print(f"  Commenters on 2+ channels: {multi}")
if multi < 20:
    print("  WARNING: Low overlap — network will be sparse")
else:
    print("  Network construction looks viable")

print(f"\nCHANNEL BREAKDOWN:")
ch_stats = videos_df.groupby('channel_name').agg(
    videos         = ('video_id',     'count'),
    total_views    = ('view_count',   'sum'),
    avg_views      = ('view_count',   'mean'),
    total_comments = ('comment_count','sum')
).sort_values('total_views', ascending=False)
print(ch_stats.to_string())

print(f"\nASX ALIGNMENT:")
print(f"  Videos with T+1 return: {videos_df['asx_return_T1'].notna().sum()}")
print(f"  Videos with T+3 return: {videos_df['asx_return_T3'].notna().sum()}")

