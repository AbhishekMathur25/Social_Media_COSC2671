# NOTEBOOK 1: DATA COLLECTION
# Financial Influencers & ASX Market Analysis
# COSC 2671 - Assignment 2


import json
import time
import os
import pandas as pd
import yfinance as yf
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# CONFIGURATION

API_KEY  = "apikey"
YOUTUBE  = build('youtube', 'v3', developerKey=API_KEY)

# Search queries to find Australian finance influencer videos
QUERIES = [
    "ASX investing 2024 Australia",
    "ASX investing 2025 Australia",
    "Australian stock market tips 2024",
    "Australian stock market tips 2025",
    "ASX shares buy 2024",
    "ASX shares buy 2025",
    "Australian finance investing advice",
    "BHP CBA shares Australia 2024",
    "BHP CBA shares Australia 2025",
    "ASX ETF investing Australia",
    "Australian property stocks 2024",
    "ASX200 market analysis 2024",
    "ASX200 market analysis 2025",
    "best ASX stocks to buy 2024",
    "best ASX stocks to buy 2025",
    "Australian dividend investing stocks",
    "Rask Australia investing",
    "Equity Mates podcast ASX",
    "ausbiz ASX market",
    "Livewire Markets ASX stocks",
]

# How many comments total to collect
TARGET_COMMENTS = 60000

# ASX tickers to track
ASX_TICKERS = [
    "^AXJO",   # ASX 200
    "BHP.AX",  # BHP Group
    "CBA.AX",  # Commonwealth Bank
    "WBC.AX",  # Westpac
    "ANZ.AX",  # ANZ Bank
    "NAB.AX",  # NAB
    "FMG.AX",  # Fortescue Metals
    "CSL.AX",  # CSL Limited
    "WES.AX",  # Wesfarmers
    "RIO.AX",  # Rio Tinto
]

# Date range
START_DATE = "2024-01-01"
END_DATE   = "2026-01-01"

# COLLECT VIDEOS + COMMENTS

def collect_videos_and_comments():

    all_data               = []
    total_comments         = 0
    seen_video_ids         = set()

    for query in QUERIES:

        if total_comments >= TARGET_COMMENTS:
            break

        print(f"\nSearching: {query}")

        try:
            search_response = YOUTUBE.search().list(
                q          = query,
                part       = "snippet",
                type       = "video",
                maxResults = 50,
                order      = "relevance",
                relevanceLanguage = "en",
                regionCode = "AU"
            ).execute()
        except HttpError as e:
            print(f"  Search error: {e}")
            continue

        for item in search_response.get('items', []):

            if total_comments >= TARGET_COMMENTS:
                break

            video_id    = item['id']['videoId']
            snippet     = item['snippet']

            # Skip duplicates from overlapping search queries
            if video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)

            video_title       = snippet.get('title', '')
            channel_title     = snippet.get('channelTitle', '')
            channel_id        = snippet.get('channelId', '')
            published_at      = snippet.get('publishedAt', '')
            description       = snippet.get('description', '')

            #  Get full video stats 
            try:
                stats_response = YOUTUBE.videos().list(
                    part = "statistics,contentDetails",
                    id   = video_id
                ).execute()

                stats = {}
                if stats_response.get('items'):
                    stats = stats_response['items'][0].get('statistics', {})

                view_count    = int(stats.get('viewCount',    0))
                like_count    = int(stats.get('likeCount',    0))
                comment_count = int(stats.get('commentCount', 0))

            except HttpError:
                view_count = like_count = comment_count = 0

            #  Get transcript 
            transcript_text = ""
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(
                    video_id,
                    languages=['en', 'en-AU', 'en-GB', 'en-US']
                )
                transcript_text = ' '.join([
                    seg['text'] for seg in transcript_list
                ])
            except (TranscriptsDisabled, NoTranscriptFound):
                transcript_text = ""
            except Exception:
                transcript_text = ""

            # Get comments 
            video_comments    = []
            next_page_token   = None

            while len(video_comments) < 1000:
                try:
                    comment_response = YOUTUBE.commentThreads().list(
                        part        = "snippet",
                        videoId     = video_id,
                        maxResults  = 100,
                        pageToken   = next_page_token,
                        textFormat  = "plainText",
                        order       = "relevance"
                    ).execute()

                    for c_item in comment_response.get('items', []):
                        top     = c_item['snippet']['topLevelComment']['snippet']
                        video_comments.append({
                            "comment_id":      c_item['id'],
                            "commenter_id":    top.get('authorChannelId', {}).get('value', 'unknown'),
                            "commenter_name":  top.get('authorDisplayName', ''),
                            "comment_text":    top.get('textDisplay', ''),
                            "like_count":      top.get('likeCount', 0),
                            "reply_count":     c_item['snippet'].get('totalReplyCount', 0),
                            "published_at":    top.get('publishedAt', '')
                        })
                        total_comments += 1

                    next_page_token = comment_response.get('nextPageToken')
                    if not next_page_token:
                        break

                except HttpError:
                    break

            #  Store everything for this video
            all_data.append({
                "video_id":       video_id,
                "channel_id":     channel_id,
                "channel_name":   channel_title,
                "title":          video_title,
                "description":    description,
                "published_at":   published_at,
                "view_count":     view_count,
                "like_count":     like_count,
                "comment_count":  comment_count,
                "transcript":     transcript_text,
                "comments":       video_comments
            })

            print(f"  Video: {video_title[:60]}")
            print(f"  Channel: {channel_title} | Views: {view_count:,} | Comments collected: {len(video_comments)}")
            print(f"  Total comments so far: {total_comments}")

            time.sleep(0.5)

    #  Save raw combined data 
    with open('raw_youtube_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)


    print(f"COLLECTION COMPLETE")
  
    print(f"Total videos:   {len(all_data)}")
    print(f"Total comments: {total_comments}")

    return all_data


#FLATTEN INTO CSV FILES

def flatten_to_csv(all_data):

    # Videos CSV
    videos_rows = []
    for video in all_data:
        videos_rows.append({
            "video_id":      video["video_id"],
            "channel_id":    video["channel_id"],
            "channel_name":  video["channel_name"],
            "title":         video["title"],
            "description":   video["description"],
            "published_at":  video["published_at"],
            "view_count":    video["view_count"],
            "like_count":    video["like_count"],
            "comment_count": video["comment_count"],
            "has_transcript": len(video["transcript"]) > 100,
            "transcript_word_count": len(video["transcript"].split()) if video["transcript"] else 0
        })

    videos_df = pd.DataFrame(videos_rows)
    videos_df.to_csv('videos_clean.csv', index=False)
    print(f"Saved videos_clean.csv  — {len(videos_df)} rows")

    # Comments CSV 
    comments_rows = []
    for video in all_data:
        for comment in video["comments"]:
            comments_rows.append({
                "video_id":       video["video_id"],
                "channel_id":     video["channel_id"],
                "channel_name":   video["channel_name"],
                "comment_id":     comment["comment_id"],
                "commenter_id":   comment["commenter_id"],
                "commenter_name": comment["commenter_name"],
                "comment_text":   comment["comment_text"],
                "like_count":     comment["like_count"],
                "reply_count":    comment["reply_count"],
                "published_at":   comment["published_at"]
            })

    comments_df = pd.DataFrame(comments_rows)
    comments_df.to_csv('comments_clean.csv', index=False)
    print(f"Saved comments_clean.csv — {len(comments_df)} rows")

    # Transcripts CSV
    transcript_rows = []
    for video in all_data:
        if video["transcript"] and len(video["transcript"]) > 100:
            transcript_rows.append({
                "video_id":     video["video_id"],
                "channel_id":   video["channel_id"],
                "channel_name": video["channel_name"],
                "title":        video["title"],
                "published_at": video["published_at"],
                "transcript":   video["transcript"],
                "word_count":   len(video["transcript"].split())
            })

    transcripts_df = pd.DataFrame(transcript_rows)
    transcripts_df.to_csv('transcripts_clean.csv', index=False)
    print(f"Saved transcripts_clean.csv — {len(transcripts_df)} rows")

    return videos_df, comments_df, transcripts_df


# COLLECT ASX PRICE DATA

def collect_asx_data():

    print(f"\nDownloading ASX price data ({START_DATE} to {END_DATE})...")

    all_rows = []

    for ticker in ASX_TICKERS:
        try:
            data = yf.download(
                ticker,
                start    = START_DATE,
                end      = END_DATE,
                progress = False,
                auto_adjust = True
            )

            if data.empty:
                print(f"  No data: {ticker}")
                continue

            data = data.reset_index()
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
            data['ticker']         = ticker
            data['daily_return']   = data['Close'].pct_change()

            all_rows.append(data)
            print(f"  Downloaded: {ticker} — {len(data)} trading days")
            time.sleep(0.3)

        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    if all_rows:
        asx_df = pd.concat(all_rows, ignore_index=True)
        asx_df.to_csv('asx_prices.csv', index=False)
        print(f"\nSaved asx_prices.csv — {len(asx_df)} rows")
        return asx_df

    return pd.DataFrame()

#SUMMARY STATS

def print_summary(videos_df, comments_df, transcripts_df, asx_df):


    print("FINAL SUMMARY")
  

    print(f"\nYOUTUBE DATA:")
    print(f"  Videos collected:    {len(videos_df)}")
    print(f"  Comments collected:  {len(comments_df)}")
    print(f"  Transcripts:         {len(transcripts_df)}")
    print(f"  Unique channels:     {videos_df['channel_name'].nunique()}")
    print(f"  Unique commenters:   {comments_df['commenter_id'].nunique()}")

    print(f"\nCHANNELS FOUND:")
    channel_counts = videos_df['channel_name'].value_counts()
    for channel, count in channel_counts.items():
        print(f"  {channel:<40} {count} videos")

    print(f"\nNETWORK READINESS:")
    commenter_channel = comments_df.groupby('commenter_id')['channel_name'].nunique()
    multi_channel     = (commenter_channel > 1).sum()
    print(f"  Commenters on 2+ channels: {multi_channel}")

    if multi_channel < 20:
        print("  WARNING: Low overlap — network may be sparse")
        print("  Consider increasing TARGET_COMMENTS or adding more queries")
    else:
        print("  Network construction looks viable")

    print(f"\nASX DATA:")
    if not asx_df.empty:
        asx200 = asx_df[asx_df['ticker'] == '^AXJO']
        print(f"  Tickers downloaded:  {asx_df['ticker'].nunique()}")
        print(f"  ASX200 trading days: {len(asx200)}")
        print(f"  Date range:          {asx_df['Date'].min()} to {asx_df['Date'].max()}")

    print(f"\nFILES SAVED:")
    for fname in ['raw_youtube_data.json', 'videos_clean.csv',
                  'comments_clean.csv', 'transcripts_clean.csv', 'asx_prices.csv']:
        exists = os.path.exists(fname)
        size   = os.path.getsize(fname) / 1024 if exists else 0
        status = f"{size:.1f} KB" if exists else "MISSING"
        print(f"  {fname:<35} {status}")


if __name__ == "__main__":


    print("FINANCIAL INFLUENCERS & ASX — DATA COLLECTION")


    #  1: Collect YouTube data
    all_data = collect_videos_and_comments()

    #  2: Flatten to CSVs
    videos_df, comments_df, transcripts_df = flatten_to_csv(all_data)

    #  3: ASX price data
    asx_df = collect_asx_data()

    #  4: Summary
    print_summary(videos_df, comments_df, transcripts_df, asx_df)
