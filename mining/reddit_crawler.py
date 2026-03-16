import praw
import pandas as pd
import datetime
import re
import os

# --- Configuration ---
# You need to get these from https://www.reddit.com/prefs/apps
CLIENT_ID = 'YOUR_CLIENT_ID'
CLIENT_SECRET = 'YOUR_CLIENT_SECRET'
USER_AGENT = 'RedditDataCrawler/1.0 by /u/YOUR_USERNAME'

def get_reddit_instance():
    return praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

def extract_links(text):
    """Extracts external URLs from text using regex."""
    if not text:
        return []
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def is_media_url(url):
    """Checks if a URL likely points to media content."""
    media_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov', '.webm']
    return any(url.lower().endswith(ext) for ext in media_extensions) or 'v.redd.it' in url or 'imgur.com' in url

def crawl_subreddit(subreddit_name, limit=10):
    reddit = get_reddit_instance()
    subreddit = reddit.subreddit(subreddit_name)
    
    data = []
    
    print(f"Starting crawl for r/{subreddit_name}...")
    
    for submission in subreddit.hot(limit=limit):
        # Extract post-level info
        post_id = submission.id
        sub_name = submission.subreddit.display_name
        title = submission.title
        content = submission.selftext
        author = str(submission.author) if submission.author else "[deleted]"
        ts = datetime.datetime.fromtimestamp(submission.created_utc).isoformat()
        upvotes = submission.score
        comments_count = submission.num_comments
        nsfw = submission.over_18
        flair = submission.link_flair_text
        
        # Moderation status for post
        mod_status_post = "active"
        if submission.removed_by_category:
            mod_status_post = f"removed ({submission.removed_by_category})"
        elif submission.archived:
            mod_status_post = "archived"

        # Media and External Links for post
        post_links = extract_links(content)
        media_urls = [submission.url] if is_media_url(submission.url) else []
        
        # Expand comments
        submission.comments.replace_more(limit=0) # Only get top-level comments or flat list
        for comment in submission.comments.list():
            c_author = comment.author
            c_author_name = str(c_author) if c_author else "[deleted]"
            
            # Fetch user karma (requires an extra API call per user, handle carefully)
            user_karma = 0
            try:
                if c_author:
                    user_karma = c_author.link_karma + c_author.comment_karma
            except Exception:
                user_karma = "N/A"

            # Moderation status for comment
            mod_status_comment = "active"
            if comment.removed:
                mod_status_comment = "removed"
            elif comment.body == "[deleted]":
                mod_status_comment = "deleted_by_user"

            # Compile row according to requested naming
            row = {
                'post_id': post_id,
                'subreddit': sub_name,
                'post_title': title,
                'post_content': content,
                'author_username': author,
                'timestamp': ts,
                'upvotes': upvotes,
                'comments_count': comments_count,
                'comment_id': comment.id,
                'comment_text': comment.body,
                'user_karma': user_karma,
                'external_links': ", ".join(post_links + extract_links(comment.body)),
                'media_urls': ", ".join(media_urls),
                'NSFW_flag': nsfw,
                'post_flair': flair,
                'moderation_status': f"Post: {mod_status_post} | Comment: {mod_status_comment}"
            }
            data.append(row)
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Example usage
    TARGET_SUBREDDIT = 'science' 
    CRAWL_LIMIT = 5 # Number of posts to fetch
    
    try:
        df = crawl_subreddit(TARGET_SUBREDDIT, limit=CRAWL_LIMIT)
        
        # Save to CSV
        output_file = f"reddit_{TARGET_SUBREDDIT}_data.csv"
        df.to_csv(output_file, index=False)
        print(f"Successfully crawled {len(df)} records. Saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nNote: Make sure you have installed 'praw' and 'pandas' and provided valid API credentials.")
