import os
import logging
from typing import List, Dict, Any, Optional

import tweepy
from datetime import datetime

# -----------------------------------------------------------
# Social Media API Ingestion Module (Twitter/X First Version)
# -----------------------------------------------------------
# Purpose:
#   - Fetch engagement metrics, comments, and trending topics
#   - Serve as unified ingestion layer for Metrics Engine,
#     Sentiment Engine, A/B Coach, Trend Optimizer
# -----------------------------------------------------------

class SocialIngestor:
    """
    Unified social media ingestion class.
    Currently implements Twitter/X API ingestion using Tweepy.
    Can be extended to LinkedIn, Instagram, YouTube later.
    """

    def __init__(self):
        # Load credentials from environment variables
        self.api_key = os.getenv("TWITTER_API_KEY")
        self.api_secret = os.getenv("TWITTER_API_SECRET")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_secret = os.getenv("TWITTER_ACCESS_SECRET")

        if not all([self.api_key, self.api_secret, self.access_token, self.access_secret]):
            logging.warning("Twitter API credentials missing. Ingestor will not function.")

        try:
            self.client = tweepy.Client(
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_secret
            )
            logging.info("Twitter API client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Twitter API client: {e}")
            self.client = None

    # -----------------------------------------------------------
    # Normalize Tweet Output Structure
    # -----------------------------------------------------------
    def _normalize(self, post_id: str, text: str, metrics: Dict[str, Any], comments: List[str]):
        return {
            "post_id": post_id,
            "text": text,
            "comments": comments,
            "metrics": {
                "likes": metrics.get("like_count", 0),
                "shares": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "quotes": metrics.get("quote_count", 0),
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    # -----------------------------------------------------------
    # Fetch Engagement Metrics
    # -----------------------------------------------------------
    def fetch_post_metrics(self, post_id: str) -> Dict[str, Any]:
        """
        Fetch public engagement metrics for a given post.
        """
        if not self.client:
            logging.error("Twitter client is not initialized.")
            return {}

        try:
            response = self.client.get_tweet(id=post_id, tweet_fields=["public_metrics", "text"])
            if not response or not response.data:
                logging.warning(f"No data returned for post {post_id}.")
                return {}

            tweet = response.data
            metrics = tweet.public_metrics

            return {
                "likes": metrics.get("like_count", 0),
                "shares": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "quotes": metrics.get("quote_count", 0),
                "text": tweet.text
            }

        except Exception as e:
            logging.error(f"Error fetching metrics for post {post_id}: {e}")
            return {}

    # -----------------------------------------------------------
    # Fetch Post Comments (Replies)
    # -----------------------------------------------------------
    def fetch_post_comments(self, post_id: str, limit: int = 50) -> List[str]:
        """
        Fetch comments for a given post using conversation_id query.
        """
        if not self.client:
            logging.error("Twitter client is not initialized.")
            return []

        try:
            query = f"conversation_id:{post_id}"
            response = self.client.search_recent_tweets(query=query, max_results=limit)

            comments = []
            if response.data:
                for tweet in response.data:
                    comments.append(tweet.text)

            return comments

        except Exception as e:
            logging.error(f"Error fetching comments for post {post_id}: {e}")
            return []

    # -----------------------------------------------------------
    # Fetch Trending Topics
    # -----------------------------------------------------------
    def fetch_trending_topics(self) -> List[str]:
        """
        Placeholder: Trending endpoint not available in Twitter API v2.
        Will use external API or scraping if needed.
        """
        logging.warning("Twitter API v2 does not support trending topics. Returning empty list.")
        return []

    # -----------------------------------------------------------
    # Fetch Recent Posts From a User
    # -----------------------------------------------------------
    def fetch_user_posts(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch latest posts from a user.
        """
        if not self.client:
            logging.error("Twitter client is not initialized.")
            return []

        try:
            response = self.client.get_users_tweets(
                id=user_id,
                max_results=limit,
                tweet_fields=["public_metrics", "text"]
            )

            posts = []
            if response.data:
                for tweet in response.data:
                    posts.append({
                        "post_id": tweet.id,
                        "text": tweet.text,
                        "metrics": tweet.public_metrics,
                        "timestamp": datetime.utcnow().isoformat()
                    })

            return posts

        except Exception as e:
            logging.error(f"Error fetching posts for user {user_id}: {e}")
            return []

    # -----------------------------------------------------------
    # Unified Fetch Wrapper
    # -----------------------------------------------------------
    def fetch_complete_post_data(self, post_id: str) -> Dict[str, Any]:
        """
        Fetch metrics + comments + text for a post, normalized.
        """
        metrics_data = self.fetch_post_metrics(post_id)
        if not metrics_data:
            return {}

        comments_data = self.fetch_post_comments(post_id)

        return self._normalize(
            post_id=post_id,
            text=metrics_data.get("text", ""),
            metrics=metrics_data,
            comments=comments_data
        )


# -----------------------------------------------------------
# Example usage (Remove in production)
# -----------------------------------------------------------
if __name__ == "__main__":
    ingestor = SocialIngestor()
    post_id_test = "1234567890123456789"  # Replace with actual

    print(ingestor.fetch_post_metrics(post_id_test))
    print(ingestor.fetch_post_comments(post_id_test))
