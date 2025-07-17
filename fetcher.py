#!/usr/bin/env python3
"""
Abstract Streamer Dashboard MVP
Complete working script for analyzing streamer performance with Twitch API integration
"""

# import ssl
# import certifi
# import aiohttp

# # Patch aiohttp to use certifi
# ssl_context = ssl.create_default_context(cafile=certifi.where())
# aiohttp.ClientSession = lambda *args, **kwargs: aiohttp.client.ClientSession(*args, ssl=ssl_context, **kwargs)

import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os
import sys
from typing import List, Dict, Optional
import time

# Install required packages if not available
try:
    from twitchAPI.twitch import Twitch
    from twitchAPI.helper import first
    from dotenv import load_dotenv
except ImportError as e:
    print("Missing required packages. Install with:")
    print("pip install twitchAPI pandas python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class AbstractStreamerDashboard:
    def __init__(self):
        self.twitch_client = None
        self.twitter_data = None
        self.streamer_data = pd.DataFrame()
        self.api_data = pd.DataFrame()
        self.merged_data = pd.DataFrame()
        
        # Configuration
        import streamlit as st
        self.client_id = "3thu71n6sc460qellf5yt3f5xn0bth"
        self.client_secret = "vil5t9qxf5are91ws4lf3h9ohvuifm"
        self.tweetscout_api_key = "4f0ca11f-7b12-47e2-9132-21705dbe91e8"
        self.csv_path = os.getenv('CSV_INPUT_PATH', 'streamers.csv')
        self.output_path = os.getenv('CSV_OUTPUT_PATH', 'dashboard_output.csv')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Missing Twitch API credentials. Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in .env file")

    async def initialize_twitch_client(self):
        """Initialize Twitch API client with app authentication"""
        try:
            self.twitch_client = await Twitch(self.client_id, self.client_secret)
            logging.info("✅ Twitch API client initialized successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize Twitch client: {e}")
            raise

    def set_streamer_data(self, data: pd.DataFrame):
        """Set streamer data directly from external source (e.g., database)"""
        self.streamer_data = data.copy()
        
        # Clean and validate data
        self.streamer_data = self.streamer_data.dropna(subset=['username'])
        self.streamer_data['username'] = self.streamer_data['username'].str.lower().str.strip()
        
        # Convert CSV business metrics to numeric
        numeric_columns = ['views', 'value', 'total_watch_time', 'total_playing_time', 'negative_impact']
        for col in numeric_columns:
            if col in self.streamer_data.columns:
                self.streamer_data[col] = pd.to_numeric(self.streamer_data[col], errors='coerce').fillna(0)
        
        logging.info(f"📊 Loaded {len(self.streamer_data)} streamers from external data")
        logging.info(f"📋 CSV columns: {list(self.streamer_data.columns)}")
        return self.streamer_data

    def load_csv_data(self):
        """Load CSV data - fallback when no external data provided"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
            
        self.streamer_data = pd.read_csv(self.csv_path, encoding='utf-8')
        self.streamer_data = self.streamer_data.dropna(subset=['username'])
        self.streamer_data['username'] = self.streamer_data['username'].str.lower().str.strip()
        
        logging.info(f"📊 Loaded {len(self.streamer_data)} streamers from CSV")
        return self.streamer_data

    async def fetch_twitch_data_batch(self, usernames: List[str]) -> List[Dict]:
        """Fetch Twitch data for a batch of usernames"""
        batch_data = []
        
        try:
            # Get user information
            users = []
            async for user in self.twitch_client.get_users(logins=usernames):
                # logging.info(f"🧾 Raw user object: {vars(user)}")
                users.append(user)
            
            if not users:
                logging.warning(f"No users found for batch: {usernames[:5]}...")
                return batch_data
            
            # Get stream data for all users
            user_ids = [user.id for user in users]
            streams = {}
            async for stream in self.twitch_client.get_streams(user_id=user_ids):
                # logging.info(f"🎥 Raw stream object: {vars(stream)}")
                streams[stream.user_id] = stream
            
            # Process each user
            for user in users:
                try:
                    stream = streams.get(user.id)
                    
                    # Get follower count (with error handling for new API changes)
                    follower_count = 0
                    try:
                        follower_info = await self.twitch_client.get_channel_followers(user.id)
                        # logging.info(f"👥 Raw follower info for {user.login}: {vars(follower_info)}")
                        follower_count = follower_info.total if follower_info else 0
                    except Exception as e:
                        logging.warning(f"Could not fetch followers for {user.login}: {e}")
                    
                    # Calculate account age
                    account_age_days = (datetime.now() - user.created_at.replace(tzinfo=None)).days
                    
                    user_data = {
                        'username': user.login.lower(),
                        'display_name': user.display_name,
                        'user_id': user.id,
                        'is_live': stream is not None,
                        'current_game': stream.game_name if stream else None,
                        'current_game_id': stream.game_id if stream else None,
                        'viewer_count': stream.viewer_count if stream else 0,
                        'view_count': user.view_count,
                        'stream_title': stream.title if stream else None,
                        'stream_language': stream.language if stream else None,
                        'follower_count': follower_count,
                        'profile_image': user.profile_image_url,
                        'broadcaster_type': user.broadcaster_type,
                        'account_created': user.created_at.isoformat(),
                        'account_age_days': account_age_days,
                        'last_updated': datetime.now().isoformat()
                    }
                    batch_data.append(user_data)
                    
                except Exception as e:
                    logging.warning(f"⚠️ Error processing user {user.login}: {e}")
                    continue
            
            logging.info(f"✅ Processed {len(batch_data)} users from batch")
            # logging.info(f"Batch usernames: {', '.join(usernames[:5])}...")  # Log first 5 usernames
            # logging.info(f"Batch data sample: {batch_data[:5]}")  # Log first 5 entries for debugging
            return batch_data
            
        except Exception as e:
            logging.error(f"❌ Error fetching batch data: {e}")
            return batch_data

    async def fetch_all_streamer_data(self):
        """Fetch comprehensive streamer data from Twitch API with rate limiting"""
        if self.streamer_data.empty:
            logging.error("No CSV data loaded")
            return pd.DataFrame()
        
        usernames = self.streamer_data['username'].unique().tolist()
        logging.info(f"🔄 Fetching Twitch data for {len(usernames)} streamers...")
        
        all_data = []
        batch_size = 100  # Twitch API limit
        
        for i in range(0, len(usernames), batch_size):
            batch = usernames[i:i + batch_size]
            logging.info(f"📡 Processing batch {i//batch_size + 1}/{(len(usernames)-1)//batch_size + 1}")
            
            batch_data = await self.fetch_twitch_data_batch(batch)
            all_data.extend(batch_data)
            
            # Rate limiting - be respectful to Twitch API
            if i + batch_size < len(usernames):
                await asyncio.sleep(1)
        
        self.api_data = pd.DataFrame(all_data)
        return self.api_data

    # ----------------------------------------------------------------
    # Twitter Data
    # ----------------------------------------------------------------
    async def fetch_twitter_data_batch(self, twitter_usernames: List[str]) -> List[Dict]:
        """Fetch Twitter data for a batch of usernames using TweetScout API"""
        import aiohttp
        
        batch_data = []
        
        if not self.tweetscout_api_key:
            logging.warning("TweetScout API key not found, skipping Twitter data")
            return batch_data
        
        headers = {
            'ApiKey': self.tweetscout_api_key,
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            for username in twitter_usernames:
                if not username or pd.isna(username):
                    continue
                    
                try:
                    # Clean username (remove @ if present)
                    clean_username = username.replace('@', '').strip()
                    
                    payload = {
                        "link": f"https://twitter.com/{clean_username}"
                    }
                    
                    async with session.post(
                        'https://api.tweetscout.io/v2/user-tweets',
                        json=payload,
                        headers=headers
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            twitter_metrics = self.process_twitter_response(clean_username, data)
                            if twitter_metrics:
                                batch_data.append(twitter_metrics)
                        else:
                            logging.warning(f"Failed to fetch Twitter data for {clean_username}: {response.status}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logging.warning(f"Error fetching Twitter data for {username}: {e}")
                    continue
        
        logging.info(f"✅ Processed {len(batch_data)} Twitter profiles")
        return batch_data

    def process_twitter_response(self, username: str, data: Dict) -> Optional[Dict]:
        """Process TweetScout API response and extract metrics"""
        try:
            tweets = data.get('tweets', [])
            if not tweets:
                return None
            
            # Get user data from first tweet
            user_data = tweets[0].get('user', {})

            # logging.info(f"User data for {username}: {user_data}")
            
            # Calculate engagement metrics from recent tweets
            total_engagement = 0
            total_tweets = len(tweets)
            recent_tweets = tweets[:20]  # Analyze last 20 tweets
            
            engagement_metrics = []
            for tweet in recent_tweets:
                engagement = (
                    tweet.get('reply_count', 0) + 
                    tweet.get('retweet_count', 0) + 
                    tweet.get('favorite_count', 0) + 
                    tweet.get('quote_count', 0)
                )
                total_engagement += engagement
                engagement_metrics.append(engagement)
            
            # Calculate posting frequency (tweets per day in last 20 tweets)
            if len(recent_tweets) >= 2:
                from datetime import datetime
                try:
                    first_tweet_date = datetime.strptime(recent_tweets[-1]['created_at'], '%a %b %d %H:%M:%S %z %Y')
                    last_tweet_date = datetime.strptime(recent_tweets[0]['created_at'], '%a %b %d %H:%M:%S %z %Y')
                    days_span = (last_tweet_date - first_tweet_date).days
                    posting_frequency = len(recent_tweets) / max(days_span, 1)
                except:
                    posting_frequency = 0
            else:
                posting_frequency = 0
            
            # Account age calculation
            try:
                account_created = datetime.strptime(user_data.get('created_at', ''), '%a %b %d %H:%M:%S %z %Y')
                account_age_days = (datetime.now().replace(tzinfo=account_created.tzinfo) - account_created).days
            except:
                account_age_days = 0

            # logging.info(f"Account age for {username}: {account_age_days} days")
            # logging.info(f"Total engagement for {username}: {total_engagement} over {total_tweets} tweets")
            # logging.info(f"Posting frequency for {username}: {posting_frequency} tweets/day")
            # logging.info(f"Average engagement for {username}: {total_engagement / max(total_tweets, 1)} per tweet")
            # logging.info(f"Engagement rate for {username}: {(total_engagement / max(user_data.get('followers_count', 1), 1)) * 100:.2f}%")
            # logging.info(f"Follower to following ratio for {username}: {user_data.get('followers_count', 0) / max(user_data.get('friends_count', 1), 1):.2f}")
            # logging.info(f"Tweets per day for {username}: {user_data.get('statuses_count', 0) / max(account_age_days, 1):.2f}")
            # logging.info(f"Last updated for {username}: {datetime.now().isoformat()}")
            
            return {
                'username': username.lower(),
                'twitter_username': user_data.get('screen_name', ''),
                'twitter_name': user_data.get('name', ''),
                'twitter_followers': user_data.get('followers_count', 0),
                'twitter_following': user_data.get('friends_count', 0),
                'twitter_tweets_count': user_data.get('statuses_count', 0),
                'twitter_account_age_days': account_age_days,
                'twitter_avg_engagement': total_engagement / max(total_tweets, 1),
                'twitter_posting_frequency': posting_frequency,
                'twitter_engagement_rate': (total_engagement / max(user_data.get('followers_count', 1), 1)) * 100,
                'twitter_follower_to_following_ratio': user_data.get('followers_count', 0) / max(user_data.get('friends_count', 1), 1),
                'twitter_tweets_per_day': user_data.get('statuses_count', 0) / max(account_age_days, 1),
                'twitter_last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error processing Twitter data for {username}: {e}")
            return None

    async def fetch_all_twitter_data(self):
        """Fetch Twitter data for all streamers"""
        if self.streamer_data.empty or 'twitter' not in self.streamer_data.columns:
            logging.warning("No Twitter usernames available in data")
            return pd.DataFrame()
        
        # Get Twitter usernames from CSV data
        twitter_usernames = self.streamer_data['twitter'].dropna().unique().tolist()
        twitter_usernames = [u for u in twitter_usernames if u and str(u).strip()]
        
        if not twitter_usernames:
            logging.warning("No valid Twitter usernames found")
            return pd.DataFrame()
        
        logging.info(f"🐦 Fetching Twitter data for {len(twitter_usernames)} profiles...")
        
        # Process in smaller batches for Twitter API
        all_data = []
        batch_size = 10
        
        for i in range(0, len(twitter_usernames), batch_size):
            batch = twitter_usernames[i:i + batch_size]
            logging.info(f"📱 Processing Twitter batch {i//batch_size + 1}/{(len(twitter_usernames)-1)//batch_size + 1}")
            
            batch_data = await self.fetch_twitter_data_batch(batch)
            all_data.extend(batch_data)
            
            # Rate limiting between batches
            if i + batch_size < len(twitter_usernames):
                await asyncio.sleep(2)
        
        self.twitter_data = pd.DataFrame(all_data)
        return self.twitter_data
    
    def calculate_advanced_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate realistic performance metrics based on available CSV + Twitch API data"""
        df = data.copy()
        
        try:
            # === BUSINESS METRICS (from CSV) ===
            # Ensure numeric types for business metrics
            business_numeric_cols = ['views', 'value', 'total_watch_time', 'total_playing_time']
            for col in business_numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Core business efficiency metrics
            if 'views' in df.columns and 'total_watch_time' in df.columns:
                df['engagement_rate'] = (df['total_watch_time'] / df['views'].replace(0, 1)).fillna(0)
            
            if 'value' in df.columns and 'views' in df.columns:
                df['value_per_view'] = (df['value'] / df['views'].replace(0, 1)).fillna(0)
                
            if 'total_watch_time' in df.columns and 'total_playing_time' in df.columns:
                df['watch_efficiency'] = (df['total_watch_time'] / df['total_playing_time'].replace(0, 1)).fillna(0)
            
            # Business impact metrics
            df['negative_impact'] = pd.to_numeric(df['negative_impact'], errors='coerce').fillna(0)
            if 'value' in df.columns:
                df['impact_adjusted_value'] = df['value'] * (1 - df['negative_impact'])
                df['impact_score'] = 1 - df['negative_impact']
            
            # Revenue metrics (from business data)
            if 'value' in df.columns and 'total_playing_time' in df.columns:
                df['revenue_per_hour'] = (df['value'] / (df['total_playing_time'] / 3600).replace(0, 1)).fillna(0)
                if 'total_watch_time' in df.columns:
                    df['monetization_efficiency'] = (df['impact_adjusted_value'] / df['total_watch_time'].replace(0, 1) * 1000000).fillna(0)
            
            # === TWITCH SOCIAL METRICS (from API) ===
            # Only handle metrics that actually have meaningful data
            twitch_cols = {
                'follower_count': 0,
                'viewer_count': 0,
                'account_age_days': 0,
                'is_live': False
            }
            
            for col, default_val in twitch_cols.items():
                if col in df.columns:
                    if col == 'is_live':
                        df[col] = df[col].fillna(False)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
                else:
                    df[col] = default_val
            
            # Social engagement metrics (only meaningful ones)
            df['social_reach'] = df['follower_count']
            
            # Only calculate live engagement if we have viewers
            if df['viewer_count'].max() > 0:
                df['live_engagement'] = df['viewer_count']
                df['viewer_to_follower_ratio'] = (df['viewer_count'] / df['follower_count'].replace(0, 1)).fillna(0)
                df['live_engagement_rate'] = (df['viewer_count'] / df['follower_count'].replace(0, 1) * 100).fillna(0)
            
            # Account growth metrics (only if we have age data)
            if 'account_age_days' in df.columns and df['account_age_days'].max() > 0:
                df['growth_velocity'] = (df['follower_count'] / df['account_age_days'].replace(0, 1)).fillna(0)
                df['maturity_factor'] = (df['account_age_days'] / 365).fillna(0)
                df['followers_per_year'] = (df['follower_count'] / (df['account_age_days'] / 365).replace(0, 1)).fillna(0)
            
            # Broadcaster status metrics
            if 'isVerified' in df.columns:
                df['is_verified_streamer'] = df['isVerified'].map({'Yes': True, 'No': False}).fillna(False)
            
            if 'broadcaster_type' in df.columns:
                df['is_twitch_partner'] = df['broadcaster_type'].isin(['partner']).fillna(False)
                df['is_twitch_affiliate'] = df['broadcaster_type'].isin(['affiliate']).fillna(False)
                # Broadcaster tier scoring
                df['broadcaster_tier_score'] = 0
                df.loc[df['broadcaster_type'] == 'partner', 'broadcaster_tier_score'] = 3
                df.loc[df['broadcaster_type'] == 'affiliate', 'broadcaster_tier_score'] = 2
                df.loc[df['broadcaster_type'] == '', 'broadcaster_tier_score'] = 1
            
            # Cross-platform metrics (business + social)
            if 'value' in df.columns and df['follower_count'].max() > 0:
                df['value_per_follower'] = (df['value'] / df['follower_count'].replace(0, 1)).fillna(0)
            
            if 'engagement_rate' in df.columns and df['follower_count'].max() > 0:
                df['engagement_to_social_ratio'] = (df['engagement_rate'] / (df['follower_count'] + 1)).fillna(0)
            
            # === COMPOSITE PERFORMANCE SCORES ===
            # Normalize metrics for scoring (0-1 scale)
            def safe_normalize(series):
                max_val = series.max()
                if max_val == 0:
                    return pd.Series(0, index=series.index)
                return series / max_val
            
            # Business Performance Score (based on CSV metrics)
            business_components = []
            business_weights = []
            
            if 'engagement_rate' in df.columns and df['engagement_rate'].max() > 0:
                business_components.append(safe_normalize(df['engagement_rate']) * 0.3)
                business_weights.append(0.3)
            
            if 'value_per_view' in df.columns and df['value_per_view'].max() > 0:
                business_components.append(safe_normalize(df['value_per_view']) * 0.25)
                business_weights.append(0.25)
            
            if 'watch_efficiency' in df.columns and df['watch_efficiency'].max() > 0:
                business_components.append(safe_normalize(df['watch_efficiency']) * 0.25)
                business_weights.append(0.25)
            
            if 'impact_adjusted_value' in df.columns and df['impact_adjusted_value'].max() > 0:
                business_components.append(safe_normalize(df['impact_adjusted_value']) * 0.2)
                business_weights.append(0.2)
            
            if business_components:
                total_business_weight = sum(business_weights)
                df['business_performance'] = sum(business_components) / (total_business_weight / len(business_components))
            else:
                df['business_performance'] = 0
            
            # Social Performance Score (based on available Twitch metrics only)
            social_components = []
            social_weights = []
            
            if df['follower_count'].max() > 0:
                social_components.append(safe_normalize(df['follower_count']) * 0.5)
                social_weights.append(0.5)
            
            if 'live_engagement' in df.columns and df['live_engagement'].max() > 0:
                social_components.append(safe_normalize(df['live_engagement']) * 0.3)
                social_weights.append(0.3)
            
            if 'growth_velocity' in df.columns and df['growth_velocity'].max() > 0:
                social_components.append(safe_normalize(df['growth_velocity']) * 0.2)
                social_weights.append(0.2)
            
            if social_components:
                total_social_weight = sum(social_weights)
                df['social_performance'] = sum(social_components) / (total_social_weight / len(social_components))
            else:
                df['social_performance'] = 0
            
            # Overall Performance Score (weighted by data availability)
            business_weight = 0.7 if df['business_performance'].max() > 0 else 0
            social_weight = 0.3 if df['social_performance'].max() > 0 else 0
            
            # If no social data, use 100% business
            if social_weight == 0 and business_weight > 0:
                business_weight = 1.0
            # If no business data, use 100% social  
            elif business_weight == 0 and social_weight > 0:
                social_weight = 1.0
            
            df['performance_score'] = (
                df['business_performance'] * business_weight + 
                df['social_performance'] * social_weight
            ).fillna(0)
            
            # === RANKINGS ===
            if df['business_performance'].max() > 0:
                df['business_rank'] = df['business_performance'].rank(method='dense', ascending=False)
            
            if df['social_performance'].max() > 0:
                df['social_rank'] = df['social_performance'].rank(method='dense', ascending=False)
            
            if df['performance_score'].max() > 0:
                df['overall_rank'] = df['performance_score'].rank(method='dense', ascending=False)
                df['overall_percentile'] = df['performance_score'].rank(pct=True)
            
            # Category rankings (only if we have categories and performance data)
            if 'field' in df.columns and df['performance_score'].max() > 0:
                df['category_rank'] = df.groupby('field')['performance_score'].rank(method='dense', ascending=False)
                df['category_percentile'] = df.groupby('field')['performance_score'].rank(pct=True)
            
            # === ACTIVITY STATUS ===
            df['is_currently_live'] = df['is_live']
            df['has_social_presence'] = df['follower_count'] > 0
            df['is_active_creator'] = df.get('total_playing_time', 0) > 0
            
            # Log what we actually calculated
            calculated_metrics = []
            if df['business_performance'].max() > 0:
                calculated_metrics.append("business")
            if df['social_performance'].max() > 0:
                calculated_metrics.append("social")
            
            logging.info("📈 Advanced metrics calculated successfully")
            logging.info(f"📊 Calculated metrics: {', '.join(calculated_metrics) if calculated_metrics else 'basic only'}")
            
            if df['business_performance'].max() > 0:
                logging.info(f"📊 Business Performance Range: {df['business_performance'].min():.3f} - {df['business_performance'].max():.3f}")
            if df['social_performance'].max() > 0:
                logging.info(f"📊 Social Performance Range: {df['social_performance'].min():.3f} - {df['social_performance'].max():.3f}")
            if df['performance_score'].max() > 0:
                logging.info(f"📊 Overall Performance Range: {df['performance_score'].min():.3f} - {df['performance_score'].max():.3f}")
            
            # === TWITTER METRICS ===
            twitter_cols = {
                'twitter_followers': 0,
                'twitter_following': 0, 
                'twitter_tweets_count': 0,
                'twitter_avg_engagement': 0,
                'twitter_posting_frequency': 0,
                'twitter_engagement_rate': 0,
                'twitter_account_age_days': 0
            }

            for col, default_val in twitter_cols.items():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
                else:
                    df[col] = default_val

            # Twitter-specific metrics
            if df['twitter_followers'].max() > 0:
                df['twitter_reach'] = df['twitter_followers']
                df['twitter_influence_score'] = (
                    df['twitter_followers'] * 0.6 + 
                    df['twitter_avg_engagement'] * 0.4
                ).fillna(0)
                df['twitter_activity_score'] = (
                    df['twitter_posting_frequency'] * 0.7 + 
                    df['twitter_engagement_rate'] * 0.3
                ).fillna(0)

            # Cross-platform social metrics (Twitch + Twitter)
            df['total_social_followers'] = df['follower_count'] + df['twitter_followers']
            if df['total_social_followers'].max() > 0:
                df['cross_platform_reach'] = df['total_social_followers']
                df['social_diversification'] = (
                    df[['follower_count', 'twitter_followers']].min(axis=1) / 
                    df[['follower_count', 'twitter_followers']].max(axis=1).replace(0, 1)
                ).fillna(0)

            # Enhanced Social Performance Score (including Twitter)
            social_components = []
            social_weights = []

            if df['follower_count'].max() > 0:
                social_components.append(safe_normalize(df['follower_count']) * 0.3)
                social_weights.append(0.3)

            if df['twitter_followers'].max() > 0:
                social_components.append(safe_normalize(df['twitter_followers']) * 0.3)
                social_weights.append(0.3)

            if 'live_engagement' in df.columns and df['live_engagement'].max() > 0:
                social_components.append(safe_normalize(df['live_engagement']) * 0.2)
                social_weights.append(0.2)

            if df['twitter_activity_score'].max() > 0:
                social_components.append(safe_normalize(df['twitter_activity_score']) * 0.2)
                social_weights.append(0.2)
            return df
            
        except Exception as e:
            logging.error(f"❌ Error calculating metrics: {e}")
            import traceback
            logging.error(f"❌ Full traceback: {traceback.format_exc()}")
            return df
    
    def merge_and_analyze_data(self):
        """Merge CSV, API, and Twitter data, then calculate comprehensive metrics"""
        try:
            if self.streamer_data.empty:
                logging.error("No CSV data available for merging")
                return pd.DataFrame()
            
            merged = self.streamer_data.copy()
            
            # Merge Twitch API data
            if not self.api_data.empty:
                csv_columns = set(self.streamer_data.columns)
                api_columns = set(self.api_data.columns)
                overlap_columns = csv_columns.intersection(api_columns) - {'username'}
                
                csv_data_clean = self.streamer_data.copy()
                for col in overlap_columns:
                    if col in csv_data_clean.columns:
                        csv_data_clean = csv_data_clean.drop(columns=[col])
                
                merged = pd.merge(csv_data_clean, self.api_data, on='username', how='left')
                logging.info(f"🔗 Merged Twitch data: {len(merged)} total records")
            
            # Merge Twitter data
            if hasattr(self, 'twitter_data') and not self.twitter_data.empty:
                # Create mapping from twitter username to main username
                twitter_mapping = {}
                for _, row in self.streamer_data.iterrows():
                    if pd.notna(row.get('twitter')):
                        twitter_username = str(row['twitter']).replace('@', '').strip().lower()
                        main_username = row['username'].lower()
                        twitter_mapping[twitter_username] = main_username
                
                # Add main username to twitter data for merging
                self.twitter_data['main_username'] = self.twitter_data['username'].map(
                    lambda x: twitter_mapping.get(x.lower(), x.lower())
                )

                existing_twitter_cols = [col for col in merged.columns if col.startswith('twitter_')]
                if existing_twitter_cols:
                    merged = merged.drop(columns=existing_twitter_cols)
                    logging.info(f"Dropped existing Twitter columns: {existing_twitter_cols}")
                
                merged = pd.merge(
                    merged, 
                    self.twitter_data.drop(columns=['username']), 
                    left_on='username', 
                    right_on='main_username', 
                    how='left'
                )
                
                # Clean up
                if 'main_username' in merged.columns:
                    merged = merged.drop(columns=['main_username'])
                
                logging.info(f"🐦 Merged Twitter data: {len(merged)} total records")
            
            # Fill missing data with defaults
            self._fill_missing_data_defaults(merged)
            
            # Calculate all metrics
            self.merged_data = self.calculate_advanced_metrics(merged)
            
            # Add report metadata
            self.merged_data['report_generated'] = datetime.now().isoformat()
            data_sources = ['csv']
            if not self.api_data.empty:
                data_sources.append('twitch_api')
            if hasattr(self, 'twitter_data') and not self.twitter_data.empty:
                data_sources.append('twitter_api')
            
            self.merged_data['data_source'] = '+'.join(data_sources)
            
            logging.info(f"✅ Analysis complete for {len(self.merged_data)} streamers")
            return self.merged_data
            
        except Exception as e:
            logging.error(f"❌ Error merging and analyzing data: {e}")
            import traceback
            logging.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise

    def _fill_missing_data_defaults(self, merged):
        """Fill missing data with appropriate defaults"""
        # Twitch defaults
        twitch_defaults = {
            'display_name': merged['username'],
            'user_id': '', 'is_live': False, 'current_game': '',
            'viewer_count': 0, 'follower_count': 0, 'broadcaster_type': '',
            'account_age_days': 0
        }
        
        # Twitter defaults  
        twitter_defaults = {
            'twitter_username': '', 'twitter_name': '', 'twitter_followers': 0,
            'twitter_following': 0, 'twitter_tweets_count': 0, 'twitter_account_age_days': 0,
            'twitter_avg_engagement': 0, 'twitter_posting_frequency': 0,
            'twitter_engagement_rate': 0, 'twitter_follower_to_following_ratio': 0,
            'twitter_tweets_per_day': 0
        }
        
        all_defaults = {**twitch_defaults, **twitter_defaults}
        
        for col, default_val in all_defaults.items():
            if col in merged.columns:
                if col == 'display_name':
                    merged[col] = merged[col].fillna(merged['username'])
                else:
                    merged[col] = merged[col].fillna(default_val)
                    
    def export_data(self):
        """Export processed data to CSV and generate summary"""
        try:
            if self.merged_data.empty:
                logging.warning("No data to export")
                return
            
            # Export full dataset
            self.merged_data.to_csv(self.output_path, index=False)
            logging.info(f"📁 Full dataset exported to: {self.output_path}")
            
        except Exception as e:
            logging.error(f"❌ Error exporting data: {e}")

    async def run_complete_analysis(self, external_data: pd.DataFrame = None):
        """Execute the complete dashboard analysis workflow"""
        try:
            print("🚀 Starting Abstract Streamer Dashboard Analysis...")
            print("=" * 60)
            
            # Step 1: Initialize Twitch API
            await self.initialize_twitch_client()
            
            # Step 2: Load streamer data
            print("\n📂 Loading streamer data...")
            if external_data is not None and not external_data.empty:
                print("   Using provided data from database...")
                self.set_streamer_data(external_data)
            else:
                print("   Loading from CSV file...")
                self.load_csv_data()
            
            print(f"   Loaded {len(self.streamer_data)} streamers")
            
            # Step 3: Fetch Twitch data
            print("\n🌐 Fetching live Twitch data...")
            await self.fetch_all_streamer_data()

            # Step 3.5: Fetch Twitter data
            print("\n🐦 Fetching Twitter data...")
            await self.fetch_all_twitter_data()
            
            # Step 4: Merge and analyze
            print("\n🔄 Merging data and calculating metrics...")
            self.merge_and_analyze_data()
            
            # Step 5: Export results
            print("\n📊 Generating reports and exports...")
            self.export_data()
            
            print("\n✅ Analysis complete! Check output files for detailed results.")
            
        except Exception as e:
            logging.error(f"❌ Analysis failed: {e}")
            raise
        finally:
            if self.twitch_client:
                await self.twitch_client.close()

