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
        self.streamer_data = pd.DataFrame()
        self.api_data = pd.DataFrame()
        self.merged_data = pd.DataFrame()
        
        # Configuration
        import streamlit as st
        self.client_id = st.secrets["TWITCH_CLIENT_ID"]
        self.client_secret = st.secrets["TWITCH_CLIENT_SECRET"]
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
            
            return df
            
        except Exception as e:
            logging.error(f"❌ Error calculating metrics: {e}")
            import traceback
            logging.error(f"❌ Full traceback: {traceback.format_exc()}")
            return df
    
    def merge_and_analyze_data(self):
        """Merge CSV and API data, then calculate comprehensive metrics"""
        try:
            if self.streamer_data.empty:
                logging.error("No CSV data available for merging")
                return pd.DataFrame()
            
            if self.api_data.empty:
                logging.warning("No API data available - using CSV data only")
                merged = self.streamer_data.copy()
            else:
                # Check for overlapping columns and handle them
                csv_columns = set(self.streamer_data.columns)
                api_columns = set(self.api_data.columns)
                overlap_columns = csv_columns.intersection(api_columns) - {'username'}  # Keep username for merge
                
                # logging.info(f"🔍 Overlapping columns found: {overlap_columns}")
                
                # Remove overlapping columns from CSV data (keep API data as it's more recent)
                csv_data_clean = self.streamer_data.copy()
                for col in overlap_columns:
                    if col in csv_data_clean.columns:
                        # logging.info(f"🗑️ Removing duplicate column from CSV: {col}")
                        csv_data_clean = csv_data_clean.drop(columns=[col])
                
                # Perform the merge
                merged = pd.merge(
                    csv_data_clean, 
                    self.api_data, 
                    on='username', 
                    how='left'
                )
                logging.info(f"🔗 Merged data: {len(merged)} total records")
                
                # Fill missing API data with defaults for streamers not found on Twitch
                api_fill_defaults = {
                    'display_name': merged['username'],
                    'user_id': '',
                    'is_live': False,
                    'current_game': '',
                    'current_game_id': '',
                    'viewer_count': 0,
                    'stream_title': '',
                    'stream_language': '',
                    'follower_count': 0,
                    'profile_image': '',
                    'broadcaster_type': '',
                    'account_created': '',
                    'account_age_days': 0,
                    'last_updated': ''
                }
                
                for col, default_val in api_fill_defaults.items():
                    if col in merged.columns:
                        if col == 'display_name':
                            merged[col] = merged[col].fillna(merged['username'])
                        else:
                            merged[col] = merged[col].fillna(default_val)
            
            # Calculate all metrics
            self.merged_data = self.calculate_advanced_metrics(merged)
            
            # Add report metadata
            self.merged_data['report_generated'] = datetime.now().isoformat()
            self.merged_data['data_source'] = 'csv+api' if not self.api_data.empty else 'csv_only'
            
            logging.info(f"✅ Analysis complete for {len(self.merged_data)} streamers")
            # logging.info(f"📊 Final columns: {list(self.merged_data.columns)}")
            return self.merged_data
            
        except Exception as e:
            logging.error(f"❌ Error merging and analyzing data: {e}")
            import traceback
            logging.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise

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

