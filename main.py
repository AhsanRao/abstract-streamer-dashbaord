import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import asyncio
import sys
from typing import Optional, Dict, List

try:
    from fetcher import AbstractStreamerDashboard
    HAS_API_INTEGRATION = True
except ImportError:
    HAS_API_INTEGRATION = False
    st.warning("API integration not available. Using data management features only.")

# Page configuration
st.set_page_config(
    page_title="Abstract Streamer Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database setup
DB_PATH = "streamer_analytics.db"

def init_database():
    """Initialize SQLite database with updated schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS streamers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            -- Original CSV columns
            views INTEGER,
            value REAL,
            total_watch_time INTEGER,
            total_playing_time REAL,
            negative_impact REAL,
            field TEXT,
            username TEXT UNIQUE,
            isVerified TEXT,
            walletAddress TEXT,
            twitter TEXT,
            -- Twitch API columns
            display_name TEXT,
            user_id TEXT,
            is_live BOOLEAN,
            current_game TEXT,
            viewer_count REAL,
            follower_count REAL,
            broadcaster_type TEXT,
            account_age_days REAL,
            -- Core calculated metrics
            engagement_rate REAL,
            value_per_view REAL,
            watch_efficiency REAL,
            impact_adjusted_value REAL,
            impact_score REAL,
            revenue_per_hour REAL,
            monetization_efficiency REAL,
            -- Social metrics (only when data exists)
            social_reach REAL,
            live_engagement REAL,
            live_engagement_rate REAL,
            viewer_to_follower_ratio REAL,
            growth_velocity REAL,
            maturity_factor REAL,
            followers_per_year REAL,
            broadcaster_tier_score INTEGER,
            -- Cross-platform metrics
            value_per_follower REAL,
            engagement_to_social_ratio REAL,
            -- Performance scores
            business_performance REAL,
            social_performance REAL,
            performance_score REAL,
            business_rank REAL,
            social_rank REAL,
            overall_rank REAL,
            overall_percentile REAL,
            category_rank REAL,
            category_percentile REAL,
            -- Status flags
            is_verified_streamer BOOLEAN,
            is_twitch_partner BOOLEAN,
            is_twitch_affiliate BOOLEAN,
            is_currently_live BOOLEAN,
            has_social_presence BOOLEAN,
            is_active_creator BOOLEAN,
            -- Metadata
            data_source TEXT,
            last_updated TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON streamers(username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_field ON streamers(field)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance ON streamers(performance_score)')
    
    conn.commit()
    conn.close()

def load_data_from_db() -> pd.DataFrame:
    """Load data from SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM streamers ORDER BY performance_score DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()

def save_data_to_db(df: pd.DataFrame):
    """Save DataFrame to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Replace existing data
        df.to_sql('streamers', conn, if_exists='replace', index=False)
        conn.close()
        st.success(f"Successfully saved {len(df)} records to database")
    except Exception as e:
        st.error(f"Error saving data to database: {e}")

def detect_csv_format(df: pd.DataFrame) -> str:
    """Detect if CSV is in initial format or processed format"""
    initial_format_columns = ['views', 'value', 'total_watch_time', 'total_playing_time', 'field', 'username']
    processed_format_columns = ['performance_score', 'follower_count', 'is_live', 'engagement_rate']
    
    initial_score = sum(1 for col in initial_format_columns if col in df.columns)
    processed_score = sum(1 for col in processed_format_columns if col in df.columns)
    
    if processed_score >= 2:
        return "processed"
    elif initial_score >= 4:
        return "initial"
    else:
        return "unknown"

def calculate_basic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic metrics for initial format CSV"""
    df = df.copy()
    
    try:
        # Fill missing values
        df['negative_impact'] = df['negative_impact'].fillna(0)
        df['total_watch_time'] = pd.to_numeric(df['total_watch_time'], errors='coerce').fillna(0)
        df['total_playing_time'] = pd.to_numeric(df['total_playing_time'], errors='coerce').fillna(0)
        df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
        
        # Calculate basic derived metrics
        df['engagement_rate'] = (df['total_watch_time'] / df['views'].replace(0, 1)).fillna(0)
        df['value_per_view'] = (df['value'] / df['views'].replace(0, 1)).fillna(0)
        df['watch_efficiency'] = (df['total_watch_time'] / df['total_playing_time'].replace(0, 1)).fillna(0)
        df['impact_adjusted_value'] = df['value'] * (1 - df['negative_impact'])
        
        # Simple performance score based on available data
        df['performance_score'] = (
            (df['engagement_rate'] / df['engagement_rate'].max() if df['engagement_rate'].max() > 0 else 0) * 0.4 +
            (df['value_per_view'] / df['value_per_view'].max() if df['value_per_view'].max() > 0 else 0) * 0.3 +
            (df['watch_efficiency'] / df['watch_efficiency'].max() if df['watch_efficiency'].max() > 0 else 0) * 0.3
        ).fillna(0)
        
        # Add ranking
        df['overall_rank'] = df['performance_score'].rank(method='dense', ascending=False)
        
        # Add placeholders for missing columns expected by dashboard
        df['follower_count'] = 0
        df['is_live'] = False
        df['current_game'] = ''
        df['viewer_count'] = 0
        df['display_name'] = df['username']
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating basic metrics: {e}")
        return df

def import_csv_data(file_path: str = None, uploaded_file = None) -> pd.DataFrame:
    """Import data from CSV file and detect format"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            return pd.DataFrame()
        
        # Detect format and process accordingly
        format_type = detect_csv_format(df)
        
        if format_type == "initial":
            st.info("📁 Detected initial CSV format. Calculating basic metrics...")
            df = calculate_basic_metrics(df)
            df['data_source'] = 'csv_initial'
        elif format_type == "processed":
            st.success("📊 Detected processed CSV format with full metrics.")
            df['data_source'] = 'csv_processed'
        else:
            st.warning("⚠️ Unknown CSV format. Attempting to load as-is...")
            df['data_source'] = 'csv_unknown'
        
        # Add timestamp columns
        df['created_at'] = datetime.now()
        df['updated_at'] = datetime.now()
        
        return df
    except Exception as e:
        st.error(f"Error importing CSV data: {e}")
        return pd.DataFrame()

def format_number(num):
    """Format numbers for display"""
    if pd.isna(num):
        return "N/A"
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.0f}"

def format_duration(seconds):
    """Format duration in seconds to readable format"""
    if pd.isna(seconds):
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

# Initialize database
init_database()

# Sidebar for navigation and data management
st.sidebar.title("🎮 Streamer Dashboard")

# Data Management Section
st.sidebar.header("📊 Data Management")

# File upload only
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV File", type=['csv'])
if uploaded_file is not None:
    if st.sidebar.button("Import & Process Data"):
        with st.spinner("Importing and processing data..."):
            try:
                df = import_csv_data(uploaded_file=uploaded_file)
                if not df.empty:
                    save_data_to_db(df)
                    
                    # If API integration available and data is initial format, process it
                    if HAS_API_INTEGRATION and df['data_source'].iloc[0] == 'csv_initial':
                        dashboard = AbstractStreamerDashboard()
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(dashboard.run_complete_analysis(df))
                        
                        # Load data
                        if os.path.exists(dashboard.output_path):
                            enhanced_df = pd.read_csv(dashboard.output_path)
                            enhanced_df['data_source'] = 'csv+api'
                            save_data_to_db(enhanced_df)
                            st.success("✅ Data imported and enhanced with Twitch API!")
                        else:
                            st.success("✅ Data imported successfully!")
                    else:
                        st.success("✅ Data imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing data: {e}")

# Refresh API data
if HAS_API_INTEGRATION:
    if st.sidebar.button("🔄 Refresh API Data"):
        with st.spinner("Refreshing data from Twitch API..."):
            try:
                current_data = load_data_from_db()
                if current_data.empty:
                    st.error("No data to refresh. Please import CSV data first.")
                else:
                    dashboard = AbstractStreamerDashboard()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(dashboard.run_complete_analysis(current_data))
                    
                    if os.path.exists(dashboard.output_path):
                        df = pd.read_csv(dashboard.output_path)
                        df['data_source'] = 'csv+api'
                        save_data_to_db(df)
                        st.success("✅ Data refreshed successfully!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error refreshing data: {e}")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "📋 Navigate to:",
    ["Overview", "Streamers Directory"] + (["Live Streams"] if HAS_API_INTEGRATION else [])
)

# Load data
df = load_data_from_db()

if df.empty:
    st.warning("📭 No data available. Please import some data using the sidebar options.")
    st.info("💡 **Quick Start Options:**")
    st.markdown("""
    1. **🔄 Load Sample Data** - Try the dashboard with test data
    2. **📁 Upload CSV** - Upload your initial CSV file (views, username, etc.)
    3. **🚀 Process with API** - Enhance basic CSV with Twitch data
    """)
    st.stop()

# Detect data format for UI adjustments
data_format = df['data_source'].iloc[0] if 'data_source' in df.columns and not df.empty else 'unknown'
is_basic_format = data_format in ['csv_initial', 'sample_data']

if is_basic_format:
    st.info("📊 **Basic Data Mode** - Upload your initial CSV and use '🚀 Process with API' for enhanced analytics!")

if page == "Overview":
    st.title("🎯 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Streamers", len(df))
    
    with col2:
        if 'is_live' in df.columns:
            live_count = len(df[df['is_live'] == True])
            st.metric("Currently Live", live_count)
        else:
            categories = df['field'].nunique() if 'field' in df.columns else 0
            st.metric("Categories", categories)
    
    with col3:
        if 'follower_count' in df.columns:
            total_followers = df['follower_count'].sum()
            st.metric("Total Followers", format_number(total_followers))
        else:
            total_value = df['value'].sum() if 'value' in df.columns else 0
            st.metric("Total Value", format_number(total_value))
    
    with col4:
        if 'performance_score' in df.columns:
            avg_performance = df['performance_score'].mean()
            st.metric("Avg Performance", f"{avg_performance:.3f}")
        else:
            avg_engagement = df['engagement_rate'].mean() if 'engagement_rate' in df.columns else 0
            st.metric("Avg Engagement", format_number(avg_engagement))
    
    # Performance distribution chart
    if 'performance_score' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Performance Distribution")
            fig = px.histogram(df, x='performance_score', nbins=15, 
                             title="Performance Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'field' in df.columns:
                st.subheader("📈 Performance by Category")
                category_performance = df.groupby('field')['performance_score'].mean().sort_values(ascending=False)
                fig = px.bar(x=category_performance.values, y=category_performance.index, 
                           orientation='h', title="Average Performance by Category")
                st.plotly_chart(fig, use_container_width=True)
    
    # Top performers table
    st.subheader("🏆 Top 10 Performers")
    
    if 'performance_score' in df.columns:
        display_cols = ['username', 'field', 'performance_score']
        
        # Add relevant columns based on what's available
        if 'business_performance' in df.columns:
            display_cols.append('business_performance')
        if 'social_performance' in df.columns:
            display_cols.append('social_performance')
        if 'is_live' in df.columns:
            display_cols.append('is_live')
        
        top_performers = df.nlargest(10, 'performance_score')[display_cols].copy()
        
        # Format for display
        for col in ['performance_score', 'business_performance', 'social_performance']:
            if col in top_performers.columns:
                top_performers[col] = top_performers[col].round(3)
        
        st.dataframe(top_performers, use_container_width=True)
    else:
        st.info("Performance scores not available. Please process data with API for complete metrics.")

elif page == "Streamers Directory":
    st.title("👥 Streamers Directory")
    
    # Show data source info
    data_source = df['data_source'].iloc[0] if 'data_source' in df.columns and len(df) > 0 else 'unknown'
    is_enhanced = data_source == 'csv+api'
    
    # Search and filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Search username", "")
    
    with col2:
        if 'field' in df.columns:
            categories = ['All'] + sorted(df['field'].dropna().unique().tolist())
            selected_category = st.selectbox("📂 Category", categories)
        else:
            selected_category = 'All'
    
    with col3:
        if 'is_live' in df.columns and is_enhanced:
            live_filter = st.selectbox("🔴 Status", ["All", "Live Only", "Offline Only"])
        else:
            live_filter = "All"
            st.selectbox("🔴 Status", ["All"], disabled=True, help="Live status available after API processing")
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_term:
        filtered_df = filtered_df[filtered_df['username'].str.contains(search_term, case=False, na=False)]
    
    if 'field' in df.columns and selected_category != 'All':
        filtered_df = filtered_df[filtered_df['field'] == selected_category]
    
    if 'is_live' in df.columns and live_filter != 'All':
        if live_filter == "Live Only":
            filtered_df = filtered_df[filtered_df['is_live'] == True]
        elif live_filter == "Offline Only":
            filtered_df = filtered_df[filtered_df['is_live'] == False]
    
    # Sort options
    if is_enhanced:
        sort_options = [
            'performance_score', 'business_performance', 'social_performance', 
            'follower_count', 'engagement_rate', 'value', 'views',
            'live_engagement_rate', 'broadcaster_tier_score', 'value_per_follower'
        ]
    else:
        sort_options = ['value', 'views', 'total_watch_time', 'engagement_rate']
    
    # Filter to only available columns
    sort_options = [col for col in sort_options if col in filtered_df.columns]
    
    if sort_options:
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("📊 Sort by", sort_options)
        with col2:
            sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
        
        ascending = sort_order == "Ascending"
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
    
    # Display mode
    st.subheader("📋 Data View")
    
    # Define what columns to show based on data availability
    if is_enhanced:
        essential_cols = ['username', 'field']
        business_cols = ['views', 'value', 'engagement_rate', 'business_performance', 'value_per_view']
        social_cols = ['follower_count', 'is_live', 'social_performance', 'broadcaster_tier_score']
        twitter_cols = ['twitter_followers', 'twitter_engagement_rate', 'twitter_posting_frequency', 'twitter_activity_score']

        cross_platform_cols = ['value_per_follower', 'engagement_to_social_ratio', 'total_social_followers', 'cross_platform_reach']
        score_cols = ['performance_score', 'overall_rank']
        
        display_mode = st.radio("View Mode:", 
                            ["Essential + Scores", "Business Focus", "Social Focus", "Twitter Focus", "Cross-Platform", "All Data"], 
                            horizontal=True)
        
        if display_mode == "Essential + Scores":
            display_columns = essential_cols + score_cols + ['follower_count', 'is_live']
        elif display_mode == "Business Focus":
            display_columns = essential_cols + business_cols
        elif display_mode == "Social Focus":
            display_columns = essential_cols + social_cols + ['live_engagement_rate']
        elif display_mode == "Cross-Platform":
            display_columns = essential_cols + cross_platform_cols + score_cols
        elif display_mode == "Twitter Focus":
            display_columns = essential_cols + twitter_cols + score_cols
        else:  # All Data
            exclude_cols = ['id', 'created_at', 'updated_at', 'data_source', 'last_updated']
            display_columns = [col for col in filtered_df.columns if col not in exclude_cols]
    else:
        # Basic CSV view
        display_columns = ['username', 'field', 'views', 'value', 'total_watch_time', 
                          'isVerified', 'walletAddress', 'twitter']
        if 'engagement_rate' in filtered_df.columns:
            display_columns.extend(['engagement_rate', 'value_per_view', 'watch_efficiency'])
    
    # Filter to only available columns
    display_columns = [col for col in display_columns if col in filtered_df.columns]
    
    # Display results
    st.write(f"📋 Showing {len(filtered_df)} streamers")
    
    if display_columns and len(filtered_df) > 0:
        display_df = filtered_df[display_columns].copy()
        
        # Format numeric columns for display
        for col in display_df.columns:
            if col in ['views', 'value', 'total_watch_time', 'follower_count', 'viewer_count']:
                display_df[col] = display_df[col].apply(format_number)
            elif col in ['engagement_rate', 'value_per_view', 'watch_efficiency', 
                        'performance_score', 'business_performance', 'social_performance',
                        'live_engagement_rate', 'value_per_follower']:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(3)
            elif col in ['broadcaster_tier_score', 'overall_rank', 'business_rank', 'social_rank']:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(0).astype('Int64')
        
        st.dataframe(display_df, use_container_width=True)
        
        # Export option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Current View",
            data=csv,
            file_name=f"streamers_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Summary stats
    if len(filtered_df) > 0:
        st.subheader("📊 Summary")
        
        if is_enhanced:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_perf = filtered_df['performance_score'].mean() if 'performance_score' in filtered_df.columns else 0
                st.metric("Avg Performance", f"{avg_perf:.3f}")
            with col2:
                total_followers = filtered_df['follower_count'].sum() if 'follower_count' in filtered_df.columns else 0
                st.metric("Total Followers", format_number(total_followers))
            with col3:
                live_count = len(filtered_df[filtered_df['is_live'] == True]) if 'is_live' in filtered_df.columns else 0
                st.metric("Live Now", live_count)
            with col4:
                total_value = filtered_df['value'].sum() if 'value' in filtered_df.columns else 0
                st.metric("Total Value", format_number(total_value))
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                total_value = filtered_df['value'].sum() if 'value' in filtered_df.columns else 0
                st.metric("Total Value", format_number(total_value))
            with col2:
                total_views = filtered_df['views'].sum() if 'views' in filtered_df.columns else 0
                st.metric("Total Views", format_number(total_views))
            with col3:
                avg_engagement = filtered_df['engagement_rate'].mean() if 'engagement_rate' in filtered_df.columns else 0
                st.metric("Avg Engagement", format_number(avg_engagement))
    
    # Raw data explorer
    if st.checkbox("🔍 Show All Available Columns"):
        st.write("**All Data Columns:**")
        all_cols = [col for col in filtered_df.columns if col not in ['id']]
        st.write(f"Available: {', '.join(all_cols)}")
        
        if st.checkbox("Show raw data table"):
            st.dataframe(filtered_df[all_cols], use_container_width=True)

elif page == "Live Streams":
    st.title("🔴 Live Streams")
    
    if is_basic_format:
        st.info("📺 **Live stream data not available in basic format.**")
        st.markdown("""
        To view live stream information:
        1. Use **🚀 Process Initial CSV with API** to fetch live data from Twitch
        2. Or upload a processed CSV file that includes live stream data
        """)
        
        # Show basic streamer info instead
        st.subheader("📋 Streamer Directory")
        if 'field' in df.columns:
            category_counts = df['field'].value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Streamers by Category:**")
                for category, count in category_counts.items():
                    st.write(f"• {category}: {count} streamers")
            
            with col2:
                if len(df) > 0:
                    fig = px.bar(x=category_counts.values, y=category_counts.index, 
                               orientation='h', title="Category Distribution")
                    st.plotly_chart(fig, use_container_width=True)
        
    
    if 'is_live' in df.columns:
        live_streamers = df[df['is_live'] == True]
        
        if len(live_streamers) > 0:
            st.write(f"🎮 Currently {len(live_streamers)} streamers are live!")
            
            # Live streamers metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                total_live_viewers = live_streamers['viewer_count'].sum() if 'viewer_count' in live_streamers.columns else 0
                st.metric("Total Live Viewers", format_number(total_live_viewers))
            
            with col2:
                avg_viewers = live_streamers['viewer_count'].mean() if 'viewer_count' in live_streamers.columns else 0
                st.metric("Avg Viewers per Stream", format_number(avg_viewers))
            
            with col3:
                if 'current_game' in live_streamers.columns:
                    unique_games = live_streamers['current_game'].nunique()
                    st.metric("Games Being Played", unique_games)
            
            # Live streamers table
            display_columns = ['username', 'current_game', 'viewer_count', 'follower_count', 'stream_title']
            available_columns = [col for col in display_columns if col in live_streamers.columns]
            
            if available_columns:
                live_display = live_streamers[available_columns].copy()
                
                # Format for display
                if 'viewer_count' in live_display.columns:
                    live_display['viewer_count'] = live_display['viewer_count'].apply(format_number)
                if 'follower_count' in live_display.columns:
                    live_display['follower_count'] = live_display['follower_count'].apply(format_number)
                
                st.dataframe(live_display.sort_values('viewer_count', ascending=False) if 'viewer_count' in live_display.columns else live_display, 
                           use_container_width=True)
            
            # Games being played
            if 'current_game' in live_streamers.columns:
                st.subheader("🎮 Popular Games Right Now")
                game_counts = live_streamers['current_game'].value_counts()
                if len(game_counts) > 0:
                    fig = px.bar(x=game_counts.values, y=game_counts.index, 
                               orientation='h',
                               title="Most Streamed Games")
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("😴 No streamers are currently live.")
    
    else:
        st.warning("Live status data not available.")


# Footer info
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tips:**")
st.sidebar.markdown("""
- Upload your CSV to get started
- API processing enhances data automatically  
- Use search and filters to find streamers
""")

# Database info
if st.sidebar.checkbox("🗃️ Database Info"):
    st.sidebar.write(f"**Records:** {len(df)}")
    if not df.empty and 'data_source' in df.columns:
        data_source = df['data_source'].iloc[0]
        st.sidebar.write(f"**Source:** {data_source}")


if st.sidebar.button("🗑️ Clear All Data"):
    if st.sidebar.checkbox("⚠️ Confirm deletion"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM streamers")
        conn.commit()
        conn.close()
        st.success("Database cleared!")
        st.rerun()