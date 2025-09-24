#!/usr/bin/env python3
"""
Twitter Women Defining AI Content Discovery Agent
Multi-pattern approach: Find high-engagement content from Women Defining AI community on Twitter
Focus: AI learning journeys, career transitions, diversity in AI, responsible AI, community support
Strategy: Target authentic personal experiences from women and non-binary individuals in AI
Captures: All tweets (including retweets) from WDAI community members
Filters out: Crypto/web3, extreme politics, adult content, MLM schemes, aggressive sales pitches
Categories: AI Learner, Career Transition, Women in AI, AI Ethics, Community Support, Work-Life Balance, AI Applications
Prints to Terminal: hook, scan date, link, category, viral analysis, author info

COMMANDS:
python viral_twitter_WDAI.py          - Run normal scan
python viral_twitter_WDAI.py debug    - Run with detailed filtering breakdown and samples
python viral_twitter_WDAI.py reset    - Reset posted status (allow reprocessing)
python viral_twitter_WDAI.py clear    - Clear entire database
python viral_twitter_WDAI.py stats    - Show database statistics
python viral_twitter_WDAI.py help     - Show available commands

REQUIRED DEPENDENCIES:
pip install beautifulsoup4 requests langdetect tweepy openai

If langdetect is missing, language filtering will be skipped.
"""

import tweepy
print(f"📦 Tweepy version: {tweepy.__version__}")

# Handle OpenAI imports for different versions
try:
    from openai import OpenAI
    import openai
    print(f"📦 OpenAI version: {openai.__version__}")
except ImportError as e:
    print(f"❌ OpenAI import error: {e}")
    print("   Try: pip install --upgrade openai")
    import sys
    sys.exit(1)

import json
import re
import sqlite3
import os
import sys
import requests
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional
import time
import hashlib

# Optional dependencies - will gracefully handle if missing
try:
    from langdetect import detect, DetectorFactory
    # Set seed for consistent results
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

@dataclass
class WDAIPost:
    original_content: str  # AI-extracted story/insight
    platform_found: str
    engagement_score: float
    category: str
    classification_analysis: Dict
    timestamp: datetime
    source_url: Optional[str] = None
    raw_original_text: Optional[str] = None  # Original tweet text
    story_type: Optional[str] = None  # Type of story
    viral_analysis: Optional[Dict] = None  # Why this content works
    author_info: Optional[Dict] = None  # Author profile information

class TwitterWDAIAgent:
    def __init__(self, creds_file: str = "creds.json", debug_mode: bool = False):
        """Initialize with credentials from JSON file"""
        self.creds = self.load_credentials(creds_file)
        self.setup_apis()
        self.init_database()
        self.debug_mode = debug_mode
        
        # Debug tracking
        self.debug_stats = {
            'total_tweets': 0,
            'english_filter_fails': 0,
            'wdai_member_fails': 0,
            'engagement_filter_fails': 0,
            'exclusion_filter_fails': 0,
            'duplicate_filter_fails': 0,
            'wdai_indicator_fails': 0,
            'ai_classification_fails': 0,
            'samples': {
                'raw_tweets': [],
                'non_english': [],
                'non_wdai_users': [],
                'low_engagement': [],
                'excluded_content': [],
                'no_wdai_indicators': [],
                'ai_rejected': []
            }
        }
        
        # ENGAGEMENT THRESHOLDS - Lower for community content
        self.engagement_thresholds = {
            'twitter': {'min': 10, 'max': 50000},  # Lower minimum for inclusive community content
        }
        
        # EXCLUSION FILTERS - Modified for WDAI community
        self.exclusion_indicators = [
            # Crypto/Web3/Trading (keep all)
            'crypto', 'cryptocurrency', 'bitcoin', 'ethereum', 'blockchain', 'web3',
            'nft', 'defi', 'dao', 'token', 'coin', 'hodl', 'presale', '#web3', '#crypto',
            'trading', 'trader', 'portfolio', 'stocks', 'investment portfolio',
            'prop firm', 'prop trading', 'funded account', 'profit target', 'airdrop', 'staking',
            'btc', 'eth', '$sol', 'solana', 'doge', 'dogecoin', 'matic', 'avax',
            
            # Extreme political content only
            'extremist', 'hate speech', 'violence', 'terrorism', 'radical',
            
            # Adult/Sexual Content (keep all)
            'onlyfans', 'onlyfans.com', 'only fans', 'sexy', 'porn', 'porno', 'pornography',
            'sexual', 'xxx', 'adult content', 'nsfw', 'cam girl', 'cam model',
            'escort', 'sugar daddy', 'sugar baby', 'strip club', 'webcam', 'adult site',
            
            # MLM/Pyramid Schemes
            'mlm', 'multi level marketing', 'pyramid scheme', 'downline', 'upline',
            'get rich quick', 'make money fast', 'guaranteed income',
            
            # Direct money requests
            'paypal', 'cashapp', 'venmo', 'zelle', 'western union', 'moneygram',
            'send money', 'wire transfer', 'bitcoin payment', 'crypto payment',
            
            # Aggressive sales pitches only
            'limited time offer', 'act now', 'buy now', 'special price',
            'discount expires', 'hurry up', 'last chance', 'claim your spot'
        ]
        
        # TARGET WDAI CATEGORIES
        self.valid_wdai_categories = {
            'ai_learner': [
                'learning ai', 'ai bootcamp', 'ai course', 'certification',
                'first project', 'tutorial', 'study group', 'online learning',
                'machine learning', 'deep learning', 'neural network', 'python'
            ],
            'career_transition': [
                'career change', 'pivot', 'transition', 'new role', 'job search',
                'interview', 'networking', 'upskilling', 'reskilling',
                'mid-career', 'career switch', 'breaking into ai', 'career milestone'
            ],
            'women_in_ai': [
                'women in tech', 'diversity', 'inclusion', 'representation',
                'mentorship', 'role model', 'leadership', 'empowerment',
                'gender gap', 'breaking barriers', 'underrepresented', 'sisterhood'
            ],
            'ai_ethics': [
                'ethical ai', 'responsible ai', 'bias', 'fairness', 'transparency',
                'ai for good', 'social impact', 'inclusive ai', 'accessible ai',
                'ai governance', 'ai regulation', 'trustworthy ai', 'explainable ai'
            ],
            'community_support': [
                'community', 'support network', 'mentorship', 'collaboration',
                'workshop', 'event', 'conference', 'meetup', 'panel',
                'networking', 'wdai', 'women defining ai', 'slack community'
            ],
            'work_life_balance': [
                'balance', 'flexibility', 'remote work', 'boundaries',
                'self-care', 'wellness', 'productivity', 'time management',
                'working mom', 'parenting', 'mental health', 'burnout'
            ],
            'ai_applications': [
                'ai project', 'use case', 'implementation', 'solution',
                'innovation', 'automation', 'optimization', 'analysis',
                'ai tool', 'chatgpt', 'claude', 'generative ai', 'prompt engineering'
            ]
        }
        
        # WDAI role indicators - Inclusive of various backgrounds
        self.wdai_role_indicators = [
            # AI Learning & Career Development
            'ai learner', 'learning ai', 'studying ai', 'ai student', 'ai enthusiast',
            'career transition', 'career change', 'mid-career', 'career pivot',
            'upskilling', 'reskilling', 'professional development',
            
            # Women in AI/Tech Identity
            'women in ai', 'woman in tech', 'female founder', 'women in stem',
            'non-binary', 'she/her', 'they/them', 'diversity in ai', 'women defining ai',
            'wdai', 'underrepresented', 'minority in tech', 'woman leader',
            
            # AI Roles (broader than just technical)
            'ai product', 'ai strategy', 'ai ethics', 'responsible ai', 'ai policy',
            'ai consultant', 'ai researcher', 'ml engineer', 'data scientist',
            'ai educator', 'ai trainer', 'ai facilitator', 'ai advocate',
            'ai practitioner', 'ai professional', 'ai specialist',
            
            # Non-technical backgrounds transitioning to AI
            'marketer', 'designer', 'writer', 'educator', 'consultant',
            'business analyst', 'project manager', 'operations', 'hr professional',
            'healthcare', 'finance', 'legal', 'nonprofit', 'government',
            'teacher', 'artist', 'creative', 'entrepreneur', 'researcher',
            
            # AI Application Areas
            'ai for good', 'ethical ai', 'inclusive ai', 'accessible ai',
            'ai in education', 'ai in healthcare', 'ai for social impact',
            
            # Learning & Community
            'bootcamp', 'online course', 'certification', 'workshop',
            'community', 'mentor', 'mentee', 'support network',
            'student', 'learner', 'beginner', 'self-taught'
        ]

    def call_openai_api(self, prompt: str, max_tokens: int = 600, temperature: float = 0.1) -> str:
        """Call OpenAI API with compatibility for different versions"""
        try:
            # Try modern API structure first
            if hasattr(self.openai_client, 'chat') and hasattr(self.openai_client.chat, 'completions'):
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            
            # Try legacy API structure for older versions
            elif hasattr(self.openai_client, 'ChatCompletion'):
                response = self.openai_client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            
            # Last resort - very old version
            else:
                import openai
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"    ⚠️  OpenAI API call failed: {e}")
            raise

    def is_english_content(self, text: str) -> bool:
        """Check if content is in English using language detection"""
        if not LANGDETECT_AVAILABLE:
            print("      ⚠️  Language detection not available, assuming English")
            return True
        
        if not text or len(text.strip()) < 20:
            return False
        
        try:
            # Clean text for better detection
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text.strip())
            
            if len(clean_text) < 10:
                return False
            
            detected_lang = detect(clean_text)
            is_english = detected_lang == 'en'
            
            if not is_english:
                print(f"      🚫 Non-English content detected: {detected_lang}")
                print(f"          Text preview: {text[:100]}...")
            
            return is_english
            
        except Exception as e:
            print(f"      ⚠️  Language detection error: {e}, assuming English")
            return True

    def load_credentials(self, creds_file: str) -> Dict:
        """Load API credentials from JSON file"""
        if not os.path.exists(creds_file):
            print(f"❌ Credentials file '{creds_file}' not found!")
            print("Please create a creds.json file with your API keys.")
            self.create_sample_creds_file()
            sys.exit(1)
        
        try:
            with open(creds_file, 'r') as f:
                creds = json.load(f)
            
            required_keys = ['twitter_bearer_token', 'openai_api_key_LI']
            missing_keys = [key for key in required_keys if key not in creds]
            
            if missing_keys:
                print(f"❌ Missing required keys in {creds_file}: {missing_keys}")
                sys.exit(1)
                
            return creds
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {creds_file}: {e}")
            sys.exit(1)

    def create_sample_creds_file(self):
        """Create sample credentials file"""
        sample_creds = {
            "twitter_bearer_token": "your_twitter_bearer_token_here",
            "openai_api_key_LI": "your_openai_api_key_here",
            "slack_webhook": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        }
        
        with open('creds.json', 'w') as f:
            json.dump(sample_creds, f, indent=2)
        
        print("✅ Created sample creds.json file.")

    def setup_apis(self):
        """Initialize API clients"""
        try:
            print("🔧 Setting up API clients...")
            
            # Twitter - very basic initialization
            print("  🐦 Initializing Twitter client...")
            bearer_token = self.creds['twitter_bearer_token']
            
            if not bearer_token or bearer_token == "your_twitter_bearer_token_here":
                print("❌ Twitter bearer token not configured properly")
                sys.exit(1)
            
            # Create Twitter client with only the bearer token
            self.twitter_client = tweepy.Client(bearer_token=bearer_token)
            print("  ✅ Twitter client created")
            
            # OpenAI - try different initialization methods
            print("  🤖 Initializing OpenAI client...")
            openai_key = self.creds['openai_api_key_LI']
            
            if not openai_key or openai_key == "your_openai_api_key_here":
                print("❌ OpenAI API key not configured properly")
                sys.exit(1)
            
            try:
                # Try basic initialization first
                self.openai_client = OpenAI(api_key=openai_key)
                print("  ✅ OpenAI client created")
            except TypeError as e:
                print(f"  ⚠️  Standard OpenAI init failed: {e}")
                print("  🔄 Trying compatibility mode for older OpenAI versions...")
                
                try:
                    # For older OpenAI versions, try different approaches
                    import openai
                    print(f"      OpenAI version: {openai.__version__}")
                    
                    if hasattr(openai, 'OpenAI'):
                        # Try with minimal parameters
                        self.openai_client = openai.OpenAI(api_key=openai_key)
                        print("  ✅ OpenAI client created with compatibility mode")
                    else:
                        # Very old version - use legacy approach
                        openai.api_key = openai_key
                        self.openai_client = openai  # Use module directly
                        print("  ✅ OpenAI legacy client configured")
                        
                except Exception as e2:
                    print(f"  ❌ All OpenAI initialization methods failed: {e2}")
                    print("  💡 Try upgrading OpenAI: pip install --upgrade openai")
                    raise
            
            # Test language detection dependencies
            if LANGDETECT_AVAILABLE:
                print("✅ Language detection dependencies available")
            else:
                print(f"⚠️  Language detection dependencies missing")
                print("    Install with: pip install langdetect")
                print("    Note: Non-English content filtering will be skipped")
            
            print("✅ All APIs initialized successfully")
            
        except Exception as e:
            print(f"❌ Error setting up APIs: {e}")
            print(f"    Error type: {type(e).__name__}")
            print(f"    Error message: {str(e)}")
            
            # Check if it's an OpenAI-specific issue
            if "openai" in str(e).lower() or "client" in str(e).lower():
                print("\n🔍 Troubleshooting OpenAI API setup:")
                print("    1. Check your OpenAI API key is valid")
                print("    2. Try regenerating your OpenAI API key")
                print("    3. Check for any proxy/network settings")
                print(f"    4. OpenAI key format check: {openai_key[:7]}..." if openai_key and len(openai_key) > 7 else "    4. OpenAI key seems invalid")
                
                # Check for environment variables that might interfere
                import os
                proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
                active_proxies = {var: os.environ.get(var) for var in proxy_vars if os.environ.get(var)}
                if active_proxies:
                    print(f"    5. Active proxy settings detected: {active_proxies}")
                    print("       Try running: unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy")
                else:
                    print("    5. No proxy environment variables detected")
            
            sys.exit(1)

    def init_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('twitter_wdai_content.db')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS wdai_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_content TEXT,
                platform_found TEXT,
                engagement_score REAL,
                category TEXT,
                classification_analysis TEXT,
                timestamp TEXT,
                source_url TEXT,
                raw_original_text TEXT,
                content_hash TEXT UNIQUE,
                posted_to_slack BOOLEAN DEFAULT 0,
                story_type TEXT,
                viral_analysis TEXT,
                target_audience TEXT,
                author_info TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add missing columns to existing database if they don't exist
        try:
            cursor = self.conn.execute("PRAGMA table_info(wdai_content)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'story_type' not in columns:
                print("  🔄 Adding story_type column to existing database")
                self.conn.execute('ALTER TABLE wdai_content ADD COLUMN story_type TEXT')
            
            if 'viral_analysis' not in columns:
                print("  🔄 Adding viral_analysis column to existing database")
                self.conn.execute('ALTER TABLE wdai_content ADD COLUMN viral_analysis TEXT')
            
            if 'target_audience' not in columns:
                print("  🔄 Adding target_audience column to existing database")
                self.conn.execute('ALTER TABLE wdai_content ADD COLUMN target_audience TEXT')
            
            if 'author_info' not in columns:
                print("  🔄 Adding author_info column to existing database")
                self.conn.execute('ALTER TABLE wdai_content ADD COLUMN author_info TEXT')
            
            self.conn.commit()
            
        except Exception as e:
            print(f"  ⚠️  Warning updating database schema: {e}")
        
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_content_hash ON wdai_content(content_hash)
        ''')
        
        self.conn.commit()
        print("✅ Database initialized")

    def generate_content_hash(self, content: str) -> str:
        """Generate hash for deduplication"""
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()

    def reset_posted_status(self):
        """Reset posted_to_slack status for all records"""
        try:
            cursor = self.conn.execute('UPDATE wdai_content SET posted_to_slack = 0')
            affected_rows = cursor.rowcount
            self.conn.commit()
            print(f"✅ Reset posted status for {affected_rows} records - they can be posted to Slack again")
            return affected_rows
        except Exception as e:
            print(f"❌ Error resetting posted status: {e}")
            return 0

    def clear_database(self):
        """Clear all records from database (use with caution)"""
        try:
            cursor = self.conn.execute('DELETE FROM wdai_content')
            affected_rows = cursor.rowcount
            self.conn.commit()
            print(f"✅ Cleared {affected_rows} records from database")
            return affected_rows
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            return 0

    def get_database_stats(self):
        """Get statistics about the database"""
        try:
            cursor = self.conn.execute('SELECT COUNT(*) FROM wdai_content')
            total_records = cursor.fetchone()[0]
            
            cursor = self.conn.execute('SELECT COUNT(*) FROM wdai_content WHERE posted_to_slack = 1')
            posted_records = cursor.fetchone()[0]
            
            cursor = self.conn.execute('SELECT COUNT(*) FROM wdai_content WHERE posted_to_slack = 0')
            unposted_records = cursor.fetchone()[0]
            
            print(f"📊 Database Stats:")
            print(f"  Total records: {total_records}")
            print(f"  Posted to Slack: {posted_records}")
            print(f"  Not posted: {unposted_records}")
            
            return {
                'total': total_records,
                'posted': posted_records,
                'unposted': unposted_records
            }
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return None

    def is_already_posted_to_slack(self, content_hash: str) -> bool:
        """Check if content was already posted to Slack"""
        try:
            cursor = self.conn.execute(
                'SELECT id FROM wdai_content WHERE content_hash = ? AND posted_to_slack = 1',
                (content_hash,)
            )
            return cursor.fetchone() is not None
        except Exception as e:
            print(f"⚠️  Database query warning: {e}")
            return False

    def has_exclusion_indicators(self, text: str) -> bool:
        """Check if text contains excluded topics"""
        text_lower = text.lower()
        
        # Check individual exclusion keywords
        for indicator in self.exclusion_indicators:
            # For single words, use word boundaries to avoid partial matches
            if len(indicator.split()) == 1 and not indicator.startswith('#'):
                pattern = r'\b' + re.escape(indicator) + r'\b'
                if re.search(pattern, text_lower):
                    print(f"    🚫 Excluded: '{indicator}' detected as full word")
                    return True
            else:
                # For phrases and hashtags, use exact matching
                if indicator in text_lower:
                    print(f"    🚫 Excluded: '{indicator}' detected")
                    return True
        
        # Check for aggressive sales pitch patterns only
        sales_pitch_patterns = [
            r'limited time offer',
            r'act now before',
            r'buy now and save',
            r'special price today only',
            r'discount expires',
            r'hurry up and',
            r'last chance to',
            r'claim your spot before'
        ]
        
        for pattern in sales_pitch_patterns:
            if re.search(pattern, text_lower):
                print(f"    🚫 Excluded: Aggressive sales pitch pattern detected")
                return True
                
        return False

    def is_valid_category(self, category: str) -> bool:
        """Check if category is one of our valid WDAI categories"""
        return category in self.valid_wdai_categories.keys()

    def has_wdai_indicators(self, content: str) -> bool:
        """Check for WDAI-relevant indicators before AI classification"""
        if not content or len(content) < 50:
            return False
            
        content_lower = content.lower()
        
        # Strong WDAI indicators that should pass on their own
        strong_indicators = [
            'women in ai', 'diversity in tech', 'career transition', 'ai career',
            'learned ai', 'ai certification', 'responsible ai', 'ethical ai',
            'ai bootcamp', 'breaking into ai', 'pivot to tech', 'wdai',
            'women defining ai', 'imposter syndrome', 'work-life balance',
            'mentorship', 'ai for good', 'inclusive ai', 'bias in ai'
        ]
        
        # Check for strong indicators first - if found, pass immediately
        for indicator in strong_indicators:
            if indicator in content_lower:
                print(f"    ✅ Strong WDAI indicator found: '{indicator}'")
                return True
        
        # WDAI business/career words
        wdai_words = [
            # Career development
            'career', 'job', 'role', 'position', 'promotion', 'transition',
            'interview', 'hired', 'opportunity', 'skill', 'learning',
            
            # AI/Tech terms
            'ai', 'ml', 'machine learning', 'artificial intelligence',
            'data', 'algorithm', 'model', 'python', 'code', 'tech',
            'chatgpt', 'claude', 'prompt', 'generative',
            
            # Community & support
            'community', 'mentor', 'network', 'support', 'workshop',
            'conference', 'meetup', 'event', 'speaker', 'panel',
            
            # Personal growth
            'journey', 'growth', 'challenge', 'achievement', 'milestone',
            'confidence', 'leadership', 'impact', 'contribution',
            
            # Learning
            'course', 'bootcamp', 'certification', 'tutorial', 'study',
            'practice', 'project', 'portfolio', 'beginner', 'intermediate'
        ]
        
        # Count indicators (lower threshold for WDAI)
        indicator_count = sum(1 for word in wdai_words if word in content_lower)
        
        # Lower threshold to 2 for inclusivity
        has_indicators = indicator_count >= 2
        
        if not has_indicators:
            print(f"    ⚡ Skipping - insufficient WDAI indicators ({indicator_count}/2+)")
        
        return has_indicators

    def has_wdai_language_in_content(self, tweet_text: str) -> bool:
        """Check if tweet content contains WDAI-relevant language"""
        if not tweet_text:
            return False
            
        text_lower = tweet_text.lower()
        
        # WDAI content indicators
        wdai_content_indicators = [
            # AI Learning Journey
            'started learning ai', 'my ai journey', 'ai bootcamp', 'ai course',
            'first ai project', 'learning python', 'learning machine learning',
            'career transition to ai', 'pivot to ai', 'breaking into ai',
            
            # Overcoming Challenges
            'imposter syndrome', 'self-doubt', 'confidence', 'overcame',
            'breakthrough', 'aha moment', 'finally understood', 'clicked for me',
            
            # Community & Support
            'supportive community', 'found my tribe', 'women supporting women',
            'mentorship', 'role model', 'inspired by', 'grateful for',
            
            # Work-Life Integration
            'work-life balance', 'working mom', 'balancing', 'juggling',
            'remote work', 'flexibility', 'boundaries',
            
            # Diversity & Inclusion
            'diversity in ai', 'representation matters', 'inclusive ai',
            'bias in ai', 'ethical ai', 'responsible ai', 'ai for good',
            
            # Career Advancement
            'got promoted', 'new role', 'career milestone', 'salary increase',
            'negotiated', 'leadership position', 'speaking at', 'invited to',
            
            # Women in AI specific
            'women in ai', 'woman in tech', 'female founder', 'women defining ai',
            'gender gap', 'breaking barriers', 'underrepresented'
        ]
        
        # Count indicators
        indicator_count = sum(1 for indicator in wdai_content_indicators if indicator in text_lower)
        
        # Even one strong indicator suggests WDAI content
        return indicator_count >= 1

    def is_wdai_member(self, user_info: Dict, tweet_text: str = "") -> bool:
        """Check if user appears to be part of WDAI community - INCLUSIVE APPROACH"""
        
        # Content-first approach
        # If tweet contains WDAI language, pass regardless of bio
        if tweet_text and self.has_wdai_language_in_content(tweet_text):
            print(f"    ✅ WDAI content detected in tweet - auto-approved")
            return True
        
        # Fallback to bio analysis if no strong WDAI content in tweet
        if not user_info:
            print(f"    ⚠️  No user data and no WDAI content in tweet")
            return False
        
        bio = user_info.get('description', '').lower() if user_info.get('description') else ''
        name = user_info.get('name', '').lower() if user_info.get('name') else ''
        
        # If no bio available, be more permissive
        if not bio:
            print(f"    ⚠️  No bio available, checking for any AI/tech context...")
            # Very basic check - if they're tweeting about AI topics, probably relevant
            ai_context = any(word in tweet_text.lower() for word in [
                'ai', 'artificial intelligence', 'machine learning', 'women', 'diversity'
            ]) if tweet_text else False
            return ai_context
        
        # Combine bio and name for analysis
        profile_text = f"{bio} {name}"
        
        # Check for WDAI role indicators
        wdai_score = 0
        for role in self.wdai_role_indicators:
            if role in profile_text:
                wdai_score += 1
        
        # Additional WDAI indicators
        wdai_specific_indicators = [
            'ai', 'artificial intelligence', 'machine learning', 'data',
            'women', 'diversity', 'inclusion', 'ethics', 'responsible',
            'learning', 'student', 'career', 'transition', 'community'
        ]
        
        for indicator in wdai_specific_indicators:
            if indicator in profile_text:
                wdai_score += 0.5
        
        # Check for company/brand indicators (we want to exclude these)
        company_indicators = [
            'official', 'we are', 'we help', 'our team', 'our company',
            'follow us', 'contact us', 'visit our', 'inc.', 'llc', 'ltd'
        ]
        
        for indicator in company_indicators:
            if indicator in profile_text:
                print(f"    🚫 Company/brand account detected: '{indicator}'")
                return False
        
        # Very permissive threshold - if they have any WDAI context, let them through
        is_wdai = wdai_score >= 0.5  # Very low threshold - let content quality decide
        
        if not is_wdai:
            print(f"    🚫 Not WDAI community member (score: {wdai_score})")
            print(f"        Bio: {bio[:100]}...")
        else:
            print(f"    ✅ Passed WDAI filter (score: {wdai_score})")
        
        return is_wdai

    def classify_wdai_content(self, text: str, author_info: Dict = None) -> Optional[Dict]:
        """Use AI to classify if content is relevant to WDAI community"""
        try:
            # Pre-filter: Check for WDAI indicators to save API calls
            if not self.has_wdai_indicators(text):
                return None
            
            # Check if content is in English before processing
            if not self.is_english_content(text):
                return None
            
            author_context = ""
            if author_info:
                bio = author_info.get('description', '')
                if bio:
                    author_context = f"\nAuthor bio: {bio[:200]}"
            
            prompt = f"""
            EXTRACT (don't create) REAL STORIES/INSIGHTS from this Twitter post relevant to the Women Defining AI community.
            Focus on AI learning journeys, career transitions, diversity & inclusion, community support, and responsible AI.

            Content: "{text[:800]}"{author_context}

            STRICT RULES:
            1. ONLY extract stories/insights that are EXPLICITLY in the content
            2. NEVER create, assume, or infer stories not directly stated
            3. PRIORITIZE personal experiences, learning journeys, community stories
            4. INCLUDE discussions about ethics, diversity, work-life balance, challenges
            5. If no relevant WDAI content exists, return "is_wdai_relevant": false

            Look for EXPLICITLY STATED:
            - AI learning experiences and educational journeys
            - Career transitions into AI/tech
            - Diversity, inclusion, representation in AI
            - Ethical AI, responsible AI discussions
            - Community support, mentorship, networking
            - Work-life balance, personal challenges
            - AI tools, applications, projects

            ABSOLUTELY EXCLUDE:
            - Crypto/blockchain content
            - Extreme political content
            - Adult/sexual content
            - MLM/pyramid schemes
            - Aggressive sales pitches

            VALID CATEGORIES (only if content explicitly fits):
            - ai_learner: AI learning journeys, courses, certifications
            - career_transition: Career changes, job search, upskilling
            - women_in_ai: Diversity, representation, empowerment
            - ai_ethics: Responsible AI, bias, fairness, social impact
            - community_support: Mentorship, events, networking
            - work_life_balance: Balance, wellness, personal challenges
            - ai_applications: Projects, tools, implementations

            Return ONLY this JSON:
            {{
                "is_wdai_relevant": true/false,
                "category": "category_name or none",
                "hook": "EXACT story/insight from content or null",
                "story_type": "learning|career|diversity|ethics|community|balance|application|insight",
                "confidence": "high|medium|low",
                "reasoning": "why this is/isn't relevant to WDAI community (max 30 words)",
                "is_personal_story": true/false,
                "is_educational": true/false
            }}
            """
            
            response_content = self.call_openai_api(prompt, max_tokens=600, temperature=0.1)
            
            try:
                # Remove markdown code blocks if present
                result = response_content
                if result.startswith('```'):
                    result = re.sub(r'```\w*\n?', '', result).replace('```', '').strip()
                
                # Try to extract JSON from the response if there's extra text
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    result = json_match.group(0)
                
                # Clean up common JSON issues
                result = result.replace('true/false', 'false')
                result = re.sub(r',\s*}', '}', result)
                
                # Handle truncated JSON
                if not result.endswith('}'):
                    if '"reasoning":' in result and not result.endswith('"'):
                        result = result.rstrip('.') + '","is_educational":false}'
                    elif '"reasoning":"' in result:
                        last_comma = result.rfind(',')
                        if last_comma > 0:
                            result = result[:last_comma] + ',"is_educational":false}'
                        else:
                            result += ',"is_educational":false}'
                    else:
                        result += '}'
                
                parsed = json.loads(result)
                
                # Validation: Must be WDAI relevant
                if (parsed.get('is_wdai_relevant') and 
                    self.is_valid_category(parsed.get('category', 'none')) and
                    parsed.get('category') != 'none' and
                    parsed.get('hook') and parsed.get('hook') != 'null'):
                    
                    return parsed
                else:
                    # Log why it was rejected
                    if not parsed.get('is_wdai_relevant'):
                        print(f"    🚫 Rejected: Not WDAI relevant")
                    elif not self.is_valid_category(parsed.get('category', 'none')):
                        print(f"    🚫 Rejected: Invalid category '{parsed.get('category')}'")
                    elif parsed.get('category') == 'none':
                        print(f"    🚫 Rejected: Category is 'none'")
                    elif not parsed.get('hook') or parsed.get('hook') == 'null':
                        print(f"    🚫 Rejected: No valid hook extracted")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"    ⚠️  Could not parse AI response: {e}")
                print(f"    📝 Raw response: {result[:300]}...")
                return None
                
        except Exception as e:
            print(f"    ⚠️  AI classification error: {e}")
            return None

    def analyze_viral_factors(self, content: str, engagement_score: float, author_info: Dict = None) -> Dict:
        """Analyze why a post works and identify target audience"""
        try:
            author_context = ""
            if author_info:
                bio = author_info.get('description', '')
                followers = author_info.get('public_metrics', {}).get('followers_count', 0)
                if bio:
                    author_context = f"\nAuthor: {bio[:100]} ({followers} followers)"
            
            prompt = f"""
            Analyze this WDAI community content and explain why it resonates:
            Content: "{content}"
            Engagement: {engagement_score}{author_context}
            
            Provide analysis in this exact JSON format:
            {{
                "context_summary": "Brief 1-sentence summary of what the post is about",
                "why_this_works": "2-3 sentence explanation of why this content resonates with WDAI community",
                "target_audience": "Primary audience (e.g., 'AI learners', 'Women in tech', 'Career transitioners')"
            }}
            
            Focus on:
            - Community connection and support
            - Relatability of challenges and experiences
            - Educational value and practical insights
            - Representation and diversity aspects
            - Authenticity and vulnerability
            """
            
            response_content = self.call_openai_api(prompt, max_tokens=300, temperature=0.2)
            
            # Clean JSON
            result = response_content
            if result.startswith('```'):
                result = re.sub(r'```\w*', '', result).replace('```', '').strip()
            
            # Extract JSON if wrapped in text
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                result = json_match.group(0)
            
            try:
                analysis = json.loads(result)
                return analysis
            except json.JSONDecodeError:
                # Fallback analysis
                return {
                    "context_summary": "WDAI community story or insight",
                    "why_this_works": "Authentic sharing that resonates with community values",
                    "target_audience": "Women Defining AI community members"
                }
                
        except Exception as e:
            print(f"    ⚠️  Error analyzing viral factors: {e}")
            return {
                "context_summary": "WDAI community content",
                "why_this_works": "Relevant to AI learning and community support",
                "target_audience": "WDAI community"
            }

    def handle_rate_limit(self, retry_func, *args, **kwargs):
        """Handle API rate limiting with exponential backoff"""
        max_retries = 1
        base_delay = 900  # 15 minutes for Basic plan reset
        
        for attempt in range(max_retries):
            try:
                return retry_func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    if attempt < max_retries - 1:
                        delay = base_delay
                        print(f"  ⚠️  Rate limited! Basic plan: waiting {delay//60} minutes for reset")
                        time.sleep(delay)
                    else:
                        print(f"  ❌ Rate limit exceeded - Basic plan has strict limits")
                        raise
                else:
                    raise
        
        return None

    def scan_twitter_for_wdai_content(self, hours_back: int = 24) -> List[WDAIPost]:
        """Scan Twitter for WDAI community content"""
        
        # WDAI-SPECIFIC CONTENT DISCOVERY
        search_terms = [
            # 1. WOMEN IN AI & DIVERSITY
            '"women in ai" OR "women defining ai" OR "wdai" OR "female founder" ai',
            
            # 2. AI LEARNING & EDUCATION
            '"learning ai" OR "ai bootcamp" OR "ai course" OR "started learning" machine learning',
            
            # 3. CAREER TRANSITIONS
            '"career transition" OR "breaking into ai" OR "pivot to tech" OR "career change" ai',
            
            # 4. AI ETHICS & RESPONSIBLE AI
            '"ethical ai" OR "responsible ai" OR "bias in ai" OR "inclusive ai" OR "ai for good"',
            
            # 5. COMMUNITY & MENTORSHIP
            '"ai community" OR "women supporting women" OR "mentorship" OR "role model" tech',
            
            # 6. WORK-LIFE BALANCE & CHALLENGES
            '"imposter syndrome" OR "work-life balance" OR "working mom" ai tech',
            
            # 7. AI TOOLS & APPLICATIONS
            '"chatgpt" OR "claude" OR "generative ai" OR "prompt engineering" learning',
            
            # 8. SUCCESS STORIES & MILESTONES
            '"got promoted" OR "new role" OR "career milestone" OR "achievement" ai women'
        ]
        
        print(f"🌟 Scanning Twitter for Women Defining AI Community Content")
        print(f"📊 Using {len(search_terms)} WDAI-specific search patterns")
        print(f"🎯 Target: AI learners, career transitioners, women in tech, ethical AI advocates")
        print(f"🇺🇸 Language: English content only")
        print(f"🚫 Excluding: Crypto, extreme politics, adult content, MLM schemes")
        print(f"⚠️  Basic Plan: 60 requests/15min = ~4 requests/min max")
        print(f"⏱️  Using 20-second delays to stay under rate limits")
        
        wdai_posts = []
        seen_tweets = set()
        seen_content_hashes = set()
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        successful_searches = 0
        total_tweets_processed = 0
        
        for i, search_term in enumerate(search_terms):
            try:
                print(f"  🔍 Searching ({i+1}/{len(search_terms)}): {search_term}")
                
                # Rate limiting for Basic plan
                if i > 0:
                    print(f"    ⏱️  Basic Plan rate limiting pause (20 seconds)...")
                    time.sleep(20)
                
                try:
                    response = self.handle_rate_limit(
                        self.twitter_client.search_recent_tweets,
                        query=search_term,
                        tweet_fields=['created_at', 'public_metrics', 'author_id'],
                        user_fields=['description', 'name', 'public_metrics', 'verified'],
                        expansions=['author_id'],
                        max_results=100,
                        start_time=start_time,
                        end_time=end_time
                    )
                except Exception as e:
                    print(f"  ❌ Failed after retries: {e}")
                    continue
                
                if not response.data:
                    print(f"  📊 No tweets found for this search")
                    continue
                
                # Get user data from includes
                users = {}
                if hasattr(response, 'includes') and response.includes and hasattr(response.includes, 'users'):
                    for user in response.includes.users:
                        users[user.id] = {
                            'name': user.name,
                            'description': user.description,
                            'verified': getattr(user, 'verified', False),
                            'public_metrics': user.public_metrics._asdict() if hasattr(user, 'public_metrics') else {}
                        }
                
                # Show top 10 tweets from this search with links
                print(f"  📋 Top 10 tweets from this search:")
                sorted_tweets = sorted(response.data, key=lambda t: (
                    t.public_metrics['like_count'] + 
                    t.public_metrics['retweet_count'] * 2 + 
                    t.public_metrics['reply_count'] + 
                    t.public_metrics['quote_count']
                ), reverse=True)
                
                for idx, tweet in enumerate(sorted_tweets[:10], 1):
                    metrics = tweet.public_metrics
                    engagement = metrics['like_count'] + metrics['retweet_count'] * 2 + metrics['reply_count'] + metrics['quote_count']
                    tweet_url = f"https://twitter.com/user/status/{tweet.id}"
                    user_info = users.get(tweet.author_id, {})
                    author_name = user_info.get('name', 'Unknown')
                    
                    print(f"    {idx}. [{engagement} eng] {author_name}")
                    print(f"       Text: {tweet.text[:120]}{'...' if len(tweet.text) > 120 else ''}")
                    print(f"       Link: {tweet_url}")
                    
                    # Quick filter status
                    is_english = self.is_english_content(tweet.text)
                    is_wdai = self.is_wdai_member(user_info, tweet.text) if is_english else False
                    has_indicators = self.has_wdai_indicators(tweet.text) if is_wdai else False
                    
                    status_parts = []
                    if not is_english:
                        status_parts.append("❌ Non-English")
                    elif not is_wdai:
                        status_parts.append("❌ Not WDAI community")
                    elif not has_indicators:
                        status_parts.append("❌ No WDAI indicators")
                    else:
                        status_parts.append("✅ Passed pre-filters")
                    
                    print(f"       Status: {' | '.join(status_parts)}")
                    print()
                
                tweet_count = 0
                for tweet in response.data:
                    tweet_count += 1
                    self.debug_stats['total_tweets'] += 1
                    
                    if tweet.id in seen_tweets:
                        continue
                    seen_tweets.add(tweet.id)
                    
                    # Store raw sample for debug
                    if len(self.debug_stats['samples']['raw_tweets']) < 10:
                        self.debug_stats['samples']['raw_tweets'].append({
                            'text': tweet.text[:200],
                            'engagement': tweet.public_metrics['like_count'] + tweet.public_metrics['retweet_count'],
                            'author_id': tweet.author_id
                        })
                    
                    # Skip retweets to avoid duplicates
                    if tweet.text.startswith('RT @'):
                        if self.debug_mode:
                            print(f"    🔄 Skipping retweet - focusing on original content only")
                        continue
                    
                    # Check if tweet text is in English first
                    if not self.is_english_content(tweet.text):
                        self.debug_stats['english_filter_fails'] += 1
                        if len(self.debug_stats['samples']['non_english']) < 5:
                            self.debug_stats['samples']['non_english'].append(tweet.text[:200])
                        continue
                    
                    # Get user info for this tweet
                    user_info = users.get(tweet.author_id, {})
                    
                    # Check if author is WDAI community member - INCLUSIVE approach
                    if not self.is_wdai_member(user_info, tweet.text):
                        self.debug_stats['wdai_member_fails'] += 1
                        if len(self.debug_stats['samples']['non_wdai_users']) < 10:
                            self.debug_stats['samples']['non_wdai_users'].append({
                                'text': tweet.text[:200],
                                'bio': user_info.get('description', 'No bio')[:100],
                                'name': user_info.get('name', 'No name')
                            })
                        continue
                    
                    metrics = tweet.public_metrics
                    engagement_score = (
                        metrics['like_count'] +
                        metrics['retweet_count'] * 2 +
                        metrics['reply_count'] +
                        metrics['quote_count']
                    )
                    
                    # Apply engagement thresholds
                    threshold_key = 'twitter'
                    min_eng = self.engagement_thresholds[threshold_key]['min']
                    max_eng = self.engagement_thresholds[threshold_key]['max']
                    
                    if not (min_eng <= engagement_score <= max_eng):
                        self.debug_stats['engagement_filter_fails'] += 1
                        if len(self.debug_stats['samples']['low_engagement']) < 5:
                            self.debug_stats['samples']['low_engagement'].append({
                                'text': tweet.text[:200],
                                'engagement': engagement_score
                            })
                        if self.debug_mode:
                            print(f"    📊 Tweet {tweet.id}: {engagement_score} engagement (outside range {min_eng}-{max_eng})")
                        continue
                    
                    # Skip if has exclusion indicators
                    if self.has_exclusion_indicators(tweet.text):
                        self.debug_stats['exclusion_filter_fails'] += 1
                        if len(self.debug_stats['samples']['excluded_content']) < 5:
                            self.debug_stats['samples']['excluded_content'].append(tweet.text[:200])
                        continue
                    
                    # Check for duplicates
                    content_hash = self.generate_content_hash(tweet.text)
                    if content_hash in seen_content_hashes or self.is_already_posted_to_slack(content_hash):
                        self.debug_stats['duplicate_filter_fails'] += 1
                        if self.debug_mode:
                            print(f"    ⏭️  Duplicate content - skipping")
                        continue
                    
                    # Check for WDAI indicators
                    if not self.has_wdai_indicators(tweet.text):
                        self.debug_stats['wdai_indicator_fails'] += 1
                        if len(self.debug_stats['samples']['no_wdai_indicators']) < 10:
                            self.debug_stats['samples']['no_wdai_indicators'].append({
                                'text': tweet.text[:200],
                                'author': user_info.get('name', 'Unknown')
                            })
                        continue
                    
                    if self.debug_mode:
                        print(f"    📊 Processing: {engagement_score} engagement from WDAI community")
                        print(f"        Text: {tweet.text[:100]}...")
                    
                    # AI Classification
                    classification = self.classify_wdai_content(tweet.text, user_info)
                    
                    if classification:
                        tweet_url = f"https://twitter.com/user/status/{tweet.id}"
                        print(f"    ✅ WDAI content found: {classification['category']}")
                        print(f"    🔗 URL: {tweet_url}")
                        print(f"    📝 Hook: {classification['hook'][:100]}{'...' if len(classification['hook']) > 100 else ''}")
                        
                        # Viral factors analysis for high-engagement content
                        viral_analysis = None
                        if engagement_score >= 50:
                            print(f"    🔍 Analyzing viral factors...")
                            viral_analysis = self.analyze_viral_factors(
                                classification['hook'], 
                                engagement_score,
                                user_info
                            )
                        
                        seen_content_hashes.add(content_hash)
                        
                        wdai_post = WDAIPost(
                            original_content=classification['hook'],
                            platform_found='twitter',
                            engagement_score=engagement_score,
                            category=classification['category'],
                            classification_analysis=classification,
                            timestamp=tweet.created_at,
                            source_url=tweet_url,
                            raw_original_text=tweet.text,
                            story_type=classification.get('story_type'),
                            viral_analysis=viral_analysis,
                            author_info=user_info
                        )
                        
                        wdai_posts.append(wdai_post)
                    else:
                        self.debug_stats['ai_classification_fails'] += 1
                        if len(self.debug_stats['samples']['ai_rejected']) < 10:
                            self.debug_stats['samples']['ai_rejected'].append({
                                'text': tweet.text[:200],
                                'author': user_info.get('name', 'Unknown'),
                                'bio': user_info.get('description', 'No bio')[:50]
                            })
                        if self.debug_mode:
                            print(f"    ❌ Not valid WDAI content")
                            print(f"        Text: {tweet.text[:100]}...")
                
                print(f"  📊 Processed {tweet_count} tweets from this search")
                total_tweets_processed += tweet_count
                successful_searches += 1
                    
            except Exception as e:
                print(f"  ⚠️  Error with search term: {e}")
                continue
        
        # Sort by engagement score
        wdai_posts.sort(key=lambda x: x.engagement_score, reverse=True)
        
        print(f"\n📊 SCAN SUMMARY:")
        print(f"  ✅ Successful searches: {successful_searches}/{len(search_terms)}")
        print(f"  📝 Total tweets processed: {total_tweets_processed}")
        print(f"  🎯 WDAI content found: {len(wdai_posts)}")
        print(f"  🇺🇸 Language filter: English only")
        print(f"  👥 WDAI community filter: Applied")
        print(f"  🔄 Retweets excluded: Focus on original content")
        
        # Show debug statistics if enabled
        if self.debug_mode:
            self.show_debug_stats()
        
        return wdai_posts

    def show_debug_stats(self):
        """Show detailed breakdown of filtering results"""
        print(f"\n🔍 DEBUG FILTER BREAKDOWN:")
        print(f"  📊 Total tweets: {self.debug_stats['total_tweets']}")
        print(f"  🌍 Non-English filtered: {self.debug_stats['english_filter_fails']}")
        print(f"  👤 Non-WDAI community: {self.debug_stats['wdai_member_fails']}")
        print(f"  📉 Low engagement: {self.debug_stats['engagement_filter_fails']}")
        print(f"  🚫 Excluded content: {self.debug_stats['exclusion_filter_fails']}")
        print(f"  🔄 Duplicates: {self.debug_stats['duplicate_filter_fails']}")
        print(f"  💡 No WDAI indicators: {self.debug_stats['wdai_indicator_fails']}")
        print(f"  🤖 AI rejected: {self.debug_stats['ai_classification_fails']}")
        
        # Show samples
        print(f"\n📝 SAMPLE TWEETS BY FILTER STAGE:")
        
        if self.debug_stats['samples']['raw_tweets']:
            print(f"\n1️⃣ RAW TWEETS (first 10):")
            for i, sample in enumerate(self.debug_stats['samples']['raw_tweets'][:5], 1):
                print(f"  {i}. [{sample['engagement']} eng] {sample['text'][:150]}...")
        
        if self.debug_stats['samples']['non_wdai_users']:
            print(f"\n❌ NON-WDAI COMMUNITY CONTENT (sample):")
            for i, sample in enumerate(self.debug_stats['samples']['non_wdai_users'][:5], 1):
                print(f"  {i}. {sample['name']}")
                print(f"     Bio: {sample['bio']}")
                print(f"     Tweet: {sample['text'][:100]}...")
                print(f"     Reason: No WDAI language in content AND no AI/diversity bio")
                print()
        
        if self.debug_stats['samples']['low_engagement']:
            print(f"\n📉 LOW ENGAGEMENT TWEETS:")
            for i, sample in enumerate(self.debug_stats['samples']['low_engagement'][:3], 1):
                print(f"  {i}. [{sample['engagement']} eng] {sample['text'][:150]}...")
        
        if self.debug_stats['samples']['no_wdai_indicators']:
            print(f"\n💡 NO WDAI INDICATORS (from community members):")
            for i, sample in enumerate(self.debug_stats['samples']['no_wdai_indicators'][:5], 1):
                print(f"  {i}. {sample['author']}: {sample['text'][:150]}...")
        
        if self.debug_stats['samples']['ai_rejected']:
            print(f"\n🤖 AI REJECTED (had WDAI indicators but failed classification):")
            for i, sample in enumerate(self.debug_stats['samples']['ai_rejected'][:5], 1):
                print(f"  {i}. {sample['author']} ({sample['bio'][:30]}...)")
                print(f"     Tweet: {sample['text'][:150]}...")
                print()

    def save_to_database(self, posts: List[WDAIPost]):
        """Save posts to database"""
        saved_count = 0
        for post in posts:
            try:
                content_hash = self.generate_content_hash(post.raw_original_text)
                
                self.conn.execute('''
                    INSERT OR IGNORE INTO wdai_content 
                    (original_content, platform_found, engagement_score, category, 
                     classification_analysis, timestamp, source_url, raw_original_text, 
                     content_hash, posted_to_slack, story_type, viral_analysis, 
                     target_audience, author_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                ''', (
                    post.original_content,
                    post.platform_found,
                    post.engagement_score,
                    post.category,
                    json.dumps(post.classification_analysis),
                    post.timestamp.isoformat(),
                    post.source_url,
                    post.raw_original_text,
                    content_hash,
                    post.story_type,
                    json.dumps(post.viral_analysis) if post.viral_analysis else None,
                    post.viral_analysis.get('target_audience') if post.viral_analysis else None,
                    json.dumps(post.author_info) if post.author_info else None
                ))
                saved_count += 1
            except Exception as e:
                print(f"    ⚠️  Error saving: {e}")
        
        self.conn.commit()
        print(f"💾 Saved {saved_count} posts to database")

    def format_slack_message(self, posts: List[WDAIPost]) -> list:
        """Format posts for Slack"""
        if not posts:
            return [{
                "text": "🤔 No WDAI community content found this scan",
                "username": "Twitter WDAI Agent",
                "icon_emoji": ":woman-technologist:"
            }]
        
        messages = []
        
        # Summary message
        summary_text = f"🌟 Found {len(posts)} Women Defining AI community posts (English only, no retweets)"
        
        messages.append({
            "text": summary_text,
            "username": "Twitter WDAI Agent",
            "icon_emoji": ":woman-technologist:"
        })
        
        # Individual posts
        sorted_posts = sorted(posts, key=lambda x: x.engagement_score, reverse=True)
        
        for post in sorted_posts[:20]:  # Top 20 posts
            # First 30 words from raw text
            raw_text = post.raw_original_text or post.original_content
            words = raw_text.split()
            first_30_words = ' '.join(words[:30])
            if len(words) > 30:
                first_30_words += "..."
            
            category_display = post.category.replace('_', ' ').title()
            story_type_display = post.story_type.replace('_', ' ').title() if post.story_type else "Story"
            
            # Author info
            author_name = "WDAI Community Member"
            author_bio = ""
            if post.author_info:
                author_name = post.author_info.get('name', 'WDAI Community Member')
                bio = post.author_info.get('description', '')
                if bio:
                    author_bio = bio[:100] + "..." if len(bio) > 100 else bio
            
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*WDAI Story:* {post.original_content}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Category:* {category_display}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Type:* {story_type_display}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Author:* {author_name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Engagement:* {post.engagement_score:.0f}"
                        }
                    ]
                }
            ]
            
            # Add author bio if available
            if author_bio:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Author Bio:* {author_bio}"
                    }
                })
            
            # Add viral analysis for high-engagement posts
            if post.viral_analysis and post.engagement_score >= 50:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Why This Resonates:* {post.viral_analysis.get('why_this_works', 'Community connection')}"
                    }
                })
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Target Audience:* {post.viral_analysis.get('target_audience', 'WDAI community')}"
                    }
                })
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Original Tweet (first 30 words):* {first_30_words}"
                }
            })
            
            blocks.extend([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Link:* <{post.source_url}|View Original>"
                    }
                },
                {
                    "type": "divider"
                }
            ])
            
            messages.append({
                "text": "WDAI Community Content Found",
                "username": "Twitter WDAI Agent",
                "icon_emoji": ":woman-technologist:",
                "blocks": blocks
            })
        
        return messages

    def post_to_slack(self, posts: List[WDAIPost]):
        """Post results to Slack"""
        if 'slack_webhook' not in self.creds:
            print("❌ Slack webhook not configured")
            return False
        
        try:
            # Filter out posts already posted to Slack
            new_posts = []
            for post in posts:
                content_hash = self.generate_content_hash(post.raw_original_text)
                if not self.is_already_posted_to_slack(content_hash):
                    new_posts.append(post)
                else:
                    print(f"  ⏭️  Skipping already posted: {post.original_content[:50]}...")
            
            if not new_posts:
                print("ℹ️  No new posts to send to Slack (all were already posted)")
                return True
            
            # Sort by engagement score
            new_posts.sort(key=lambda x: x.engagement_score, reverse=True)
            
            print(f"📱 Posting {len(new_posts)} new WDAI community posts to Slack...")
            
            messages = self.format_slack_message(new_posts)
            posted_content_hashes = []
            
            for i, message in enumerate(messages):
                response = requests.post(
                    self.creds['slack_webhook'],
                    json=message,
                    timeout=10
                )
                
                if response.status_code == 200:
                    print(f"✅ Message {i+1} posted to Slack")
                    
                    # Track which posts were successfully posted
                    if i > 0:  # Skip summary message
                        post_index = i - 1
                        if post_index < len(new_posts):
                            content_hash = self.generate_content_hash(new_posts[post_index].raw_original_text)
                            posted_content_hashes.append(content_hash)
                else:
                    print(f"❌ Message {i+1} failed: {response.status_code}")
                
                time.sleep(0.5)  # Rate limiting
            
            # Mark successfully posted items in database
            for content_hash in posted_content_hashes:
                try:
                    self.conn.execute(
                        'UPDATE wdai_content SET posted_to_slack = 1 WHERE content_hash = ?',
                        (content_hash,)
                    )
                except Exception as e:
                    print(f"⚠️  Error marking as posted: {e}")
            
            if posted_content_hashes:
                self.conn.commit()
                print(f"✅ Marked {len(posted_content_hashes)} posts as sent to Slack")
            
            return True
            
        except Exception as e:
            print(f"❌ Slack error: {e}")
            return False

    def display_results(self, posts: List[WDAIPost]):
        """Display results in terminal"""
        print("\n" + "="*80)
        print("🌟 WOMEN DEFINING AI COMMUNITY CONTENT (ENGLISH ONLY)")
        print("="*80)
        
        if not posts:
            print("🤔 No WDAI community content found")
            return
        
        # Sort all posts by engagement first
        all_posts_sorted = sorted(posts, key=lambda x: x.engagement_score, reverse=True)
        
        print(f"📊 Top {min(15, len(all_posts_sorted))} WDAI Community Posts:")
        print("-" * 60)
        
        for i, post in enumerate(all_posts_sorted[:15], 1):
            category_display = post.category.replace('_', ' ').title()
            story_type_display = post.story_type.replace('_', ' ').title() if post.story_type else "Story"
            
            # Author info
            author_name = "WDAI Community Member"
            author_bio = ""
            if post.author_info:
                author_name = post.author_info.get('name', 'WDAI Community Member')
                bio = post.author_info.get('description', '')
                if bio:
                    author_bio = f" | {bio[:50]}..." if len(bio) > 50 else f" | {bio}"
            
            print(f"\n#{i} | {post.engagement_score:.0f} engagement | {category_display} | {story_type_display}")
            print(f"👤 {author_name}{author_bio}")
            print(f"📅 {post.timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"📝 Story: {post.original_content[:150]}{'...' if len(post.original_content) > 150 else ''}")
            
            # Show viral analysis for high-engagement posts
            if post.viral_analysis and post.engagement_score >= 50:
                print(f"🎯 Target: {post.viral_analysis.get('target_audience', 'WDAI community')}")
                print(f"💡 Why it resonates: {post.viral_analysis.get('why_this_works', 'Community connection')[:100]}...")
            
            print(f"🔗 URL: {post.source_url}")
            if i < len(all_posts_sorted[:15]):
                print("-" * 60)
        
        # Show breakdown by category and story type
        by_category = {}
        by_story_type = {}
        
        for post in posts:
            # Category breakdown
            if post.category not in by_category:
                by_category[post.category] = 0
            by_category[post.category] += 1
            
            # Story type breakdown
            story_type = post.story_type or 'other'
            if story_type not in by_story_type:
                by_story_type[story_type] = 0
            by_story_type[story_type] += 1
        
        print(f"\n📈 CATEGORY BREAKDOWN:")
        for category, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            category_display = category.replace('_', ' ').title()
            print(f"  {category_display}: {count} posts")
        
        print(f"\n📊 STORY TYPE BREAKDOWN:")
        for story_type, count in sorted(by_story_type.items(), key=lambda x: x[1], reverse=True):
            story_display = story_type.replace('_', ' ').title()
            print(f"  {story_display}: {count} posts")

    def run_scan(self, hours_back: int = 24):
        """Main scanning function"""
        print(f"\n🌟 WOMEN DEFINING AI COMMUNITY CONTENT DISCOVERY")
        print(f"⏰ Scanning last {hours_back} hours")
        print(f"🇺🇸 Language: English content only")
        print(f"👥 Target: AI learners, women in tech, career transitioners, ethical AI advocates")
        print(f"📖 Focus: Learning journeys, diversity, community support, work-life balance")
        print(f"🚫 Filtering: Crypto, extreme politics, adult content, MLM schemes, aggressive sales")
        print(f"📋 Valid categories: {', '.join(self.valid_wdai_categories.keys())}")
        print(f"⚠️  BASIC PLAN: Limited to 8 searches with 20-second delays")
        print(f"💡 Expected runtime: ~3-4 minutes (original content only)")
        print("="*60)
        
        # Scan Twitter for WDAI content
        wdai_posts = self.scan_twitter_for_wdai_content(hours_back)
        
        print(f"\n📊 WOMEN DEFINING AI CONTENT DISCOVERY RESULTS:")
        print(f"  📝 Total WDAI posts found: {len(wdai_posts)}")
        print(f"  🎯 Strategy: Inclusive approach for AI learners and community members")
        
        if wdai_posts:
            # Save to database
            self.save_to_database(wdai_posts)
            
            # Print results to terminal instead of Slack
            print(f"📱 Displaying {len(wdai_posts)} WDAI community posts in terminal...")
            
            # Display results
            self.display_results(wdai_posts)
            
            print(f"\n✅ SUCCESS! Found {len(wdai_posts)} WDAI community posts")
        else:
            print(f"\n🤔 No WDAI community content found this scan")
        
        print(f"💾 Data saved to: twitter_wdai_content.db")

def main():
    """Main execution"""
    import sys
    
    print("🌟 Twitter Women Defining AI Content Discovery Agent")
    print("=" * 70)
    
    try:
        # Check for debug mode
        debug_mode = 'debug' in [arg.lower() for arg in sys.argv[1:]]
        
        agent = TwitterWDAIAgent(debug_mode=debug_mode)
        
        # Check for command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'debug':
                print("🔍 DEBUG MODE ENABLED - Will show detailed filtering breakdown")
                agent.run_scan(hours_back=24)
                return
            
            elif command == 'reset':
                print("🔄 Resetting posted status for all records...")
                agent.reset_posted_status()
                print("✅ All records can now be posted to Slack again")
                return
                
            elif command == 'clear':
                print("⚠️  WARNING: This will delete ALL records from the database!")
                confirm = input("Type 'DELETE' to confirm: ")
                if confirm == 'DELETE':
                    agent.clear_database()
                    print("✅ Database cleared")
                else:
                    print("❌ Operation cancelled")
                return
                
            elif command == 'stats':
                agent.get_database_stats()
                return
                
            elif command == 'help':
                print("📋 Available commands:")
                print("  python viral_twitter_WDAI.py          - Run normal scan (English only)")
                print("  python viral_twitter_WDAI.py debug    - Run with detailed filtering breakdown")
                print("  python viral_twitter_WDAI.py reset    - Reset posted status (allow reprocessing)")
                print("  python viral_twitter_WDAI.py clear    - Clear entire database")
                print("  python viral_twitter_WDAI.py stats    - Show database statistics")
                print("  python viral_twitter_WDAI.py help     - Show this help")
                return
                
            else:
                print(f"❌ Unknown command: {command}")
                print("Run 'python viral_twitter_WDAI.py help' for available commands")
                return
        
        # Normal scan execution
        agent.run_scan(hours_back=24)
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
