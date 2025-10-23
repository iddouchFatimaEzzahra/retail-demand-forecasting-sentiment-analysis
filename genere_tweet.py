import os
import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd
from groq import Groq
from kafka import KafkaProducer
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
import mysql.connector
from mysql.connector import Error

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tweet_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTweetGenerator:
    def __init__(self, groq_api_key: str, kafka_config: Dict):
        self.client = Groq(api_key=groq_api_key)
        self.kafka_config = kafka_config
        self.lock = threading.Lock()
        self.generated_count = 0
        self.daily_token_limit = 100000
        self.tokens_used_today = 0
        self.last_reset_date = datetime.now().date()
        
        # MySQL Configuration
        self.mysql_config = {
            'host': 'host.docker.internal',
            'database': 'retail_db',
            'user': 'root',
            'password': '1234',
            'port': 3306,
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        }
        self.mysql_conn = None
        self.mysql_cursor = None
        
        # Initialize MySQL connection
        self.init_mysql_connection()
        
        # Rate limiting parameters
        self.requests_per_minute = 30  # Conservative limit
        self.tokens_per_request = 150  # Average estimate
        self.request_times = []
        
        # Kafka Producer with enhanced configuration
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=kafka_config['bootstrap_servers'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                batch_size=16384,
                linger_ms=10,
                compression_type='gzip'
            )
        except Exception as e:
            logger.warning(f"Kafka producer initialization failed: {e}. Continuing without Kafka.")
            self.producer = None
        
        # Load products and stores from the original dataset
        self.products = [
            "Apple iPhone 13", 
            "Samsung Galaxy S21", 
            "Nike Air Max", 
            "Adidas Ultraboost", 
            "Sony WH-1000XM4"
        ]
        
        self.stores = [
            "New York", "Los Angeles", "Chicago", "Houston", "Miami", 
            "San Francisco", "Seattle", "Boston", "Denver", "Atlanta"
        ]
        
        # Enhanced prompt templates for better variety
        self.prompt_templates = self.generate_comprehensive_prompts()
        
        # Sentiment weights for realistic distribution
        self.sentiment_weights = {
            'positive': 0.35,  # 35% positive
            'neutral': 0.30,  # 35% neutral
            'negative': 0.35  # 30% negative
        }
        
        # Focus distribution for realistic variety
        self.focus_weights = {
            'product_only': 0.35,  # 35% product-only tweets
            'store_only': 0.25,    # 25% store-only tweets  
            'both': 0.40           # 40% tweets mentioning both
        }
    
    def init_mysql_connection(self):
        """Initialize MySQL connection and create table if needed"""
        try:
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            self.mysql_cursor = self.mysql_conn.cursor()
            
            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS generated_tweets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tweet_id VARCHAR(255) UNIQUE NOT NULL,
                tweet_text TEXT NOT NULL,
                product VARCHAR(255),
                store VARCHAR(255),
                sentiment VARCHAR(50) NOT NULL,
                context VARCHAR(100),
                focus VARCHAR(50) NOT NULL,
                generated_at DATETIME NOT NULL,
                character_count INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_sentiment (sentiment),
                INDEX idx_product (product),
                INDEX idx_store (store),
                INDEX idx_generated_at (generated_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            self.mysql_cursor.execute(create_table_query)
            self.mysql_conn.commit()
            logger.info("MySQL connection established and table created/verified")
            
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            self.mysql_conn = None
            self.mysql_cursor = None
    
    def check_daily_reset(self):
        """Reset token counter if it's a new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.tokens_used_today = 0
            self.last_reset_date = current_date
            logger.info(f"Daily token counter reset for {current_date}")
    
    def can_make_request(self) -> bool:
        """Check if we can make a request based on rate limits"""
        self.check_daily_reset()
        
        # Check daily token limit
        if self.tokens_used_today + self.tokens_per_request > self.daily_token_limit:
            logger.warning(f"Daily token limit would be exceeded. Used: {self.tokens_used_today}, Limit: {self.daily_token_limit}")
            return False
        
        # Check per-minute rate limit
        current_time = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            logger.info(f"Per-minute rate limit reached. Waiting...")
            return False
        
        return True
    
    def wait_for_rate_limit(self):
        """Wait until we can make another request"""
        while not self.can_make_request():
            # Check if we hit daily limit
            if self.tokens_used_today + self.tokens_per_request > self.daily_token_limit:
                # Calculate time until midnight
                now = datetime.now()
                tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                wait_seconds = (tomorrow - now).total_seconds()
                
                logger.info(f"Daily token limit reached. Waiting {wait_seconds/3600:.1f} hours until reset...")
                time.sleep(min(wait_seconds, 3600))  # Wait max 1 hour at a time
                continue
            
            # Wait for per-minute limit
            current_time = time.time()
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            if len(self.request_times) >= self.requests_per_minute:
                # Calculate wait time until oldest request is > 1 minute old
                oldest_request = min(self.request_times)
                wait_time = 60 - (current_time - oldest_request) + 1
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
    
    def record_request(self):
        """Record that a request was made"""
        current_time = time.time()
        self.request_times.append(current_time)
        self.tokens_used_today += self.tokens_per_request
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=300),
        reraise=True
    )
    def make_api_request(self, formatted_prompt: str, expected_product: str = None, expected_store: str = None) -> str:
        """Make API request with retry logic and rate limiting"""
        self.wait_for_rate_limit()
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at generating authentic social media content. Create realistic customer tweets in English. Avoid hashtags unless they naturally fit the context. Make the tweets sound genuine and conversational, like real customer experiences."
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.9,
                max_tokens=100,
                top_p=0.9
            )
            
            self.record_request()
            tweet_text = response.choices[0].message.content.strip()
            
            # Clean up the tweet text
            tweet_text = tweet_text.strip('"').strip("'")
            if len(tweet_text) > 280:
                tweet_text = tweet_text[:277] + "..."
            
            return tweet_text
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                # Extract wait time from error message if available
                if "Please try again in" in error_msg:
                    try:
                        wait_part = error_msg.split("Please try again in")[1].split(".")[0]
                        if "m" in wait_part:
                            minutes = int(wait_part.split("m")[0])
                            wait_seconds = minutes * 60 + 30  # Add buffer
                        else:
                            wait_seconds = 300  # Default 5 minutes
                    except:
                        wait_seconds = 300
                else:
                    wait_seconds = 300
                
                logger.warning(f"Rate limit hit, waiting {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                raise  # Re-raise to trigger retry
            else:
                logger.error(f"API request failed: {e}")
                raise
    
    def generate_comprehensive_prompts(self) -> Dict[str, List[Dict]]:
        """Generate comprehensive prompt templates categorized by sentiment and focus"""
        return {
            'positive': {
                'product_only': [
                    {
                        "template": "Generate a positive tweet about {product}. Customer should praise the product's features, quality, or performance. Don't mention any specific store. Keep it authentic and under 280 characters.",
                        "context": "product_praise",
                        "focus": "product_only"
                    },
                    {
                        "template": "Create a tweet where someone recommends {product} to their followers. Focus on what makes this product great. No store mentions needed. Make it conversational.",
                        "context": "product_recommendation",
                        "focus": "product_only"
                    },
                    {
                        "template": "Write an excited tweet about unboxing or using {product}. Express satisfaction with the product itself. Keep focus only on the product experience.",
                        "context": "product_unboxing",
                        "focus": "product_only"
                    }
                ],
                'store_only': [
                    {
                        "template": "Generate a positive tweet praising the shopping experience at {store}. Focus on customer service, store atmosphere, or shopping convenience. Don't mention specific products.",
                        "context": "store_praise",
                        "focus": "store_only"
                    },
                    {
                        "template": "Create a tweet recommending {store} to followers. Highlight what makes this store special for shopping. Keep focus on the store experience only.",
                        "context": "store_recommendation", 
                        "focus": "store_only"
                    },
                    {
                        "template": "Write a tweet about excellent customer service received at {store}. Focus on staff helpfulness, store policies, or overall service quality.",
                        "context": "store_service",
                        "focus": "store_only"
                    }
                ],
                'both': [
                    {
                        "template": "Generate a positive tweet about purchasing {product} from {store}. Express satisfaction with both the product quality and shopping experience. Keep it authentic and under 280 characters.",
                        "context": "combined_satisfaction",
                        "focus": "both"
                    },
                    {
                        "template": "Create a tweet recommending both {product} and {store}. Mention why this product at this specific store is a great combination.",
                        "context": "combined_recommendation",
                        "focus": "both"
                    },
                    {
                        "template": "Write a tweet about finding the perfect {product} at {store}. Express happiness about both the product and where you found it.",
                        "context": "perfect_match",
                        "focus": "both"
                    }
                ]
            },
            'negative': {
                'product_only': [
                    {
                        "template": "Generate a disappointed tweet about {product}. Express frustration with product quality, features, or performance. Don't mention any store. Keep it realistic.",
                        "context": "product_disappointment",
                        "focus": "product_only"
                    },
                    {
                        "template": "Create a tweet about issues with {product}. Focus on specific product problems or defects. Avoid mentioning where it was purchased.",
                        "context": "product_defect",
                        "focus": "product_only"
                    },
                    {
                        "template": "Write a tweet about {product} not meeting expectations. Focus only on the product's shortcomings or issues.",
                        "context": "product_unmet_expectations",
                        "focus": "product_only"  
                    }
                ],
                'store_only': [
                    {
                        "template": "Generate a complaint tweet about poor service at {store}. Focus on customer service issues, store policies, or shopping experience problems. Don't mention specific products.",
                        "context": "store_complaint",
                        "focus": "store_only"
                    },
                    {
                        "template": "Create a tweet about bad shopping experience at {store}. Mention store-related issues like long waits, unhelpful staff, or poor store conditions.",
                        "context": "store_bad_experience", 
                        "focus": "store_only"
                    },
                    {
                        "template": "Write a frustrated tweet about {store}'s customer service or policies. Keep focus on store-related problems only.",
                        "context": "store_service_issue",
                        "focus": "store_only"
                    }
                ],
                'both': [
                    {
                        "template": "Generate a disappointed tweet about both {product} and the experience at {store}. Express frustration with both the product quality and store service.",
                        "context": "combined_disappointment",
                        "focus": "both"
                    },
                    {
                        "template": "Create a complaint tweet about purchasing {product} from {store}. Mention issues with both the product and the shopping experience.",
                        "context": "combined_complaint", 
                        "focus": "both"
                    },
                    {
                        "template": "Write a tweet about problems with {product} bought at {store}. Include issues with both product and store experience.",
                        "context": "combined_issues",
                        "focus": "both"
                    }
                ]
            },
            'neutral': {
                'product_only': [
                    {
                        "template": "Generate a neutral, informational tweet about {product}. Share factual information or ask for opinions about this product. Keep tone balanced.",
                        "context": "product_info",
                        "focus": "product_only"
                    },
                    {
                        "template": "Create a tweet asking followers about their experience with {product}. Make it a genuine question seeking product advice.",
                        "context": "product_inquiry",
                        "focus": "product_only"
                    },
                    {
                        "template": "Write a neutral tweet sharing thoughts about {product}. Keep it factual and balanced without strong emotions.",
                        "context": "product_thoughts",
                        "focus": "product_only"
                    }
                ],
                'store_only': [
                    {
                        "template": "Generate a neutral tweet about shopping at {store}. Share factual information about the store experience without strong emotions.",
                        "context": "store_info",
                        "focus": "store_only"
                    },
                    {
                        "template": "Create a tweet asking about others' experiences shopping at {store}. Make it an informational inquiry about the store.",
                        "context": "store_inquiry",
                        "focus": "store_only"
                    },
                    {
                        "template": "Write a balanced tweet about {store}. Share neutral observations about the shopping experience there.",
                        "context": "store_observation",
                        "focus": "store_only"
                    }
                ],
                'both': [
                    {
                        "template": "Generate a factual tweet about finding {product} at {store}. Share the information objectively without strong emotional language.",
                        "context": "combined_factual",
                        "focus": "both"
                    },
                    {
                        "template": "Create a tweet asking for opinions about {product} available at {store}. Make it a balanced question seeking advice.",
                        "context": "combined_inquiry",
                        "focus": "both"
                    },
                    {
                        "template": "Write a neutral tweet about the process of buying {product} at {store}. Keep it informational and balanced.",
                        "context": "combined_process",
                        "focus": "both"
                    }
                ]
            }
        }
    
    def select_weighted_sentiment(self) -> str:
        """Select sentiment based on realistic distribution weights"""
        rand = random.random()
        if rand < self.sentiment_weights['positive']:
            return 'positive'
        elif rand < self.sentiment_weights['positive'] + self.sentiment_weights['neutral']:
            return 'neutral'
        else:
            return 'negative'
    
    def select_weighted_focus(self) -> str:
        """Select focus type based on realistic distribution weights"""
        rand = random.random()
        if rand < self.focus_weights['product_only']:
            return 'product_only'
        elif rand < self.focus_weights['product_only'] + self.focus_weights['store_only']:
            return 'store_only'
        else:
            return 'both'
    
    def generate_single_tweet(self, sentiment: str, focus: str, product: str, store: str) -> Tuple[str, str, str, str]:
        """Generate a single tweet with specified sentiment and focus"""
        try:
            # Select random prompt template for the sentiment and focus
            prompt_info = random.choice(self.prompt_templates[sentiment][focus])
            
            # Format prompt based on focus type
            if focus == 'product_only':
                formatted_prompt = prompt_info["template"].format(product=product)
            elif focus == 'store_only':
                formatted_prompt = prompt_info["template"].format(store=store)
            else:  # both
                formatted_prompt = prompt_info["template"].format(product=product, store=store)
            
            # Make API request with rate limiting
            tweet = self.make_api_request(formatted_prompt)
            tweet = tweet.strip('"').strip("'")
            
            # Ensure tweet length is reasonable
            if len(tweet) > 280:
                tweet = tweet[:277] + "..."
            
            return tweet, sentiment, prompt_info["context"], focus
            
        except Exception as e:
            logger.error(f"Error generating tweet: {e}")
            return None, None, None, None
    
    def save_to_mysql(self, tweet_data: Dict):
        """Save a tweet to MySQL database"""
        if not self.mysql_conn or not self.mysql_cursor:
            logger.warning("MySQL connection not available")
            return False
        
        try:
            # Check if connection is still alive
            if not self.mysql_conn.is_connected():
                logger.info("MySQL connection lost, attempting to reconnect...")
                self.init_mysql_connection()
            
            query = """
            INSERT INTO generated_tweets (
                tweet_id, tweet_text, product, store, sentiment, 
                context, focus, generated_at, character_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                tweet_text = VALUES(tweet_text),
                product = VALUES(product),
                store = VALUES(store),
                sentiment = VALUES(sentiment),
                context = VALUES(context),
                focus = VALUES(focus),
                generated_at = VALUES(generated_at),
                character_count = VALUES(character_count)
            """
            
            values = (
                tweet_data.get("tweet_id"),
                tweet_data.get("tweet_text"),
                tweet_data.get("product"),
                tweet_data.get("store"),
                tweet_data.get("sentiment"),
                tweet_data.get("context"),
                tweet_data.get("focus"),
                tweet_data.get("generated_at"),
                tweet_data.get("character_count")
            )
            
            self.mysql_cursor.execute(query, values)
            self.mysql_conn.commit()
            
            logger.debug(f"Tweet {tweet_data.get('tweet_id')} saved to MySQL")
            return True
            
        except Error as e:
            logger.error(f"MySQL save error: {e}")
            # Try to reconnect on error
            try:
                self.init_mysql_connection()
            except:
                pass
            return False
    
    def send_to_kafka(self, tweet_data: Dict):
        """Send tweet to Kafka topic"""
        try:
            if self.producer:
                self.producer.send('tweets_topic', tweet_data)
        except Exception as e:
            logger.error(f"Error sending to Kafka: {e}")
    
    def save_to_csv_chunked(self, tweets_data: List[Dict], filename: str = 'synthetic_tweets_daily.csv', chunk_size: int = 50000):
        """Save tweets to CSV in chunks to handle large datasets efficiently"""
        try:
            logger.info(f"Saving {len(tweets_data)} tweets to {filename}")
            
            # Create DataFrame
            df = pd.DataFrame(tweets_data)
            
            # Reorder columns for better readability
            column_order = ['tweet_id', 'tweet_text', 'product', 'store', 'sentiment', 'focus', 'context', 'generated_at', 'character_count']
            df = df[column_order]
            
            # Save to CSV
            df.to_csv(filename, index=False, encoding='utf-8')
            
            # Generate and log statistics
            self.log_dataset_statistics(df)
            
            logger.info(f"Successfully saved {len(tweets_data)} tweets to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
    
    def log_dataset_statistics(self, df: pd.DataFrame):
        """Log comprehensive statistics about the generated dataset"""
        logger.info("\n" + "="*50)
        logger.info("DATASET STATISTICS")
        logger.info("="*50)
        logger.info(f"Total tweets generated: {len(df):,}")
        
        logger.info("\nSentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {sentiment.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        logger.info("\nFocus Distribution:")
        focus_counts = df['focus'].value_counts()
        for focus, count in focus_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {focus.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
        
        logger.info(f"\nAverage tweet length: {df['character_count'].mean():.1f} characters")
        logger.info(f"Tokens used today: {self.tokens_used_today}")
        logger.info("="*50)
    
    def run_complete_pipeline(self, target_tweets: int = 100, output_file: str = 'synthetic_tweets_100.csv'):
        """Run the complete tweet generation pipeline"""
        start_time = time.time()
        tweets_data = []
        successful_saves = 0
        
        logger.info(f"Starting generation of {target_tweets} tweets...")
        logger.info(f"MySQL Database: {self.mysql_config['database']}")
        logger.info(f"Output CSV: {output_file}")
        
        try:
            for i in range(target_tweets):
                try:
                    # Select parameters for this tweet
                    sentiment = self.select_weighted_sentiment()
                    focus = self.select_weighted_focus()
                    product = random.choice(self.products)
                    store = random.choice(self.stores)
                    
                    # Select and format prompt
                    prompt_info = random.choice(self.prompt_templates[sentiment][focus])
                    
                    if focus == 'product_only':
                        formatted_prompt = prompt_info["template"].format(product=product)
                        expected_product = product
                        expected_store = None
                    elif focus == 'store_only':
                        formatted_prompt = prompt_info["template"].format(store=store)
                        expected_product = None
                        expected_store = store
                    else:  # both
                        formatted_prompt = prompt_info["template"].format(product=product, store=store)
                        expected_product = product
                        expected_store = store
                    
                    # Generate tweet
                    tweet_text = self.make_api_request(formatted_prompt, expected_product, expected_store)
                    
                    if tweet_text and len(tweet_text.strip()) >= 10:
                        # Generate realistic timestamp
                        base_time = datetime.now() - timedelta(days=random.randint(0, 180))
                        generated_time = base_time - timedelta(
                            hours=random.randint(0, 23),
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59)
                        )
                        
                        # Create tweet data
                        tweet_data = {
                            "tweet_id": f"tweet_{i+1}_{int(time.time()*1000)}",
                            "tweet_text": tweet_text,
                            "product": expected_product,
                            "store": expected_store,
                            "sentiment": sentiment,
                            "context": prompt_info["context"],
                            "focus": focus,
                            "generated_at": generated_time.isoformat(),
                            "character_count": len(tweet_text)
                        }
                        
                        # Save to MySQL
                        if self.save_to_mysql(tweet_data):
                            successful_saves += 1
                        
                        # Send to Kafka
                        if self.producer:
                            try:
                                self.producer.send('tweets_topic', tweet_data)
                                self.producer.flush()
                            except Exception as e:
                                logger.error(f"Kafka send error: {e}")
                        
                        # Add to local collection
                        tweets_data.append(tweet_data)
                        
                        # Log progress
                        if (i + 1) % 10 == 0:
                            logger.info(f"Progress: {i+1}/{target_tweets} tweets generated. MySQL saves: {successful_saves}")
                        
                    else:
                        logger.warning(f"Tweet {i+1} generation failed - invalid content")
                    
                    # Small delay to respect rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error generating tweet {i+1}: {e}")
                    continue
            
            # Save to CSV
            if tweets_data:
                df = pd.DataFrame(tweets_data)
                column_order = ['tweet_id', 'tweet_text', 'product', 'store', 'sentiment', 'focus', 'context', 'generated_at', 'character_count']
                df = df[column_order]
                df.to_csv(output_file, index=False, encoding='utf-8')
                
                # Log final statistics
                self.log_dataset_statistics(df)
                logger.info(f"Successfully saved {len(tweets_data)} tweets to {output_file}")
                logger.info(f"MySQL successful saves: {successful_saves}/{len(tweets_data)}")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        
        finally:
            # Cleanup connections
            if self.mysql_conn and self.mysql_conn.is_connected():
                self.mysql_cursor.close()
                self.mysql_conn.close()
                logger.info("MySQL connection closed")
            
            if self.producer:
                self.producer.close()
                logger.info("Kafka producer closed")
            
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed in {total_time:.1f} seconds")
            logger.info(f"Average time per tweet: {total_time/max(len(tweets_data), 1):.2f} seconds")

if __name__ == "__main__":
    # Configuration
    api_key = os.getenv('GROQ_API_KEY')
    #api_key = os.getenv("GROQ_API_KEY", "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")
    kafka_config = {'bootstrap_servers': ['kafka:29092']}
    
    # Initialize generator
    generator = EnhancedTweetGenerator(api_key, kafka_config)
    
    # Run pipeline to generate 100 tweets
    generator.run_complete_pipeline(target_tweets=100, output_file='generated_tweets.csv')