import spacy
import re
from typing import Dict, List, Tuple
import pandas as pd
import logging
import json
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import mysql.connector
from mysql.connector import Error
import os
import sys
from datetime import datetime

# Configure logging for Airflow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs are visible in Airflow
    ]
)
logger = logging.getLogger(__name__)

class RetailNERModel:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            logger.error("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            raise
        
        # Product and store dictionaries
        self.products = {
            "apple iphone 13": ["iphone 13", "iphone13", "apple iphone", "iphone"],
            "samsung galaxy s21": ["galaxy s21", "samsung galaxy", "galaxy", "samsung"],
            "nike air max": ["air max", "nike air", "nike shoes", "nike"],
            "adidas ultraboost": ["ultraboost", "adidas ultra", "adidas shoes", "adidas"],
            "sony wh-1000xm4": ["wh-1000xm4", "sony headphones", "sony wh", "sony"]
        }
        
        self.stores = {
            "new york": ["nyc", "ny", "new york city", "manhattan"],
            "los angeles": ["la", "los angeles", "california"],
            "chicago": ["chicago", "chi-town"],
            "houston": ["houston", "texas"],
            "miami": ["miami", "florida", "fl"],
            "san francisco": ["sf", "san francisco", "bay area"],
            "seattle": ["seattle", "washington"],
            "boston": ["boston", "massachusetts"],
            "denver": ["denver", "colorado"],
            "atlanta": ["atlanta", "georgia"]
        }
        
        # Regex patterns for improved detection
        self.product_patterns = self._create_product_patterns()
        self.store_patterns = self._create_store_patterns()
    
    def _create_product_patterns(self) -> List[Tuple[str, str]]:
        """Create regex patterns for products"""
        patterns = []
        for main_product, variants in self.products.items():
            for variant in variants:
                pattern = r'\b' + re.escape(variant.lower()) + r'\b'
                patterns.append((pattern, main_product))
        return patterns
    
    def _create_store_patterns(self) -> List[Tuple[str, str]]:
        """Create regex patterns for stores"""
        patterns = []
        for main_store, variants in self.stores.items():
            for variant in variants:
                pattern = r'\b' + re.escape(variant.lower()) + r'\b'
                patterns.append((pattern, main_store))
        return patterns
    
    def extract_entities(self, text: str) -> Dict:
        """Extract all entities from a tweet"""
        entities = {
            'products': [],
            'stores': [],
            'organizations': [],
            'persons': [],
            'money': [],
            'dates': []
        }
        
        text_lower = text.lower()
        
        # Extract products using patterns
        for pattern, product_name in self.product_patterns:
            if re.search(pattern, text_lower):
                if product_name not in entities['products']:
                    entities['products'].append(product_name)
        
        # Extract stores using patterns
        for pattern, store_name in self.store_patterns:
            if re.search(pattern, text_lower):
                if store_name not in entities['stores']:
                    entities['stores'].append(store_name)
        
        # Extract other entities using spaCy
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities['organizations'].append(ent.text)
            elif ent.label_ == "PERSON":
                entities['persons'].append(ent.text)
            elif ent.label_ == "MONEY":
                entities['money'].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                entities['dates'].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def analyze_tweet_comprehensive(self, tweet_text: str) -> Dict:
        """Comprehensive analysis of a tweet (entities + metadata)"""
        entities = self.extract_entities(tweet_text)
        focus = self._determine_focus(entities)
        entity_counts = {
            'total_products': len(entities['products']),
            'total_stores': len(entities['stores']),
            'total_entities': sum(len(v) for v in entities.values())
        }
        
        return {
            'tweet_text': tweet_text,
            'entities': entities,
            'focus': focus,
            'entity_counts': entity_counts,
            'has_product': len(entities['products']) > 0,
            'has_store': len(entities['stores']) > 0
        }
    
    def _determine_focus(self, entities: Dict) -> str:
        """Determine tweet focus based on extracted entities"""
        has_products = len(entities['products']) > 0
        has_stores = len(entities['stores']) > 0
        if has_products and has_stores:
            return 'both'
        elif has_products:
            return 'product_only'
        elif has_stores:
            return 'store_only'
        else:
            return 'general'
    
    def batch_analyze_tweets(self, tweets_df: pd.DataFrame, text_column: str = 'tweet_text') -> pd.DataFrame:
        """Analyze a batch of tweets"""
        logger.info(f"🔍 Analyzing NER for {len(tweets_df)} tweets...")
        results = []
        for idx, row in tweets_df.iterrows():
            try:
                analysis = self.analyze_tweet_comprehensive(row[text_column])
                result_row = {
                    'tweet_id': row.get('tweet_id', f'tweet_{idx}'),
                    'tweet_text': analysis['tweet_text'],
                    'products_found': '|'.join(analysis['entities']['products']),
                    'stores_found': '|'.join(analysis['entities']['stores']),
                    'organizations_found': '|'.join(analysis['entities']['organizations']),
                    'focus_detected': analysis['focus'],
                    'has_product': analysis['has_product'],
                    'has_store': analysis['has_store'],
                    'total_entities': analysis['entity_counts']['total_entities']
                }
                for col in row.index:
                    if col not in result_row:
                        result_row[col] = row[col]
                results.append(result_row)
            except Exception as e:
                logger.error(f"Error analyzing tweet {idx}: {e}")
                continue
        result_df = pd.DataFrame(results)
        logger.info(f"✅ NER analysis completed. {len(result_df)} tweets processed.")
        return result_df

def setup_kafka_topic(topic_name: str, bootstrap_servers: List[str]):
    """Create Kafka topic if it doesn't exist"""
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        topic_list = [NewTopic(name=topic_name, num_partitions=1, replication_factor=1)]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
        logger.info(f"Kafka topic '{topic_name}' created successfully")
        admin_client.close()
    except Exception as e:
        if "TopicAlreadyExists" in str(e):
            logger.info(f"Kafka topic '{topic_name}' already exists")
        else:
            logger.error(f"Failed to create Kafka topic '{topic_name}': {e}")
            raise

def save_to_mysql(results_df: pd.DataFrame, mysql_config: Dict):
    """Save NER results to MySQL"""
    try:
        connection = mysql.connector.connect(**mysql_config)
        cursor = connection.cursor()
        
        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ner_results (
            tweet_id VARCHAR(255) PRIMARY KEY,
            tweet_text TEXT,
            products_found TEXT,
            stores_found TEXT,
            organizations_found TEXT,
            focus_detected VARCHAR(50),
            has_product BOOLEAN,
            has_store BOOLEAN,
            total_entities INT,
            generated_at DATETIME
        )
        """
        cursor.execute(create_table_query)
        
        # Insert results
        insert_query = """
        INSERT INTO ner_results (
            tweet_id, tweet_text, products_found, stores_found, organizations_found,
            focus_detected, has_product, has_store, total_entities, generated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for _, row in results_df.iterrows():
            cursor.execute(insert_query, (
                row['tweet_id'],
                row['tweet_text'],
                row['products_found'],
                row['stores_found'],
                row['organizations_found'],
                row['focus_detected'],
                row['has_product'],
                row['has_store'],
                row['total_entities'],
                row.get('generated_at', datetime.now().isoformat())
            ))
        
        connection.commit()
        logger.info(f"Successfully saved {len(results_df)} NER results to MySQL")
        
    except Error as e:
        logger.error(f"Error saving to MySQL: {e}")
        raise
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def process_ner_from_kafka():
    """Process tweets from Kafka topic and store NER results"""
    # Kafka configuration
    kafka_config = {
        'bootstrap_servers': ['kafka:29092'],
        'consumer_group': 'ner_processor_group',
        'input_topic': 'tweets_topic',
        'output_topic': 'ner_results_topic'
    }
    
    # MySQL configuration (placeholder, update with actual credentials)
    mysql_config = {
        'host': 'host.docker.internal',
        'port': 3306,
        'user': 'root',
        'password': '1234',
        'database': 'retail_db'
    }
    
    # Initialize NER model
    ner_model = RetailNERModel()
    
    # Setup output Kafka topic
    setup_kafka_topic(kafka_config['output_topic'], kafka_config['bootstrap_servers'])
    
    # Initialize Kafka consumer
    try:
        consumer = KafkaConsumer(
            kafka_config['input_topic'],
            bootstrap_servers=kafka_config['bootstrap_servers'],
            group_id=kafka_config['consumer_group'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            consumer_timeout_ms=10000  # Timeout after 10 seconds of no messages
        )
    except Exception as e:
        logger.error(f"Failed to initialize Kafka consumer: {e}")
        raise
    
    # Initialize Kafka producer
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            retries=3,
            retry_backoff_ms=1000
        )
    except Exception as e:
        logger.error(f"Failed to initialize Kafka producer: {e}")
        raise
    
    # Process tweets in batches
    batch_size = 50  # Match the generation batch size
    tweets_batch = []
    
    logger.info(f"Starting NER processing from Kafka topic '{kafka_config['input_topic']}'...")
    
    try:
        for message in consumer:
            tweet_data = message.value
            tweets_batch.append(tweet_data)
            
            if len(tweets_batch) >= batch_size:
                # Convert to DataFrame
                tweets_df = pd.DataFrame(tweets_batch)
                
                # Analyze batch
                results_df = ner_model.batch_analyze_tweets(tweets_df)
                
                # Send results to Kafka
                for _, row in results_df.iterrows():
                    result_dict = row.to_dict()
                    producer.send(kafka_config['output_topic'], result_dict)
                
                # Save to MySQL (optional, based on DAG documentation)
                try:
                    save_to_mysql(results_df, mysql_config)
                except Exception as e:
                    logger.warning(f"MySQL saving failed, continuing without saving: {e}")
                
                # Commit Kafka offset
                consumer.commit()
                
                logger.info(f"Processed and sent {len(tweets_batch)} NER results to '{kafka_config['output_topic']}'")
                tweets_batch = []  # Reset batch
            
            # Flush producer periodically
            producer.flush()
        
        # Process remaining tweets in batch
        if tweets_batch:
            tweets_df = pd.DataFrame(tweets_batch)
            results_df = ner_model.batch_analyze_tweets(tweets_df)
            for _, row in results_df.iterrows():
                producer.send(kafka_config['output_topic'], row.to_dict())
            try:
                save_to_mysql(results_df, mysql_config)
            except Exception as e:
                logger.warning(f"MySQL saving failed, continuing without saving: {e}")
            consumer.commit()
            logger.info(f"Processed and sent {len(tweets_batch)} final NER results to '{kafka_config['output_topic']}'")
        
    except Exception as e:
        logger.error(f"Error processing Kafka messages: {e}")
        raise
    finally:
        producer.flush()
        producer.close()
        consumer.close()
        logger.info("NER processing completed, Kafka connections closed.")

if __name__ == "__main__":
    process_ner_from_kafka()