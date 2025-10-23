import sys
import os
import logging
import json
import uuid
import pickle
import joblib
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.structs import TopicPartition
import mysql.connector
from mysql.connector import Error
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chemin du modèle principal
model_paths = [
        "/opt/airflow/models/balanced_sentiment_v3_balanced_v3.0.0.pkl",
        "/opt/airflow/models/improved_sentiment_model.pkl",
        "/opt/airflow/sentiment_model.pkl",
        "models/balanced_sentiment_v3_balanced_v3.0.0.pkl"
    ]

OFFSET_FILE = "/opt/airflow/data/kafka_offset.json"
BATCH_SIZE = 100

def create_kafka_topic(topic_name, bootstrap_servers='localhost:9092'):
    """Créer un topic Kafka s'il n'existe pas"""
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
        topic_list = [NewTopic(name=topic_name, num_partitions=1, replication_factor=1)]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
        logger.info(f"Kafka topic {topic_name} created successfully")
        admin_client.close()
    except Exception as e:
        if "TopicAlreadyExists" in str(e):
            logger.info(f"Kafka topic {topic_name} already exists")
        else:
            logger.error(f"Failed to create Kafka topic {topic_name}: {e}")

def save_kafka_offset(topic, partition, offset):
    """Sauvegarder l'offset Kafka pour éviter les doublons"""
    try:
        os.makedirs(os.path.dirname(OFFSET_FILE), exist_ok=True)
        offset_data = {
            'topic': topic,
            'partition': partition,
            'offset': offset,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(OFFSET_FILE, 'w') as f:
            json.dump(offset_data, f)
        logger.info(f"Saved Kafka offset: {offset} for topic {topic}, partition {partition}")
    except Exception as e:
        logger.error(f"Failed to save Kafka offset: {e}")

def load_kafka_offset():
    """Charger le dernier offset Kafka"""
    try:
        if os.path.exists(OFFSET_FILE):
            with open(OFFSET_FILE, 'r') as f:
                offset_data = json.load(f)
            logger.info(f"Loaded Kafka offset: {offset_data['offset']} from {offset_data['timestamp']}")
            return offset_data['offset']
        return 0
    except Exception as e:
        logger.warning(f"Could not load Kafka offset: {e}")
        return 0

def create_default_model():
    """Créer un modèle par défaut avec SGDClassifier pour l'apprentissage incrémental"""
    logger.info("Creating default incremental learning model...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )),
        ('classifier', SGDClassifier(
            loss='log_loss',  # Remplace 'log' qui est déprécié
            random_state=42,
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1000
        ))
    ])
    
    return pipeline

def load_sentiment_model():
    """Charger le modèle de sentiment ou créer un nouveau"""
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading existing model from: {MODEL_PATH}")
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                model = model_data.get('model')
                if model is None:
                    for key in ['pipeline', 'classifier', 'estimator']:
                        if key in model_data:
                            model = model_data[key]
                            break
            else:
                model = model_data
            
            if hasattr(model, 'predict'):
                logger.info(f"✅ Model loaded successfully. Type: {type(model)}")
                if hasattr(model, 'classes_'):
                    logger.info(f"Model classes: {model.classes_}")
                return model
            else:
                logger.warning("Loaded model doesn't have predict method, creating new one")
                
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
    
    logger.info("Creating new model for incremental learning")
    return create_default_model()

def save_updated_model(model, metrics=None):
    """Sauvegarder le modèle mis à jour"""
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        model_data = {
            'model': model,
            'updated_at': datetime.now().isoformat(),
            'metrics': metrics or {},
            'version': 'incremental_v1.0'
        }
        
        # Sauvegarder avec un nom temporaire puis renommer pour éviter la corruption
        temp_path = MODEL_PATH + '.tmp'
        with open(temp_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        os.replace(temp_path, MODEL_PATH)
        logger.info(f"✅ Model updated and saved to {MODEL_PATH}")
        
        if metrics:
            logger.info(f"Model metrics: {metrics}")
            
    except Exception as e:
        logger.error(f"Failed to save updated model: {e}")

def save_model_metrics_to_mysql(metrics, batch_id):
    """Sauvegarder les métriques du modèle dans MySQL"""
    conn = None
    
    try:
        conn = mysql.connector.connect(
            host='host.docker.internal',
            user="root",
            password="1234",
            database="retail_db",
            autocommit=True,
            connection_timeout=10
        )
        
        cursor = conn.cursor()
        
        # Créer la table des métriques si elle n'existe pas
        create_metrics_table_query = """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            batch_id VARCHAR(100) NOT NULL,
            model_version VARCHAR(50),
            accuracy DECIMAL(6,4),
            precision_positive DECIMAL(6,4),
            precision_negative DECIMAL(6,4),
            precision_neutral DECIMAL(6,4),
            recall_positive DECIMAL(6,4),
            recall_negative DECIMAL(6,4),
            recall_neutral DECIMAL(6,4),
            f1_positive DECIMAL(6,4),
            f1_negative DECIMAL(6,4),
            f1_neutral DECIMAL(6,4),
            training_samples INT,
            validation_samples INT,
            classes_used JSON,
            label_distribution JSON,
            confusion_matrix JSON,
            error_message TEXT,
            update_skipped BOOLEAN DEFAULT FALSE,
            skip_reason VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_batch_id (batch_id),
            INDEX idx_accuracy (accuracy),
            INDEX idx_created_at (created_at)
        )
        """
        cursor.execute(create_metrics_table_query)
        
        # Préparer les données des métriques
        model_version = "incremental_v1.0"
        accuracy = metrics.get('accuracy')
        training_samples = metrics.get('training_samples', 0)
        validation_samples = metrics.get('validation_samples', 0)
        classes_used = json.dumps(metrics.get('classes_used', []))
        label_distribution = json.dumps(metrics.get('label_distribution', {}))
        confusion_matrix_data = json.dumps(metrics.get('confusion_matrix', []))
        error_message = metrics.get('error')
        update_skipped = metrics.get('skipped', False)
        skip_reason = metrics.get('reason')
        
        # Extraire les métriques par classe
        class_metrics = metrics.get('classification_report', {})
        precision_pos = class_metrics.get('positive', {}).get('precision')
        precision_neg = class_metrics.get('negative', {}).get('precision')
        precision_neu = class_metrics.get('neutral', {}).get('precision')
        recall_pos = class_metrics.get('positive', {}).get('recall')
        recall_neg = class_metrics.get('negative', {}).get('recall')
        recall_neu = class_metrics.get('neutral', {}).get('recall')
        f1_pos = class_metrics.get('positive', {}).get('f1-score')
        f1_neg = class_metrics.get('negative', {}).get('f1-score')
        f1_neu = class_metrics.get('neutral', {}).get('f1-score')
        
        # Insérer les métriques
        insert_query = """
        INSERT INTO model_metrics 
        (batch_id, model_version, accuracy, precision_positive, precision_negative, precision_neutral,
         recall_positive, recall_negative, recall_neutral, f1_positive, f1_negative, f1_neutral,
         training_samples, validation_samples, classes_used, label_distribution, confusion_matrix,
         error_message, update_skipped, skip_reason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            batch_id, model_version, accuracy, precision_pos, precision_neg, precision_neu,
            recall_pos, recall_neg, recall_neu, f1_pos, f1_neg, f1_neu,
            training_samples, validation_samples, classes_used, label_distribution, confusion_matrix_data,
            error_message, update_skipped, skip_reason
        )
        
        cursor.execute(insert_query, values)
        
        logger.info(f"✅ Model metrics saved to MySQL for batch {batch_id}")
        
        # Optionnel: Sauvegarder aussi un résumé des performances
        save_performance_summary(cursor, batch_id, metrics)
        
    except Exception as e:
        logger.error(f"Failed to save model metrics to MySQL: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def save_performance_summary(cursor, batch_id, metrics):
    """Sauvegarder un résumé des performances dans une table séparée"""
    try:
        # Créer la table de résumé des performances
        create_summary_table_query = """
        CREATE TABLE IF NOT EXISTS performance_summary (
            id INT AUTO_INCREMENT PRIMARY KEY,
            batch_id VARCHAR(100) NOT NULL,
            total_samples INT,
            positive_samples INT,
            negative_samples INT,
            neutral_samples INT,
            model_accuracy DECIMAL(6,4),
            confidence_avg DECIMAL(6,4),
            processing_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_batch_id (batch_id),
            INDEX idx_processing_date (processing_date)
        )
        """
        cursor.execute(create_summary_table_query)
        
        # Calculer les statistiques
        label_distribution = metrics.get('label_distribution', {})
        total_samples = metrics.get('training_samples', 0) + metrics.get('validation_samples', 0)
        positive_samples = label_distribution.get('positive', 0)
        negative_samples = label_distribution.get('negative', 0)
        neutral_samples = label_distribution.get('neutral', 0)
        accuracy = metrics.get('accuracy')
        
        # Insérer le résumé
        summary_query = """
        INSERT INTO performance_summary 
        (batch_id, total_samples, positive_samples, negative_samples, neutral_samples, 
         model_accuracy, processing_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        summary_values = (
            batch_id,
            total_samples,
            positive_samples,
            negative_samples,
            neutral_samples,
            accuracy,
            datetime.now().date()
        )
        
        cursor.execute(summary_query, summary_values)
        logger.info(f"✅ Performance summary saved for batch {batch_id}")
        
    except Exception as e:
        logger.error(f"Failed to save performance summary: {e}")

def get_model_performance_history():
    """Récupérer l'historique des performances du modèle"""
    conn = None
    
    try:
        conn = mysql.connector.connect(
            host='host.docker.internal',
            user="root",
            password="1234",
            database="retail_db",
            autocommit=True,
            connection_timeout=10
        )
        
        cursor = conn.cursor(dictionary=True)
        
        # Récupérer les dernières métriques
        query = """
        SELECT 
            batch_id,
            model_version,
            accuracy,
            training_samples,
            validation_samples,
            classes_used,
            label_distribution,
            update_skipped,
            skip_reason,
            created_at
        FROM model_metrics 
        ORDER BY created_at DESC 
        LIMIT 10
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if results:
            logger.info("📊 Recent model performance history:")
            for result in results:
                logger.info(f"  Batch: {result['batch_id']}, "
                          f"Accuracy: {result['accuracy']}, "
                          f"Train/Val: {result['training_samples']}/{result['validation_samples']}, "
                          f"Date: {result['created_at']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to retrieve model performance history: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def predict_sentiment_with_model(text, model):
    """Prédiction avec le modèle ML"""
    try:
        if not hasattr(model, 'predict'):
            logger.error(f"Model doesn't have predict method. Type: {type(model)}")
            return None
        
        prediction = model.predict([text])[0]
        
        # Récupérer les probabilités si disponibles
        confidence = 0.8  # Valeur par défaut
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba([text])[0]
                confidence = float(max(probabilities))
            except Exception as prob_error:
                logger.warning(f"Could not get probabilities: {prob_error}")
        
        # Normaliser les prédictions
        sentiment = "neutral"
        
        if hasattr(model, 'classes_'):
            classes = model.classes_
            if prediction in classes:
                pred_str = str(prediction).lower()
                if pred_str in ['positive', 'pos', '1', '4']:
                    sentiment = 'positive'
                elif pred_str in ['negative', 'neg', '0']:
                    sentiment = 'negative'
                elif pred_str in ['neutral', '2']:
                    sentiment = 'neutral'
                else:
                    sentiment = pred_str
        else:
            pred_str = str(prediction).lower()
            if pred_str in ['positive', 'pos', '1', 1, 4]:
                sentiment = 'positive'
            elif pred_str in ['negative', 'neg', '0', 0]:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
        
        result = {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "model_used": "ml_model",
            "raw_prediction": str(prediction)
        }
        
        return result
            
    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        return None

def predict_sentiment_fallback(text):
    """Méthode de fallback basée sur des mots-clés"""
    if not text or text.strip() == "":
        return {"sentiment": "neutral", "confidence": 0.5, "model_used": "fallback"}
    
    text_lower = text.lower()
    
    positive_words = [
        'love', 'great', 'amazing', 'excellent', 'perfect', 'good', 'awesome',
        'fantastic', 'wonderful', 'brilliant', 'outstanding', 'superb', 'best',
        'happy', 'satisfied', 'pleased', 'delighted', 'recommend', 'quality'
    ]
    
    negative_words = [
        'hate', 'terrible', 'awful', 'bad', 'worst', 'horrible', 'sucks',
        'disappointing', 'poor', 'useless', 'angry', 'upset', 'dissatisfied',
        'broken', 'defective', 'expensive', 'problem', 'issue', 'complaint'
    ]
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        confidence = min(0.8, 0.6 + (positive_count - negative_count) * 0.1)
        return {"sentiment": "positive", "confidence": confidence, "model_used": "fallback"}
    elif negative_count > positive_count:
        confidence = min(0.8, 0.6 + (negative_count - positive_count) * 0.1)
        return {"sentiment": "negative", "confidence": confidence, "model_used": "fallback"}
    else:
        return {"sentiment": "neutral", "confidence": 0.5, "model_used": "fallback"}

def predict_sentiment(text, model=None):
    """Fonction principale de prédiction de sentiment"""
    if not text or text.strip() == "":
        return {"sentiment": "neutral", "confidence": 0.5, "model_used": "empty_text"}
    
    if model is not None:
        ml_result = predict_sentiment_with_model(text, model)
        if ml_result is not None:
            return ml_result
    
    return predict_sentiment_fallback(text)

def split_data_for_training_validation(tweets_data, train_ratio=0.7):
    """Diviser les données en ensemble d'entraînement et de validation"""
    import random
    
    # Mélanger les données
    shuffled_data = tweets_data.copy()
    random.shuffle(shuffled_data)
    
    # Calculer le point de division
    split_point = int(len(shuffled_data) * train_ratio)
    
    # Diviser les données
    train_data = shuffled_data[:split_point]
    val_data = shuffled_data[split_point:]
    
    logger.info(f"Data split: {len(train_data)} training samples, {len(val_data)} validation samples")
    
    return train_data, val_data

def prepare_training_data(tweet_results):
    """Préparer les données d'entraînement à partir des tweets avec leurs vrais sentiments"""
    training_data = []
    
    for tweet_result in tweet_results:
        # Utiliser le vrai sentiment du tweet (pas la prédiction)
        true_sentiment = tweet_result.get('true_sentiment')  # Le vrai sentiment du tweet
        tweet_text = tweet_result['tweet_text']
        
        if true_sentiment and tweet_text:
            training_data.append((tweet_text, true_sentiment))
    
    logger.info(f"Prepared {len(training_data)} samples for training with true sentiments")
    return training_data

def update_model_with_validation(model, train_data, val_data):
    """Mise à jour du modèle avec validation sur données réelles"""
    if not train_data or not val_data:
        logger.warning("Insufficient training or validation data")
        return model, {'skipped': True, 'reason': 'insufficient_data'}
    
    try:
        # Préparer les données d'entraînement
        train_texts, train_labels = zip(*train_data)
        val_texts, val_labels = zip(*val_data)
        
        # Vérifier la diversité des classes dans l'entraînement
        unique_train_labels = set(train_labels)
        train_label_counts = {label: train_labels.count(label) for label in unique_train_labels}
        
        # Vérifier la diversité des classes dans la validation
        unique_val_labels = set(val_labels)
        val_label_counts = {label: val_labels.count(label) for label in unique_val_labels}
        
        logger.info(f"Training data distribution: {train_label_counts}")
        logger.info(f"Validation data distribution: {val_label_counts}")
        
        if len(unique_train_labels) < 2:
            logger.warning(f"Only {len(unique_train_labels)} class(es) in training data")
            return model, {'skipped': True, 'reason': 'insufficient_class_diversity_train'}
        
        # Entraîner le modèle
        if not hasattr(model, 'classes_') or model.classes_ is None:
            logger.info("Initial model training...")
            model.fit(train_texts, train_labels)
        else:
            # Apprentissage incrémental
            logger.info(f"Incremental learning with {len(train_texts)} samples...")
            
            if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier'), 'partial_fit'):
                # Transformer les textes avec le TF-IDF existant
                X_transformed = model.named_steps['tfidf'].transform(train_texts)
                
                # Mise à jour incrémentale avec toutes les classes possibles
                all_classes = ['negative', 'neutral', 'positive']
                model.named_steps['classifier'].partial_fit(
                    X_transformed, 
                    train_labels, 
                    classes=all_classes
                )
            else:
                # Fallback: ré-entraînement complet
                logger.warning("Model doesn't support partial_fit, using full retraining")
                model.fit(train_texts, train_labels)
        
        # Validation sur les données de test avec les VRAIS sentiments
        val_predictions = model.predict(val_texts)
        
        # Calculer les métriques réelles
        accuracy = accuracy_score(val_labels, val_predictions)
        
        # Rapport de classification détaillé
        from sklearn.metrics import classification_report
        class_report = classification_report(val_labels, val_predictions, output_dict=True, zero_division=0)
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(val_labels, val_predictions, labels=['negative', 'neutral', 'positive'])
        
        # Calculer la distribution totale des labels (train + val)
        all_labels = list(train_labels) + list(val_labels)
        total_label_counts = {label: all_labels.count(label) for label in set(all_labels)}
        
        metrics = {
            'accuracy': float(accuracy),
            'training_samples': len(train_texts),
            'validation_samples': len(val_texts),
            'label_distribution': total_label_counts,
            'classes_used': list(unique_train_labels.union(unique_val_labels)),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'updated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Model validation completed. Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(val_labels, val_predictions)}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Failed to update model with validation: {e}")
        return model, {'error': str(e)}

def save_sentiment_results_to_kafka(results, producer):
    """Sauvegarde des résultats de sentiment dans Kafka"""
    if not producer:
        logger.warning("Kafka producer not available")
        return
    
    try:
        success_count = 0
        for result in results:
            try:
                future = producer.send('sentiment_results_topic', result)
                future.get(timeout=10)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to send result to Kafka: {e}")
        
        producer.flush()
        logger.info(f"Sent {success_count}/{len(results)} results to Kafka")
    except Exception as e:
        logger.error(f"Failed to send results to Kafka: {e}")

def save_sentiment_results_to_mysql(results):
    """Sauvegarde des résultats de sentiment dans MySQL"""
    conn = None
    
    try:
        conn = mysql.connector.connect(
            host='host.docker.internal',
            user="root",
            password="1234",
            database="retail_db",
            autocommit=True,
            connection_timeout=10
        )
        
        cursor = conn.cursor()
        
        # Créer la table si elle n'existe pas
        create_table_query = """
        CREATE TABLE IF NOT EXISTS sentiment_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tweet_id VARCHAR(255) NOT NULL,
            tweet_text TEXT NOT NULL,
            true_sentiment VARCHAR(20) NOT NULL,
            predicted_sentiment VARCHAR(20) NOT NULL,
            confidence DECIMAL(5,4),
            model_used VARCHAR(50),
            raw_prediction VARCHAR(100),
            is_correct BOOLEAN,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            batch_id VARCHAR(100),
            INDEX idx_tweet_id (tweet_id),
            INDEX idx_true_sentiment (true_sentiment),
            INDEX idx_predicted_sentiment (predicted_sentiment),
            INDEX idx_batch_id (batch_id),
            INDEX idx_is_correct (is_correct)
        )
        """
        cursor.execute(create_table_query)
        
        # Générer un ID de batch unique
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success_count = 0
        for result in results:
            try:
                true_sentiment = result['true_sentiment']
                predicted_sentiment = result['sentiment_analysis']['sentiment']
                is_correct = (true_sentiment == predicted_sentiment)
                
                query = """
                INSERT INTO sentiment_results 
                (tweet_id, tweet_text, true_sentiment, predicted_sentiment, confidence, 
                 model_used, raw_prediction, is_correct, batch_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    result['tweet_id'],
                    result['tweet_text'][:1000],
                    true_sentiment,
                    predicted_sentiment,
                    result['sentiment_analysis']['confidence'],
                    result['sentiment_analysis']['model_used'],
                    result['sentiment_analysis'].get('raw_prediction', ''),
                    is_correct,
                    batch_id
                )
                cursor.execute(query, values)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to insert result: {e}")
        
        logger.info(f"Saved {success_count}/{len(results)} results to MySQL (batch: {batch_id})")
        return batch_id
        
    except Exception as e:
        logger.error(f"Failed to save results to MySQL: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def process_tweets_sentiment():
    """Traiter les tweets pour l'analyse de sentiment avec apprentissage incrémental"""
    logger.info("🚀 Starting incremental sentiment analysis pipeline")
    
    try:
        # Créer les topics nécessaires
        create_kafka_topic('tweets_topic')
        create_kafka_topic('sentiment_results_topic')
        
        # Afficher l'historique des performances
        get_model_performance_history()
        
        # Charger le modèle de sentiment
        logger.info("📂 Loading sentiment model...")
        sentiment_model = load_sentiment_model()
        
        # Charger le dernier offset pour éviter les doublons
        last_offset = load_kafka_offset()
        logger.info(f"Starting from Kafka offset: {last_offset}")
        
        # Initialiser le consumer Kafka avec offset manuel
        consumer = KafkaConsumer(
            bootstrap_servers=['kafka:29092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id=None,
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            consumer_timeout_ms=30000,
            max_poll_records=BATCH_SIZE
        )
        
        # Assigner manuellement la partition et définir l'offset
        topic_partition = TopicPartition('tweets_topic', 0)
        consumer.assign([topic_partition])
        consumer.seek(topic_partition, last_offset)
        
        # Initialiser le producer Kafka
        producer = KafkaProducer(
            bootstrap_servers=['kafka:29092'],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        
        results = []
        processed_count = 0
        current_offset = last_offset
        
        logger.info(f"🔍 Starting to consume tweets from offset {last_offset}")
        
        try:
            start_time = time.time()
            timeout_duration = 60
            
            for message in consumer:
                try:
                    if time.time() - start_time > timeout_duration:
                        logger.info(f"Timeout reached after {timeout_duration} seconds")
                        break
                    tweet_data = message.value
                    current_offset = message.offset + 1
                    
                    # Valider les données du tweet
                    if not all(key in tweet_data for key in ['tweet_id', 'tweet_text', 'sentiment']):
                        logger.warning(f"Missing required fields in tweet data: {tweet_data}")
                        continue
                    
                    # Analyser le sentiment avec le modèle ML
                    sentiment_result = predict_sentiment(tweet_data['tweet_text'], sentiment_model)
                    
                    # Créer le résultat final
                    result = {
                        'tweet_id': tweet_data['tweet_id'],
                        'tweet_text': tweet_data['tweet_text'],
                        'true_sentiment': tweet_data['sentiment'],
                        'sentiment_analysis': sentiment_result,
                        'processed_at': datetime.now().isoformat(),
                        'offset': message.offset
                    }
                    
                    results.append(result)
                    processed_count += 1
                    
                    # Traitement par batch
                    if len(results) >= BATCH_SIZE:
                        logger.info(f"Processing batch of {len(results)} tweets...")
                        break
                    
                    # Log du progrès
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count} tweets...")
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
        
        finally:
            consumer.close()
        
        logger.info(f"✅ Consumed {len(results)} tweets for processing")
        
        if not results:
            logger.info("No new tweets to process")
            return
        
        # Sauvegarder les résultats
        logger.info("💾 Saving sentiment results...")
        
        # Sauvegarder dans MySQL
        batch_id = save_sentiment_results_to_mysql(results)
        
        # Sauvegarder dans Kafka
        save_sentiment_results_to_kafka(results, producer)
        
        # Sauvegarder l'offset
        save_kafka_offset('tweets_topic', 0, current_offset)
        
        # Préparer les données pour l'entraînement incrémental
        logger.info("🧠 Preparing incremental learning data...")
        
        # Préparer les données avec les vrais sentiments
        training_data = prepare_training_data(results)
        
        if len(training_data) >= 10:  # Minimum de données pour l'entraînement
            # Diviser en données d'entraînement et de validation
            train_data, val_data = split_data_for_training_validation(training_data, train_ratio=0.7)
            
            if len(train_data) >= 5 and len(val_data) >= 3:
                logger.info("🔄 Updating model with incremental learning...")
                
                # Mettre à jour le modèle
                updated_model, metrics = update_model_with_validation(sentiment_model, train_data, val_data)
                
                # Sauvegarder le modèle mis à jour
                save_updated_model(updated_model, metrics)
                
                # Sauvegarder les métriques dans MySQL
                if batch_id:
                    save_model_metrics_to_mysql(metrics, batch_id)
                
                # Analyser les performances
                analyze_model_performance(metrics, results)
                
                logger.info("✅ Incremental learning completed successfully")
            else:
                logger.warning(f"Insufficient data for training: train={len(train_data)}, val={len(val_data)}")
        else:
            logger.info(f"Not enough data for training: {len(training_data)} samples (minimum 10 required)")
        
        # Nettoyage
        producer.close()
        
        
        logger.info("🎉 Sentiment analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in sentiment analysis pipeline: {e}")
        raise

def analyze_model_performance(metrics, results):
    """Analyser et afficher les performances du modèle"""
    try:
        accuracy = metrics.get('accuracy', 0)
        total_samples = len(results)
        
        # Compter les prédictions correctes
        correct_predictions = sum(1 for result in results 
                                if result['true_sentiment'] == result['sentiment_analysis']['sentiment'])
        
        actual_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        logger.info("📊 MODEL PERFORMANCE ANALYSIS")
        logger.info(f"   Validation Accuracy: {accuracy:.4f}")
        logger.info(f"   Actual Batch Accuracy: {actual_accuracy:.4f}")
        logger.info(f"   Correct Predictions: {correct_predictions}/{total_samples}")
        
        # Analyser par classe
        sentiment_analysis = {}
        for result in results:
            true_sent = result['true_sentiment']
            pred_sent = result['sentiment_analysis']['sentiment']
            
            if true_sent not in sentiment_analysis:
                sentiment_analysis[true_sent] = {'total': 0, 'correct': 0}
            
            sentiment_analysis[true_sent]['total'] += 1
            if true_sent == pred_sent:
                sentiment_analysis[true_sent]['correct'] += 1
        
        logger.info("   Performance by Class:")
        for sentiment, stats in sentiment_analysis.items():
            class_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            logger.info(f"     {sentiment.capitalize()}: {class_accuracy:.4f} "
                       f"({stats['correct']}/{stats['total']})")
        
        # Analyser la distribution des modèles utilisés
        model_usage = {}
        for result in results:
            model_used = result['sentiment_analysis']['model_used']
            model_usage[model_used] = model_usage.get(model_used, 0) + 1
        
        logger.info("   Model Usage:")
        for model, count in model_usage.items():
            percentage = (count / total_samples) * 100
            logger.info(f"     {model}: {count} samples ({percentage:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error analyzing model performance: {e}")



def main():
    """Fonction principale"""
    try:
        logger.info("🚀 Starting Incremental Sentiment Analysis System")
        
        
        
        # Traiter les tweets
        process_tweets_sentiment()
        
        logger.info("✅ System completed successfully")
        
    except KeyboardInterrupt:
        logger.info("🛑 System interrupted by user")
    except Exception as e:
        logger.error(f"💥 System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()