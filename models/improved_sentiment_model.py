# improved_sentiment_model.py
import pandas as pd
import numpy as np
import pickle
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime
import os
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedSentimentModelTrainer:
    def __init__(self, model_name="retail_sentiment_v1"):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Créer le dossier des modèles s'il n'existe pas
        os.makedirs('/opt/airflow/models', exist_ok=True)
    
    def preprocess_text(self, text):
        """Préprocesser le texte pour l'analyse de sentiment"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convertir en string si ce n'est pas déjà le cas
        text = str(text)
        
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Supprimer les caractères spéciaux et chiffres
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Si le texte est vide après nettoyage
        if not text:
            return ""
        
        try:
            # Tokenisation
            tokens = word_tokenize(text)
            
            # Supprimer les mots vides et lemmatisation
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            
            return ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Erreur lors du preprocessing: {e}")
            return text
    
    def load_sentiment140_data(self, data_path="data/training.1600000.processed.noemoticon.csv", sample_size=200000):
        """Charger les données Sentiment140"""
        logger.info(f"📂 Chargement des données depuis {data_path}")
        
        # Colonnes du dataset Sentiment140
        columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        
        try:
            # Vérifier si le fichier existe
            if not os.path.exists(data_path):
                logger.error(f"❌ Fichier non trouvé: {data_path}")
                logger.info("🔄 Création de données synthétiques...")
                return self._create_synthetic_data()
            
            # Charger un échantillon des données
            data = pd.read_csv(data_path, encoding='latin-1', names=columns, 
                             nrows=sample_size, header=None)
            
            # Mapper les sentiments (0=négatif, 4=positif) - pas de neutre dans Sentiment140
            sentiment_mapping = {0: 'negative', 4: 'positive'}
            data['sentiment'] = data['sentiment'].map(sentiment_mapping)
            
            # Supprimer les lignes avec sentiment manquant
            data = data.dropna(subset=['sentiment', 'text'])
            
            # Filtrer les textes vides
            data = data[data['text'].str.strip() != '']
            
            logger.info(f"✅ Données chargées: {len(data)} tweets")
            logger.info(f"Distribution des sentiments:\n{data['sentiment'].value_counts()}")
            
            # Ajouter des données neutres synthétiques pour équilibrer
            neutral_data = self._create_neutral_data(len(data) // 10)  # 10% de neutres
            
            final_data = pd.concat([data[['text', 'sentiment']], neutral_data], ignore_index=True)
            
            logger.info(f"✅ Données finales avec neutres: {len(final_data)} tweets")
            logger.info(f"Distribution finale:\n{final_data['sentiment'].value_counts()}")
            
            return final_data
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            # Créer des données synthétiques si le fichier n'existe pas
            return self._create_synthetic_data()
    
    def _create_neutral_data(self, size):
        """Créer des données neutres synthétiques"""
        neutral_texts = [
            "Product arrived on time as expected",
            "Standard quality for the price point",
            "Nothing special but does the job",
            "Average experience neither good nor bad",
            "Product is okay meets basic requirements",
            "Item works as described in listing",
            "Delivery was on schedule no issues",
            "Product matches the description given",
            "Regular customer service experience",
            "Standard packaging and shipping process",
            "Item received in expected condition",
            "Process was straightforward and simple",
            "No complaints about the transaction",
            "Everything went according to plan",
            "Product functions as it should",
            "Typical online shopping experience",
            "Item arrived in reasonable timeframe",
            "Customer service was adequate",
            "Product quality meets expectations",
            "No major issues with the order"
        ]
        
        # Répliquer pour atteindre la taille demandée
        multiplier = (size // len(neutral_texts)) + 1
        expanded_texts = (neutral_texts * multiplier)[:size]
        
        return pd.DataFrame({
            'text': expanded_texts,
            'sentiment': ['neutral'] * len(expanded_texts)
        })
    
    def _create_synthetic_data(self):
        """Créer des données synthétiques pour l'entraînement"""
        logger.info("🔄 Création de données synthétiques...")
        
        synthetic_data = [
            # Tweets positifs
            ("I love this product amazing quality and fast shipping", "positive"),
            ("Great service at this store highly recommend to everyone", "positive"),
            ("Excellent customer experience will definitely buy again", "positive"),
            ("Best purchase I have made this year absolutely perfect", "positive"),
            ("Outstanding quality and incredibly fast delivery service", "positive"),
            ("Amazing product quality exceeded all my expectations completely", "positive"),
            ("Fantastic customer support team helped me resolve issues quickly", "positive"),
            ("Love the design and functionality works perfectly every time", "positive"),
            ("Incredible value for money definitely worth every penny spent", "positive"),
            ("Superb quality construction and attention to detail evident", "positive"),
            
            # Tweets négatifs
            ("Terrible product quality complete waste of money and time", "negative"),
            ("Poor customer service very disappointed with entire experience", "negative"),
            ("Product broke after one day awful quality control", "negative"),
            ("Never shopping here again horrible experience overall", "negative"),
            ("Completely unsatisfied with this purchase regret buying", "negative"),
            ("Worst customer service ever encountered extremely frustrating", "negative"),
            ("Product does not work as advertised misleading description", "negative"),
            ("Overpriced for such poor quality materials and construction", "negative"),
            ("Delivery was delayed multiple times terrible communication", "negative"),
            ("Product arrived damaged and return process was complicated", "negative"),
            
            # Tweets neutres
            ("Product arrived on time as expected nothing special", "neutral"),
            ("Standard quality for the price point meets basic needs", "neutral"),
            ("Nothing special but does the job adequately", "neutral"),
            ("Average experience neither particularly good nor bad", "neutral"),
            ("Product is okay meets basic requirements sufficiently", "neutral"),
            ("Item works as described in the listing", "neutral"),
            ("Delivery was on schedule no major issues", "neutral"),
            ("Product matches the description provided by seller", "neutral"),
            ("Regular customer service experience nothing noteworthy", "neutral"),
            ("Standard packaging and shipping process typical", "neutral"),
        ]
        
        # Répliquer les données pour avoir plus d'échantillons
        synthetic_data = synthetic_data * 2000  # 60,000 échantillons
        
        return pd.DataFrame(synthetic_data, columns=['text', 'sentiment'])
    
    def train_model(self, data, test_size=0.2):
        """Entraîner le modèle de sentiment"""
        logger.info("🚀 Début de l'entraînement du modèle")
        
        # Préprocesser les textes
        logger.info("🔄 Préprocessing des textes...")
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        
        # Supprimer les textes vides après preprocessing
        data = data[data['processed_text'].str.strip() != '']
        
        if len(data) == 0:
            raise ValueError("Aucun texte valide après preprocessing")
        
        # Séparer les données
        X = data['processed_text']
        y = data['sentiment']
        
        logger.info(f"📊 Données d'entraînement: {len(X)} échantillons")
        logger.info(f"📊 Distribution: {y.value_counts().to_dict()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Créer le pipeline avec paramètres optimisés
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b\w+\b'
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=2000,
                class_weight='balanced',
                C=1.0,
                solver='liblinear'
            ))
        ])
        
        # Entraîner le modèle
        logger.info("🔄 Entraînement en cours...")
        self.model.fit(X_train, y_train)
        
        # Évaluer le modèle
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"✅ Entraînement terminé - Accuracy: {accuracy:.4f}")
        logger.info("\n📊 Rapport de classification:")
        logger.info(classification_report(y_test, y_pred))
        
        return accuracy, y_test, y_pred
    
    def predict_sentiment(self, text):
        """Prédire le sentiment d'un texte - méthode compatible avec sentiment_processor.py"""
        if self.model is None:
            raise ValueError("Modèle non entraîné. Chargez un modèle ou entraînez-en un.")
        
        if not text or str(text).strip() == "":
            return {"sentiment": "neutral", "confidence": 0.5}
        
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {"sentiment": "neutral", "confidence": 0.5}
            
            prediction = self.model.predict([processed_text])[0]
            probabilities = self.model.predict_proba([processed_text])[0]
            confidence = max(probabilities)
            
            return {
                'sentiment': prediction,
                'confidence': float(confidence)
            }
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def predict_sentiment_long_text(self, text):
        """Prédire le sentiment d'un texte (méthode étendue)"""
        result = self.predict_sentiment(text)
        
        if self.model is not None:
            try:
                processed_text = self.preprocess_text(text)
                if processed_text:
                    probabilities = self.model.predict_proba([processed_text])[0]
                    result['probabilities'] = dict(zip(self.model.classes_, probabilities))
            except Exception as e:
                logger.warning(f"Impossible d'obtenir les probabilités: {e}")
        
        return result
    
    def save_model(self, path):
        """Sauvegarder le modèle avec gestion d'erreurs améliorée"""
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        try:
            # Créer le dossier si nécessaire
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat(),
                'version': '2.0.0'
            }
            
            # Sauvegarder avec pickle
            with open(path, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Vérifier que le fichier a été créé correctement
            if os.path.exists(path) and os.path.getsize(path) > 0:
                logger.info(f"💾 Modèle sauvegardé avec succès: {path}")
                logger.info(f"📏 Taille du fichier: {os.path.getsize(path)} bytes")
            else:
                raise Exception("Fichier non créé ou vide")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
            raise
    
    def load_model(self, path):
        """Charger un modèle sauvegardé avec gestion d'erreurs améliorée"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fichier modèle non trouvé: {path}")
            
            if os.path.getsize(path) == 0:
                raise ValueError(f"Fichier modèle vide: {path}")
            
            logger.info(f"📂 Chargement du modèle depuis: {path}")
            logger.info(f"📏 Taille du fichier: {os.path.getsize(path)} bytes")
            
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            if not isinstance(model_data, dict):
                raise ValueError("Format de modèle invalide")
            
            if 'model' not in model_data:
                raise ValueError("Modèle manquant dans les données")
            
            self.model = model_data['model']
            self.model_name = model_data.get('model_name', 'loaded_model')
            
            logger.info(f"✅ Modèle chargé avec succès: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise

class SentimentModelPipeline:
    def __init__(self, model_name="retail_sentiment_v2"):
        self.model_name = model_name
        self.trainer = ImprovedSentimentModelTrainer(model_name)
        self.model_version = "2.0.0"
        
        # Initialiser MLflow (optionnel)
        try:
            mlflow.set_experiment("retail_sentiment_analysis")
        except Exception as e:
            logger.warning(f"MLflow non disponible: {e}")
    
    def train_and_save_model(self, data_path="data/training.1600000.processed.noemoticon.csv"):
        """Entraîner et sauvegarder le modèle final"""
        logger.info("🚀 Début de l'entraînement du modèle de sentiment")
        
        # Charger et préprocesser les données
        data = self.trainer.load_sentiment140_data(data_path)
        
        # Entraîner le modèle
        accuracy, y_test, y_pred = self.trainer.train_model(data, test_size=0.2)
        
        # Sauvegarder le modèle
        model_path = f"/opt/airflow/models/{self.model_name}_v{self.model_version}.pkl"
        self.trainer.save_model(model_path)
        
        # Sauvegarder les métriques
        self._save_model_metrics(accuracy, y_test, y_pred)
        
        logger.info(f"✅ Modèle sauvegardé: {model_path}")
        logger.info(f"📊 Accuracy finale: {accuracy:.4f}")
        
        return model_path, accuracy
    
    def _save_model_metrics(self, accuracy, y_test, y_pred):
        """Sauvegarder les métriques dans MLflow"""
        try:
            with mlflow.start_run():
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_param("model_version", self.model_version)
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("training_date", datetime.now().isoformat())
                
                # Log classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                for label, metrics in report.items():
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            mlflow.log_metric(f"{label}_{metric}", value)
                
                # Sauvegarder le modèle dans MLflow
                mlflow.sklearn.log_model(self.trainer.model, "model")
                
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder dans MLflow: {e}")
    
    def test_model_on_sample_tweets(self, model_path):
        """Tester le modèle sur des tweets d'exemple"""
        # Charger le modèle
        trainer = ImprovedSentimentModelTrainer()
        trainer.load_model(model_path)
        
        # Tweets de test
        test_tweets = [
            "Just got my iPhone 13 from the Apple Store in NYC - amazing device and great service!",
            "Terrible experience at Best Buy today, long wait times and completely unhelpful staff",
            "Thinking about buying the new Samsung Galaxy, any recommendations from users?",
            "Love my new Nike shoes, super comfortable for running and great design!",
            "Had serious issues with my Sony headphones, sound quality is really poor",
            "Great customer service at the Miami store, very helpful and friendly staff",
            "Product arrived on time and works exactly as expected, nothing special",
            "Worst purchase ever made, completely disappointed with the quality and service"
        ]
        
        logger.info("🧪 Test du modèle sur des tweets d'exemple:")
        for i, tweet in enumerate(test_tweets, 1):
            result = trainer.predict_sentiment_long_text(tweet)
            logger.info(f"{i}. Tweet: {tweet[:60]}...")
            logger.info(f"   Sentiment: {result['sentiment']} (confiance: {result['confidence']:.3f})")
            logger.info("-" * 70)

if __name__ == "__main__":
    # Entraîner le modèle final
    pipeline = SentimentModelPipeline()
    model_path, accuracy = pipeline.train_and_save_model()
    
    # Tester le modèle
    pipeline.test_model_on_sample_tweets(model_path)