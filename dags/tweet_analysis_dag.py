from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os
import logging
import subprocess

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Vérification de l'environnement et des dépendances"""
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Vérification des variables d'environnement importantes
    important_vars = ['AIRFLOW_HOME', 'PYTHONPATH', 'PATH']
    for var in important_vars:
        logger.info(f"{var}: {os.environ.get(var, 'NOT SET')}")
    
    # Test d'import des packages critiques
    critical_packages = ['pandas', 'sklearn', 'json', 'subprocess', 'kafka', 'mysql.connector']
    for package in critical_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError as e:
            logger.error(f"✗ {package} import failed: {e}")
    
    return True


def generate_tweets():
    """Génération des tweets avec la solution alternative (sans bibliothèque groq)"""
    try:
        # Vérifier si le module alternatif existe
        script_path = '/opt/airflow/genere_tweet_alternative.py'
        if not os.path.exists(script_path):
            logger.error(f"genere_tweet_alternative.py not found at {script_path}")
            raise FileNotFoundError(f"Required script {script_path} not found")
        
        from airflow.models import Variable
        
        # Récupérer la clé API depuis les Variables Airflow
        api_key = Variable.get("GROQ_API_KEY", default_var=None)
        
        if not api_key:
            logger.error("GROQ_API_KEY not found in Airflow Variables")
            raise Exception("GROQ_API_KEY is required for tweet generation")
        
        # Configuration Kafka
        kafka_config = {
            'bootstrap_servers': ['kafka:29092']
        }
        
        # Import et utilisation de la solution alternative
        sys.path.append('/opt/airflow')
        from genere_tweet_alternative import AlternativeTweetGenerator
        
        # Initialiser le générateur alternatif
        generator = AlternativeTweetGenerator(api_key, kafka_config)
        
        # Exécuter le pipeline
        generator.run_complete_pipeline(
            target_tweets=50,
            output_file='synthetic_tweets_daily.csv'
        )
        logger.info("Tweet generation completed successfully with alternative method")
        
    except Exception as e:
        logger.error(f"Alternative tweet generation failed: {e}")
        raise

def process_ner():
    """Fonction de traitement NER utilisant le script ner_processor.py"""
    logger.info("Starting NER processing...")
    
    # Vérifier que le script de traitement NER existe
    script_path = "/opt/airflow/ner_processor.py"
    if not os.path.exists(script_path):
        logger.error(f"NER processor script not found at {script_path}")
        raise FileNotFoundError(f"Required script {script_path} not found")
    
    # Configuration de l'environnement
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': '/opt/airflow',
        'PYTHONUNBUFFERED': '1',  # Pour voir les logs en temps réel
    })
    
    try:
        # Exécuter le script de traitement NER
        cmd = [sys.executable, script_path]
        
        logger.info(f"Executing NER command: {' '.join(cmd)}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Exécuter avec timeout et capture des logs
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd="/opt/airflow"
        )
        
        # Lire les logs en temps réel
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(f"NER Process output: {output.strip()}")
        
        # Attendre la fin du processus
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("NER processing completed successfully")
        else:
            logger.error(f"NER processing failed with return code {return_code}")
            raise Exception(f"NER process failed with return code {return_code}")
        
    except subprocess.TimeoutExpired:
        logger.error("NER processing timed out")
        process.kill()
        raise Exception("NER process timed out")
    except FileNotFoundError as e:
        logger.error(f"NER script or executable not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in process_ner: {e}")
        raise

def process_sentiment():
    """Fonction de traitement de sentiment utilisant le script sentiment_processor.py"""
    logger.info("Starting sentiment processing...")
    
    script_path = "/opt/airflow/sentiment_processor.py"
    if not os.path.exists(script_path):
        logger.error(f"Sentiment processor script not found at {script_path}")
        raise FileNotFoundError(f"Required script {script_path} not found")
    
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': '/opt/airflow:/opt/airflow/models',
        'PYTHONUNBUFFERED': '1',
    })
    
    try:
        cmd = [sys.executable, script_path]
        logger.info(f"Executing sentiment command: {' '.join(cmd)}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd="/opt/airflow"
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(f"Sentiment Process output: {output.strip()}")
        
        return_code = process.wait()
        if return_code == 0:
            logger.info("Sentiment processing completed successfully")
        else:
            logger.error(f"Sentiment processing failed with return code {return_code}")
            raise Exception(f"Sentiment process failed with return code {return_code}")
        
    except subprocess.TimeoutExpired:
        logger.error("Sentiment processing timed out")
        process.kill()
        raise Exception("Sentiment process timed out")
    except FileNotFoundError as e:
        logger.error(f"Sentiment script or executable not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in process_sentiment: {e}")
        raise

def cleanup_temp_files():
    """Nettoyage des fichiers temporaires"""
    temp_files = [
        '/tmp/spark-checkpoint',
        'synthetic_tweets_daily.csv',  # Updated to relative path
        '/opt/airflow/synthetic_tweets_daily.json'
    ]
    
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                logger.info(f"Cleaned up: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up {file_path}: {e}")

# Configuration par défaut du DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2025, 6, 1),
    'execution_timeout': timedelta(minutes=30),
}

# Définition du DAG
with DAG(
    'tweet_analysis_pipeline_v3',
    default_args=default_args,
    description='Pipeline d\'analyse de tweets avec NER et sentiment séparés (Version v3)',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['tweets', 'ml', 'analytics', 'ner', 'sentiment', 'v3']
) as dag:

    # Tâche de vérification de l'environnement
    check_env_task = PythonOperator(
        task_id='check_environment',
        python_callable=check_environment,
        doc_md="""
        ### Vérification de l'environnement
        - Vérifie la configuration Python
        - Teste les imports de base
        - Affiche les variables d'environnement importantes
        """,
        execution_timeout=timedelta(minutes=5)
    )

  

    # Tâche de génération des tweets
    generate_tweets_task = PythonOperator(
        task_id='generate_tweets',
        python_callable=generate_tweets,
        doc_md="""
        ### Génération des tweets
        - Génère 50 tweets synthétiques par batch
        - Utilise l'API Groq pour la génération
        - Envoie les tweets vers Kafka topic 'tweets_topic'
        - Sauvegarde dans synthetic_tweets_daily.csv
        """,
        execution_timeout=timedelta(minutes=10)
    )

    # Tâche de traitement NER
    process_ner_task = PythonOperator(
        task_id='process_ner',
        python_callable=process_ner,
        doc_md="""
        ### Traitement NER (Named Entity Recognition)
        - Extrait les entités nommées des tweets
        - Identifie les produits et magasins
        - Sauvegarde les résultats dans Kafka et MySQL
        """,
        execution_timeout=timedelta(minutes=20)
    )

    process_sentiment_task = PythonOperator(
        task_id='process_sentiment',
        python_callable=process_sentiment,
        doc_md="""### Analyse de sentiment
        - Analyse le sentiment des tweets (positif, négatif, neutre)
        - Lit depuis Kafka topic 'tweets_topic'
        - Sauvegarde les résultats dans Kafka topic 'sentiment_results_topic'""",
        execution_timeout=timedelta(minutes=20)
    )

    # Tâche de nettoyage
    cleanup_task = PythonOperator(
        task_id='cleanup_temp_files',
        python_callable=cleanup_temp_files,
        doc_md="""
        ### Nettoyage
        - Supprime les fichiers temporaires
        - Libère l'espace disque
        """,
        trigger_rule='all_done',
        execution_timeout=timedelta(minutes=5)
    )

    # Définition des dépendances
    check_env_task  >> generate_tweets_task
    generate_tweets_task >> [process_ner_task, process_sentiment_task]
    [process_ner_task, process_sentiment_task] >> cleanup_task