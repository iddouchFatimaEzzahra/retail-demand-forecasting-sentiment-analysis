from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os
import mlflow
import pymysql

# Ajouter le chemin absolu au PYTHONPATH
sys.path.append('/opt/airflow')

# CORRECTION: Wrapper functions pour éviter l'exécution immédiate
def run_kafka_producer():
    """Wrapper pour exécuter le producer Kafka"""
    from producer.producer import send_to_kafka
    return send_to_kafka()

def run_spark_aggregator():
    """Wrapper pour exécuter l'agrégateur Spark"""
    from spark_aggregator import main as spark_main
    return spark_main()

def run_prophet_model():
    """Wrapper pour exécuter le modèle Prophet"""
    from model_prophet import main as prophet_main
    return prophet_main()

def store_real_sales():
    from kafka import KafkaConsumer
    import json
    import pymysql
    import logging
    from datetime import datetime

    # Initialiser le logger
    logger = logging.getLogger('retail_pipeline_dag')

    consumer = KafkaConsumer(
        'retail_aggregated',
        bootstrap_servers='pfa_mf-kafka-1:29092',
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id='store_sales_group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=10000
    )

    conn = pymysql.connect(
        host='host.docker.internal',
        user='root',
        password='manal',
        database='retail_forecast'
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS real_sales (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ds DATETIME NOT NULL,
                    Product_ID VARCHAR(255) NOT NULL,
                    Store_ID VARCHAR(255) NOT NULL,
                    y REAL NOT NULL
                )
            """)

            for message in consumer:
                try:
                    record = json.loads(message.value.decode('utf-8'))
                    logger.info(f"Processing record: {record}")  # Info

                    timestamp = datetime.strptime(record['Date'], '%Y-%m-%dT%H:%M:%S.%fZ')
                    product_id = record['Product_ID']
                    store_id = record['Store_ID']
                    quantity = float(record['Total_Quantity_Sold'])

                    cursor.execute("""
                        INSERT INTO real_sales 
                        (ds, Product_ID, Store_ID, y)
                        VALUES (%s, %s, %s, %s)
                    """, (timestamp, product_id, store_id, quantity))

                except KeyError as e:
                    logger.warning(f"Missing expected field in record: {e}")
                    logger.warning(f"Problematic record: {record}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

            conn.commit()
            logger.info("Successfully stored all records")
    finally:
        conn.close()


# Définition des arguments par défaut
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,  # ✅ IMPORTANT: Pas de retry pour éviter la double exécution
    'retry_delay': timedelta(minutes=5),
}

# Création du DAG
dag = DAG(
    'retail_pipeline',
    default_args=default_args,
    description='Pipeline de données retail avec Kafka, Spark et Prophet',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,  # ✅ IMPORTANT: Une seule instance du DAG à la fois
)

# 1. Tâche Producer Kafka - CORRIGÉE
producer_task = PythonOperator(
    task_id='kafka_producer',
    python_callable=run_kafka_producer,  # ✅ SANS parenthèses
    dag=dag
)

# Tâche pour supprimer et recréer le topic retail_aggregated
reset_kafka_topic = BashOperator(
    task_id='reset_retail_aggregated_topic',
    bash_command="""
    docker exec pfa_mf-kafka-1 kafka-topics.sh --bootstrap-server kafka:29092 --delete --topic retail_aggregated || true
    sleep 2
    docker exec pfa_mf-kafka-1 kafka-topics.sh --bootstrap-server kafka:29092 --create --topic retail_aggregated --partitions 1 --replication-factor 1
    """,
    dag=dag
)

# 2. Tâche Spark Aggregator - CORRIGÉE
spark_task = PythonOperator(
    task_id='spark_aggregator',
    python_callable=run_spark_aggregator,  # ✅ SANS parenthèses
    dag=dag
)

store_real_sales_task = PythonOperator(
    task_id='store_real_sales',
    python_callable=store_real_sales,
    dag=dag
)

# 3. Tâche Prophet Model - CORRIGÉE
prophet_task = PythonOperator(
    task_id='prophet_model',
    python_callable=run_prophet_model,  # ✅ SANS parenthèses
    dag=dag
)

# Définition des dépendances
producer_task >> reset_kafka_topic >> spark_task >> store_real_sales_task >> prophet_task