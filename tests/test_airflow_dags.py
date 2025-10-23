# tests/test_retail_pipeline_dag.py
import unittest
import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
import json
import sys
import os

# Ajouter le chemin pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dags.retail_pipeline_dag import (
    run_kafka_producer,
    run_spark_aggregator,
    run_prophet_model,
    store_real_sales,
    dag
)

class TestRetailPipelineDAG:
    """Tests pour le DAG du pipeline retail"""
    
    def test_dag_configuration(self):
        """Test de la configuration du DAG"""
        assert dag.dag_id == 'retail_pipeline'
        assert dag.schedule_interval == timedelta(days=1)
        assert dag.catchup is False
        assert dag.max_active_runs == 1
        assert dag.default_args['retries'] == 0
    
    def test_dag_tasks_count(self):
        """Test du nombre de tâches dans le DAG"""
        assert len(dag.tasks) == 5
        
    def test_dag_task_ids(self):
        """Test de la présence des tâches attendues"""
        task_ids = [task.task_id for task in dag.tasks]
        expected_tasks = [
            'kafka_producer',
            'reset_retail_aggregated_topic',
            'spark_aggregator',
            'store_real_sales',
            'prophet_model'
        ]
        for task_id in expected_tasks:
            assert task_id in task_ids
    
    def test_task_dependencies(self):
        """Test des dépendances entre tâches"""
        producer_task = dag.get_task('kafka_producer')
        reset_task = dag.get_task('reset_retail_aggregated_topic')
        spark_task = dag.get_task('spark_aggregator')
        store_task = dag.get_task('store_real_sales')
        prophet_task = dag.get_task('prophet_model')
        
        # Vérifier les dépendances upstream
        assert reset_task in producer_task.downstream_list
        assert spark_task in reset_task.downstream_list
        assert store_task in spark_task.downstream_list
        assert prophet_task in store_task.downstream_list

class TestKafkaProducer:
    """Tests pour la fonction Kafka Producer"""
    
    @mock.patch('producer.producer.send_to_kafka')
    def test_run_kafka_producer(self, mock_send_to_kafka):
        """Test du wrapper du producer Kafka"""
        mock_send_to_kafka.return_value = "Success"
        
        result = run_kafka_producer()
        
        mock_send_to_kafka.assert_called_once()
        assert result == "Success"
    
    @mock.patch('producer.producer.send_to_kafka')
    def test_kafka_producer_exception(self, mock_send_to_kafka):
        """Test de gestion d'exception du producer Kafka"""
        mock_send_to_kafka.side_effect = Exception("Kafka connection failed")
        
        with pytest.raises(Exception) as exc_info:
            run_kafka_producer()
        
        assert "Kafka connection failed" in str(exc_info.value)

class TestSparkAggregator:
    """Tests pour la fonction Spark Aggregator"""
    
    @mock.patch('spark_aggregator.main')
    def test_run_spark_aggregator(self, mock_spark_main):
        """Test du wrapper de l'agrégateur Spark"""
        mock_spark_main.return_value = "Aggregation completed"
        
        result = run_spark_aggregator()
        
        mock_spark_main.assert_called_once()
        assert result == "Aggregation completed"
    
    @mock.patch('spark_aggregator.main')
    def test_spark_aggregator_exception(self, mock_spark_main):
        """Test de gestion d'exception de l'agrégateur Spark"""
        mock_spark_main.side_effect = Exception("Spark job failed")
        
        with pytest.raises(Exception) as exc_info:
            run_spark_aggregator()
        
        assert "Spark job failed" in str(exc_info.value)

class TestProphetModel:
    """Tests pour la fonction Prophet Model"""
    
    @mock.patch('model_prophet.main')
    def test_run_prophet_model(self, mock_prophet_main):
        """Test du wrapper du modèle Prophet"""
        mock_prophet_main.return_value = "Model trained successfully"
        
        result = run_prophet_model()
        
        mock_prophet_main.assert_called_once()
        assert result == "Model trained successfully"
    
    @mock.patch('model_prophet.main')
    def test_prophet_model_exception(self, mock_prophet_main):
        """Test de gestion d'exception du modèle Prophet"""
        mock_prophet_main.side_effect = Exception("Model training failed")
        
        with pytest.raises(Exception) as exc_info:
            run_prophet_model()
        
        assert "Model training failed" in str(exc_info.value)

class TestStoreRealSales(unittest.TestCase):
    """Tests pour la fonction de stockage des ventes réelles"""
    
    @mock.patch('dags.retail_pipeline_dag.pymysql.connect')
    @mock.patch('kafka.KafkaConsumer')
    def test_store_real_sales_success(self, mock_kafka_consumer, mock_db_connect):
        """Test du stockage réussi des ventes"""
        # Mock des données Kafka
        mock_message = mock.MagicMock()  # <-- Utiliser MagicMock au lieu de Mock
        mock_message.value = json.dumps({
            'Date': '2023-01-01T10:00:00.000Z',
            'Product_ID': 'P001',
            'Store_ID': 'S001',
            'Total_Quantity_Sold': 100
        }).encode('utf-8')

        # Créer un MagicMock pour supporter __iter__
        mock_consumer_instance = mock.MagicMock()
        mock_consumer_instance.__iter__.return_value = iter([mock_message])
        mock_kafka_consumer.return_value = mock_consumer_instance

        # Configuration du mock MySQL
        mock_cursor = mock.MagicMock()
        mock_connection = mock.MagicMock()
        
        # Support du context manager
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_db_connect.return_value = mock_connection

        from dags.retail_pipeline_dag import store_real_sales
        store_real_sales()

        # Vérifications
        mock_cursor.execute.assert_any_call(mock.ANY)  # CREATE TABLE
        mock_cursor.execute.assert_any_call(
            mock.ANY,
            (datetime(2023, 1, 1, 10, 0), 'P001', 'S001', 100.0)
        )
        mock_connection.commit.assert_called_once()
    
    @mock.patch('dags.retail_pipeline_dag.pymysql.connect')
    @mock.patch('kafka.KafkaConsumer')
    def test_store_real_sales_missing_field(self, mock_kafka_consumer, mock_db_connect):
        """Test de gestion des champs manquants"""

        import json
        import logging
        from unittest import mock

        # Mock du message Kafka avec champ manquant
        mock_message = mock.MagicMock()
        mock_message.value = json.dumps({
            'Date': '2023-01-01T10:00:00.000Z',
            'Product_ID': 'P001',
            # Store_ID manquant intentionnellement
            'Total_Quantity_Sold': 100
        }).encode('utf-8')

        # Configuration du mock Kafka
        mock_consumer_instance = mock.MagicMock()
        mock_consumer_instance.__iter__.return_value = iter([mock_message])
        mock_kafka_consumer.return_value = mock_consumer_instance

        # Configuration du mock MySQL
        mock_cursor = mock.MagicMock()
        mock_connection = mock.MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_db_connect.return_value = mock_connection

        # Initialisation du logger (pas utile ici si on utilise print, sinon adapter le code à logger.warning)
        logger = logging.getLogger('dags.retail_pipeline_dag')
        logger.setLevel(logging.WARNING)

        from dags.retail_pipeline_dag import store_real_sales
        store_real_sales()

        # Vérifie que la table a été créée une fois
        creation_calls = [
            call for call in mock_cursor.execute.call_args_list
            if "CREATE TABLE IF NOT EXISTS real_sales" in call[0][0]
        ]
        assert len(creation_calls) == 1

        # Vérifie qu’aucune insertion n’a été faite
        insert_calls = [
            call for call in mock_cursor.execute.call_args_list
            if "INSERT INTO real_sales" in call[0][0]
        ]
        assert len(insert_calls) == 0


class TestIntegration:
    """Tests d'intégration"""
    
    @mock.patch('producer.producer.send_to_kafka')
    @mock.patch('spark_aggregator.main')
    @mock.patch('model_prophet.main')
    def test_pipeline_components_integration(self, mock_prophet, mock_spark, mock_kafka):
        """Test d'intégration des composants du pipeline"""
        mock_kafka.return_value = "Kafka OK"
        mock_spark.return_value = "Spark OK"
        mock_prophet.return_value = "Prophet OK"
        
        # Test de chaque composant
        kafka_result = run_kafka_producer()
        spark_result = run_spark_aggregator()
        prophet_result = run_prophet_model()
        
        assert kafka_result == "Kafka OK"
        assert spark_result == "Spark OK"
        assert prophet_result == "Prophet OK"
        
        # Vérifier que chaque fonction a été appelée
        mock_kafka.assert_called_once()
        mock_spark.assert_called_once()
        mock_prophet.assert_called_once()

# tests/test_data_quality.py
class TestDataQuality:
    """Tests de qualité des données"""
    
    def test_validate_kafka_message_structure(self):
        """Test de validation de la structure des messages Kafka"""
        valid_message = {
            'Date': '2023-01-01T10:00:00.000Z',
            'Product_ID': 'P001',
            'Store_ID': 'S001',
            'Total_Quantity_Sold': 100
        }
        
        required_fields = ['Date', 'Product_ID', 'Store_ID', 'Total_Quantity_Sold']
        
        for field in required_fields:
            assert field in valid_message
        
        # Test du format de date
        from datetime import datetime
        try:
            datetime.strptime(valid_message['Date'], '%Y-%m-%dT%H:%M:%S.%fZ')
            date_valid = True
        except ValueError:
            date_valid = False
        
        assert date_valid
        
        # Test des types
        assert isinstance(valid_message['Product_ID'], str)
        assert isinstance(valid_message['Store_ID'], str)
        assert isinstance(valid_message['Total_Quantity_Sold'], (int, float))
        assert valid_message['Total_Quantity_Sold'] >= 0
    
    def test_database_schema_validation(self):
        """Test de validation du schéma de base de données"""
        expected_columns = ['id', 'ds', 'Product_ID', 'Store_ID', 'y']
        
        # Simulation de la structure de table
        table_structure = {
            'id': 'INT AUTO_INCREMENT PRIMARY KEY',
            'ds': 'DATETIME NOT NULL',
            'Product_ID': 'VARCHAR(255) NOT NULL',
            'Store_ID': 'VARCHAR(255) NOT NULL',
            'y': 'REAL NOT NULL'
        }
        
        for column in expected_columns:
            assert column in table_structure

# Configuration pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])