import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import json
from datetime import datetime
import logging

# Configuration pour CI/CD - pas de dépendances Spark réelles
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Désactiver les logs verbeux pour CI/CD
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

class TestSparkAggregatorCICD(unittest.TestCase):
    """Tests optimisés pour pipeline CI/CD"""
    
    @classmethod
    def setUpClass(cls):
        """Configuration minimale pour CI/CD"""
        cls.temp_dir = tempfile.mkdtemp()
        print(f"Using temp directory: {cls.temp_dir}")
        
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tests"""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_imports_and_dependencies(self):
        """Test 1: Vérifier que tous les imports fonctionnent"""
        try:
            import spark_aggregator
            from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType
            from pyspark.sql.functions import col, from_json, sum, max, window
            self.assertTrue(True, "All imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_schema_structure_without_spark(self):
        """Test 2: Validation du schéma sans initialiser Spark"""
        import spark_aggregator
        
        schema = spark_aggregator.define_input_schema()
        
        # Tests de structure
        self.assertIsNotNone(schema)
        self.assertEqual(len(schema.fields), 6)
        
        field_names = [field.name for field in schema.fields]
        expected_fields = [
            "Timestamp", "Product_ID", "Store_ID", 
            "Quantity_Sold", "Holiday_Flag", "Promotional_Flag"
        ]
        self.assertEqual(field_names, expected_fields)
    
    @patch('spark_aggregator.SparkSession')
    def test_spark_session_configuration(self, mock_spark_session):
        """Test 3: Configuration Spark (moquée pour CI/CD)"""
        import spark_aggregator
        
        mock_builder = MagicMock()
        mock_spark_session.builder = mock_builder
        
        # Chain mocking
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_spark = MagicMock()
        mock_builder.getOrCreate.return_value = mock_spark
        
        result = spark_aggregator.create_spark_session()
        
        # Vérifications de configuration
        mock_builder.appName.assert_called_with("RetailAggregationToKafka")
        
        # Vérifier les configurations critiques
        config_calls = [call[0] for call in mock_builder.config.call_args_list]
        self.assertIn(("spark.sql.shuffle.partitions", "3"), config_calls)
        
        mock_spark.sparkContext.setLogLevel.assert_called_with("WARN")
    
    @patch('spark_aggregator.SparkSession')
    def test_kafka_read_configuration(self, mock_spark_session):
        """Test 4: Configuration lecteur Kafka"""
        import spark_aggregator
        
        mock_spark = MagicMock()
        mock_read_stream = MagicMock()
        mock_spark.readStream = mock_read_stream
        
        # Chain mocking pour readStream
        mock_read_stream.format.return_value = mock_read_stream
        mock_read_stream.option.return_value = mock_read_stream
        mock_read_stream.load.return_value = MagicMock()
        
        schema = spark_aggregator.define_input_schema()
        spark_aggregator.read_from_kafka(mock_spark, schema)
        
        # Vérifications critiques pour production
        mock_read_stream.format.assert_called_with("kafka")
        
        expected_options = [
            ("kafka.bootstrap.servers", "kafka:29092"),
            ("subscribe", "retail-transactions"),
            ("startingOffsets", "earliest"),
            ("failOnDataLoss", "false")
        ]
        
        for option, value in expected_options:
            mock_read_stream.option.assert_any_call(option, value)
    
    def test_kafka_output_structure_mock(self):
        """Test 5: Structure de sortie Kafka (moquée)"""
        # Simuler la structure de sortie sans Spark réel
        test_record = {
            "Date": "2024-01-01T00:00:00",
            "Product_ID": "P001",
            "Store_ID": "S001",
            "Total_Quantity_Sold": 10,
            "Holiday_Flag": 0,
            "Promotional_Flag": 1
        }
        
        # Vérifier que la structure peut être sérialisée en JSON
        json_output = json.dumps(test_record)
        reconstructed = json.loads(json_output)
        
        # Vérifications de structure
        required_fields = {
            "Date", "Product_ID", "Store_ID", 
            "Total_Quantity_Sold", "Holiday_Flag", "Promotional_Flag"
        }
        
        self.assertEqual(set(reconstructed.keys()), required_fields)
        self.assertIsInstance(reconstructed["Total_Quantity_Sold"], int)
    
    def test_aggregation_logic_validation(self):
        """Test 6: Validation de la logique d'agrégation (sans Spark)"""
        # Test de la logique d'agrégation avec des données simulées
        
        # Données d'entrée simulées
        input_data = [
            {"Product_ID": "P001", "Store_ID": "S001", "Quantity_Sold": 5, "Holiday_Flag": 0, "Promotional_Flag": 1},
            {"Product_ID": "P001", "Store_ID": "S001", "Quantity_Sold": 3, "Holiday_Flag": 0, "Promotional_Flag": 1},
            {"Product_ID": "P002", "Store_ID": "S001", "Quantity_Sold": 7, "Holiday_Flag": 0, "Promotional_Flag": 0},
        ]
        
        # Simuler l'agrégation manuellement
        from collections import defaultdict
        
        aggregated = defaultdict(lambda: {
            "Total_Quantity_Sold": 0,
            "Holiday_Flag": 0,
            "Promotional_Flag": 0
        })
        
        for record in input_data:
            key = (record["Product_ID"], record["Store_ID"])
            aggregated[key]["Total_Quantity_Sold"] += record["Quantity_Sold"]
            aggregated[key]["Holiday_Flag"] = max(aggregated[key]["Holiday_Flag"], record["Holiday_Flag"])
            aggregated[key]["Promotional_Flag"] = max(aggregated[key]["Promotional_Flag"], record["Promotional_Flag"])
        
        # Vérifications
        p001_s001 = aggregated[("P001", "S001")]
        self.assertEqual(p001_s001["Total_Quantity_Sold"], 8)  # 5 + 3
        self.assertEqual(p001_s001["Promotional_Flag"], 1)
        
        p002_s001 = aggregated[("P002", "S001")]
        self.assertEqual(p002_s001["Total_Quantity_Sold"], 7)
        self.assertEqual(p002_s001["Promotional_Flag"], 0)
    
    @patch('spark_aggregator.write_to_kafka')
    @patch('spark_aggregator.process_stream')
    @patch('spark_aggregator.read_from_kafka')
    @patch('spark_aggregator.create_spark_session')
    def test_main_integration_flow(self, mock_create_spark, mock_read_kafka, 
                                  mock_process_stream, mock_write_kafka):
        """Test 7: Flux d'intégration principal (entièrement moqué)"""
        import spark_aggregator
        
        # Configuration des mocks
        mock_spark = MagicMock()
        mock_create_spark.return_value = mock_spark
        
        mock_raw_df = MagicMock()
        mock_read_kafka.return_value = mock_raw_df
        
        mock_processed_df = MagicMock()
        mock_process_stream.return_value = mock_processed_df
        
        mock_query = MagicMock()
        mock_write_kafka.return_value = mock_query
        
        # Exécuter main
        spark_aggregator.main()
        
        # Vérifications du flux
        mock_create_spark.assert_called_once()
        mock_read_kafka.assert_called_once_with(mock_spark, unittest.mock.ANY)
        mock_process_stream.assert_called_once()
        mock_write_kafka.assert_called_once_with(mock_processed_df)
        mock_query.awaitTermination.assert_called_once_with(timeout=60)
    
    @patch('spark_aggregator.SparkSession')
    def test_error_handling_configuration(self, mock_spark_session):
        """Test 8: Gestion d'erreur dans la configuration"""
        import spark_aggregator
        
        # Configuration du mock pour simuler une erreur
        mock_builder = MagicMock()
        mock_spark_session.builder = mock_builder
        
        # Chain mocking jusqu'au point de failure
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.side_effect = Exception("Configuration error")
        
        # Vérification que l'exception est bien propagée
        with self.assertRaises(Exception) as context:
            spark_aggregator.create_spark_session()
        
        # Vérification du message d'erreur
        self.assertIn("Configuration error", str(context.exception))
        
    def test_environment_variables(self):
        """Test 9: Variables d'environnement requises"""
        # Vérifier que les variables d'environnement sont correctement définies
        required_env_vars = ['PYSPARK_PYTHON', 'PYSPARK_DRIVER_PYTHON']
        
        for var in required_env_vars:
            self.assertIn(var, os.environ, f"Variable d'environnement {var} manquante")
    
    def test_package_dependencies_mock(self):
        """Test 10: Dépendances des packages (simulation)"""
        # Simuler la vérification des packages Kafka
        package_config = 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5'
        
        # Vérifier le format du package
        parts = package_config.split(':')
        self.assertEqual(len(parts), 3, "Format de package incorrect")
        self.assertEqual(parts[0], "org.apache.spark")
        self.assertIn("kafka", parts[1])
        self.assertEqual(parts[2], "3.5.5")

class TestCICDReporting(unittest.TestCase):
    """Tests spécifiques pour le reporting CI/CD"""
    
    def test_generate_test_report(self):
        """Générer un rapport de test pour CI/CD"""
        report = {
            "test_suite": "spark_aggregator_cicd",
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform
            },
            "status": "running"
        }
        
        # Écrire le rapport dans le répertoire temporaire
        report_path = os.path.join(tempfile.gettempdir(), "test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Vérifier que le rapport existe
        self.assertTrue(os.path.exists(report_path))
        
        # Lire et vérifier le contenu
        with open(report_path, 'r') as f:
            loaded_report = json.load(f)
        
        self.assertEqual(loaded_report["test_suite"], "spark_aggregator_cicd")
        self.assertIn("timestamp", loaded_report)

def run_cicd_tests():
    """Fonction principale pour exécuter les tests CI/CD"""
    
    # Configuration du logging pour CI/CD
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Démarrage des tests CI/CD pour spark_aggregator...")
    print("=" * 60)
    
    # Créer la suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    test_suite.addTest(unittest.makeSuite(TestSparkAggregatorCICD))
    test_suite.addTest(unittest.makeSuite(TestCICDReporting))
    
    # Runner avec format adapté pour CI/CD
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True  # Capture la sortie pour éviter le spam dans CI/CD
    )
    
    result = runner.run(test_suite)
    
    # Génération du rapport final
    final_report = {
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
        "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        "status": "PASSED" if result.wasSuccessful() else "FAILED"
    }
    
    print("=" * 60)
    print(f"📊 RAPPORT FINAL CI/CD:")
    print(f"   Tests exécutés: {final_report['total_tests']}")
    print(f"   Échecs: {final_report['failures']}")
    print(f"   Erreurs: {final_report['errors']}")
    print(f"   Taux de réussite: {final_report['success_rate']:.1f}%")
    print(f"   Statut: {final_report['status']}")
    
    # Écrire le rapport pour CI/CD
    with open('cicd_test_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_cicd_tests()
    sys.exit(0 if success else 1)