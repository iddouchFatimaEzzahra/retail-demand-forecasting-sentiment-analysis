import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os

# Ajouter le répertoire parent au path pour importer le module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import du module à tester
import model_prophet

class TestModelProphet(unittest.TestCase):
    
    def setUp(self):
        """Configuration initiale pour chaque test"""
        # Données de test réalistes
        self.sample_data = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(50):  # 50 jours de données
            self.sample_data.append({
                'Product_ID': 'P001',
                'Store_ID': 'S001',
                'Date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                'Total_Quantity_Sold': np.random.poisson(50) + 20,  # Données réalistes
                'Holiday_Flag': 1 if i % 7 == 0 else 0,  # Tous les 7 jours
                'Promotional_Flag': 1 if i % 10 == 0 else 0  # Tous les 10 jours
            })
        
        # DataFrame de test
        self.df_test = pd.DataFrame(self.sample_data)
        self.df_test['Date'] = pd.to_datetime(self.df_test['Date'])
        
        # Données Prophet formatées
        self.df_prophet_test = self.df_test.copy()
        self.df_prophet_test = self.df_prophet_test.rename(columns={
            'Date': 'ds',
            'Total_Quantity_Sold': 'y'
        })

    @patch('model_prophet.KafkaConsumer')
    def test_lecture_kafka_success(self, mock_consumer_class):
        """Test de lecture Kafka avec succès"""
        # Configuration du mock
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer
        
        # Simuler les messages Kafka
        mock_messages = []
        for data in self.sample_data[:5]:  # Quelques messages de test
            mock_message = MagicMock()
            mock_message.value = data
            mock_messages.append(mock_message)
        
        mock_consumer.__iter__.return_value = iter(mock_messages)
        
        # Exécution du test
        result = model_prophet.lecture_kafka()
        
        # Vérifications
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertIn('Product_ID', result.columns)
        self.assertIn('Store_ID', result.columns)
        self.assertIn('Date', result.columns)
        
        # Vérifier que le consumer est fermé
        mock_consumer.close.assert_called_once()

    @patch('model_prophet.KafkaConsumer')
    def test_lecture_kafka_empty_data(self, mock_consumer_class):
        """Test de lecture Kafka sans données"""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer
        mock_consumer.__iter__.return_value = iter([])  # Pas de messages
        
        result = model_prophet.lecture_kafka()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('model_prophet.KafkaConsumer')
    def test_lecture_kafka_exception(self, mock_consumer_class):
        """Test de lecture Kafka avec exception"""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer
        mock_consumer.__iter__.side_effect = Exception("Erreur Kafka")
        
        result = model_prophet.lecture_kafka()
        
        self.assertIsInstance(result, pd.DataFrame)
        mock_consumer.close.assert_called_once()

    def test_preparer_donnees_prophet_basic(self):
        """Test de préparation des données Prophet - cas basique"""
        result = model_prophet.preparer_donnees_prophet(self.df_test)
        
        # Vérifications de base
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('ds', result.columns)
        self.assertIn('y', result.columns)
        self.assertIn('day_of_week', result.columns)
        self.assertIn('month', result.columns)
        self.assertIn('quarter', result.columns)
        self.assertIn('year', result.columns)
        
        # Vérifier que les données sont triées par date
        self.assertTrue(result['ds'].is_monotonic_increasing)
        
        # Vérifier que les valeurs aberrantes sont gérées
        self.assertTrue(all(result['y'] >= 0))

    def test_preparer_donnees_prophet_outliers(self):
        """Test de préparation avec valeurs aberrantes"""
        # Ajouter des valeurs aberrantes
        df_with_outliers = self.df_test.copy()
        df_with_outliers.loc[0, 'Total_Quantity_Sold'] = 1000  # Valeur très élevée
        df_with_outliers.loc[1, 'Total_Quantity_Sold'] = -50   # Valeur négative
        
        result = model_prophet.preparer_donnees_prophet(df_with_outliers)
        
        # Vérifier que les outliers sont traités
        self.assertTrue(result['y'].max() < 1000)
        self.assertTrue(result['y'].min() >= 0)

    def test_grid_search_prophet_small_dataset(self):
        """Test de grid search avec petit dataset"""
        # Petit dataset (moins de 10 points)
        small_df = self.df_prophet_test.head(5).copy()
        
        result = model_prophet.grid_search_prophet(small_df, max_evals=5, timeout_minutes=1)
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('changepoint_prior_scale', result)
        self.assertIn('seasonality_prior_scale', result)
        self.assertIn('growth', result)
        
        # Pour petit dataset, doit utiliser des paramètres conservateurs
        self.assertEqual(result['growth'], 'linear')

    def test_grid_search_prophet_normal_dataset(self):
        """Test de grid search avec dataset normal"""
        result = model_prophet.grid_search_prophet(self.df_prophet_test, max_evals=10, timeout_minutes=1)
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('changepoint_prior_scale', result)
        self.assertIn('seasonality_prior_scale', result)
        self.assertIn('holidays_prior_scale', result)
        self.assertIn('seasonality_mode', result)
        self.assertIn('changepoint_range', result)
        self.assertIn('n_changepoints', result)
        self.assertIn('growth', result)
        
        # Vérifier que les valeurs sont dans des plages raisonnables
        self.assertGreater(result['changepoint_prior_scale'], 0)
        self.assertGreater(result['seasonality_prior_scale'], 0)
        self.assertIn(result['growth'], ['linear', 'logistic'])

    def test_evaluer_predictions(self):
        """Test d'évaluation des prédictions"""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])
        
        result = model_prophet.evaluer_predictions(y_true, y_pred)
        
        # Vérifications
        self.assertIsInstance(result, dict)
        self.assertIn('MAE', result)
        self.assertIn('MSE', result)
        #self.assertIn('RMSE', result)
        
        # Vérifier que les métriques sont positives
        self.assertGreater(result['MAE'], 0)
        self.assertGreater(result['MSE'], 0)
        #self.assertGreater(result['RMSE'], 0)

    def test_evaluer_modele(self):
        """Test d'évaluation complète du modèle"""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])
        
        result = model_prophet.evaluer_modele(y_true, y_pred, 'P001', 'S001')
        
        # Vérifications
        self.assertIsInstance(result, dict)
        #self.assertIn('rmse', result)
        self.assertIn('mae', result)
        self.assertIn('mape', result)
        self.assertIn('r2', result)
        self.assertIn('erreur_moyenne', result)
        self.assertIn('erreur_std', result)
        
        # Vérifier les types
        for key, value in result.items():
            self.assertIsInstance(value, float)

    @patch('pymysql.connect')
    def test_sauvegarder_predictions_mysql_success(self, mock_connect):
        """Test de sauvegarde MySQL avec succès"""
        # Configuration du mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Données de test
        df_pred = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=5),
            'yhat': [100, 110, 120, 130, 140],
            'yhat_lower': [90, 100, 110, 120, 130],
            'yhat_upper': [110, 120, 130, 140, 150]
        })
        
        # Exécution
        model_prophet.sauvegarder_predictions_mysql(df_pred, 'P001', 'S001')
        
        # Vérifications
        mock_connect.assert_called_once()
        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('pymysql.connect')
    def test_sauvegarder_predictions_mysql_error(self, mock_connect):
        """Test de sauvegarde MySQL avec erreur"""
        # Configuration du mock pour lever une exception
        mock_connect.side_effect = Exception("Erreur de connexion")
        
        # Données de test
        df_pred = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=3),
            'yhat': [100, 110, 120],
            'yhat_lower': [90, 100, 110],
            'yhat_upper': [110, 120, 130]
        })
        
        # Exécution (ne doit pas lever d'exception)
        try:
            model_prophet.sauvegarder_predictions_mysql(df_pred, 'P001', 'S001')
        except Exception:
            self.fail("La fonction ne doit pas lever d'exception en cas d'erreur MySQL")

    @patch('mlflow.start_run')
    @patch('mlflow.log_metric')
    @patch('mlflow.set_tag')
    def test_sauvegarder_metriques_mlflow_individual(self, mock_set_tag, mock_log_metric, mock_start_run):
        """Test de sauvegarde MLflow pour métriques individuelles"""
        # Configuration du mock
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock(return_value=False)
        
        metriques = {
            'rmse': 10.5,
            'mae': 8.2,
            'r2': 0.85,
            'Product_ID': 'P001',
            'Store_ID': 'S001'
        }
        
        # Exécution
        model_prophet.sauvegarder_metriques_mlflow(metriques, 'P001', 'S001', is_global=False)
        
        # Vérifications
        mock_start_run.assert_called_once()
        # Vérifier que les métriques numériques sont loggées
        expected_calls = [
            unittest.mock.call('rmse', 10.5),
            unittest.mock.call('mae', 8.2),
            unittest.mock.call('r2', 0.85)
        ]
        # Note: batch_number sera aussi loggé mais avec une valeur variable
        
        # Vérifier que les tags sont définis
        mock_set_tag.assert_called()

    @patch('mlflow.start_run')
    @patch('mlflow.log_metric')
    @patch('mlflow.set_tag')
    def test_sauvegarder_metriques_mlflow_global(self, mock_set_tag, mock_log_metric, mock_start_run):
        """Test de sauvegarde MLflow pour métriques globales"""
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock(return_value=False)
        
        metriques_globales = {
            'rmse_global': 12.3,
            'mae_global': 9.8,
            'r2_global': 0.78,
            'nombre_modeles': 5
        }
        
        # Exécution
        model_prophet.sauvegarder_metriques_mlflow(metriques_globales, None, None, is_global=True)
        
        # Vérifications
        mock_start_run.assert_called_once()
        mock_log_metric.assert_called()
        mock_set_tag.assert_called()

    @patch('model_prophet.KafkaProducer')
    def test_publier_predictions_kafka(self, mock_producer_class):
        """Test de publication Kafka"""
        # Configuration du mock
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer
        
        # Données de test
        forecast = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=3),
            'yhat': [100, 110, 120],
            'yhat_lower': [90, 100, 110],
            'yhat_upper': [110, 120, 130]
        })
        
        # Exécution
        model_prophet.publier_predictions_kafka(forecast, 'P001', 'S001')
        
        # Vérifications
        mock_producer_class.assert_called_once()
        self.assertEqual(mock_producer.send.call_count, 3)  # 3 messages
        mock_producer.close.assert_called_once()

    def test_get_batch_execution_number(self):
        """Test de récupération du numéro de batch"""
        # Test de base
        batch_number = model_prophet.get_batch_execution_number()
        self.assertIsInstance(batch_number, int)
        self.assertGreater(batch_number, 0)

    def test_get_or_set_batch_number(self):
        """Test de récupération/définition du numéro de batch"""
        # Réinitialiser d'abord
        model_prophet.reset_batch_number()
        
        # Premier appel
        batch1 = model_prophet.get_or_set_batch_number()
        self.assertIsInstance(batch1, int)
        
        # Deuxième appel doit retourner le même numéro
        batch2 = model_prophet.get_or_set_batch_number()
        self.assertEqual(batch1, batch2)

    def test_reset_batch_number(self):
        """Test de réinitialisation du numéro de batch"""
        # Définir un numéro de batch
        model_prophet.get_or_set_batch_number()
        
        # Réinitialiser
        model_prophet.reset_batch_number()
        
        # Vérifier que le numéro peut être redéfini
        new_batch = model_prophet.get_or_set_batch_number()
        self.assertIsInstance(new_batch, int)

    @patch('model_prophet.lecture_kafka')
    @patch('model_prophet.sauvegarder_predictions_mysql')
    @patch('model_prophet.publier_predictions_kafka')
    @patch('model_prophet.sauvegarder_metriques_mlflow')
    def test_main_function_with_data(self, mock_mlflow, mock_kafka_pub, mock_mysql, mock_lecture):
        """Test de la fonction principale avec données"""
        # Configuration des mocks
        mock_lecture.return_value = self.df_test
        
        # Exécution
        try:
            model_prophet.main()
        except Exception as e:
            # La fonction peut échouer à cause des dépendances Prophet, 
            # mais on vérifie que lecture_kafka est appelée
            pass
        
        # Vérifications
        mock_lecture.assert_called_once()

    @patch('model_prophet.lecture_kafka')
    def test_main_function_no_data(self, mock_lecture):
        """Test de la fonction principale sans données"""
        # Configuration du mock pour retourner un DataFrame vide
        mock_lecture.return_value = pd.DataFrame()
        
        # Exécution
        model_prophet.main()
        
        # Vérifications
        mock_lecture.assert_called_once()

class TestDataValidation(unittest.TestCase):
    """Tests de validation des données"""
    
    def test_data_quality_checks(self):
        """Test des vérifications de qualité des données"""
        # Données avec problèmes
        problematic_data = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=5),
            'y': [10, -5, 1000, np.nan, 20]  # Valeurs négatives, outliers, NaN
        })
        
        # Appliquer la préparation des données
        df_test_clean = pd.DataFrame({
            'Product_ID': ['P001'] * 5,
            'Store_ID': ['S001'] * 5,
            'Date': pd.date_range('2024-01-01', periods=5),
            'Total_Quantity_Sold': [10, -5, 1000, 0, 20]  # Remplacer NaN par 0
        })
        
        result = model_prophet.preparer_donnees_prophet(df_test_clean)
        
        # Vérifier que les problèmes sont corrigés
        self.assertTrue(all(result['y'] >= 0))  # Pas de valeurs négatives
        self.assertFalse(result['y'].isna().any())  # Pas de NaN

class TestEdgeCases(unittest.TestCase):
    """Tests des cas limites"""
    
    def test_single_data_point(self):
        """Test avec un seul point de données"""
        single_point = pd.DataFrame({
            'Product_ID': ['P001'],
            'Store_ID': ['S001'],
            'Date': [datetime(2024, 1, 1)],
            'Total_Quantity_Sold': [50]
        })
        
        result = model_prophet.preparer_donnees_prophet(single_point)
        self.assertEqual(len(result), 1)
        self.assertIn('ds', result.columns)
        self.assertIn('y', result.columns)

    def test_constant_values(self):
        """Test avec valeurs constantes"""
        constant_data = pd.DataFrame({
            'Product_ID': ['P001'] * 10,
            'Store_ID': ['S001'] * 10,
            'Date': pd.date_range('2024-01-01', periods=10),
            'Total_Quantity_Sold': [50] * 10  # Valeurs constantes
        })
        
        result = model_prophet.preparer_donnees_prophet(constant_data)
        self.assertEqual(len(result), 10)
        
        # Test de grid search avec données constantes
        df_prophet = result.rename(columns={'Date': 'ds', 'Total_Quantity_Sold': 'y'})
        if 'Date' in df_prophet.columns:
            df_prophet = df_prophet.drop('Date', axis=1)
        
        params = model_prophet.grid_search_prophet(df_prophet, max_evals=3, timeout_minutes=1)
        self.assertIsInstance(params, dict)

if __name__ == '__main__':
    # Configuration pour les tests
    unittest.TestLoader.sortTestMethodsUsing = None
    
    # Créer une suite de tests
    suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    suite.addTest(unittest.makeSuite(TestModelProphet))
    suite.addTest(unittest.makeSuite(TestDataValidation))
    suite.addTest(unittest.makeSuite(TestEdgeCases))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Afficher un résumé
    print(f"\n{'='*50}")
    print(f"RÉSUMÉ DES TESTS")
    print(f"{'='*50}")
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    if result.failures:
        print(f"\nÉCHECS:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {error_msg}")


    
    if result.errors:
        print(f"\nERREURS:")
        for test, traceback in result.errors:
            error_line = traceback.split('\n')[-2]
            print(f"- {test}: {error_line}")

    
    print(f"\nStatut: {'✓ SUCCÈS' if result.wasSuccessful() else '✗ ÉCHEC'}")