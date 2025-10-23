import json
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from sqlalchemy import create_engine
import mlflow
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,root_mean_squared_error
mlflow.set_tracking_uri("http://host.docker.internal:5000")
# Lecture des données Kafka
def lecture_kafka():
    consumer = KafkaConsumer(
        'retail_aggregated',
        bootstrap_servers='kafka:29092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=5000
    )
    data = []
    try:
        for message in consumer:
            print(f"Message brut reçu (type: {type(message.value)}): {message.value}")  # Debug
            data.append(message.value)
    except Exception as e:
        print(f"Erreur Kafka : {e}")
    finally:
        consumer.close()

    # Debug: Afficher la structure des données avant conversion
    print(f"\nExemple de données brutes (premier élément): {data[0] if data else 'AUCUNE DONNÉE'}")
    
    try:
        df_agreger = pd.DataFrame(data)
        print("DataFrame créé avec succès. Colonnes:", df_agreger.columns.tolist())
    except Exception as e:
        print(f"Échec de la création du DataFrame : {e}")
        return pd.DataFrame()  # Retourne un DataFrame vide pour éviter des erreurs ensuite

    # Conversion de la date
    try:
        df_agreger['Date'] = pd.to_datetime(df_agreger['Date']).dt.tz_localize(None)
        print("Conversion de date réussie.")
    except Exception as e:
        print(f"Échec de la conversion de date : {e}")
        df_agreger['Date'] = pd.NaT  # Ou une valeur par défaut

    # Trier (seulement si le DataFrame n'est pas vide)
    if not df_agreger.empty:
        df_agreger = df_agreger.sort_values(['Product_ID', 'Store_ID', 'Date'])
    else:
        print("Avertissement : DataFrame vide, impossible de trier.")

    return df_agreger

# Prépare les données pour Prophet
def preparer_donnees_prophet(df_groupe):
    # Préparation plus avancée des données
    df_prophet = df_groupe.copy()
    
    # Trier par date
    df_prophet = df_prophet.sort_values('Date')
    
    # Ajouter des features temporelles
    df_prophet['day_of_week'] = df_prophet['Date'].dt.dayofweek
    df_prophet['month'] = df_prophet['Date'].dt.month
    df_prophet['quarter'] = df_prophet['Date'].dt.quarter
    df_prophet['year'] = df_prophet['Date'].dt.year
    
    # Gestion plus sophistiquée des outliers
    # Utiliser une moyenne mobile pour lisser les données
    df_prophet['Total_Quantity_Sold'] = df_prophet['Total_Quantity_Sold'].rolling(window=7, min_periods=1).mean()
    
    # Détecter et gérer les valeurs aberrantes avec une méthode plus robuste
    Q1 = df_prophet['Total_Quantity_Sold'].quantile(0.25)
    Q3 = df_prophet['Total_Quantity_Sold'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remplacer les valeurs aberrantes par les bornes
    df_prophet.loc[df_prophet['Total_Quantity_Sold'] < lower_bound, 'Total_Quantity_Sold'] = lower_bound
    df_prophet.loc[df_prophet['Total_Quantity_Sold'] > upper_bound, 'Total_Quantity_Sold'] = upper_bound
    
    # Renommer les colonnes pour Prophet
    df_prophet = df_prophet.rename(columns={
        'Date': 'ds',
        'Total_Quantity_Sold': 'y'
    })
    
    return df_prophet

# Recherche des meilleurs paramètres pour Prophet
def grid_search_prophet(df_prophet, max_evals=20, timeout_minutes=5):
    import random
    import time
    
    print(f"\n=== STARTING GRID SEARCH FOR {len(df_prophet)} DATA POINTS ===")
    
    # Data quality check first
    print(f"Date range: {df_prophet['ds'].min()} to {df_prophet['ds'].max()}")
    print(f"Target variable stats - Mean: {df_prophet['y'].mean():.2f}, Std: {df_prophet['y'].std():.2f}")
    print(f"Min: {df_prophet['y'].min()}, Max: {df_prophet['y'].max()}")
    
    # Check for insufficient data
    if len(df_prophet) < 10:
        print("WARNING: Very limited data - using simple linear model")
        return {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 0.1,
            'holidays_prior_scale': 0.1,
            'seasonality_mode': 'additive',
            'changepoint_range': 0.8,
            'n_changepoints': 5,
            'growth': 'linear'
        }
    
    # Adaptive parameter grid based on data size
    if len(df_prophet) < 30:
        # Small dataset - conservative parameters
        param_grid = {
            'changepoint_prior_scale': [0.01, 0.05, 0.1],
            'seasonality_prior_scale': [0.1, 1.0],
            'holidays_prior_scale': [0.1, 1.0],
            'seasonality_mode': ['additive'],  # More stable for small datasets
            'changepoint_range': [0.8],
            'n_changepoints': [5, 10],
            'growth': ['linear']  # Remove logistic for small datasets
        }
    else:
        # Larger dataset - full parameter space
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_range': [0.8, 0.9],
            'n_changepoints': [10, 20, 30],
            'growth': ['linear', 'logistic']
        }
    
    # Calculate total combinations
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    
    # Generate parameter combinations
    if total_combinations <= max_evals:
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    else:
        all_params = []
        for _ in range(max_evals):
            params = {k: random.choice(v) for k, v in param_grid.items()}
            if params not in all_params:
                all_params.append(params)
    
    # IMPROVED VALIDATION SPLIT - Time-based instead of random
    total_days = (df_prophet['ds'].max() - df_prophet['ds'].min()).days
    
    if total_days < 30:
        # Very short time series - use last 20% but minimum 2 points
        val_size = max(2, len(df_prophet) // 5)
        cutoff = len(df_prophet) - val_size
    else:
        # Longer time series - use time-based split (last 20% of time period)
        cutoff_date = df_prophet['ds'].max() - pd.Timedelta(days=max(7, total_days // 5))
        cutoff = len(df_prophet[df_prophet['ds'] <= cutoff_date])
    
    df_train = df_prophet.iloc[:cutoff].copy()
    df_val = df_prophet.iloc[cutoff:].copy()
    
    print(f"Training set: {len(df_train)} points")
    print(f"Validation set: {len(df_val)} points")
    
    # Check if validation set is reasonable
    if len(df_val) < 2:
        print("WARNING: Validation set too small - adjusting split")
        cutoff = len(df_prophet) - 2
        df_train = df_prophet.iloc[:cutoff].copy()
        df_val = df_prophet.iloc[cutoff:].copy()
    
    best_rmse = float('inf')
    best_params = None
    best_r2 = -float('inf')
    successful_runs = 0
    
    start_time = time.time()
    timeout = timeout_minutes * 60
    
    print(f"Testing {len(all_params)} parameter combinations...")
    
    for i, params in enumerate(all_params):
        if time.time() - start_time > timeout:
            print(f"\nTimeout reached after {i} evaluations")
            break
            
        try:
            # Handle logistic growth properly
            df_train_model = df_train.copy()
            df_val_model = df_val.copy()
            
            if params['growth'] == 'logistic':
                # Set capacity as 120% of historical maximum
                cap_value = df_prophet['y'].max() * 1.2
                if cap_value <= df_prophet['y'].max():
                    # If max is 0 or negative, set a reasonable cap
                    cap_value = max(df_prophet['y'].max() + df_prophet['y'].std(), 1)
                
                df_train_model['cap'] = cap_value
                df_val_model['cap'] = cap_value
            
            # Create model with error handling
            m = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                changepoint_range=params['changepoint_range'],
                n_changepoints=params['n_changepoints'],
                growth=params['growth'],
                daily_seasonality=True if len(df_prophet) >= 14 else False,  # Adaptive seasonality
                yearly_seasonality=False  # Disable to avoid the warning
            )
            
            # Add seasonalities only if we have enough data
            if len(df_prophet) >= 60:  # At least 2 months
                m.add_seasonality(name='monthly', period=30.5, fourier_order=3)
            
            if len(df_prophet) >= 180:  # At least 6 months
                m.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
            
            # Add regressors that exist in the data
            available_regressors = []
            for col in ['day_of_week', 'month', 'quarter', 'year']:
                if col in df_train_model.columns:
                    m.add_regressor(col)
                    available_regressors.append(col)
            
            if 'Holiday_Flag' in df_train_model.columns:
                m.add_regressor('Holiday_Flag')
                available_regressors.append('Holiday_Flag')
            if 'Promotional_Flag' in df_train_model.columns:
                m.add_regressor('Promotional_Flag')
                available_regressors.append('Promotional_Flag')
            
            # Fit model
            m.fit(df_train_model)
            
            # Create future dataframe for validation
            future = m.make_future_dataframe(periods=len(df_val_model))
            
            # Add capacity for logistic growth
            if params['growth'] == 'logistic':
                future['cap'] = cap_value
            
            # Add regressors to future dataframe
            for col in available_regressors:
                if col in df_prophet.columns:
                    future[col] = df_prophet[col].values[:len(future)]
            
            # Make predictions
            forecast = m.predict(future)
            
            # Extract validation predictions
            y_true = df_val_model['y'].values
            y_pred = forecast.iloc[cutoff:]['yhat'].values
            
            # Calculate metrics with error handling
            if len(y_true) != len(y_pred):
                print(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
                continue
            
            # Handle edge cases
            if np.var(y_true) == 0:
                r2 = 0 if np.mean((y_true - y_pred) ** 2) == 0 else -float('inf')
            else:
                r2 = r2_score(y_true, y_pred)
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Update best parameters
            if r2 > best_r2 and not np.isnan(r2) and not np.isinf(r2):
                best_r2 = r2
                best_rmse = rmse
                best_params = params.copy()
                print(f"New best score (R²: {best_r2:.3f}, RMSE: {best_rmse:.2f}) - {params}")
            
            successful_runs += 1
            
        except Exception as e:
            error_msg = str(e)
            if "cap" in error_msg.lower():
                print(f"Logistic growth error (expected): {error_msg}")
            else:
                print(f"Error with params {params}: {error_msg}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"\nGrid search completed in {elapsed_time:.1f} seconds")
    print(f"Successful runs: {successful_runs}/{len(all_params)}")
    
    # Return best parameters or fallback
    if best_params is None:
        print("WARNING: No successful parameter combination found, using default parameters")
        best_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 0.1,
            'holidays_prior_scale': 0.1,
            'seasonality_mode': 'additive',
            'changepoint_range': 0.8,
            'n_changepoints': 10,
            'growth': 'linear'
        }
        best_r2 = 0.0
        best_rmse = float('inf')
    
    print(f"Best parameters found (R²: {best_r2:.3f}, RMSE: {best_rmse:.2f}):")
    print(best_params)
    
    return best_params

# Évaluation des prédictions
def evaluer_predictions(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    metriques = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        #'RMSE': root_mean_squared_error(y_true, y_pred, squared=False)
    }
    return metriques

# Sauvegarder les prédictions dans MySQL
# Sauvegarder les prédictions dans MySQL
def sauvegarder_predictions_mysql(df_pred, product_id, store_id):
    try:
        import pymysql
        from datetime import datetime
        
        # Créer une copie explicite du DataFrame et formater les colonnes
        df_to_save = df_pred.copy()
        df_to_save['Product_ID'] = str(product_id)
        df_to_save['Store_ID'] = str(store_id)
        df_to_save['forecast_made_on'] = datetime.now()  # Ajouter la date de création de la prédiction
        
        # S'assurer que les colonnes correspondent exactement à la table
        df_to_save = df_to_save[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'Product_ID', 'Store_ID', 'forecast_made_on']]
        
        # Établir la connexion
        conn = pymysql.connect(
            host='host.docker.internal',
            user='root',
            password='manal',
            database='retail_forecast'
        )
        
        # Créer un curseur
        cursor = conn.cursor()
        
        # Préparer les données pour l'insertion
        values = []
        for _, row in df_to_save.iterrows():
            values.append((
                row['ds'].strftime('%Y-%m-%d %H:%M:%S'),
                row['yhat'],  # already int from your preprocessing
                row['yhat_lower'],
                row['yhat_upper'],
                str(row['Product_ID']),
                str(row['Store_ID']),
                row['forecast_made_on'].strftime('%Y-%m-%d %H:%M:%S')
            ))
        
        # Requête SQL d'insertion
        sql = """
        INSERT INTO predictions_prophet 
        (ds, yhat, yhat_lower, yhat_upper, Product_ID, Store_ID, forecast_made_on)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # Exécuter l'insertion par lots
        cursor.executemany(sql, values)
        
        # Valider la transaction
        conn.commit()
        
        print(f"Prédictions sauvegardées pour Produit {product_id} et Magasin {store_id}")
        
        # Fermer la connexion
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Échec de sauvegarde MySQL pour {product_id}/{store_id}: {str(e)}")
        if 'conn' in locals():
            conn.close()

# Sauvegarder les métriques dans MLflow
# Sauvegarder les métriques dans MLflow

def get_batch_execution_number():
    """Récupère le numéro d'exécution du batch actuel en se basant sur les runs MLflow existants"""
    try:
        import mlflow
        
        # Rechercher tous les runs pour obtenir le numéro de batch le plus élevé
        runs = mlflow.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name("Default").experiment_id],
            output_format="pandas"
        )
        
        if runs.empty:
            return 1
        
        # Extraire les numéros de batch des noms de runs
        batch_numbers = []
        for run_name in runs['tags.mlflow.runName'].dropna():
            if '#' in run_name:
                try:
                    number = int(run_name.split('#')[-1])
                    batch_numbers.append(number)
                except ValueError:
                    continue
        
        if not batch_numbers:
            return 1
            
        # Retourner le numéro de batch le plus élevé + 1
        return max(batch_numbers) + 1
    
    except Exception as e:
        print(f"Erreur lors de la récupération du numéro de batch: {e}")
        # Fallback: utiliser timestamp comme numéro unique
        from datetime import datetime
        return int(datetime.now().strftime("%Y%m%d%H"))

# Variable globale pour stocker le numéro de batch pendant l'exécution du DAG
_current_batch_number = None

def get_or_set_batch_number():
    """Obtient ou définit le numéro de batch pour toute la session d'exécution"""
    global _current_batch_number
    
    if _current_batch_number is None:
        _current_batch_number = get_batch_execution_number()
    
    return _current_batch_number

def sauvegarder_metriques_mlflow(metriques, product_id, store_id, is_global=False):
    try:
        # Configurer MLflow pour ne pas utiliser Git
        import os
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        
        # Obtenir le numéro de batch (même pour tous les runs du même groupe)
        batch_number = get_or_set_batch_number()
        
        if is_global:
            run_name = f"Global_Metrics_#{batch_number}"
        else:
            run_name = f"Prophet_{product_id}_{store_id}_#{batch_number}"
            
        with mlflow.start_run(run_name=run_name):
            # Log du numéro de batch comme métrique
            mlflow.log_metric("batch_number", batch_number)
            
            # Convertir les métriques en float
            for metric_name, metric_value in metriques.items():
                if metric_name not in ['Product_ID', 'Store_ID']:
                    mlflow.log_metric(metric_name, float(metric_value))
            
            if not is_global:
                # Sauvegarder product_id et store_id comme tags pour les runs individuels
                mlflow.set_tag('Product_ID', str(product_id))
                mlflow.set_tag('Store_ID', str(store_id))
                mlflow.set_tag('batch_number', str(batch_number))
            else:
                mlflow.set_tag('Type', 'Global_Metrics')
                mlflow.set_tag('batch_number', str(batch_number))
                
        if is_global:
            print(f"Métriques globales sauvegardées dans MLflow (Batch #{batch_number})")
        else:
            print(f"Métriques sauvegardées dans MLflow pour Produit {product_id} et Magasin {store_id} (Batch #{batch_number})")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde MLflow: {str(e)}")

def reset_batch_number():
    """Réinitialise le numéro de batch (à appeler au début d'un nouveau batch si nécessaire)"""
    global _current_batch_number
    _current_batch_number = None
    print("Numéro de batch réinitialisé pour le prochain groupe de données")

# Publier les prédictions dans Kafka
def publier_predictions_kafka(forecast, product_id, store_id):
    producer = KafkaProducer(
        bootstrap_servers='kafka:29092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    for _, row in forecast.iterrows():
        message = {
            'Product_ID': product_id,
            'Store_ID': store_id,
            'Date': row['ds'].strftime('%Y-%m-%d'),
            'yhat': row['yhat'],
            'yhat_lower': row['yhat_lower'],
            'yhat_upper': row['yhat_upper']
        }
        producer.send('retail_predictions', value=message)
    
    producer.close()
    print(f"Prédictions publiées dans Kafka pour Produit {product_id} et Magasin {store_id}")

# Fonction principale
def main():
    df_agreger = lecture_kafka()
    print(f"Taille du DataFrame : {len(df_agreger)} lignes")
    
    if df_agreger.empty:
        print("Aucune donnée n'a été récupérée. Arrêt du programme.")
        return
    
    resultats = []
    reset_batch_number()
    for (prod_id, store_id), groupe in df_agreger.groupby(['Product_ID', 'Store_ID']):
        print(f"\nTraitement de Produit {prod_id} - Magasin {store_id}")
        print(f"Nombre d'observations : {len(groupe)}")
        
        try:
        # Préparation des données
            df_prophet = preparer_donnees_prophet(groupe)
            print("Recherche des meilleurs hyperparamètres...")
            best_params = grid_search_prophet(df_prophet)

            # Création et entraînement du modèle avec les meilleurs paramètres
            model = Prophet(
                changepoint_prior_scale=best_params['changepoint_prior_scale'],
                seasonality_prior_scale=best_params['seasonality_prior_scale'],
                holidays_prior_scale=best_params['holidays_prior_scale'],
                changepoint_range=best_params['changepoint_range'],
                n_changepoints=best_params['n_changepoints'],
                growth=best_params['growth'],
                daily_seasonality=True
            )
            
            # Vérifier quels régresseurs sont disponibles dans les données
            available_regressors = []
            if 'Holiday_Flag' in groupe.columns:
                model.add_regressor('Holiday_Flag')
                available_regressors.append('Holiday_Flag')
            if 'Promotional_Flag' in groupe.columns:
                model.add_regressor('Promotional_Flag')
                available_regressors.append('Promotional_Flag')
        
            print("Entraînement du modèle...")
            model.fit(df_prophet)
            
            # Obtenir la dernière date des données
            last_date = df_prophet['ds'].max()
            
            # Créer le DataFrame futur uniquement pour les 30 prochains jours après la dernière date
            future = model.make_future_dataframe(periods=30, freq='D', include_history=False)
            future['ds'] = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
            
            # Ajouter les régresseurs au DataFrame futur
            for regressor in available_regressors:
                if regressor in df_prophet.columns:
                    future[regressor] = df_prophet[regressor].iloc[-1]  # Utiliser la dernière valeur connue
                
            forecast = model.predict(future)
            
            # Arrondir les prédictions en nombres entiers
            forecast['yhat'] = forecast['yhat'].round().astype(int)
            forecast['yhat_lower'] = forecast['yhat_lower'].round().astype(int)
            forecast['yhat_upper'] = forecast['yhat_upper'].round().astype(int)
            
            # S'assurer que les prédictions ne sont pas négatives
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
            
            # Sauvegarder les prédictions dans MySQL
            sauvegarder_predictions_mysql(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], prod_id, store_id)
            
            # Publier les prédictions dans Kafka
            publier_predictions_kafka(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], prod_id, store_id)
            
            # Calculer les métriques sur les données historiques uniquement
            y_true = df_prophet['y'].values
            
            # Préparer les données historiques pour la prédiction
            historical = df_prophet[['ds'] + available_regressors]
            y_pred = model.predict(historical)['yhat'].round().astype(int).values
            
            metriques = evaluer_modele(y_true, y_pred, prod_id, store_id)
            
            # Sauvegarder les métriques individuelles dans MLflow
            sauvegarder_metriques_mlflow(metriques, prod_id, store_id)
            
            # Ajouter les identifiants pour le résumé
            metriques['Product_ID'] = str(prod_id)
            metriques['Store_ID'] = str(store_id)
            resultats.append(metriques)
            
        except Exception as e:
            print(f"Erreur pour Produit {prod_id} - Magasin {store_id}: {str(e)}")
            continue
            
    print("\nTraitement terminé!")
    if resultats:
        print("\nRésumé des performances:")
        df_resultats = pd.DataFrame(resultats)
        
        # Calculer les métriques globales
        metriques_globales = {
            'rmse_global': float(df_resultats['rmse'].mean()),
            'mae_global': float(df_resultats['mae'].mean()),
            'r2_global': float(df_resultats['r2'].mean()),
            'rmse_std': float(df_resultats['rmse'].std()),
            'mae_std': float(df_resultats['mae'].std()),
            'r2_std': float(df_resultats['r2'].std()),
            'nombre_modeles': len(df_resultats)
        }
        
        # Sauvegarder les métriques globales dans MLflow
        sauvegarder_metriques_mlflow(metriques_globales, None, None, is_global=True)
        
        # Afficher le résumé
        print("\nMétriques par modèle:")
        print(df_resultats.groupby(['Product_ID', 'Store_ID'])[['mae', 'rmse', 'r2']].mean())
        print("\nMétriques globales:")
        for metric, value in metriques_globales.items():
            print(f"{metric}: {value:.4f}")

def evaluer_modele(y_true, y_pred, prod_id, store_id):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    
    # Calculer toutes les métriques
    metriques = {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mape': float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        'r2': float(r2_score(y_true, y_pred))
    }
    
    # Analyse détaillée des performances
    if metriques['r2'] < 0.5:
        print(f"\nAttention: Faible R² ({metriques['r2']:.3f}) pour Produit {prod_id} - Magasin {store_id}")
        print("Suggestions d'amélioration:")
        print("1. Vérifier la qualité des données d'entrée")
        print("2. Examiner la présence de tendances ou motifs non capturés")
        print("3. Considérer l'ajout de variables externes (météo, événements, etc.)")
        print("4. Augmenter le nombre de changepoints")
        print("5. Ajuster les prior scales pour la saisonnalité")
    
    if metriques['mape'] > 20:
        print(f"\nAttention: MAPE élevé ({metriques['mape']:.2f}%) pour Produit {prod_id} - Magasin {store_id}")
        print("Suggestions d'amélioration:")
        print("1. Vérifier la présence d'outliers")
        print("2. Ajuster la gestion des valeurs aberrantes")
        print("3. Considérer une transformation des données")
    
    # Calculer la distribution des erreurs
    erreurs = y_true - y_pred
    metriques['erreur_moyenne'] = float(np.mean(erreurs))
    metriques['erreur_std'] = float(np.std(erreurs))
    
    # Vérifier la présence de biais systématique
    if abs(metriques['erreur_moyenne']) > np.std(y_true) * 0.1:
        print(f"\nAttention: Présence possible de biais systématique (erreur moyenne: {metriques['erreur_moyenne']:.2f})")
        print("Suggestions d'amélioration:")
        print("1. Vérifier la normalisation des données")
        print("2. Ajuster les paramètres de croissance du modèle")
        print("3. Considérer l'ajout de régresseurs supplémentaires")
    
    return metriques

if __name__ == "__main__":
    main()