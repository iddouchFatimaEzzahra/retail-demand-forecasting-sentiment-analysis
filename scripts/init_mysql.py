import pymysql

def init_mysql():
    # Connexion à MySQL
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='manal'
    )
    
    cursor = conn.cursor()
    
    # Création de la base de données
    cursor.execute("CREATE DATABASE IF NOT EXISTS retail_forecasting")
    cursor.execute("USE retail_forecasting")
    
    # Création de la table des prédictions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions_prophet (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ds DATETIME NOT NULL,
        yhat DOUBLE NOT NULL,
        yhat_lower DOUBLE NOT NULL,
        yhat_upper DOUBLE NOT NULL,
        Product_ID VARCHAR(255) NOT NULL,
        Store_ID VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_product_store (Product_ID, Store_ID),
        INDEX idx_date (ds)
    )
    """)
    
    # Création de la table des métriques
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Product_ID VARCHAR(50) NOT NULL,
        Store_ID VARCHAR(50) NOT NULL,
        mae FLOAT NOT NULL,
        rmse FLOAT NOT NULL,
        r2 FLOAT NOT NULL,
        run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_product_store (Product_ID, Store_ID),
        INDEX idx_date (run_date)
    )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("Base de données initialisée avec succès!")

if __name__ == "__main__":
    init_mysql()