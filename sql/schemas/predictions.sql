CREATE TABLE predictions_prophet (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ds DATETIME NOT NULL,                 
    yhat DOUBLE NOT NULL,                 
    yhat_lower DOUBLE NOT NULL,          
    yhat_upper DOUBLE NOT NULL,          
    Product_ID VARCHAR(255) NOT NULL,    
    Store_ID VARCHAR(255) NOT NULL,      
    forecast_made_on DATETIME NOT NULL   
);
