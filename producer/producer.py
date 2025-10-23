import pandas as pd
from kafka import KafkaProducer
import json
import time
import os

KAFKA_TOPIC = "retail-transactions"
KAFKA_BOOTSTRAP_SERVERS = "pfa_mf-kafka-1:9092"
CSV_FILE = '/opt/airflow/synthetic_retail_sales_enhanced.csv'

def preprocess(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Convert string to datetime
    df['Holiday_Flag'] = df['Holiday_Flag'].astype(int)
    df['Promotional_Flag'] = df['Promotional_Flag'].astype(int)
    return df

def send_to_kafka():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8")
    )

    df = pd.read_csv(CSV_FILE)
    df = preprocess(df)

    batch_size = 150000
    offset_dir = "/opt/airflow/data/offset"
    offset_file = os.path.join(offset_dir, "offset.txt")
    
    # Create directory if it doesn't exist
    os.makedirs(offset_dir, exist_ok=True)
    
    # Initialize offset file if it doesn't exist
    if not os.path.exists(offset_file):
        with open(offset_file, "w") as f:
            f.write("0")
    
    offset = int(open(offset_file).read())

    next_offset = offset + batch_size
    for _, row in df.iloc[offset:next_offset].iterrows():
        producer.send(KAFKA_TOPIC, row.to_dict())
        time.sleep(0.0001)

    producer.flush()
    with open(offset_file, "w") as f:
        f.write(str(next_offset))

if __name__ == "__main__":
    send_to_kafka()