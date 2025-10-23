import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 pyspark-shell'

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, sum, max, window, to_json, struct
from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType

def create_spark_session():
    """Create and configure Spark session"""
    spark = SparkSession.builder \
        .appName("RetailAggregationToKafka") \
        .config("spark.sql.shuffle.partitions", "3") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

def define_input_schema():
    """Define schema for incoming Kafka messages"""
    return StructType() \
        .add("Timestamp", TimestampType()) \
        .add("Product_ID", StringType()) \
        .add("Store_ID", StringType()) \
        .add("Quantity_Sold", IntegerType()) \
        .add("Holiday_Flag", IntegerType()) \
        .add("Promotional_Flag", IntegerType())

def read_from_kafka(spark, schema):
    """Read streaming data from Kafka"""
    return spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "retail-transactions") \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()

def process_stream(raw_df, schema):
    """Process the streaming data with aggregations"""
    # Parse JSON from Kafka value
    json_df = raw_df.selectExpr("CAST(value AS STRING) as json_string") \
        .select(from_json(col("json_string"), schema).alias("data")) \
        .select("data.*")
    
    # Apply watermark for late data handling
    json_df = json_df.withWatermark("Timestamp", "1 day")
    
    # Daily aggregations by product and store
    agg_df = json_df.groupBy(
        window(col("Timestamp"), "1 day").alias("day_window"),
        col("Product_ID"),
        col("Store_ID")
    ).agg(
        sum("Quantity_Sold").alias("Total_Quantity_Sold"),
        max("Holiday_Flag").alias("Holiday_Flag"),
        max("Promotional_Flag").alias("Promotional_Flag")
    )
    
    # Flatten the window structure
    return agg_df.select(
        col("day_window.start").alias("Date"),
        "Product_ID",
        "Store_ID",
        "Total_Quantity_Sold",
        "Holiday_Flag",
        "Promotional_Flag"
    )

def write_to_kafka(df):
    """Write aggregated results back to Kafka"""
    # Convert to JSON string for Kafka
    kafka_df = df.selectExpr(
        "CAST(NULL AS STRING) AS key",  # Optional key
        "to_json(struct(*)) AS value"
    )
    
    return kafka_df.writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("topic", "retail_aggregated") \
        .option("checkpointLocation", "checkpoints/retail_agg_kafka") \
        .outputMode("complete") \
        .trigger(processingTime="30 seconds") \
        .start()

def main():
    """Main execution function"""
    spark = create_spark_session()
    schema = define_input_schema()
    raw_df = read_from_kafka(spark, schema)
    processed_df = process_stream(raw_df, schema)
    query = write_to_kafka(processed_df)
    
    print("Stream processing started. Aggregated results being written to Kafka...")
    query.awaitTermination(timeout=60)

if __name__ == "__main__":
    main()