FROM apache/airflow:2.6.3-python3.9

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    wget \
    gnupg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Apache Spark
ARG SPARK_VERSION=3.3.0
ARG HADOOP_VERSION=3
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

USER airflow

# Create requirements file with compatible versions
RUN echo "pydantic>=1.10.0,<2.0" > /tmp/requirements.txt && \
    echo "typing-extensions>=4.6.0" >> /tmp/requirements.txt && \
    echo "groq>=0.4.0,<0.29.0" >> /tmp/requirements.txt && \
    echo "pyspark==3.3.0" >> /tmp/requirements.txt && \
    echo "kafka-python" >> /tmp/requirements.txt && \
    echo "mlflow" >> /tmp/requirements.txt && \
    echo "prophet" >> /tmp/requirements.txt && \
    echo "scikit-learn>=1.0.0" >> /tmp/requirements.txt && \
    echo "pymysql" >> /tmp/requirements.txt && \
    echo "mysql-connector-python" >> /tmp/requirements.txt && \
    echo "pandas>=1.3.0" >> /tmp/requirements.txt && \
    echo "spacy==3.4.4" >> /tmp/requirements.txt && \
    echo "tenacity" >> /tmp/requirements.txt && \
    echo "backoff" >> /tmp/requirements.txt && \
    echo "apache-airflow-providers-apache-kafka" >> /tmp/requirements.txt

# Install Python packages with proper dependency resolution
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Clean up
RUN rm /tmp/requirements.txt