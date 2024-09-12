FROM python:3.12

# Instale as dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de requirements para dentro do container
COPY requirements.txt /opt/ml/code/

# Instala as bibliotecas Python listadas no requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Copie os scripts necessários da pasta train/
COPY preprocessing.py /opt/ml/code/preprocessing.py
COPY cnn.py /opt/ml/code/train_pipeline.py
COPY evaluate.py /opt/ml/code/evaluate.py

# Defina o diretório de trabalho
WORKDIR /opt/ml/code

# Define o comando padrão para executar o entrypoint.py
CMD ["python3"]