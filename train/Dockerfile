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
COPY train_pca.py /opt/ml/code/train_pca.py
COPY postprocessing_pca.py /opt/ml/code/postprocessing_pca.py
COPY train_kmeans_hpo.py /opt/ml/code/train_kmeans_hpo.py
COPY kmeans_postprocessing.py /opt/ml/code/kmeans_postprocessing.py
COPY entrypoint.py /opt/ml/code/entrypoint.py

# Defina o diretório de trabalho
WORKDIR /opt/ml/code

# Define o comando padrão para executar o entrypoint.py
CMD ["python3", "/opt/ml/code/entrypoint.py"]