FROM python:3.12

# Instale as dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copie o arquivo de requirements para dentro do container
COPY requirements.txt /opt/ml/code/

# Instala as bibliotecas Python listadas no requirements.txt e o sagemaker_training_toolkit
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Copie seu código para o contêiner
COPY train.py /opt/ml/code/train.py

# Defina o diretório de trabalho
WORKDIR /opt/ml/code