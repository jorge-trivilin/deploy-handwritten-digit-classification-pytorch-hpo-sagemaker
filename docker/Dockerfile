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

# Copie os scripts necessários
COPY train_pca.py /opt/ml/code/train_pca.py

# Define a variável de ambiente para o script principal
ENV SAGEMAKER_PROGRAM train_pca.py

# Defina o diretório de trabalho
WORKDIR /opt/ml/code

# Defina o comando padrão
CMD ["python3"]
