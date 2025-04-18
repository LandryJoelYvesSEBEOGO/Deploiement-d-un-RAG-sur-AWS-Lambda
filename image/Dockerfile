FROM public.ecr.aws/lambda/python:3.11

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Required to make SQLlite3 work for Chroma.
RUN pip install pysqlite3-binary

# Install the specified packages
RUN pip install -r requirements.txt --upgrade

# Crée le dossier nomic dans le home Lambda
RUN mkdir -p /tmp/.nomic /home/sbx_user1051 \
    && chmod 777 /tmp/.nomic \
    && ln -sfn /tmp/.nomic /home/sbx_user1051/.nomic

# For local testing.
EXPOSE 8000

# Set IS_USING_IMAGE_RUNTIME Environment Variable
ENV IS_USING_IMAGE_RUNTIME=True
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}:${LAMBDA_TASK_ROOT}/src

# Copy all files in ./src
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY data/chroma_db/ ${LAMBDA_TASK_ROOT}/data/chroma_db/
COPY data/raw/ ${LAMBDA_TASK_ROOT}/data/raw/
COPY .env ${LAMBDA_TASK_ROOT}/

CMD ["src.app_api_handler.handler"]