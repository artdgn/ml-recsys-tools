FROM python:3.6-slim

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV APP_DIR=/ml_recsys_tools

ADD . ${APP_DIR}

WORKDIR ${APP_DIR}

RUN pip install -i file://$(realpath .) .

