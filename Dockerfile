ARG ROOT_CONTAINER=aurenore/runai-job
FROM $ROOT_CONTAINER

USER root
ENV HOME=/workspace
RUN mkdir -p "$HOME" && \
    chown -R "$NB_UID:$NB_GID" "$HOME"
WORKDIR "$HOME"
ENV PATH="${PATH}:${HOME}/.local/bin"

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    python3.10 python3.10-dev \
    libcurl4-gnutls-dev \
    libgnutls28-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    sudo ln -s /usr/bin/python3.10 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade pip


USER $NB_USER

COPY requirements.txt /tmp/requirements.txt
RUN pip install --timeout 3600 -r /tmp/requirements.txt

ENV PYTHONPATH=/workspace/.local/lib/python3.10/site-packages

COPY ./entrypoint.sh /entrypoint.sh
# RUN chmod a+x /entrypoint.sh
ENTRYPOINT ["tini", "-g", "--", "/bin/bash", "/entrypoint.sh"]

# # export WANDB_API_KEY from $ENV_FILE
# RUN export $(grep -v '^#' $ENV_FILE | xargs) && \
#     export WANDB_API_KEY=$SECRET_WANDB_API_KEY 


# add visual code extension

# copy source code into the container
# COPY . .

# # copy the folder data into the container
# COPY data data
# COPY utils utils

# # download datasets into the container once
# RUN python -c "from data import BreizhCrops; BreizhCrops('train'); BreizhCrops('valid'); BreizhCrops('eval')"