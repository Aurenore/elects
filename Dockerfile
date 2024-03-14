ARG ROOT_CONTAINER=jjurm/runai-job
FROM $ROOT_CONTAINER

USER root
ENV HOME=/workdir
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

COPY requirements.txt /tmp/requirements-freeze-v1.txt
RUN pip install --timeout 3600 -r /tmp/requirements-freeze-v1.txt

# add visual code extension

# copy source code into the container
# COPY . .

# download datasets into the container once (~400 MB)
# RUN python -c "from data import BavarianCrops; BavarianCrops('train'); BavarianCrops('valid'); BavarianCrops('eval')"