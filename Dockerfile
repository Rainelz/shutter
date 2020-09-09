FROM ubuntu:18.04

#
# --- Set env variables ---
#

RUN apt-get update && apt-get -y install locales apt-utils dumb-init
RUN locale-gen en_US.UTF-8  
ENV TZ="Europe/Rome"
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

#
#
# --- Add user and and allow random uid ---
#
ENV APP_ROOT="/opt/app"
ENV USER_NAME="app"
ENV USER_UID="10001"
ENV PATH="${APP_ROOT}/bin:$PATH"
#COPY docker/bin/ ${APP_ROOT}/bin/
# allow to run random user id (openshift compatibility)
RUN useradd -l -u ${USER_UID} -r -g 0 -d ${APP_ROOT} -s /sbin/nologin -c "${USER_NAME} user" ${USER_NAME} && \
    mkdir -p ${APP_ROOT}/bin && \
    chmod -R u+x ${APP_ROOT}/bin && \
    chgrp -R 0 ${APP_ROOT} && \
    chmod -R g=u ${APP_ROOT} /etc/passwd

## Fonts install
    
RUN apt-get update && apt-get -qq -y install curl bzip2 wget cabextract xfonts-utils fontconfig
RUN wget http://ftp.uk.debian.org/debian/pool/contrib/m/msttcorefonts/ttf-mscorefonts-installer_3.7_all.deb

RUN apt-get purge ttf-mscorefonts-installer

RUN dpkg -i ttf-mscorefonts-installer_3.7_all.deb

RUN fc-cache -f -v

#
# --- Pyenv and pipenv ---
#
ENV PY_VERSION="3.7.4"
ENV PYENV_ROOT="${APP_ROOT}/pyenv"
# add shim to path, this avoid running pyenv init -
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:$PATH"
# install env in project with
ENV PIPENV_VENV_IN_PROJECT=true
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -y install git build-essential zlib1g-dev \
        libffi-dev libssl-dev libbz2-dev libreadline-dev libsm6 curl libsqlite3-dev "libglib2.0-0" libgl1-mesa-dev libxrender1
RUN git clone --branch v1.2.13 https://github.com/yyuu/pyenv.git ${PYENV_ROOT} && \
    pyenv install "$PY_VERSION" && \
    pyenv global ${PY_VERSION} && \
    pip install --upgrade pip && \
    pip install pipenv && \
    pyenv rehash 


ENV PROJ_ROOT=${APP_ROOT}/shutter

COPY Pipfile ${PROJ_ROOT}/Pipfile
COPY Pipfile.lock ${PROJ_ROOT}/Pipfile.lock

WORKDIR ${PROJ_ROOT}

RUN pipenv sync 

COPY . ${PROJ_ROOT}



