###########
# BUILDER #
###########

# pull official base image
FROM python:3.10-slim-buster as builder

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install python dependencies
COPY server/requirements.txt requirements.txt
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

#########
# FINAL #
#########

FROM python:3.10-slim-buster

WORKDIR /app

# copy sensor data
COPY --chown=1001:1001 sensors_list.json . 
COPY --chown=1001:1001 measurements/outer_sensors_per_sensor_importance.json ./importance.json

COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache /wheels/*

COPY --chown=1001:1001 server/app.py app.py

USER 1001

CMD [ "gunicorn", "-w", "10", "--bind", "0.0.0.0:80", "app:server" ]