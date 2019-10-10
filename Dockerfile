FROM python:3

ADD main.py /
ADD requirements.txt /

RUN pip install --proxy=http-proxy.sbb.spk-berlin.de:3128 -r requirements.txt

ENTRYPOINT ["python", "./main.py"]
