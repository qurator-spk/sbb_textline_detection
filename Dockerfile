FROM python:3

ADD requirements.txt /
RUN pip install --proxy=http-proxy.sbb.spk-berlin.de:3128 -r requirements.txt

COPY . /usr/src/sbb_textline_detector
RUN pip install /usr/src/sbb_textline_detector

ENTRYPOINT ["sbb_textline_detector"]
