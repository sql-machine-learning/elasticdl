FROM tensorflow/tensorflow:2.0.0-py3
RUN pip install requests
RUN pip install flask
RUN pip install pyyaml
RUN pip install atomiclong
RUN pip install kubernetes
ADD ./master.py /master.py
ADD ./template.yaml /template.yaml
ENTRYPOINT ["sleep","9999999"]
