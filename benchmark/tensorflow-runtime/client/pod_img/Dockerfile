FROM allreduce
RUN pip install requests
RUN pip install flask
ADD pod.py ./
ENTRYPOINT ["python", "pod.py"]
