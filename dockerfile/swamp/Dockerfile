FROM reg.docker.alibaba-inc.com/alipay-collie/cuda8-396.26-driver:20190905

RUN yum install -y tops-python37.x86_64; ln -sf /home/tops/bin/python3 /home/tops/bin/python
RUN /home/tops/bin/pip3 install http://download.pytorch.org/whl/cu80/torch-1.0.0-cp37-cp37m-linux_x86_64.whl torchvision
RUN echo 'export PATH=/home/tops/bin:${PATH}' >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]
