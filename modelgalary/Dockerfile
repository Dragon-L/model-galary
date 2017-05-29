from ubuntu:16.04

RUN apt-get update && apt-get install -y python3-pip
# Put all required packages here to make building faster
RUN cd /work/modelgalary && pip3 install -r scipy==0.19.0 django==1.11.1 \
    requests==2.16.5 grpcio==1.3.5 numpy==1.12.1 tensorflow==1.0.0 pillow==4.1.1

ADD . /work/modelgalary

WORKDIR /work/modelgalary
ENV PYTHONPATH=./tf_serving

CMD python3 manage.py runserver 0.0.0.0:8000