from ubuntu:16.04

RUN apt-get update && apt-get install -y python3-pip
ADD ../modelgalary /work/modelgalary
RUN cd /work/modelgalary && pip3 install -r requirements.txt

WORKDIR /work/modelgalary
ENV PYTHONPATH=./tf_serving

CMD python3 manage.py runserver 8000