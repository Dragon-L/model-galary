from stayrascal/django:latest

ADD . /work/modelgalary

WORKDIR /work/modelgalary
ENV PYTHONPATH=./tf_serving

EXPOSE 8000

CMD python3 manage.py runserver 0.0.0.0:8000