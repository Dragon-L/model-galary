Model galary is website to show the AI models you've built.

## Prerequisite

To get the model running locally. You would need the below:

- docker environment
- python3
- node 7

## How to add new model

### Export your model

Refer to [tensorflow serving examples](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example) to write some code to export your model.

To test the exporting, you can do some experiments through iPython.
We've built a tensorflow serving docker image and pushed to docker hub `twaiclub/tf-serving` to help with this. If you want to know how the image is built, see [Extended information](#Extended information) below.

- Start serving your model: `docker run --rm -p 8021:8000 -v /path/to/exported-model/:/work twaiclub/tf-serving /tensorflow_model_server --port=8000 --model_name=YOUR_MODEL_NAME --model_base_path=/work` 
- Switch to project directory
- `export PYTHONPATH=./modelgalary/tf_serving:$PYTHONPATH`
- Refer to file `modelgalary/models/mnist/mnist_cliet.py` to import and run test

### Expose your model as HTTP endpoints

We have setup a Django application to help with this.

To add a new model, you would need to follow the current folder conventions. The followings need to be done:

- Run `python3 manage.py startapp YOUR_MODEL_NAME` from `/modelgalary/models` folder
- Edit `/modelgalary/models/YOUR_MODEL_NAME/urls.py` to add new endpoints
- Edit `/modelgalary/models/YOUR_MODEL_NAME/views.py` to add the implementations
- Edit `/modelgalary/modelgalary/urls.py` to let the application know your endpoints

To test your endpoints, you can start the application and do some curl requests like below:

- Start your applcation by running `python3 manage.py runserver 8080`
- Send requests by curl, eg. `curl http://127.0.0.1:8000/inception/infer?url=http%3A%2F%2Flocalhost%3A3001%2Fassets%2Fmnist%2F1.png` (In the example, you need to have a file named `assets/mnist.png` and make it available through HTTP which can be done by running `python3 http.server 3001`)

### Design and implement your webpage

We have setup a react isomorphic application for doing this. The application is created from [react starter kit](https://github.com/kriasoft/react-starter-kit#readme) and placed in the repository as a submodule named `webapp`.

The application has a graphql based backend service which can proxy the requests from client to the Django application. The reason why we need to add this complexity is that we can do ML tasks in python environment much more easily while node gives us fantastic asynchrony and isomorphic support.

You should follow the current design to implement your web page. To add your page:

- Create a folder `webapp/src/routes/model-galary/YOUR_MODEL_NAME` for your components
- Refer to existing code to implement your own components
- Expose your model page by adding the entry in file `webapp/src/routes/model-galary/ModelGalary.js`
- Create graphql query implementations in `webapp/src/data`

To test the page you've built, do the followings:

- Run `yarn start` to start a development environment for the application
- View your page and test

## How to release the application with your model included

We don't have a CI pipeline for this yet. Please contact me to release it after you fully tested your changes. I've got some scripts to get it up and running in our local server and I'd be happy to get it done.

## Extended information

### Build your own tf-serving docker image

To build the image on your own, you need to build the tensorflow_model_server binary file following [tensorflow_serving documents](https://tensorflow.github.io/serving/). After you finished building, you'll have a file named `tensorflow_model_server` and placed at `bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server`. Follow the commands below to create your own image:

- Copy the binary file to current folder
- Create a file named `Dockerfile` with contents
```
from ubuntu:16.04

ADD tensorflow_model_server /

CMD /tensorflow_serving
```
- Build the image by running `docker build . -t tf-serving`

## TODO

- Setup a pipeline
- Add support to deploy the application to cloud







