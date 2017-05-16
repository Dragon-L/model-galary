from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def infer(request):
    from . import mnist_client
    return HttpResponse(mnist_client.do_inference('http://localhost:8082/' + request.GET['url']))
