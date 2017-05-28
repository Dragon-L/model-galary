from django.http import HttpResponse, JsonResponse


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def infer(request):
    from . import mnist_client
    return JsonResponse(mnist_client.do_inference(request.GET['url']))
