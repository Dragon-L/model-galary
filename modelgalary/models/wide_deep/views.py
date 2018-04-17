from django.http import HttpResponse, JsonResponse


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def infer(request):
    from . import wide_and_client
    return JsonResponse(wide_and_client.do_inference(request.GET['url']))
