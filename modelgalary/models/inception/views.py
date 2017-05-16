from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.


def infer(request):
    from . import inception_client
    return JsonResponse(inception_client.do_inference('http://localhost:8082/' + request.GET['url']), safe=False)
