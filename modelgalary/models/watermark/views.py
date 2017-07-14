from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.


def infer(request):
    from . import watermark_client
    image_url = request.GET['url']
    if not image_url.startswith('http'):
        image_url = 'http://localhost:8082/' + image_url
    return JsonResponse(watermark_client.do_inference(image_url))
