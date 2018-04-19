from django.http import HttpResponse, JsonResponse


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def infer(request):
    query_obj = {}
    print(request.GET)
    for key in request.GET:
        print(key)

    for key in ['age', 'capital_gain', 'capital_loss', 'education', 'education_num', 'gender',
                'hours_per_week', 'native_country', 'occupation', 'relationship', 'workclass']:
        if key in request.GET:
            query_obj[key] = request.GET[key]
    from . import wide_and_client
    return JsonResponse(wide_and_client.do_inference(query_obj))
