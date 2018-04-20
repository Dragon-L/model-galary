from django.http import JsonResponse


def infer(request):
    query_obj = {}

    for key in ['age', 'capital_gain', 'capital_loss', 'education', 'education_num', 'gender',
                'hours_per_week', 'native_country', 'occupation', 'relationship', 'workclass']:
        if key in request.GET:
            query_obj[key] = request.GET[key]
    from . import wide_and_deep_client
    return JsonResponse(wide_and_deep_client.do_inference(query_obj))


def infer_movielens(request):
    query_obj = {}

    for key in ['user', 'item', 'age', 'gender', 'occupation', 'zipcode',
                'action', 'adventure', 'animation', 'child', 'comedy', 'crime',
                'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical',
                'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western',
                'delay_days', 'watch_year', 'watch_month', 'watch_day', 'watch_wd',
                'watch_season', 'relase_year', 'release_month', 'release_day',
                'release_wd', 'watch_span', 'age_span']:
        if key in request.GET:
            query_obj[key] = request.GET[key]
    from . import wide_and_deep_client_movielens
    return JsonResponse(wide_and_deep_client_movielens.do_inference(query_obj))
