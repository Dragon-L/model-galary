from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^infer$', views.infer, name='infer'),
    url(r'^movie$', views.infer_movielens, name='infer_movielens'),
]