from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^infer$', views.infer, name='infer'),
]