from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

from .views import *

urlpatterns = [
    path('', image_view, name="image_view"),
    path('success', success, name="success"),
]