from django.urls import path
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path("studio/", views.studio_view, name="studio"),
    path("upload/", views.upload_view, name="upload"),
    path("delete_session/", views.delete_session_view, name="delete_session"),
    path("inference/", views.inference_view, name="inference"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
