"""app URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings


from homepage.views import *


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',homepage_view, name='homepage' ),
    path('login/',login_handle, name='login_handle' ),
    path('user/upload-image/',user_upload_image_view, name= 'user_upload_image'),
    path('user/history/',user_history_view,name='user_history'),
    path('user/profile/',user_profile_view,name='user_profile'),
    path('logout/',user_logout,name='user_logout'),
    path('user/history/delete/<int:id>',user_history_delete, name='user_history_delete')
] 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
