from django.shortcuts import render,redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.core.paginator import Paginator
from django.utils import timezone


from .models import History_image

from datetime import datetime


# Create your views here.
def homepage_view(request):
	return render(request,'home.html',{})

def login_handle(request):

	if request.method == 'POST':
		username = request.POST['username']
		password = request.POST['password']

		user = authenticate(username=username,password=password)
		print(user)

		if user is not None:
			login(request, user)

			messages.success(request, 'you are logged in')
			return redirect('/user/upload-image')

	messages.error(request, 'wrong username or password')

	page = request.GET.get('page')

	return redirect('/')

@login_required
def user_upload_image_view(request):

	if(request.method == 'POST'):
		try:
			image = request.FILES['image']
			print('file retrieved')

			#validation of image type
			imageType = ['jpg']
			if(str(image.name)[-3:] not in imageType):
				messages.error(request, 'only jpg file allowed')
				return redirect('/user/upload-image')


			now = timezone.now()

			history = History_image(
				date = now,
				image = image,
				status= 'None',
				user = request.user
			)

			history.save()

			messages.success(request, 'upload successful, see the result below')
			return render(request,'user_upload_image.html', {'result':'null','image':history.image})


		except:
			print('file not retrieved')
			messages.error(request, 'upload unsuccessful')
			return render(request,'user_upload_image.html', {})


	return render(request,'user_upload_image.html', {'result':''})


@login_required
def user_history_view(request):
	history_list = History_image.objects.filter(user=request.user)

	paginator = Paginator(history_list, 3)

	page = request.GET.get('page')
	history = paginator.get_page(page)

	print(history)
	return render(request,'user_history.html', {'history':history})

@login_required
def user_history_delete(request, id):
	history = History_image.objects.get(id=id)
	history.delete()
	return redirect('/user/history')

@login_required
def user_profile_view(request):
	return render(request,'user_profile.html', {})

@login_required
def user_consultation_view(request):
	return render(request,'user_consultation.html', {})

@login_required
def user_profile_view(request):
	return render(request, 'user_profile.html', {'user':request.user})


@login_required
def user_logout(request):
	 logout(request)
	 return redirect('/')



	

	






