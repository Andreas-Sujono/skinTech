from django.shortcuts import render,redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.core.paginator import Paginator
from django.utils import timezone

from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16
import pandas as pd

from PIL import Image

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
		image = request.FILES['image']
		print('file retrieved')

		#validation of image type
		imageType = ['jpg']
		if(str(image.name)[-3:] not in imageType):
			messages.error(request, 'only jpg file allowed')
			return redirect('/user/upload-image')


		now = timezone.now()

		
		#ML
		# Load the json file that contains the model's structure
		f = Path("model_structure.json")

		import ssl
		ssl._create_default_https_context = ssl._create_unverified_context

		#ML
		# Load the json file that contains the model's structure

		from archicture import model
	

		# Re-load the model's trained weights
		model.load_weights("model_weights.h5")


		img = image.resize((64,64), Image,ANTIALIAS)

		
		# Convert the image to a numpy array
		image_array = np.array(img)
		#image_array = image.img_to_array(img)

		# Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
		images = np.expand_dims(image_array, axis=0)

		# Normalize the data
		images = vgg16.preprocess_input(images)


		# Use the pre-trained neural network to extract features from our test image (the same way we did to train the model)
		feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

		features = feature_extraction_model.predict(images)
		# Given the extracted features, make a final prediction using our own model
		results = model.predict(features)

		# Since we are only testing one image with possible class, we only need to check the first result's first element
		single_result = results[0][0]	

		
		history = History_image(
			date = now,
			image = image,
			status= 'None',
			user = request.user
		)

		history.save()
		
		single_result = 0
		messages.success(request, 'upload successful, see the result below')
		return render(request,'user_upload_image.html', {'result':{
			'percent_malignant':single_result*100,
			'percent_benign':100-single_result*100,
			},'image':history.image})


		print('file not retrieved')
		messages.error(request, 'upload unsuccessful')
		return render(request,'user_upload_image.html', {})


	return render(request,'user_upload_image.html', {'result':None})


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



	

	






