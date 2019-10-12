from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class History_image(models.Model):
    date = models.DateTimeField()
    image = models.ImageField(blank=True,upload_to='skinImages')
    status = models.TextField()
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user')
