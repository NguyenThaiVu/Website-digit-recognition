from django.db import models

# Create your models here
class MyImage(models.Model):
    name = models.CharField(max_length=100)
    img = models.ImageField(upload_to="images/")