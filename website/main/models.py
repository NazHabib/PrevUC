from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    perfil = models.CharField(max_length=20)

    def __str__(self):
        return self.user.username

from django.db import models
from django.contrib.auth.models import User

class Prevision(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    math_score = models.FloatField(null=True, blank=True)  # Allows null and blank entries
    reading_score = models.FloatField(null=True, blank=True)
    writing_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)  # Automatically set the field to now when the object is first created

