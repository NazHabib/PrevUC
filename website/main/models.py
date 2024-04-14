from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    perfil = models.CharField(max_length=20)

    def __str__(self):
        return self.user.username

class Prevision(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    math_score = models.FloatField()
    reading_score = models.FloatField()
    writing_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prevision {self.id} by {self.user.username}"