from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.core.validators import MaxValueValidator, MinValueValidator


class NewsletterSubscriber(models.Model):
    email = models.EmailField(unique=True)
    subscribed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.email


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    PERFIL_CHOICES = (
        ('professor', 'Professor'),
        ('data scientist', 'DataScientist'),
        ('user', 'User'),
    )
    perfil = models.CharField(max_length=15, choices=PERFIL_CHOICES)

    def __str__(self):
        return f"{self.user.name}'s Profile"

class CommonInfo(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]
    LUNCH_CHOICES = [
        ('Standard', 'Standard'),
        ('Free/Reduced', 'Free/Reduced'),
    ]
    TEST_PREPARATION_CHOICES = [
        ('None', 'None'),
        ('Completed', 'Completed'),
    ]
    RACE_ETHNICITY_CHOICES = [
        ('Group A', 'Group A'),
        ('Group B', 'Group B'),
        ('Group C', 'Group C'),
        ('Group D', 'Group D'),
        ('Group E', 'Group E'),
    ]
    PARENTAL_LEVEL_OF_EDUCATION_CHOICES = [
        ('High School', 'High School'),
        ('Some College', 'Some College'),
        ('Some High School', 'Some High School'),
        ("Bachelor's Degree", "Bachelor's Degree"),
        ("Master's Degree", "Master's Degree"),
        ("Associate's Degree", "Associate's Degree"),
    ]

    gender = models.CharField(max_length=20, choices=GENDER_CHOICES)
    lunch = models.CharField(max_length=20, choices=LUNCH_CHOICES)
    test_preparation_course = models.CharField(max_length=20, choices=TEST_PREPARATION_CHOICES)
    race_ethnicity = models.CharField(max_length=20, choices=RACE_ETHNICITY_CHOICES)
    parental_level_of_education = models.CharField(max_length=50, choices=PARENTAL_LEVEL_OF_EDUCATION_CHOICES)
    math_score = models.FloatField(validators=[MinValueValidator(1), MaxValueValidator(100)], null=True, blank=True)
    reading_score = models.FloatField(validators=[MinValueValidator(1), MaxValueValidator(100)], null=True, blank=True)
    writing_score = models.FloatField(validators=[MinValueValidator(1), MaxValueValidator(100)], null=True, blank=True)

    class Meta:
        abstract = True

class PredictionDataForm(CommonInfo):
    validated = models.BooleanField(default=False)

class Prevision(CommonInfo):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)



@receiver(post_save, sender=Prevision)
def save_prediction_data(sender, instance, created, **kwargs):
    if created:
        prediction_data = PredictionDataForm.objects.create(
            gender=instance.gender,
            lunch=instance.lunch,
            test_preparation_course=instance.test_preparation_course,
            race_ethnicity=instance.race_ethnicity,
            parental_level_of_education=instance.parental_level_of_education,
            math_score=instance.math_score,
            reading_score=instance.reading_score,
            writing_score=instance.writing_score
        )
class ChangeLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

class Notification(models.Model):
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.message


