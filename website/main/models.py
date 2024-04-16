from django.db import models
from django.contrib.auth.models import User

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


from django.core.validators import MaxValueValidator, MinValueValidator


class PredictionDataForm(models.Model):
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

    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    lunch = models.CharField(max_length=20, choices=LUNCH_CHOICES)
    test_preparation_course = models.CharField(max_length=20, choices=TEST_PREPARATION_CHOICES)
    race_ethnicity = models.CharField(max_length=20, choices=RACE_ETHNICITY_CHOICES)
    parental_level_of_education = models.CharField(max_length=50, choices=PARENTAL_LEVEL_OF_EDUCATION_CHOICES)
    math_score = models.FloatField(validators=[MinValueValidator(1), MaxValueValidator(100)])
    reading_score = models.FloatField(validators=[MinValueValidator(1), MaxValueValidator(100)])
    writing_score = models.FloatField(validators=[MinValueValidator(1), MaxValueValidator(100)])


class Prevision(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    math_score = models.FloatField(null=True, blank=True)
    reading_score = models.FloatField(null=True, blank=True)
    writing_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

