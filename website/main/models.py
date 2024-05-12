from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.core.validators import MaxValueValidator, MinValueValidator
from jsonfield import ListField, JSONField


class NewsletterSubscriber(models.Model):
    email = models.EmailField(unique=True)
    subscribed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.email


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    PERFIL_CHOICES = (
        ('professor', 'Professor'),
        ('data scientist', 'Data Scientist'),
        ('user', 'User'),
    )
    perfil = models.CharField(max_length=15, choices=PERFIL_CHOICES)

    def __str__(self):
        return f"{self.user.username}'s Profile"


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

    gender = models.CharField(max_length=20, choices=GENDER_CHOICES)
    lunch = models.CharField(max_length=20, choices=LUNCH_CHOICES)
    test_preparation_course = models.CharField(max_length=20, choices=TEST_PREPARATION_CHOICES)
    race_ethnicity = models.CharField(max_length=20, choices=RACE_ETHNICITY_CHOICES)
    parental_level_of_education = models.CharField(max_length=50, choices=PARENTAL_LEVEL_OF_EDUCATION_CHOICES)
    math_score = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(100)], null=True, blank=True)
    reading_score = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(100)], null=True, blank=True)
    writing_score = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(100)], null=True, blank=True)
    validated = models.BooleanField(default=False)


class Prevision(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    gender = models.CharField(max_length=20, choices=PredictionDataForm.GENDER_CHOICES)
    lunch = models.CharField(max_length=20, choices=PredictionDataForm.LUNCH_CHOICES)
    test_preparation_course = models.CharField(max_length=20, choices=PredictionDataForm.TEST_PREPARATION_CHOICES)
    race_ethnicity = models.CharField(max_length=20, choices=PredictionDataForm.RACE_ETHNICITY_CHOICES)
    parental_level_of_education = models.CharField(max_length=50, choices=PredictionDataForm.PARENTAL_LEVEL_OF_EDUCATION_CHOICES)
    math_score = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(100)], null=True, blank=True)
    reading_score = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(100)], null=True, blank=True)
    writing_score = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(100)], null=True, blank=True)



@receiver(post_save, sender=Prevision)
def save_prediction_data(sender, instance, created, **kwargs):
    if created:
        PredictionDataForm.objects.create(
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


class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback from {self.user.username} on {self.created_at.strftime('%Y-%m-%d')}"


class ModelConfiguration(models.Model):
    num_layers = models.IntegerField(default=3)
    neurons_per_layer = models.CharField(max_length=255, default='[64, 64, 64]')
    epochs = models.IntegerField(default=10)
    learning_rate = models.FloatField(default=0.01)
    batch_size = models.IntegerField(default=16)
    rating = models.IntegerField(default=0, blank=True, null=True)  # Rating field added
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Configuration {self.id} - Rating: {self.rating or 'Not rated'}"

class ModelMetrics(models.Model):
    configuration = models.ForeignKey(ModelConfiguration, on_delete=models.CASCADE)
    loss = models.FloatField()
    mse = models.FloatField()
    mae = models.FloatField()
    rmse = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Metrics for Configuration {self.configuration.id}"


class NeuronLayer(models.Model):
    model_config = models.ForeignKey('ModelConfigurationTesting', on_delete=models.CASCADE)
    neurons = models.IntegerField()

class ModelConfigurationTesting(models.Model):
    num_layers = models.IntegerField(verbose_name='Number of Layers', default=1)
    epochs = models.IntegerField(verbose_name='Epochs', default=10)
    learning_rate = models.FloatField(verbose_name='Learning Rate', default=0.01)
    batch_size = models.IntegerField(verbose_name='Batch Size', default=32)

    # Storing results as JSON
    loss = models.JSONField(blank=True, null=True, verbose_name='Loss')
    mae = models.JSONField(blank=True, null=True, verbose_name='Mean Absolute Error')
    rmse = models.JSONField(blank=True, null=True, verbose_name='Root Mean Squared Error')
    mse = models.JSONField(blank=True, null=True, verbose_name='Mean Squared Error')

    def __str__(self):
        return f"Configuration #{self.pk} with {self.num_layers} layers"