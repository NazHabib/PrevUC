from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserChangeForm

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    perfil = forms.CharField(required=True)
    class Meta:
        model=User
        fields = ["username", "email", "perfil", "password1", "password2"]

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["username", "email"]

class UserForm(UserChangeForm):
    password = None  # Prevents the password fields from being displayed
    class Meta:
        model = User
        fields = ['username', 'email']

class PrevisionForm(forms.Form):
    GENDER_CHOICES = [
        ('True', 'Male'),
        ('False', 'Female'),
    ]
    
    LUNCH_CHOICES = [
        ('True', 'Standard'),
        ('False', 'Free/Reduced'),
    ]
    
    TEST_PREPARATION_CHOICES = [
        ('False', 'None'),
        ('True', 'Completed'),
    ]
    
    RACE_ETHNICITY_CHOICES = [
        ('group A', 'Group A'),
        ('group B', 'Group B'),
        ('group C', 'Group C'),
        ('group D', 'Group D'),
        ('group E', 'Group E'),
    ]
    
    PARENTAL_LEVEL_OF_EDUCATION_CHOICES = [
        ('high school', 'High School'),
        ('some college', 'Some College'),
        ('some high school', 'Some High School'),
        ("bachelor's degree", "Bachelor's Degree"),
        ("master's degree", "Master's Degree"),
        ("associate's degree", "Associate's degree"),
    ]
    
    gender = forms.ChoiceField(choices=GENDER_CHOICES)
    lunch = forms.ChoiceField(choices=LUNCH_CHOICES)
    test_preparation_course = forms.ChoiceField(choices=TEST_PREPARATION_CHOICES)
    race_ethnicity = forms.ChoiceField(choices=RACE_ETHNICITY_CHOICES)
    parental_level_of_education = forms.ChoiceField(choices=PARENTAL_LEVEL_OF_EDUCATION_CHOICES)