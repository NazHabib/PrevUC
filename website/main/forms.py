from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile, PredictionDataForm
from .models import ChangeLog
from .models import Notification

class NotificationForm(forms.ModelForm):
    class Meta:
        model = Notification
        fields = ['message']
class ChangeForm(forms.ModelForm):
    class Meta:
        model = ChangeLog
        fields = ['name', 'description']

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    perfil = forms.ChoiceField(choices=Profile.PERFIL_CHOICES, required=True)

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

    def save(self, commit=True):
        user = super(RegisterForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            profile = Profile(user=user, perfil=self.cleaned_data['perfil'])
            profile.save()
        return user

class ProfileForm(forms.ModelForm):
    perfil = forms.ChoiceField(choices=Profile.PERFIL_CHOICES)

    class Meta:
        model = Profile
        fields = ['perfil']


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['perfil']


# For editing User model fields
class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["username", "email"]


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

class PrevisionInputForm(forms.ModelForm):
    class Meta:
        model = PredictionDataForm
        fields = ['gender', 'lunch', 'test_preparation_course', 'race_ethnicity', 'parental_level_of_education', 'math_score', 'reading_score', 'writing_score']