from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserChangeForm
from .models import Profile

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    perfil = forms.CharField(required=True)

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

    def save(self, commit=True):
        user = super(RegisterForm, self).save(commit=False)
        if commit:
            user.save()
            profile = Profile(user=user, perfil=self.cleaned_data['perfil'])
            profile.save()
        return user


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['perfil']


# For editing User model fields
class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["username", "email"]

# For editing Profile model fields
class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['perfil']


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