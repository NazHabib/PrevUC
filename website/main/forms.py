from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.models import User
from .models import Profile, PredictionDataForm, Feedback, ModelConfigurationTesting
from .models import ChangeLog
from .models import Notification
from .models import ModelConfiguration
from django.core.exceptions import ValidationError
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import get_user_model
from django.core.mail import send_mail
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator as token_generator, PasswordResetTokenGenerator
from django.urls import reverse


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

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise ValidationError("This email is already in use.")
        return email

    def save(self, commit=True):
        user = super(RegisterForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        user.is_active = False  # User must be activated before they can log in
        if commit:
            user.save()
            profile = Profile.objects.create(user=user, perfil=self.cleaned_data['perfil'])
            profile.save()
        return user

def send_activation_email(user, request):
    token_generator = PasswordResetTokenGenerator()
    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = token_generator.make_token(user)
    link = request.build_absolute_uri(reverse('activate_account', args=[uid, token]))
    subject = 'Activate your account'
    message = f'Hi {user.username}, please activate your account by clicking this link: {link}'
    send_mail(subject, message, 'from@example.com', [user.email])

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


class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Your feedback...'})
        }


class ModelConfigurationForm(forms.ModelForm):
    class Meta:
        model = ModelConfiguration
        fields = ['num_layers', 'neurons_per_layer', 'epochs', 'learning_rate', 'batch_size']

class ModelSelectionForm(forms.Form):
    model_type = forms.ChoiceField(choices=[
        ('model_math', 'model_math'),
        ('model_reading', 'model_reading'),
        ('model_writing', 'model_writing')
    ])


class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    perfil = forms.ChoiceField(choices=Profile.PERFIL_CHOICES, required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2", "perfil")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
            Profile.objects.create(user=user, perfil=self.cleaned_data['perfil'])
        return user

class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = User
        fields = '__all__'


class ModelConfigurationFormTesting(forms.ModelForm):
    neurons_per_layer = forms.CharField(widget=forms.Textarea)

    class Meta:
        model = ModelConfigurationTesting
        fields = ['num_layers', 'neurons_per_layer', 'epochs', 'learning_rate', 'batch_size']