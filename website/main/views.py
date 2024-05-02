import numpy as np
from django.contrib.auth import login
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import PrevisionInputForm, ChangeForm, NotificationForm
from .forms import RegisterForm
from .predictor import predict_scores
from .forms import UserForm, ProfileForm
from .models import Profile, Prevision, NewsletterSubscriber
from .decorators import role_required
from .models import Notification
from django.contrib.staticfiles.storage import staticfiles_storage
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.contrib import messages
from .forms import PredictionDataForm
from django.shortcuts import render, redirect
from .forms import PrevisionForm
import pandas as pd
import numpy as np
import json


@login_required
def list_data_entries(request):
    data_entries = PredictionDataForm.objects.filter(validated=False)
    return render(request, 'main/validate_data.html', {'data_entries': data_entries})


@login_required
@require_POST
def delete_data(request, entry_id):
    data_entry = get_object_or_404(PredictionDataForm, id=entry_id)
    if data_entry.validated == False:
        data_entry.delete()
        messages.success(request, "Data deleted successfully.")
        return redirect('list_data_entries')
    else:
        return redirect('list_data_entries')


@login_required
@require_POST
def validate_data(request, entry_id):
    data_entry = get_object_or_404(PredictionDataForm, id=entry_id)
    data_entry.validated = True
    data_entry.save()
    messages.success(request, "Data validated successfully.")
    return redirect('list_data_entries')


@login_required
def create_notification(request):
    if request.method == 'POST':
        form = NotificationForm(request.POST)
        if form.is_valid():
            notification = form.save(commit=False)
            notification.save()
            return redirect('home')
    else:
        form = NotificationForm()
    return render(request, 'main/create_notification.html', {'form': form})


def notify_users(change):
    message = f"{change.user.username} updated the model '{change.name}' at {change.created_at.strftime('%Y-%m-%d %H:%M:%S')}: {change.description}"
    notification = Notification.objects.create(message=message)
    for user in User.objects.exclude(id=change.user.id):
        notification.users_notified.add(user)
    notification.save()


@login_required
def change_documentation(request):
    if request.method == 'POST':
        form = ChangeForm(request.POST)
        if form.is_valid():
            change = form.save(commit=False)
            change.user = request.user
            change.save()
            # Notify other users
            notify_users(change)
            return redirect('home')
    else:
        form = ChangeForm()
    return render(request, 'main/change_documentation.html', {'form': form})


@login_required
def view_notifications(request):
    notifications = Notification.objects.all()
    return render(request, 'main/notifications.html', {'notifications': notifications})


@role_required('professor')
def data_input(request):
    if request.method == 'POST':
        form = PrevisionInputForm(request.POST)
        if form.is_valid():
            new_data = PredictionDataForm(
                gender=form.cleaned_data['gender'],
                lunch=form.cleaned_data['lunch'],
                test_preparation_course=form.cleaned_data['test_preparation_course'],
                race_ethnicity=form.cleaned_data['race_ethnicity'],
                parental_level_of_education=form.cleaned_data['parental_level_of_education'],
                math_score=form.cleaned_data['math_score'],
                reading_score=form.cleaned_data['reading_score'],
                writing_score=form.cleaned_data['writing_score']
            )
            new_data.save()
            return redirect('home')
    else:
        form = PrevisionInputForm()

    return render(request, 'main/data_input.html', {'form': form})


@login_required
def edit_account(request):
    try:
        profile = request.user.profile
    except Profile.DoesNotExist:
        profile = Profile(user=request.user)
        profile.save()

    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, instance=profile)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect('profile')
    else:
        user_form = UserForm(instance=request.user)
        profile_form = ProfileForm(instance=profile)

    return render(request, 'main/edit_account.html', {
        'user_form': user_form,
        'profile_form': profile_form,
    })


def base(request):
    lavender_image_url = staticfiles_storage.url('main/lavend.jpg')
    context = {'lavender_image_url': lavender_image_url}
    return render(request, 'main/base.html', context)


def home(request):
    return render(request, 'main/home.html')


def sign_up(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('/home')
    else:
        form = RegisterForm()

    return render(request, 'registration/sign_up.html', {"form": form})


def privacy_policy(request):
    return render(request, 'main/privacy_policy.html')


def terms_of_service(request):
    return render(request, 'main/terms_of_service.html')


def profile(request):
    return render(request, 'main/profile.html')


@login_required
def account_delete(request):
    if request.method == 'POST':
        user = request.user
        user.delete()
        messages.success(request, 'Your account has been deleted.')
        return redirect('login')

    return redirect('main/profile.html')


def convert_bool_string_to_numeric(df, boolean_fields):
    pd.set_option('future.no_silent_downcasting', True)

    for field in boolean_fields:
        df[field] = df[field].replace({'True': True, 'False': False}).astype(bool).astype(int)
    return df


def guest_prevision(request):
    if request.method == 'POST':
        form = PrevisionForm(request.POST)
        if form.is_valid():
            input_data = form.cleaned_data
            df_input = pd.DataFrame([input_data])

            df_input.rename(columns={
                'race_ethnicity': 'race/ethnicity',
                'parental_level_of_education': 'parental level of education'
            }, inplace=True)

            boolean_fields = ['gender', 'lunch', 'test_preparation_course']

            df_input = convert_bool_string_to_numeric(df_input, boolean_fields)

            categorical_columns = ['race/ethnicity', 'parental level of education']
            df_input_encoded = pd.get_dummies(df_input, columns=categorical_columns)

            expected_columns = [
                'gender', 'lunch', 'test preparation course', 'race/ethnicity_group A',
                'race/ethnicity_group B', 'race/ethnicity_group C',
                'race/ethnicity_group D', 'race/ethnicity_group E',
                "parental level of education_associate's degree",
                "parental level of education_bachelor's degree",
                'parental level of education_high school',
                "parental level of education_master's degree",
                'parental level of education_some college',
                'parental level of education_some high school'
            ]

            df_input_encoded = df_input_encoded.reindex(columns=expected_columns, fill_value=0).astype('int32')

            prediction_result = predict_scores(df_input_encoded)

            prediction_result_serializable = {k: int(v) for k, v in prediction_result.items()}

            request.session['prediction_result'] = prediction_result_serializable

            return redirect('prediction_results')
    else:
        form = PrevisionForm()
    return render(request, 'main/guest_main.html', {'form': form})


def main_prevision(request):
    if request.method == 'POST':
        form = PrevisionForm(request.POST)
        if form.is_valid():
            input_data = form.cleaned_data

            # Create DataFrame from cleaned data
            df_input = pd.DataFrame([input_data])

            # Rename columns directly in the view function
            df_input.rename(columns={
                'race_ethnicity': 'race/ethnicity',
                'parental_level_of_education': 'parental level of education'
            }, inplace=True)

            # Convert boolean fields from string representation to numeric
            boolean_fields = ['gender', 'lunch', 'test_preparation_course']
            for field in boolean_fields:
                df_input[field] = df_input[field].replace({'True': 1, 'False': 0}).astype(int)

            # One-hot encoding categorical columns
            categorical_columns = ['race/ethnicity', 'parental level of education']
            df_input_encoded = pd.get_dummies(df_input, columns=categorical_columns)

            # Ensure DataFrame matches expected structure
            expected_columns = [
                'gender', 'lunch', 'test preparation course', 'race/ethnicity_group A',
                'race/ethnicity_group B', 'race/ethnicity_group C',
                'race/ethnicity_group D', 'race/ethnicity_group E',
                "parental level of education_associate's degree",
                "parental level of education_bachelor's degree",
                'parental level of education_high school',
                "parental level of education_master's degree",
                'parental level of education_some college',
                'parental level of education_some high school'
            ]
            df_input_encoded = df_input_encoded.reindex(columns=expected_columns, fill_value=0).astype('int32')

            # Obtain prediction results
            prediction_result = predict_scores(df_input_encoded)

            # Ensure the prediction result is serializable
            prediction_result_serializable = {k: int(v) for k, v in prediction_result.items()}

            # Store serialized prediction result and input data in session
            request.session['prediction_result'] = prediction_result_serializable
            request.session['prediction_data'] = input_data

            # Redirect to the prediction results page
            return redirect('prediction_results')
        else:
            # If the form is not valid, render the form with errors
            return render(request, 'main/main_prevision.html', {'form': form})
    else:
        # For GET requests, simply display the form
        form = PrevisionForm()
    return render(request, 'main/main_prevision.html', {'form': form})

def prediction_results(request):
    prediction_result = request.session.get('prediction_result', {})
    prediction_data = request.session.get('prediction_data', {})
    return render(request, 'main/results.html', {
        'prediction_result': prediction_result,
        'prediction_data': prediction_data
    })

@login_required
def saved_previsions(request):
    if request.method == 'POST':
        if 'prediction_data' in request.session and 'prediction_result' in request.session:
            prediction_data = request.session['prediction_data']
            prediction_result = request.session['prediction_result']

            new_prevision = Prevision.objects.create(
                user=request.user,
                gender=prediction_data['gender'],
                lunch=prediction_data['lunch'],
                test_preparation_course=prediction_data['test_preparation_course'],
                race_ethnicity=prediction_data['race_ethnicity'],
                parental_level_of_education=prediction_data['parental_level_of_education'],
                math_score=prediction_result.get('math_score'),
                reading_score=prediction_result.get('reading_score'),
                writing_score=prediction_result.get('writing_score'),
            )
            print("New prediction saved:", new_prevision)
            request.session.pop('prediction_data', None)
            request.session.pop('prediction_result', None)
            return redirect('main_prevision')

    previsions = Prevision.objects.filter(user=request.user)
    return render(request, 'main/saved_previsions.html', {'previsions': previsions})


def subscribe_newsletter(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        if email:
            subscriber, created = NewsletterSubscriber.objects.get_or_create(email=email)
            if created:
                success_message = "Subscription successful!"
            else:
                success_message = "You are already subscribed!"
            return HttpResponseRedirect(reverse('base'))
    return HttpResponseRedirect(reverse('base'))
