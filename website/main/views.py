import pandas as pd
from django.contrib import messages
from django.contrib.auth import login
# from .models import SavedPrevision
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from .forms import PrevisionForm
from .forms import RegisterForm
from .predictor import predict_scores
from .forms import UserForm, ProfileForm
from django.contrib.auth.models import User
from .models import Profile, Prevision


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
            # Replace 'some_view' with the name of the view you want to redirect to after saving
            return redirect('profile')
    else:
        user_form = UserForm(instance=request.user)
        profile_form = ProfileForm(instance=profile)

    return render(request, 'main/edit_account.html', {
        'user_form': user_form,
        'profile_form': profile_form,
    })


def base(request):
    return render(request, 'main/base.html')

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
def saved_previsions(request):
    if request.user.is_authenticated:
        previsions = Prevision.objects.filter(user=request.user).order_by('-created_at')
        return render(request, 'main/saved_previsions.html', {'previsions': previsions})
    else:
        # Redirect or handle non-authenticated users
        return redirect('login')

@login_required
def account_delete(request):
    if request.method == 'POST':
        user = request.user
        user.delete()
        messages.success(request, 'Your account has been deleted.')
        return redirect(' ')  

    return redirect('main/profile.html')

def convert_bool_string_to_numeric(df, boolean_fields):
    for field in boolean_fields:
        # Convert fields in DataFrame from 'True'/'False' to 1/0
        df[field] = df[field].replace({'True': 1, 'False': 0}).astype(int)
    return df

def guest_prevision(request):
    if request.method == 'POST':
        form = PrevisionForm(request.POST)
        if form.is_valid():
            input_data = form.cleaned_data
            df_input = pd.DataFrame([input_data])

            # Rename columns to match the expected format
            df_input.rename(columns={
                'race_ethnicity': 'race/ethnicity', 
                'parental_level_of_education': 'parental level of education'
            }, inplace=True)

            # Specify the fields that should be treated as boolean
            boolean_fields = ['gender', 'lunch', 'test_preparation_course']
            
            # Convert boolean string values to numeric directly in df_input
            df_input = convert_bool_string_to_numeric(df_input, boolean_fields)
            
            # Proceed with one-hot encoding for categorical columns
            categorical_columns = ['race/ethnicity', 'parental level of education']
            df_input_encoded = pd.get_dummies(df_input, columns=categorical_columns)
            
            # Ensure the DataFrame matches the expected structure
            expected_columns = [
                'gender', 'lunch', 'test preparation course','race/ethnicity_group A',
                'race/ethnicity_group B', 'race/ethnicity_group C',
                'race/ethnicity_group D', 'race/ethnicity_group E',
                "parental level of education_associate's degree",
                "parental level of education_bachelor's degree",
                'parental level of education_high school',
                "parental level of education_master's degree",
                'parental level of education_some college',
                'parental level of education_some high school'
            ]

            df_input_encoded = df_input_encoded.reindex(columns=expected_columns, fill_value=0).astype('float32')
            
            # Use the prepared DataFrame for prediction
            prediction_result = predict_scores(df_input_encoded)
            # Assuming prediction_result is a dictionary with values that might be of type float32
            prediction_result_serializable = {k: float(v) for k, v in prediction_result.items()}

            # Now, save the serializable version to the session
            request.session['prediction_result'] = prediction_result_serializable

            return redirect('prediction_results')
    else:
        form = PrevisionForm()
    return render(request, 'main/guest_main.html', {'form': form})



def prediction_results(request):
    prediction_result = request.session.get('prediction_result', {})
    return render(request, 'main/results.html', {'prediction_result': prediction_result})


@login_required  # Ensures only authenticated users can access this view
def main_prevision(request):
    if request.method == 'POST':
        form = PrevisionForm(request.POST)
        if form.is_valid():
            input_data = form.cleaned_data
            df_input = pd.DataFrame([input_data])

            # Adjust columns to the expected format, similar to guest_prevision
            df_input.rename(columns={
                'race_ethnicity': 'race/ethnicity',
                'parental_level_of_education': 'parental level of education'
            }, inplace=True)

            # Convert boolean fields and one-hot encode as before
            df_input = convert_bool_string_to_numeric(df_input, ['gender', 'lunch', 'test_preparation_course'])
            df_input_encoded = pd.get_dummies(df_input, columns=['race/ethnicity', 'parental level of education'])

            # Ensure DataFrame matches expected structure
            expected_columns = [
                # Add or remove columns based on model requirements
                'gender', 'lunch', 'test preparation course', 'race/ethnicity_group A',
                'race/ethnicity_group B', 'race/ethnicity_group C', 'race/ethnicity_group D',
                'race/ethnicity_group E', "parental level of education_associate's degree",
                "parental level of education_bachelor's degree", 'parental level of education_high school',
                "parental level of education_master's degree", 'parental level of education_some college',
                'parental level of education_some high school'
            ]
            df_input_encoded = df_input_encoded.reindex(columns=expected_columns, fill_value=0).astype('float32')

            # Predict and store results in the session
            prediction_result = predict_scores(df_input_encoded)
            request.session['prediction_result'] = {k: float(v) for k, v in prediction_result.items()}

            return redirect('prediction_results')
    else:
        form = PrevisionForm()
    return render(request, 'main/main_prevision.html', {'form': form})
@login_required
def saved_previsions(request):
    if request.method == 'POST':
        # Save a new prevision
        prediction_result = request.session.get('prediction_result', {})
        new_prevision = Prevision(
            user=request.user,
            math_score=prediction_result.get('math_score'),
            reading_score=prediction_result.get('reading_score'),
            writing_score=prediction_result.get('writing_score')
        )
        new_prevision.save()
        return redirect('saved_previsions')

    # Fetch all previsions for the current user to display
    previsions = Prevision.objects.filter(user=request.user)
    return render(request, 'main/saved_previsions.html', {'previsions': previsions})