import pandas as pd
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import redirect, render
from .forms import PrevisionForm, PrevisionInputForm, ChangeForm, NotificationForm
from .forms import RegisterForm
from .predictor import predict_scores
from .forms import UserForm, ProfileForm
from .models import Profile, Prevision
from .models import PredictionDataForm
from .decorators import role_required
from .models import Notification
from django.core.paginator import Paginator

def validate_data(request):
    if request.method == 'GET':
        # Retrieve all data entries for validation
        data_entries = PredictionDataForm.objects.all()
        return render(request, 'main/validate_data.html', {'data_entries': data_entries})
    elif request.method == 'POST':
        # Handle form submission for data validation
        data_entry_id = request.POST.get('data_entry_id')
        action = request.POST.get('action')

        if action == 'validate':
            # Update data entry status to validated
            data_entry = PredictionDataForm.objects.get(id=data_entry_id)
            data_entry.validated = True
            data_entry.save()
        elif action == 'delete':
            # Delete the data entry
            data_entry = PredictionDataForm.objects.get(id=data_entry_id)
            data_entry.delete()

        return redirect('validate_data')

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
def view_notifications(request):
    # Retrieve all notifications from the database
    notifications = Notification.objects.all().order_by('-created_at')
    return render(request, 'main/notifications.html', {'notifications': notifications})

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
            return redirect('home')  # Redirect to an appropriate page
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
def account_delete(request):
    if request.method == 'POST':
        user = request.user
        user.delete()
        messages.success(request, 'Your account has been deleted.')
        return redirect('login')

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



@login_required
def saved_previsions(request):
    if request.method == 'POST':
        # Save a new prevision
        prediction_result = request.session.get('prediction_result', {})
        new_prevision = Prevision.objects.create(
            user=request.user,
            math_score=prediction_result.get('math_score'),
            reading_score=prediction_result.get('reading_score'),
            writing_score=prediction_result.get('writing_score'),
            gender=prediction_result.get('gender'),
            lunch=prediction_result.get('lunch'),
            test_preparation_course=prediction_result.get('test_preparation_course'),
            race_ethnicity=prediction_result.get('race_ethnicity'),
            parental_level_of_education=prediction_result.get('parental_level_of_education')
        )
        return redirect('saved_previsions')

    # Fetch all previsions for the current user to display
    previsions = Prevision.objects.filter(user=request.user)
    return render(request, 'main/saved_previsions.html', {'previsions': previsions})


@login_required
def main_prevision(request):
    if request.method == 'POST':
        form = PrevisionForm(request.POST)
        if form.is_valid():
            # Extract the form data without modifications
            gender = form.cleaned_data['gender']
            lunch = form.cleaned_data['lunch']
            test_preparation_course = form.cleaned_data['test_preparation_course']
            race_ethnicity = form.cleaned_data['race_ethnicity']
            parental_level_of_education = form.cleaned_data['parental_level_of_education']

            # Prepare the input data for the predictive model, possibly modifying it
            df_input = pd.DataFrame([form.cleaned_data])
            df_input.rename(columns={
                'race_ethnicity': 'race/ethnicity',
                'parental_level_of_education': 'parental level of education',
            }, inplace=True)
            df_input = convert_bool_string_to_numeric(df_input, ['gender', 'lunch', 'test_preparation_course'])
            df_input_encoded = pd.get_dummies(df_input, columns=['race/ethnicity', 'parental level of education'])

            # Make sure the DataFrame matches the expected structure
            expected_columns = [
                # Define the expected structure for the predictive model
            ]
            df_input_encoded = df_input_encoded.reindex(columns=expected_columns, fill_value=0).astype('float32')

            # Get the prediction results
            prediction_result = predict_scores(df_input_encoded)

            # Now, save the original user input and prediction result
            new_prevision = Prevision.objects.create(
                user=request.user,
                gender=gender,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                math_score=prediction_result.get('math_score'),
                reading_score=prediction_result.get('reading_score'),
                writing_score=prediction_result.get('writing_score'),
            )

            # Store prediction results in the session to pass to the results page
            request.session['prediction_result'] = prediction_result

            return redirect('prediction_results')

    else:
        form = PrevisionForm()

    return render(request, 'main/main_prevision.html', {'form': form})


def prediction_results(request):
    prediction_result = request.session.get('prediction_result', {})
    additional_params = request.session.get('additional_params', {})
    return render(request, 'main/results.html', {
        'prediction_result': prediction_result,
        'gender': additional_params.get('gender', 'Not specified'),
        'lunch': additional_params.get('lunch', 'Not specified'),
        'test_preparation_course': additional_params.get('test_preparation_course', 'Not specified'),
        'race_ethnicity': additional_params.get('race_ethnicity', 'Not specified'),
        'parental_level_of_education': additional_params.get('parental_level_of_education', 'Not specified')
    })


@login_required
def saved_previsions(request):
    if request.method == 'POST':
        # Create a new prevision instance with data from the form
        new_prevision = Prevision(
            user=request.user,
            gender=request.session.get('prediction_data', {}).get('gender', 'Not specified'),
            lunch=request.session.get('prediction_data', {}).get('lunch', 'Not specified'),
            test_preparation_course=request.session.get('prediction_data', {}).get('test_preparation_course', 'Not specified'),
            race_ethnicity=request.session.get('prediction_data', {}).get('race_ethnicity', 'Not specified'),
            parental_level_of_education=request.session.get('prediction_data', {}).get('parental_level_of_education', 'Not specified'),
            math_score=request.POST.get('math_score'),
            reading_score=request.POST.get('reading_score'),
            writing_score=request.POST.get('writing_score'),
        )
        new_prevision.save()

        # Redirect to the page that shows all saved previsions
        return redirect('saved_previsions')

    # If it's not a POST request, fetch all previsions to display
    previsions = Prevision.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'main/saved_previsions.html', {'previsions': previsions})
