import keras
from django.contrib.sites.shortcuts import get_current_site
from django.db import transaction
from django.contrib.auth import login, authenticate, get_user_model
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.template.loader import render_to_string
from django.urls import reverse
from keras import Sequential, Input
from django.utils.encoding import force_str
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from .forms import PrevisionInputForm, ChangeForm, NotificationForm, ModelSelectionForm, ModelConfigurationFormTesting
from .forms import RegisterForm
from .predictor import predict_scores
from .forms import UserForm, ProfileForm
from .models import Profile, Prevision, NewsletterSubscriber, ChangeLog, ModelMetrics, ModelConfigurationTesting, \
    NeuronLayer, ModelParameters
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
from .models import Feedback
from .forms import FeedbackForm
from .models import ModelConfiguration
from .forms import ModelConfigurationForm
from django.urls import reverse
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator as token_generator, default_token_generator
import logging
from django.core.mail import send_mail


def calculate_metrics(model, X_train, y_train, X_test, y_test, loss_fn):
    if model is None:
        raise ValueError('The model variable cannot be None.')
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_train_loss = loss_fn(y_train, y_pred_train)
    loss_train = np.mean(y_pred_train_loss)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    loss_test = model.evaluate(X_test, y_test, verbose=0)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    return {
        'loss_train': loss_train,
        'mse_train': mse_train,
        'mae_train': mae_train,
        'r2_train': r2_train,
        'loss_test': loss_test,
        'mse_test': mse_test,
        'mae_test': mae_test,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
    }


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




@login_required
def view_changes(request):
    changes = ChangeLog.objects.all()
    return render(request, 'main/view_changes.html', {'changes': changes})


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if not user.is_active:
                messages.error(request, 'This account is inactive. Please check your email to activate.')
                return render(request, 'login.html', {'form': form})
            if user is not None:
                login(request, user)
                return HttpResponseRedirect('/home')
            else:
                messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})



@login_required
def change_documentation(request):
    if request.method == 'POST':
        form = ChangeForm(request.POST)
        if form.is_valid():
            change = form.save(commit=False)
            change.user = request.user
            change.save()
            return redirect('home')
    else:
        form = ChangeForm()
    return render(request, 'main/change_documentation.html', {'form': form})


@login_required
def view_notifications(request):
    notifications = Notification.objects.all()
    return render(request, 'main/notifications.html', {'notifications': notifications})



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


def send_activation_email(user, request):
    from django.utils.http import urlsafe_base64_encode
    from django.utils.encoding import force_bytes
    from django.contrib.auth.tokens import default_token_generator
    from django.core.mail import send_mail
    from django.urls import reverse

    uidb64 = urlsafe_base64_encode(force_bytes(user.pk))
    token = default_token_generator.make_token(user)
    link = request.build_absolute_uri(reverse('activate_account', args=[uidb64, token]))

    subject = 'Activate your account'
    message = f'Olá {user.username}, ative a conta clicando no link: {link}'
    send_mail(subject, message, 'from@example.com', [user.email])


def sign_up(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            form.save_m2m()
            send_activation_email(user, request)
            return redirect('account_activation_sent')
    else:
        form = RegisterForm()

    return render(request, 'registration/sign_up.html', {'form': form})

def account_activation_sent(request):
    return render(request, 'registration/account_activation_sent.html')


def activate_account(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = get_user_model().objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, get_user_model().DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        return redirect('login')
    else:
        return HttpResponse('Activation link is invalid!')


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


def guest_prevision_form(request):
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
            df_input = pd.DataFrame([input_data])
            df_input.rename(columns={
                'race_ethnicity': 'race/ethnicity',
                'parental_level_of_education': 'parental level of education'
            }, inplace=True)
            boolean_fields = ['gender', 'lunch', 'test_preparation_course']
            for field in boolean_fields:
                df_input[field] = df_input[field].replace({'True': 1, 'False': 0}).astype(int)
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
            request.session['prediction_data'] = input_data
            return redirect('prediction_results')
        else:
            return render(request, 'main/main_prevision.html', {'form': form})
    else:
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
def delete_prevision(request, prevision_id):
    prevision = get_object_or_404(Prevision, pk=prevision_id, user=request.user)
    prevision.delete()
    return JsonResponse({'message': 'Prevision deleted successfully'})


@login_required
def saved_previsions(request):
    if request.method == 'POST':
        prediction_data = request.session.get('prediction_data')
        prediction_result = request.session.get('prediction_result')

        if prediction_data and prediction_result:
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
        else:
            messages.error(request, "Missing prediction data.")
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
            return HttpResponseRedirect(reverse('home'))
    return HttpResponseRedirect(reverse('home'))


def add_feedback(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            feedback = form.save(commit=False)
            feedback.user = request.user
            feedback.save()
            return redirect('home')
    else:
        form = FeedbackForm()

    return render(request, 'main/add_feedback.html', {'form': form})

def dashboard_view(request):
    feedback_list = Feedback.objects.all()
    return render(request, 'main/feedback.html', {
        'feedback_list': feedback_list
    })


def model_configuration_view(request, pk=None):
    instance = ModelConfiguration.objects.get(pk=pk) if pk else None
    if request.method == 'POST':
        form = ModelConfigurationForm(request.POST, instance=instance)
        if form.is_valid():
            form.save()
            return redirect('model_metrics_list')
    else:
        form = ModelConfigurationForm(instance=instance)

    return render(request, 'main/model_configuration_form.html', {'form': form})


def preprocess_data(data):
    data['gender'] = data['gender'].map({'male': 1, 'female': 0})
    data['lunch'] = data['lunch'].map({'standard': 1, 'free/reduced': 0})
    data['test preparation course'] = data['test preparation course'].map({'completed': 1, 'none': 0})
    return pd.get_dummies(data, columns=['race/ethnicity', 'parental level of education'])

def get_features_and_labels(data, model_type):
    feature_columns = data.columns.drop(['math score', 'reading score', 'writing score'])
    X = data[feature_columns].astype('int32')
    if model_type == 'model_math':
        y = data['math score'].astype('int32')
    elif model_type == 'model_reading':
        y = data['reading score'].astype('int32')
    else:
        y = data['writing score'].astype('int32')
    return X, y

def save_metrics(y_test, y_pred, config, model_type):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    metrics = ModelMetrics(configuration=config, model_type=model_type, loss=0, mse=mse, mae=mae, rmse=rmse)
    metrics.save()
    return metrics

def model_configuration_list_view(request):
    configurations = ModelConfiguration.objects.all()
    return render(request, 'main/model_metrics_list.html', {'configurations': configurations})


def model_comparison_view(request):
    metrics = ModelMetrics.objects.all()
    return render(request, 'main/model_comparison.html', {'metrics': metrics})

def model_metrics_list(request):
    configurations = ModelConfiguration.objects.all().order_by('-created_at')
    return render(request, 'main/model_metrics_list.html', {'configurations': configurations})


def train_model(model: Sequential, X_train: pd.DataFrame, y_train: pd.Series, epochs: int, batch_size: int, learning_rate: float) -> dict:
    """Train the model"""
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return history


def preprocess_data(file_path: str) -> pd.DataFrame:
    """Preprocess the data"""
    data = pd.read_csv(file_path)
    data['gender'] = data['gender'].map({'male': 1, 'female': 0})
    data['lunch'] = data['lunch'].map({'standard': 1, 'free/reduced': 0})
    data['test preparation course'] = data['test preparation course'].map({'completed': 1, 'none': 0})
    data = pd.get_dummies(data, columns=['race/ethnicity', 'parental level of education'])
    return data

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[pd.Series]]:
    """Split the data into features and targets"""
    feature_columns = data.columns.drop(['math score', 'reading score', 'writing score'])
    X_target = data[feature_columns].astype('int32')
    y_targets = [data['math score'].astype('int32'), data['reading score'].astype('int32'), data['writing score'].astype('int32')]
    return X_target, y_targets

def train_test_split_data(X_target: pd.DataFrame, y_targets: List[pd.Series]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.Series], List[pd.Series]]:
    """Split the data into training and testing sets"""
    X_train, X_test, y_trains, y_tests = [], [], [], []
    for i, y_target in enumerate(y_targets):
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_target, y_target, test_size=0.3, random_state=44 + i)
        X_train.append(X_train_i)
        X_test.append(X_test_i)
        y_trains.append(y_train_i)
        y_tests.append(y_test_i)
    return X_train, X_test, y_trains, y_tests

def build_model(neurons_per_layer: List[int], input_shape: int) -> Sequential:
    """Build the neural network model"""
    model = Sequential()
    model.add(Input(shape=input_shape))
    for num_neurons in neurons_per_layer:
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(1))
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    rmse_value = np.sqrt(loss)
    y_test_pred = model.predict(X_test)
    mse = np.mean((y_test_pred - y_test.to_numpy()) ** 2)
    return loss, mae, rmse_value, mse


from django.shortcuts import render
from django.http import JsonResponse
from .forms import ModelConfigurationFormTesting
from .models import ModelConfigurationTesting, NeuronLayer

logger = logging.getLogger(__name__)

def train_and_evaluate_model(request):
    if request.method == 'POST':
        form = ModelConfigurationFormTesting(request.POST)
        if form.is_valid():
            file_path = 'main/StudentsPerformance.csv'
            data = preprocess_data(file_path)
            X_target, y_targets = split_data(data)
            X_train, X_test, y_trains, y_tests = train_test_split_data(X_target, y_targets)
            num_layers = form.cleaned_data['num_layers']
            neurons_per_layer_str = form.cleaned_data['neurons_per_layer']
            try:
                neurons_per_layer = [int(x) for x in neurons_per_layer_str.split(',')]
            except ValueError as e:
                logger.error("Error parsing neurons_per_layer: %s", e)
                return JsonResponse({'message': 'Invalid neurons_per_layer format.'}, status=400)
            input_dim = (X_train[0].shape[1],)
            model = build_model(neurons_per_layer, input_dim)
            epochs = form.cleaned_data['epochs']
            batch_size = form.cleaned_data['batch_size']
            learning_rate = form.cleaned_data['learning_rate']
            for i in range(len(X_train)):
                train_data = X_train[i]
                train_labels = y_trains[i]
                history = train_model(model, train_data, train_labels, epochs, batch_size, learning_rate)
            loss_values, mae_values, rmse_values, mse_values = [], [], [], []
            for i in range(len(X_test)):
                loss, mae, rmse_value, mse = evaluate_model(model, X_test[i], y_tests[i])
                loss_values.append(loss)
                mae_values.append(mae)
                rmse_values.append(rmse_value)
                mse_values.append(mse)
            model_config = ModelConfigurationTesting(
                num_layers=num_layers,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                loss=loss_values,
                mae=mae_values,
                rmse=rmse_values,
                mse=mse_values
            )
            model_config.save()
            for neurons in neurons_per_layer:
                NeuronLayer(model_config=model_config, neurons=neurons).save()

            response = {
                'message': 'Model trained and evaluated successfully.',
                'loss': loss_values,
                'mae': mae_values,
                'rmse': rmse_values,
                'mse': mse_values
            }
            return HttpResponseRedirect(reverse('model_results', kwargs={'pk': model_config.pk}))
        else:
            return JsonResponse({'message': 'Form is not valid.', 'errors': form.errors}, status=400)
    else:
        form = ModelConfigurationFormTesting()
        return render(request, 'main/model_configuration_form.html', {'form': form})


def list_configurations(request):
    configurations = ModelConfigurationTesting.objects.prefetch_related('neuronlayer_set').all()
    return render(request, 'main/list_configurations.html', {'configurations': configurations})

import tensorflow as tf
from .models import ModelParameters
def get_model_parameters(model_path, model_id):
    model = tf.keras.models.load_model(model_path)
    architecture = [layer.units for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    architecture = architecture[:-1]
    learning_rate = model.optimizer.learning_rate.numpy()
    loss = model.loss


    parameters = ModelParameters.objects.get(pk=model_id)
    epochs = parameters.epochs
    batch_size = parameters.batch_size
    validation_split = parameters.validation_split

    return {
        'architecture': architecture,
        'learning_rate': learning_rate,
        'loss': loss,
        'epochs': epochs,
        'batch_size': batch_size,
        'validation_split': validation_split
    }



def model_parameters_list(request):
    model_paths = {
        10: 'main/model_math.keras',
        11: 'main/model_reading.keras',
        12: 'main/model_writing.keras',
    }


    for model_id, model_path in model_paths.items():
        parameters = ModelParameters.objects.get(pk=model_id)
        model_params = get_model_parameters(model_path, model_id)
        parameters.architecture = model_params['architecture']
        parameters.learning_rate = model_params['learning_rate']
        parameters.loss = model_params['loss']
        parameters.epochs = model_params['epochs']
        parameters.batch_size = model_params['batch_size']
        parameters.validation_split = model_params['validation_split']
        parameters.save()

    parameters = ModelParameters.objects.all()
    return render(request, 'main/model_parameters_list.html', {'parameters': parameters})


def model_results(request, pk):
    config = get_object_or_404(ModelConfigurationTesting, pk=pk)
    math_metrics = {
        'mse': format(config.mse[0], ".2f"),
        'rmse': format(config.rmse[0], ".2f"),
        'mae': format(config.mae[0], ".2f")
    }
    reading_metrics = {
        'mse': format(config.mse[1], ".2f"),
        'rmse': format(config.rmse[1], ".2f"),
        'mae': format(config.mae[1], ".2f")
    }
    writing_metrics = {
        'mse': format(config.mse[2], ".2f"),
        'rmse': format(config.rmse[2], ".2f"),
        'mae': format(config.mae[2], ".2f")
    }
    neurons_per_layer = config.neuronlayer_set.values_list('neurons', flat=True)

    return render(request, 'main/model_results.html', {
        'config': config,
        'neurons_per_layer': neurons_per_layer,
        'math_metrics': math_metrics,
        'reading_metrics': reading_metrics,
        'writing_metrics': writing_metrics
    })

def select_model(request, pk):
    if request.method == 'POST':
        form = ModelSelectionForm(request.POST)
        if form.is_valid():
            model_type = form.cleaned_data['model_type']
            return redirect('update_model', pk=pk, model_type=model_type)
    else:
        form = ModelSelectionForm()

    return render(request, 'main/select_model.html', {'form': form})

def update_model(request, pk, model_type):
    config = get_object_or_404(ModelConfigurationTesting, pk=pk)


    if hasattr(config, 'name'):
        name = config.name
    else:
        name = model_type


    model_type_id_map = {
        'model_math': 10,
        'model_reading': 11,
        'model_writing': 12,
    }


    model_id = model_type_id_map.get(model_type)


    parameters = get_object_or_404(ModelParameters, id=model_id)


    parameters.architecture = list(config.neuronlayer_set.values_list('neurons', flat=True))
    parameters.learning_rate = config.learning_rate
    parameters.loss = 'mean_squared_error'
    parameters.epochs = config.epochs
    parameters.batch_size = config.batch_size
    parameters.validation_split = 0.2
    parameters.save()

    num_layers = config.num_layers
    neurons_per_layer = config.neuronlayer_set.values_list('neurons', flat=True)
    epochs = config.epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate

    file_path = 'main/StudentsPerformance.csv'
    data = pd.read_csv(file_path)

    data['gender'] = data['gender'].map({'male': 1, 'female': 0})
    data['lunch'] = data['lunch'].map({'standard': 1, 'free/reduced': 0})
    data['test preparation course'] = data['test preparation course'].map({'completed': 1, 'none': 0})
    data = pd.get_dummies(data, columns=['race/ethnicity', 'parental level of education'])

    feature_columns = data.columns.drop(['math score', 'reading score', 'writing score'])
    X_target = data[feature_columns].astype('int32')
    y_target = data[f'{model_type.split("_")[1]} score'].astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(X_target, y_target, test_size=0.2, random_state=42)

    def build_model(architecture, input_shape):
        model = Sequential()
        model.add(Input(shape=(input_shape,)))
        for units in architecture:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(1))
        return model

    architecture = neurons_per_layer
    model = build_model(architecture, X_train.shape[1])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    model.save(f'main/{model_type}.keras')


    print(f"Model training completed for {model_type}")
    print(f"Model saved to main/{model_type}.keras")

    return HttpResponseRedirect(reverse('model_results', kwargs={'pk': pk}))



