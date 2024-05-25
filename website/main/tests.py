# website/main/tests.py

from django.test import TestCase
from django.contrib.auth.models import User
from .models import (
    NewsletterSubscriber, Profile, PredictionDataForm, Prevision, ChangeLog, Notification,
    Feedback, ModelConfiguration, ModelMetrics, NeuronLayer, ModelConfigurationTesting, ModelParameters
)
from .forms import (
    RegisterForm, ProfileForm, UserForm, PrevisionForm, PrevisionInputForm, FeedbackForm,
    ModelConfigurationForm, ModelSelectionForm, ModelConfigurationFormTesting, NotificationForm, ChangeForm
)

class NewsletterSubscriberTestCase(TestCase):
    def setUp(self):
        NewsletterSubscriber.objects.create(email="test@example.com")

    def test_newsletter_subscriber_creation(self):
        subscriber = NewsletterSubscriber.objects.get(email="test@example.com")
        self.assertEqual(subscriber.email, "test@example.com")

class ProfileTestCase(TestCase):
    def setUp(self):
        user = User.objects.create(username="testuser", email="test@example.com")
        Profile.objects.create(user=user, perfil='data scientist')

    def test_profile_creation(self):
        profile = Profile.objects.get(user__username="testuser")
        self.assertEqual(profile.perfil, 'data scientist')

class PredictionDataFormTestCase(TestCase):
    def setUp(self):
        PredictionDataForm.objects.create(
            gender='Male', lunch='Standard', test_preparation_course='None',
            race_ethnicity='Group A', parental_level_of_education='High School',
            math_score=80, reading_score=85, writing_score=90
        )

    def test_prediction_data_form_creation(self):
        prediction = PredictionDataForm.objects.get(gender='Male')
        self.assertEqual(prediction.lunch, 'Standard')
        self.assertEqual(prediction.math_score, 80)

class PrevisionTestCase(TestCase):
    def setUp(self):
        user = User.objects.create(username="testuser", email="test@example.com")
        Prevision.objects.create(
            user=user, gender='Male', lunch='Standard', test_preparation_course='None',
            race_ethnicity='Group A', parental_level_of_education='High School',
            math_score=80, reading_score=85, writing_score=90
        )

    def test_prevision_creation(self):
        prevision = Prevision.objects.get(gender='Male')
        self.assertEqual(prevision.lunch, 'Standard')
        self.assertEqual(prevision.math_score, 80)

class ChangeLogTestCase(TestCase):
    def setUp(self):
        user = User.objects.create(username="testuser", email="test@example.com")
        ChangeLog.objects.create(user=user, name="Initial Change", description="First change in the log.")

    def test_changelog_creation(self):
        changelog = ChangeLog.objects.get(name="Initial Change")
        self.assertEqual(changelog.description, "First change in the log.")

class NotificationTestCase(TestCase):
    def setUp(self):
        Notification.objects.create(message="This is a test notification.")

    def test_notification_creation(self):
        notification = Notification.objects.get(message="This is a test notification.")
        self.assertEqual(notification.message, "This is a test notification.")

class FeedbackTestCase(TestCase):
    def setUp(self):
        user = User.objects.create(username="testuser", email="test@example.com")
        Feedback.objects.create(user=user, content="This is a feedback message.")

    def test_feedback_creation(self):
        feedback = Feedback.objects.get(content="This is a feedback message.")
        self.assertEqual(feedback.content, "This is a feedback message.")

class ModelConfigurationTestCase(TestCase):
    def setUp(self):
        ModelConfiguration.objects.create(
            num_layers=3, neurons_per_layer='[64, 64, 64]', epochs=10, learning_rate=0.01, batch_size=16
        )

    def test_model_configuration_creation(self):
        config = ModelConfiguration.objects.get(num_layers=3)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.batch_size, 16)

class ModelMetricsTestCase(TestCase):
    def setUp(self):
        config = ModelConfiguration.objects.create(
            num_layers=3, neurons_per_layer='[64, 64, 64]', epochs=10, learning_rate=0.01, batch_size=16
        )
        ModelMetrics.objects.create(
            configuration=config, loss=0.05, mse=0.02, mae=0.01, rmse=0.03
        )

    def test_model_metrics_creation(self):
        metrics = ModelMetrics.objects.get(mse=0.02)
        self.assertEqual(metrics.loss, 0.05)
        self.assertEqual(metrics.rmse, 0.03)

class NeuronLayerTestCase(TestCase):
    def setUp(self):
        config = ModelConfigurationTesting.objects.create(
            num_layers=3, epochs=10, learning_rate=0.01, batch_size=32
        )
        NeuronLayer.objects.create(model_config=config, neurons=128)

    def test_neuron_layer_creation(self):
        layer = NeuronLayer.objects.get(neurons=128)
        self.assertEqual(layer.neurons, 128)

class ModelConfigurationTestingTestCase(TestCase):
    def setUp(self):
        ModelConfigurationTesting.objects.create(
            num_layers=3, epochs=10, learning_rate=0.01, batch_size=32,
            loss=[], mae=[], rmse=[], mse=[]
        )

    def test_model_configuration_testing_creation(self):
        config = ModelConfigurationTesting.objects.get(num_layers=3)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.batch_size, 32)

class ModelParametersTestCase(TestCase):
    def setUp(self):
        ModelParameters.objects.create(
            name='Test Model', architecture=[6, 61, 32], learning_rate=0.01,
            loss='mean_squared_error', epochs=79, batch_size=11, validation_split=0.2
        )

    def test_model_parameters_creation(self):
        model = ModelParameters.objects.get(name='Test Model')
        self.assertEqual(model.learning_rate, 0.01)
        self.assertEqual(model.loss, 'mean_squared_error')
        self.assertEqual(model.epochs, 79)
        self.assertEqual(model.batch_size, 11)


# main/tests.py

from django import forms
from django.contrib.auth.models import User
from django.test import TestCase
from .forms import (
    RegisterForm, ProfileForm, UserForm, PrevisionForm, PrevisionInputForm, FeedbackForm,
    ModelConfigurationForm, ModelSelectionForm, ModelConfigurationFormTesting, NotificationForm, ChangeForm
)

class RegisterFormTestCase(TestCase):
    def test_valid_register_form(self):
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password1': 'complexpassword123',
            'password2': 'complexpassword123',
            'perfil': 'data scientist'
        }
        form = RegisterForm(data=data)
        self.assertTrue(form.is_valid())

    def test_invalid_register_form(self):
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password1': 'complexpassword123',
            'password2': 'differentpassword123',
            'perfil': 'data scientist'
        }
        form = RegisterForm(data=data)
        self.assertFalse(form.is_valid())

class ProfileFormTestCase(TestCase):
    def test_valid_profile_form(self):
        user = User.objects.create(username="testuser", email="test@example.com")
        data = {'perfil': 'data scientist'}
        form = ProfileForm(data=data)
        self.assertTrue(form.is_valid())
"""
class UserFormTestCase(TestCase):
    def test_valid_user_form(self):
        user = User.objects.create(username="testuser", email="test@example.com")
        data = {'username': 'testuser', 'email': 'test@example.com'}
        form = UserForm(data=data)
        self.assertTrue(form.is_valid())
"""
class PrevisionFormTestCase(TestCase):
    def test_valid_prevision_form(self):
        data = {
            'gender': 'True',
            'lunch': 'True',
            'test_preparation_course': 'False',
            'race_ethnicity': 'group A',
            'parental_level_of_education': 'high school'
        }
        form = PrevisionForm(data=data)
        self.assertTrue(form.is_valid())

    def test_invalid_prevision_form(self):
        data = {
            'gender': '',
            'lunch': 'True',
            'test_preparation_course': 'False',
            'race_ethnicity': 'group A',
            'parental_level_of_education': 'high school'
        }
        form = PrevisionForm(data=data)
        self.assertFalse(form.is_valid())

class PrevisionInputFormTestCase(TestCase):
    def test_valid_prevision_input_form(self):
        data = {
            'gender': 'Male',
            'lunch': 'Standard',
            'test_preparation_course': 'None',
            'race_ethnicity': 'Group A',
            'parental_level_of_education': 'High School',
            'math_score': 80,
            'reading_score': 85,
            'writing_score': 90
        }
        form = PrevisionInputForm(data=data)
        self.assertTrue(form.is_valid())

class FeedbackFormTestCase(TestCase):
    def test_valid_feedback_form(self):
        user = User.objects.create(username="testuser", email="test@example.com")
        data = {'content': 'This is a feedback message.'}
        form = FeedbackForm(data=data)
        self.assertTrue(form.is_valid())

class ModelConfigurationFormTestCase(TestCase):
    def test_valid_model_configuration_form(self):
        data = {
            'num_layers': 3,
            'neurons_per_layer': '[64, 64, 64]',
            'epochs': 10,
            'learning_rate': 0.01,
            'batch_size': 16
        }
        form = ModelConfigurationForm(data=data)
        self.assertTrue(form.is_valid())

class ModelSelectionFormTestCase(TestCase):
    def test_valid_model_selection_form(self):
        data = {'model_type': 'model_math'}
        form = ModelSelectionForm(data=data)
        self.assertTrue(form.is_valid())

class ModelConfigurationFormTestingTestCase(TestCase):
    def test_valid_model_configuration_testing_form(self):
        data = {
            'num_layers': 3,
            'neurons_per_layer': '64, 64, 64',
            'epochs': 10,
            'learning_rate': 0.01,
            'batch_size': 32
        }
        form = ModelConfigurationFormTesting(data=data)
        self.assertTrue(form.is_valid())

class NotificationFormTestCase(TestCase):
    def test_valid_notification_form(self):
        data = {'message': 'This is a test notification.'}
        form = NotificationForm(data=data)
        self.assertTrue(form.is_valid())

class ChangeFormTestCase(TestCase):
    def test_valid_change_form(self):
        data = {'name': 'Initial Change', 'description': 'First change in the log.'}
        form = ChangeForm(data=data)
        self.assertTrue(form.is_valid())
