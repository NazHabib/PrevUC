from django.shortcuts import render
from django.urls import path, include
from . import views
from django.contrib import admin

# Corrected urlpatterns
urlpatterns = [
    path('', views.base,name='base'),
    path('home', views.home, name='home'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('sign-up', views.sign_up, name='sign_up'),
    path('view-changes/', views.view_changes, name='view_changes'),
    path('privacy-policy/', views.privacy_policy, name='privacy_policy'),
    path('terms-of-service/', views.terms_of_service, name='terms_of_service'),
    path('results/', views.prediction_results, name='prediction_results'),
    path('profile/', views.profile, name='profile'),
    path('delete-prevision/<int:prevision_id>/', views.delete_prevision, name='delete_prevision'),
    path('saved-previsions/', views.saved_previsions, name='saved_previsions'),
    path('delete_account/', views.account_delete, name='account_delete'),
    path('guest-main/', views.guest_prevision_form, name='guest_prevision_form'),
    path('subscribe/', views.subscribe_newsletter, name='subscribe_newsletter'),
    path('edit_account/', views.edit_account, name='edit_account'),
    path('main-prevision/', views.main_prevision, name='main_prevision'),
    path('data-input/', views.data_input, name='data_input'),
    path('change-documentation/', views.change_documentation, name='change_documentation'),
    path('notifications/', views.view_notifications, name='view_notifications'),
    path('create-notification/', views.create_notification, name='create_notification'),
    path('data_entries/', views.list_data_entries, name='list_data_entries'),
    path('validate_data/<int:entry_id>/', views.validate_data, name='validate_data'),
    path('delete_data/<int:entry_id>/', views.delete_data, name='delete_data'),
    path('feedback/', views.add_feedback, name='feedback'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('feedback-thank-you/', lambda request: render(request, 'feedback_thank_you.html'), name='feedback_thank_you'),
    path('model_configuration/', views.model_configuration_view, name='model_configuration'),
    path('model_configuration_list/', views.model_configuration_list_view, name='model_configuration_list'),
    path('configure/', views.model_configuration_view, name='configure_model'),
    path('metrics/', views.model_metrics_list, name='model_metrics_list'),
    path('admin/', admin.site.urls),
    path('configuratemodel/', views.train_and_evaluate_model, name='configure_model'),
    path('results/<int:pk>/', views.model_results, name='model_results'),
    path('configurations/', views.list_configurations, name='list_configurations'),
    path('activate/<uidb64>/<token>/', views.activate_account, name='activate_account'),
    path('account-activation-sent/', views.account_activation_sent, name='account_activation_sent'),

]
