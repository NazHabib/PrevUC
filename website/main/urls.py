from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.base,name='base'),
    path('home', views.home, name='home'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('sign-up', views.sign_up, name='sign_up'),
    path('privacy-policy/', views.privacy_policy, name='privacy_policy'),
    path('terms-of-service/', views.terms_of_service, name='terms_of_service'),
    path('profile/', views.profile, name='profile'),
    path('saved-previsions/', views.saved_previsions, name='saved_previsions'),
    path('delete_account/', views.account_delete, name='account_delete'),
    path('guest-main/', views.guest_prevision, name='guest_prevision'),
    path('results/', views.prediction_results, name='prediction_results'),
    path('submit-prevision-form/', views.guest_prevision, name='guest_prevision'),
    path('edit_account/', views.edit_account, name='edit_account'),
    path('main-prevision/', views.main_prevision, name='main_prevision'),
    path('saved-previsions/', views.saved_previsions, name='saved_previsions'),
    path('data-input/', views.data_input, name='data_input'),
    path('change-documentation/', views.change_documentation, name='change_documentation'),
    path('notifications/', views.view_notifications, name='view_notifications'),
    path('create-notification/', views.create_notification, name='create_notification'),
]