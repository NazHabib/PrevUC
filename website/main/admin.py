from django.contrib import admin
from .models import (
    NewsletterSubscriber, Profile, PredictionDataForm, Prevision,
    ChangeLog, Notification, Feedback, ModelConfiguration, ModelMetrics
)
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin, UserAdmin
from django.contrib.auth.models import User
from .models import Profile
from .forms import RegisterForm

class ProfileInline(admin.StackedInline):
    model = Profile
    can_delete = False
    verbose_name_plural = 'profile'



admin.site.unregister(User)


def get_inline_instances(self, request, obj=None):
    if not obj:
        return list()
    return super().get_inline_instances(request, obj)


admin.site.register(NewsletterSubscriber)
admin.site.register(Profile)
admin.site.register(PredictionDataForm)
admin.site.register(Prevision)
admin.site.register(ChangeLog)
admin.site.register(Notification)
admin.site.register(Feedback)
admin.site.register(ModelConfiguration)
admin.site.register(ModelMetrics)


