from django.contrib import admin
from .models import (
    NewsletterSubscriber, Profile, PredictionDataForm, Prevision,
    ChangeLog, Notification, Feedback, ModelConfiguration, ModelMetrics
)
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin, UserAdmin
from django.contrib.auth.models import User
from .models import Profile
from .forms import RegisterForm, CustomUserChangeForm, \
    CustomUserCreationForm  # Ensure your custom form is imported if needed in admin

class ProfileInline(admin.StackedInline):
    model = Profile
    can_delete = False
    verbose_name_plural = 'profile'

class CustomUserAdmin(BaseUserAdmin):
    add_form = CustomUserCreationForm
    form = CustomUserChangeForm
    inlines = (ProfileInline,)

    list_display = ('username', 'email', 'is_staff', 'is_active', 'is_superuser')
    list_select_related = ('profile',)

    def get_form(self, request, obj=None, **kwargs):
        if obj:
            kwargs['form'] = self.form
        else:
            kwargs['form'] = self.add_form
        return super().get_form(request, obj, **kwargs)

admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)

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


