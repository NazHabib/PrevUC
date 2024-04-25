from django.http import HttpResponseForbidden
from functools import wraps
from django.shortcuts import redirect

def role_required(*roles):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')
            if hasattr(request.user, 'profile') and request.user.profile.perfil in roles:
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden("You do not have permission to view this page.")
        return _wrapped_view
    return decorator

def data_scientist_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.groups.filter(name='Data Scientist').exists():
            return HttpResponseForbidden("You do not have permission to access this page.")
        return view_func(request, *args, **kwargs)
    return _wrapped_view
