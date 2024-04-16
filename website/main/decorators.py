from django.http import HttpResponseForbidden
from functools import wraps
from django.shortcuts import redirect

def role_required(*roles):
    """Ensure user has any of the specified roles."""
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')  # Redirect to login if not authenticated
            if hasattr(request.user, 'profile') and request.user.profile.perfil in roles:
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden("You do not have permission to view this page.")
        return _wrapped_view
    return decorator

