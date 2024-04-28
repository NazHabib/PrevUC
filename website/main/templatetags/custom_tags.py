from django import template
from django.urls import resolve

register = template.Library()

@register.simple_tag(takes_context=True)
def is_base_template(context):
    request = context.get('request')
    if request:
        resolved_path = resolve(request.path_info)
        return '' in resolved_path.template_name
    return False
