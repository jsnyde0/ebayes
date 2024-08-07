from django import template

register = template.Library()

@register.filter
def get_item(list_or_dict, key):
    try:
        return list_or_dict[key]
    except (KeyError, TypeError, IndexError):
        return ''