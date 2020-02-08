from django.contrib import admin
from .models import Product, AppUser, envoFriendly

admin.site.register(Product)
admin.site.register(AppUser)
admin.site.register(envoFriendly)
