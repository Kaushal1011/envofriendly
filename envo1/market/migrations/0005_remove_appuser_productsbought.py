# Generated by Django 3.0.3 on 2020-02-09 00:57

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('market', '0004_auto_20200208_2317'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='appuser',
            name='productsBought',
        ),
    ]
