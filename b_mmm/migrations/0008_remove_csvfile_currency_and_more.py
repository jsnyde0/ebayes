# Generated by Django 5.0.7 on 2024-08-12 10:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('b_mmm', '0007_csvfile_predictor_currencies'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='csvfile',
            name='currency',
        ),
        migrations.RemoveField(
            model_name='csvfile',
            name='predictor_currencies',
        ),
        migrations.AddField(
            model_name='csvfile',
            name='currencies',
            field=models.JSONField(default=dict),
        ),
    ]
