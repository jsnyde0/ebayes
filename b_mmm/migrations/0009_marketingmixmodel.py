# Generated by Django 5.0.7 on 2024-08-12 12:54

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('b_mmm', '0008_remove_csvfile_currency_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='MarketingMixModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('model_type', models.CharField(max_length=50)),
                ('parameters', models.JSONField(default=dict)),
                ('results', models.JSONField(default=dict)),
                ('csv_file', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='mmm', to='b_mmm.csvfile')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='mmm', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
