# Generated by Django 5.0.7 on 2024-08-14 13:38

import b_mmm.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('b_mmm', '0012_alter_marketingmixmodel_trace_plot'),
    ]

    operations = [
        migrations.AddField(
            model_name='marketingmixmodel',
            name='parameter_posteriors_plot',
            field=models.ImageField(blank=True, null=True, upload_to=b_mmm.models.get_plot_path),
        ),
    ]
