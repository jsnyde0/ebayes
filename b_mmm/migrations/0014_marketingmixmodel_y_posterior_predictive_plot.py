# Generated by Django 5.0.7 on 2024-08-14 14:12

import b_mmm.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('b_mmm', '0013_marketingmixmodel_parameter_posteriors_plot'),
    ]

    operations = [
        migrations.AddField(
            model_name='marketingmixmodel',
            name='y_posterior_predictive_plot',
            field=models.ImageField(blank=True, null=True, upload_to=b_mmm.models.get_plot_path),
        ),
    ]
