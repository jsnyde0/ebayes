# Generated by Django 5.0.7 on 2024-08-13 16:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('b_mmm', '0009_marketingmixmodel'),
    ]

    operations = [
        migrations.AlterField(
            model_name='marketingmixmodel',
            name='model_type',
            field=models.CharField(choices=[('linear_regression', 'Linear Regression'), ('bayesian_mmm', 'Bayesian MMM')], default='bayesian_mmm', max_length=50),
        ),
    ]
