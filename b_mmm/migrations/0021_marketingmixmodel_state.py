# Generated by Django 5.0.7 on 2024-08-19 10:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('b_mmm', '0020_alter_marketingmixmodel_parameters_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='marketingmixmodel',
            name='state',
            field=models.CharField(default='initialized', max_length=20),
        ),
    ]
