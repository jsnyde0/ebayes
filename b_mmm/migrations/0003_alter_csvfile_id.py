# Generated by Django 5.0.7 on 2024-08-06 13:27

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('b_mmm', '0002_csvfile_column_names_delete_csvdata'),
    ]

    operations = [
        migrations.AlterField(
            model_name='csvfile',
            name='id',
            field=models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False),
        ),
    ]
