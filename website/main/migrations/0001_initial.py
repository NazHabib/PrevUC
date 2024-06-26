# Generated by Django 4.2.11 on 2024-05-19 14:32

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelConfiguration',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('num_layers', models.IntegerField(default=3)),
                ('neurons_per_layer', models.CharField(default='[64, 64, 64]', max_length=255)),
                ('epochs', models.IntegerField(default=10)),
                ('learning_rate', models.FloatField(default=0.01)),
                ('batch_size', models.IntegerField(default=16)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='ModelConfigurationTesting',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('num_layers', models.IntegerField(default=1, verbose_name='Number of Layers')),
                ('epochs', models.IntegerField(default=10, verbose_name='Epochs')),
                ('learning_rate', models.FloatField(default=0.01, verbose_name='Learning Rate')),
                ('batch_size', models.IntegerField(default=32, verbose_name='Batch Size')),
                ('loss', models.JSONField(blank=True, null=True, verbose_name='Loss')),
                ('mae', models.JSONField(blank=True, null=True, verbose_name='Mean Absolute Error')),
                ('rmse', models.JSONField(blank=True, null=True, verbose_name='Root Mean Squared Error')),
                ('mse', models.JSONField(blank=True, null=True, verbose_name='Mean Squared Error')),
            ],
        ),
        migrations.CreateModel(
            name='NewsletterSubscriber',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('subscribed_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Notification',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('message', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='PredictionDataForm',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('gender', models.CharField(choices=[('Male', 'Male'), ('Female', 'Female')], max_length=20)),
                ('lunch', models.CharField(choices=[('Standard', 'Standard'), ('Free/Reduced', 'Free/Reduced')], max_length=20)),
                ('test_preparation_course', models.CharField(choices=[('None', 'None'), ('Completed', 'Completed')], max_length=20)),
                ('race_ethnicity', models.CharField(choices=[('Group A', 'Group A'), ('Group B', 'Group B'), ('Group C', 'Group C'), ('Group D', 'Group D'), ('Group E', 'Group E')], max_length=20)),
                ('parental_level_of_education', models.CharField(choices=[('High School', 'High School'), ('Some College', 'Some College'), ('Some High School', 'Some High School'), ("Bachelor's Degree", "Bachelor's Degree"), ("Master's Degree", "Master's Degree"), ("Associate's Degree", "Associate's Degree")], max_length=50)),
                ('math_score', models.IntegerField(blank=True, null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)])),
                ('reading_score', models.IntegerField(blank=True, null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)])),
                ('writing_score', models.IntegerField(blank=True, null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)])),
                ('validated', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('perfil', models.CharField(choices=[('professor', 'Professor'), ('data scientist', 'Data Scientist'), ('user', 'User')], max_length=15)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Prevision',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('gender', models.CharField(choices=[('Male', 'Male'), ('Female', 'Female')], max_length=20)),
                ('lunch', models.CharField(choices=[('Standard', 'Standard'), ('Free/Reduced', 'Free/Reduced')], max_length=20)),
                ('test_preparation_course', models.CharField(choices=[('None', 'None'), ('Completed', 'Completed')], max_length=20)),
                ('race_ethnicity', models.CharField(choices=[('Group A', 'Group A'), ('Group B', 'Group B'), ('Group C', 'Group C'), ('Group D', 'Group D'), ('Group E', 'Group E')], max_length=20)),
                ('parental_level_of_education', models.CharField(choices=[('High School', 'High School'), ('Some College', 'Some College'), ('Some High School', 'Some High School'), ("Bachelor's Degree", "Bachelor's Degree"), ("Master's Degree", "Master's Degree"), ("Associate's Degree", "Associate's Degree")], max_length=50)),
                ('math_score', models.IntegerField(blank=True, null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)])),
                ('reading_score', models.IntegerField(blank=True, null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)])),
                ('writing_score', models.IntegerField(blank=True, null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)])),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='NeuronLayer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('neurons', models.IntegerField()),
                ('model_config', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main.modelconfigurationtesting')),
            ],
        ),
        migrations.CreateModel(
            name='ModelMetrics',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('loss', models.FloatField()),
                ('mse', models.FloatField()),
                ('mae', models.FloatField()),
                ('rmse', models.FloatField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('configuration', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main.modelconfiguration')),
            ],
        ),
        migrations.CreateModel(
            name='Feedback',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='ChangeLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('description', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
