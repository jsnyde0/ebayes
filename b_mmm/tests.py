from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

class ViewTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='12345')

    def test_view_home(self):
        response = self.client.get(reverse('mmm:home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'mmm/home.html')

    # def test_view_upload_unauthenticated(self):
    #     response = self.client.get(reverse('mmm:upload'))
    #     self.assertEqual(response.status_code, 302)  # Should redirect to login

    # def test_view_upload_authenticated(self):
    #     self.client.login(username='testuser', password='12345')
    #     response = self.client.get(reverse('mmm:upload'))
    #     self.assertEqual(response.status_code, 200)
    #     self.assertTemplateUsed(response, 'mmm/upload.html')

    # Add more tests for file upload, preview, and model views