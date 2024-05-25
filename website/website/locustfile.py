import ssl
from locust import HttpUser, TaskSet, task, between

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

class UserBehavior(TaskSet):
    def on_start(self):
        """ Run on start to initialize CSRF token """
        self.client.verify = False  # Disable SSL verification
        response = self.client.get("/", headers={"Referer": self.client.base_url})
        if response.status_code == 200:
            self.csrftoken = response.cookies['csrftoken']
        else:
            print(f"Failed to load index: {response.status_code} {response.text}")

    @task(1)
    def index(self):
        response = self.client.get("/", headers={"Referer": self.client.base_url}, verify=False)
        if response.status_code == 200:
            print("Index loaded successfully")
        else:
            print(f"Failed to load index: {response.status_code} {response.text}")

    @task(2)
    def login(self):
        response = self.client.get("/login/", headers={"Referer": self.client.base_url}, verify=False)
        if response.status_code == 200:
            self.csrftoken = response.cookies['csrftoken']
            login_response = self.client.post("/login/", {
                "username": "testuser",
                "password": "password",
                "csrfmiddlewaretoken": self.csrftoken
            }, headers={"X-CSRFToken": self.csrftoken, "Referer": self.client.base_url}, verify=False)
            if login_response.status_code == 200 or login_response.status_code == 302:
                print("Logged in successfully")
            else:
                print(f"Failed to login: {login_response.status_code} {login_response.text}")
        else:
            print(f"Failed to load login page: {response.status_code} {response.text}")

    @task(3)
    def prediction(self):
        response = self.client.get("/predict/", headers={"Referer": self.client.base_url}, verify=False)
        if response.status_code == 200:
            self.csrftoken = response.cookies['csrftoken']
            predict_response = self.client.post("/predict/", {
                "gender": "Male",
                "lunch": "Standard",
                "test_preparation_course": "None",
                "race_ethnicity": "Group A",
                "parental_level_of_education": "High School",
                "math_score": 90,
                "reading_score": 85,
                "writing_score": 88,
                "csrfmiddlewaretoken": self.csrftoken
            }, headers={"X-CSRFToken": self.csrftoken, "Referer": self.client.base_url}, verify=False)
            if predict_response.status_code == 200:
                print("Prediction made successfully")
            else:
                print(f"Failed to predict: {predict_response.status_code} {predict_response.text}")
        else:
            print(f"Failed to load predict page: {response.status_code} {response.text}")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(5, 15)
    host = "https://127.0.0.1:8000"

