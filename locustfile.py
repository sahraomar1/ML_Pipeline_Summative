"""Flood test: many fake users send photos to the API.

Run: locust -f locustfile.py --host http://localhost:8080
Then open http://localhost:8089
"""

import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_IMAGES = [p for p in Path("data/test").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


class ClaimUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(5)
    def predict(self):
        image_path = random.choice(TEST_IMAGES)
        with open(image_path, "rb") as image:
            self.client.post("/predict", files={"file": (image_path.name, image, "image/jpeg")})

    @task(1)
    def health(self):
        self.client.get("/health")