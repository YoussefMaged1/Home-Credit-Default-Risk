import os

from lightning_sdk.deployment import Deployment

deployment = Deployment(name="home-credit-risk", teamspace=os.getenv("TEAMSPACE"), user=os.getenv("LIGHTNING_USERNAME"))

deployment.start(
    image=f"{os.getenv('DOCKERHUB_USERNAME')}/{os.getenv('IMAGE_NAME')}:{os.getenv('CI_COMMIT_SHORT_SHA')}",  # Ollama Docker image
    ports=[8000],
)
