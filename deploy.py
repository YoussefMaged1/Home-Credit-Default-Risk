import os

from lightning_sdk.api.deployment_api import Env
from lightning_sdk.deployment import Deployment

deployment = Deployment(
    name="home-credit-risk",
    teamspace=os.getenv("TEAMSPACE"),
    user=os.getenv("LIGHTNING_USERNAME"),
)

required_env = [
    "DOCKERHUB_USERNAME",
    "IMAGE_NAME",
    "CI_COMMIT_SHORT_SHA",
    "DAGSHUB_TOKEN",
    "DAGSHUB_USERNAME",
]
missing_env = [name for name in required_env if not os.getenv(name)]
if missing_env:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_env)}")

image = (
    f"{os.getenv('DOCKERHUB_USERNAME')}/"
    f"{os.getenv('IMAGE_NAME')}:{os.getenv('CI_COMMIT_SHORT_SHA')}"
)
runtime_env = [
    Env(name="DAGSHUB_TOKEN", value=os.environ["DAGSHUB_TOKEN"]),
    Env(name="DAGSHUB_USERNAME", value=os.environ["DAGSHUB_USERNAME"]),
]

try:
    deployment.update(image=image, ports=[8000], env=runtime_env)
    print(f"✅ Deployment updated with image: {image}")
except Exception:
    deployment.start(
        image=image,
        ports=[8000],
        env=runtime_env,
    )
    print(f"✅ Deployment started with image: {image}")
