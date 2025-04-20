import os
import time
from core.database import get_db, upload_image_to_minio
from model.muscle import Muscle

images = [
    {"name": "biceps", "path": "images/biceps.jpg"},
    {"name": "triceps", "path": "images/triceps.jpg"},
    {"name": "anterior tibialis", "path": "images/anterior tibialis.jpg"},
    {"name": "calves", "path": "images/calves.jpg"},
    {"name": "chest", "path": "images/chest.jpg"},
    {"name": "forearm", "path": "images/forearm.jpg"},
    {"name": "front delts", "path": "images/front delts.jpg"},
    {"name": "glutes", "path": "images/glutes.jpg"},
    {"name": "hamstrings", "path": "images/hamstrings.jpg"},
    {"name": "lower back", "path": "images/lower back.jpg"},
    {"name": "obliques", "path": "images/obliques.jpg"},
    {"name": "quads abductors", "path": "images/quads abductors.jpg"},
    {"name": "rear delts", "path": "images/rear delts.jpg"},
    {"name": "rhomboids", "path": "images/rhomboids.jpg"},
    {"name": "abs", "path": "images/abs.jpg"},
]


def bulk_upload():
    db = next(get_db())

    for item in images:
        with open(item["path"], "rb") as f:
            file_bytes = f.read()

        filename = f"{item['name']}_{int(time.time())}.jpg"
        image_url = upload_image_to_minio(file_bytes, filename)

        muscle = db.query(Muscle).filter(Muscle.name == item["name"]).first()
        if not muscle:
            muscle = Muscle(name=item["name"])
            db.add(muscle)

        muscle.image_url = image_url
        db.commit()
        db.refresh(muscle)
        print(f"Uploaded {item['name']} -> {image_url}")


if __name__ == "__main__":
    bulk_upload()
