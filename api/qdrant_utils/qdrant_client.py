import os
from typing import Optional, Dict, Any
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# -------------------------------
# Qdrant configuration
# -------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "product_embeddings")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

EMBEDDING_DIM = 512  # ResNet18 output dimension

# -------------------------------
# Model setup (Option 1: offline)
# -------------------------------
model = None
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def initialize_model():
    """Load ResNet18 for feature extraction (offline, no GitHub)."""
    global model
    if model is None:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()
        model.fc = torch.nn.Identity()  # remove final classification layer
    return model

# -------------------------------
# Create Qdrant client
# -------------------------------
def create_qdrant_client() -> Optional[QdrantClient]:
    try:
        client = QdrantClient(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            api_key=QDRANT_API_KEY,                  # string only
            headers={"api-key": QDRANT_API_KEY}      # correct header for Qdrant
        )
        client.get_collections()
        print("Connected to Qdrant successfully")
        return client
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print("Please ensure the Qdrant service is running and API key is correct")
        print(f"Error details: {str(e)}")
        return None

# -------------------------------
# Extract embedding
# -------------------------------
def get_image_embedding(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    model = initialize_model()
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy()
    return embedding

# -------------------------------
# Save embedding to Qdrant
# -------------------------------
def save_embedding_to_qdrant(
    client: QdrantClient,
    product_id: int,
    image_path: str,
    additional_data: Optional[Dict[str, Any]] = None
) -> bool:

    if client is None:
        print("Qdrant client not initialized")
        return False

    try:
        # Ensure collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if QDRANT_COLLECTION not in collection_names:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )

        embedding = get_image_embedding(image_path)
        payload = {
            "product_id": product_id,
            "image_path": image_path,
            **(additional_data or {})
        }

        point = PointStruct(
            id=product_id,
            vector=embedding.tolist(),
            payload=payload
        )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[point]
        )

        return True

    except Exception as e:
        print(f"Error saving to Qdrant: {str(e)}")
        return False