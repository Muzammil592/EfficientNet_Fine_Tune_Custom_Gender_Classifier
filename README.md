Gender Classification using EfficientNet-B0OverviewThis project implements a complete, end-to-end production pipeline for a binary gender classifier (Male / Female) using deep learning. The core system pairs a fine-tuned EfficientNet-B0 backbone with a lightweight OpenCV face-detection preprocessing step to ensure clean inference inputs. The entire stack is served via a FastAPI backend featuring a responsive drag-and-drop web dashboard, made public instantly using an integrated Ngrok tunnel directly from a Kaggle notebook environment.Architecture & Deployment Workflow[ User Image Upload ] 
         │
         ▼
[ FastAPI Endpoint (`/predict`) ]
         │
         ▼
[ Preprocessing: OpenCV Haar Cascade ] ──► (Falls back to full image if no face found)
         │
         ▼
[ Inference: EfficientNet-B0 ]
         │
         ▼
[ Response: JSON Data + UI Update ]
Technical Specifications1. Core Model ArchitectureBackbone: EfficientNet-B0 (Pretrained on ImageNet)Custom Classification Head:$$\text{Global Average Pool} \longrightarrow \text{Dropout}(p=0.3) \longrightarrow \text{Linear}(1280 \rightarrow 2)$$Trainable Parameters: ~5.3M2. Upstream Dataset & Training RecapDataset Source: UTKFace (crop_part1 subset containing 9,779 face images). Labels parsed automatically from raw filenames (age_gender_race_date.jpg).Data Split: Train (70% - 6,845) | Validation (15% - 1,466) | Test (15% - 1,468).Training Configuration: Optimized via AdamW ($1e^{-4}$ learning rate) and a CosineAnnealingLR scheduler over 20 epochs on a Kaggle T4 GPU.Baseline Results: * Best Validation Accuracy: 88.61%Final Test Accuracy: 90.80%Test Loss: 0.5295Production Implementations & Features⚡ Smart Preprocessing Pipeline (Face Detection)Rather than passing raw, uncropped images straight to the neural network, the backend utilizes an optimized OpenCV Haar Cascade Classifier (haarcascade_frontalface_default.xml).Region of Interest (ROI) Selection: Identifies and isolates the largest face in the bounding box layout.Padding Margin: Adds a $+20\text{px}$ bounding margin dynamically clamped within image borders to keep hair and facial structure intact.Fail-Safe Architecture: Wrapped in exception handlers—if an upload contains no identifiable face or uses complex angles, the system safely falls back to standard processing on the full image without crashing the inference worker.🌐 FastAPI Serving StackThe application logic is written entirely in pure Python, utilizing high-performance frameworks over complex, high-level orchestration wrappers./predict (POST): Validates the incoming mime-type, processes bytes with PIL, applies the face-detection layer, passes the $224 \times 224$ normalized tensor to PyTorch, and returns full confidence metrics./health (GET): Monitors host environment status, returning current hardware execution context (cuda vs cpu), active model path, and system package versions.Asynchronous Engine: Employs nest_asyncio alongside uvicorn to allow the ASGI server to run concurrently alongside active Jupyter/Kaggle notebook loops.🎨 Responsive Frontend DashboardThe user interface is a single-page app (SPA) embedded right into the FastAPI core template:Modern Aesthetics: Designed with a clean, dark-mode visual hierarchy suited for direct client presentations.Interactive Handling: Features native JavaScript drag-and-drop mechanics or direct click-to-upload workflows.Dynamic Metrics: Updates instantly upon receiving a response payload, showing class probabilities via clean animated progress bars and displaying status tags marking whether OpenCV caught a face or used the full asset.Environment Dependencies (requirements.txt)To deploy this script without version runtime errors (such as internal PyTorch mapping crashes), use the following stabilized requirements tree to ensure proper environment configurations:Plaintextnumpy<2.0.0
torch
torchvision
fastapi
uvicorn[standard]
python-multipart
facenet-pytorch
Pillow
matplotlib
