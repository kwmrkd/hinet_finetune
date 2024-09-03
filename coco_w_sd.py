import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import AutoTokenizer
from pycocotools.coco import COCO
from torch.optim import AdamW
from tqdm import tqdm
import time

# Paths to the COCO annotations and images
annotation_file = 'coco/annotations/instances_val2017.json'  # Update this to your annotations file
image_dir = 'test/image/steg'

# Initialize the COCO API
coco = COCO(annotation_file)

# Load category information
categories = coco.loadCats(coco.getCatIds())
category_id_to_name = {cat['id']: cat['name'] for cat in categories}

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 512x512 or another size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] range for stable diffusion
])

# Dataset class
class COCODataset(Dataset):
    def __init__(self, coco, image_dir, transform=None):
        self.coco = coco
        self.image_dir = image_dir
        self.image_ids = coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_filename = image_info['file_name']
        image_path = os.path.join(self.image_dir, image_filename)
        
        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Get category names for the current image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        category_names = set()
        for annotation in annotations:
            category_id = annotation['category_id']
            category_name = category_id_to_name[category_id]
            category_names.add(category_name)
        
        # Use the first category name (or modify as needed)
        category_name = sorted(list(category_names))[0] if category_names else "object"
        
        prompt = f"a photo of a {category_name}"
        
        return {"image": image, "prompt": prompt}

# Initialize the dataset and dataloader
dataset = COCODataset(coco, image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model_id = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline = pipeline.to("cuda")  # Assuming you have a GPU



# Tokenizer for text prompts
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Fine-tuning setup
optimizer = AdamW(pipeline.unet.parameters(), lr=1e-5)

num_epochs = 5

# Training loop
pipeline.unet.train()
for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        images = batch['image'].to(pipeline.device)
        prompts = batch['prompt']

        # Tokenize the prompts
        text_inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        text_input_ids = text_inputs.input_ids.to(pipeline.device)

        # Encode text prompts to get the text embeddings
        encoder_hidden_states = pipeline.text_encoder(text_input_ids)[0]

        # Encode images to latents
        latents = pipeline.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # scaling factor

        # Sample a random timestep for each image in the batch
        timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device).long()

        # Generate random noise
        noise = torch.randn_like(latents).to(pipeline.device)

        # Add noise to the latents based on the current timestep
        noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise using U-Net
        noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Compute the loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(f"Epoch {epoch + 1}/{num_epochs} completed, Loss: {loss.item()}")

# Save the fine-tuned model
pipeline.save_pretrained("./fine-tuned-stable-diffusion-watermarked")