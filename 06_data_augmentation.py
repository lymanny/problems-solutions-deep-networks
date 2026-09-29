from PIL import Image
from torchvision import transforms

# Load image from the images folder
image = Image.open("images/cat.jpg")

# Create Data Augmentation
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # Flip image
    transforms.RandomRotation(20),          # Rotate image
    transforms.RandomResizedCrop(224),      # Crop and resize
    transforms.ColorJitter(brightness=0.2)  # Change brightness
])

# Apply augmentation
augmented_image = augmentation(image)

# Show original image
image.show()

# Show augmented image
augmented_image.show()