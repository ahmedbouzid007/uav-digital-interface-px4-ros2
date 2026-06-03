import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the inspection image
img_path = os.path.expanduser("~/PX4-Autopilot/inspection_shot.jpg")
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"Image shape: {img.shape}")
print(f"Resolution: {img.shape[1]}x{img.shape[0]} pixels")

# Create output directory for analysis
out_dir = os.path.expanduser("~/PX4-Autopilot/analysis_output")
os.makedirs(out_dir, exist_ok=True)

# 1. Basic color channel analysis
b, g, r = cv2.split(img)
print(f"\nMean channel values:")
print(f"  Red:   {r.mean():.1f}")
print(f"  Green: {g.mean():.1f}")
print(f"  Blue:  {b.mean():.1f}")

# 2. Simple vegetation index (like NDVI approximation from RGB)
# For real NDVI you need NIR band, but we can estimate greenness
greenness = g.astype(float) / (r.astype(float) + b.astype(float) + 1)
greenness_norm = (greenness / greenness.max() * 255).astype(np.uint8)

# 3. Detect vegetation areas (green pixels)
green_mask = (g > r) & (g > b) & (g > 100)
vegetation_ratio = green_mask.sum() / green_mask.size * 100
print(f"\nVegetation coverage: {vegetation_ratio:.1f}%")

# 4. Detect shadows (dark areas)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
shadow_mask = gray < 80
shadow_ratio = shadow_mask.sum() / shadow_mask.size * 100
print(f"Shadow coverage: {shadow_ratio:.1f}%")

# 5. Edge detection (for structure detection)
edges = cv2.Canny(gray, 100, 200)

# 6. Save analysis plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0,0].imshow(img_rgb)
axes[0,0].set_title('Original Image')
axes[0,0].axis('off')

axes[0,1].imshow(green_mask, cmap='Greens')
axes[0,1].set_title(f'Vegetation Mask ({vegetation_ratio:.1f}%)')
axes[0,1].axis('off')

axes[0,2].imshow(shadow_mask, cmap='gray')
axes[0,2].set_title(f'Shadow Mask ({shadow_ratio:.1f}%)')
axes[0,2].axis('off')

axes[1,0].imshow(greenness_norm, cmap='YlGn')
axes[1,0].set_title('Greenness Index')
axes[1,0].axis('off')

axes[1,1].imshow(edges, cmap='gray')
axes[1,1].set_title('Edge Detection')
axes[1,1].axis('off')

# Histogram
axes[1,2].hist(r.flatten(), bins=50, color='red', alpha=0.5, label='Red')
axes[1,2].hist(g.flatten(), bins=50, color='green', alpha=0.5, label='Green')
axes[1,2].hist(b.flatten(), bins=50, color='blue', alpha=0.5, label='Blue')
axes[1,2].set_title('Color Histogram')
axes[1,2].legend()

plt.tight_layout()
analysis_path = os.path.join(out_dir, 'analysis_overview.png')
plt.savefig(analysis_path, dpi=150)
print(f"\nAnalysis saved to: {analysis_path}")

# 7. Save individual masks for further use
cv2.imwrite(os.path.join(out_dir, 'vegetation_mask.png'), green_mask.astype(np.uint8) * 255)
cv2.imwrite(os.path.join(out_dir, 'shadow_mask.png'), shadow_mask.astype(np.uint8) * 255)
cv2.imwrite(os.path.join(out_dir, 'edges.png'), edges)

print(f"\nAll outputs saved to: {out_dir}")
print(f"  - analysis_overview.png (combined view)")
print(f"  - vegetation_mask.png")
print(f"  - shadow_mask.png")
print(f"  - edges.png")
