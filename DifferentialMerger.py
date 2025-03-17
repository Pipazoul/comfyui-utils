import torch
import numpy as np

class DifferentialImageMerger:
    """
    Node that compares a batch of images against a source image,
    identifies pixel differences, and creates a unique final image
    by merging only the different pixels from each image in the batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_images": ("IMAGE",),
                "source_image": ("IMAGE",),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001}),
                "blend_mode": (["replace", "additive", "maximum"], {"default": "replace"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)
    FUNCTION = "mergeDifferentialPixels"
    CATEGORY = "pipazoul/image"

    def mergeDifferentialPixels(self, batch_images, source_image, threshold=0.05, blend_mode="replace"):
        # Ensure source image is a single image
        if len(source_image.shape) == 4 and source_image.shape[0] == 1:  # [1, C, H, W]
            source_image = source_image[0]  # -> [C, H, W]
        elif len(source_image.shape) == 4 and source_image.shape[0] > 1:
            # If source is a batch, use only the first image
            print("Warning: Source is a batch with multiple images. Using only the first image.")
            source_image = source_image[0]  # -> [C, H, W]
        
        # Get shapes and check compatibility
        if len(batch_images.shape) != 4:  # [B, C, H, W]
            raise ValueError("Batch images must be a 4D tensor [batch, channels, height, width]")
        
        # Ensure dimensions match
        batch_size, channels, height, width = batch_images.shape
        if source_image.shape != (channels, height, width):
            raise ValueError(f"Source image shape {source_image.shape} doesn't match batch image dimensions ({channels}, {height}, {width})")
        
        # Create result tensor starting with source image
        result = source_image.clone().unsqueeze(0)  # -> [1, C, H, W]
        
        # Process each image in the batch
        for i in range(batch_size):
            # Get current image
            current_image = batch_images[i]  # [C, H, W]
            
            # Calculate absolute pixel differences
            diff = torch.abs(current_image - source_image)  # [C, H, W]
            
            # Create mask of pixels that differ more than threshold
            # Sum across channels and check if any channel differs above threshold
            diff_mask = (diff > threshold).any(dim=0).float()  # [H, W]
            
            # Apply the mask according to blend mode
            if blend_mode == "replace":
                # Replace pixels where diff_mask is 1
                for c in range(channels):
                    pixel_mask = diff_mask.unsqueeze(0)  # [1, H, W]
                    result[0, c] = result[0, c] * (1 - pixel_mask) + current_image[c] * pixel_mask
            
            elif blend_mode == "additive":
                # Add values where pixels differ
                for c in range(channels):
                    pixel_mask = diff_mask.unsqueeze(0)  # [1, H, W]
                    # Only add the difference (not full value) and clamp to valid range
                    result[0, c] = torch.clamp(
                        result[0, c] + (current_image[c] - source_image[c]) * pixel_mask,
                        0.0, 1.0
                    )
            
            elif blend_mode == "maximum":
                # Take maximum value between source and current image where pixels differ
                for c in range(channels):
                    pixel_mask = diff_mask.unsqueeze(0)  # [1, H, W]
                    max_values = torch.maximum(result[0, c], current_image[c])
                    result[0, c] = result[0, c] * (1 - pixel_mask) + max_values * pixel_mask
        
        # Return the merged image
        return (result,)


class MultiDifferentialMerger:
    """
    More advanced node that compares a batch of images against a source image,
    creates a differential mask for each image, and then applies them sequentially
    with various advanced options.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_images": ("IMAGE",),
                "source_image": ("IMAGE",),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001}),
                "blend_mode": (["normal", "add", "multiply", "screen", "overlay", "difference"], {"default": "normal"}),
                "mask_processing": (["per_image", "cumulative", "weighted"], {"default": "per_image"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("merged_image", "difference_mask")
    FUNCTION = "advancedMergeDifferentialPixels"
    CATEGORY = "pipazoul/image"

    def advancedMergeDifferentialPixels(self, batch_images, source_image, threshold=0.05, 
                                        blend_mode="normal", mask_processing="per_image", 
                                        blend_strength=1.0):
        # Ensure source image is a single image
        if len(source_image.shape) == 4 and source_image.shape[0] == 1:  # [1, C, H, W]
            source_image = source_image[0]  # -> [C, H, W]
        elif len(source_image.shape) == 4 and source_image.shape[0] > 1:
            print("Warning: Source is a batch with multiple images. Using only the first image.")
            source_image = source_image[0]  # -> [C, H, W]
        
        # Get shapes and check compatibility
        if len(batch_images.shape) != 4:  # [B, C, H, W]
            raise ValueError("Batch images must be a 4D tensor [batch, channels, height, width]")
        
        # Ensure dimensions match
        batch_size, channels, height, width = batch_images.shape
        if source_image.shape != (channels, height, width):
            raise ValueError(f"Source image shape {source_image.shape} doesn't match batch image dimensions ({channels}, {height}, {width})")
        
        # Create result tensor starting with source image
        result = source_image.clone().unsqueeze(0)  # -> [1, C, H, W]
        
        # Initialize cumulative mask for the "cumulative" mode
        cumulative_mask = torch.zeros((height, width), device=batch_images.device)
        combined_diff_mask = torch.zeros((1, height, width), device=batch_images.device)
        
        # Process each image in the batch
        for i in range(batch_size):
            # Get current image
            current_image = batch_images[i]  # [C, H, W]
            
            # Calculate absolute pixel differences
            diff = torch.abs(current_image - source_image)  # [C, H, W]
            
            # Create mask of pixels that differ more than threshold
            # Average across channels to get a per-pixel difference amount
            avg_diff = diff.mean(dim=0)  # [H, W]
            diff_mask = (avg_diff > threshold).float()  # [H, W]
            
            # Add to combined mask for visualization
            combined_diff_mask = torch.maximum(combined_diff_mask, diff_mask.unsqueeze(0))
            
            # Handle mask processing mode
            if mask_processing == "cumulative":
                # Accumulate mask, but don't overlap with previous masks
                new_mask = diff_mask * (1 - cumulative_mask)
                cumulative_mask = torch.clamp(cumulative_mask + new_mask, 0.0, 1.0)
                active_mask = new_mask
            elif mask_processing == "weighted":
                # Weight mask by position in batch (later images have more influence)
                weight = (i + 1) / batch_size
                active_mask = diff_mask * weight
            else:  # per_image
                active_mask = diff_mask
            
            # Apply the mask with selected blend mode
            active_mask = active_mask.unsqueeze(0) * blend_strength  # Adjust by strength
            
            if blend_mode == "normal":
                # Standard alpha blending
                for c in range(channels):
                    result[0, c] = result[0, c] * (1 - active_mask) + current_image[c] * active_mask
            
            elif blend_mode == "add":
                # Add values where pixels differ
                for c in range(channels):
                    result[0, c] = torch.clamp(result[0, c] + current_image[c] * active_mask, 0.0, 1.0)
            
            elif blend_mode == "multiply":
                # Multiply values
                for c in range(channels):
                    # Lerp between original and multiplied
                    multiplied = result[0, c] * current_image[c]
                    result[0, c] = result[0, c] * (1 - active_mask) + multiplied * active_mask
            
            elif blend_mode == "screen":
                # Screen blend: 1-(1-a)*(1-b)
                for c in range(channels):
                    screened = 1 - (1 - result[0, c]) * (1 - current_image[c])
                    result[0, c] = result[0, c] * (1 - active_mask) + screened * active_mask
            
            elif blend_mode == "overlay":
                # Overlay blend
                for c in range(channels):
                    # Apply overlay formula: base < 0.5 ? 2*base*blend : 1-2*(1-base)*(1-blend)
                    base = result[0, c]
                    blend = current_image[c]
                    
                    overlay = torch.where(
                        base < 0.5,
                        2 * base * blend,
                        1 - 2 * (1 - base) * (1 - blend)
                    )
                    
                    result[0, c] = base * (1 - active_mask) + overlay * active_mask
            
            elif blend_mode == "difference":
                # Absolute difference
                for c in range(channels):
                    difference = torch.abs(result[0, c] - current_image[c])
                    result[0, c] = result[0, c] * (1 - active_mask) + difference * active_mask
        
        # Create visualization of the difference mask (expand to 3 channels)
        vis_mask = torch.cat([combined_diff_mask] * 3, dim=0).unsqueeze(0)
        
        # Return the merged image and visualization mask
        return (result, vis_mask)