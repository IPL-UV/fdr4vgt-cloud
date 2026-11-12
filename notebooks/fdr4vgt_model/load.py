"""
load.py

Module for loading ensemble models (STAC compatible) and performing
optimized inference on large geospatial imagery using dynamic batching
and Gaussian blending.
"""

import math
import pathlib
import itertools
from typing import Literal, Tuple, List

import torch
import torch.nn
import numpy as np
import pystac
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==============================================================================
# 1. HELPER CLASSES & FUNCTIONS
# ==============================================================================

class EnsembleModel(torch.nn.Module):
    """
    Runtime ensemble model for combining multiple model outputs.
    Used when loading multiple separate .pt2 files.
    """
    def __init__(self, *models, mode="max"):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList(models)
        self.mode = mode
        if mode not in ["min", "mean", "median", "max", "none"]:
            raise ValueError("Mode must be 'none', 'min', 'mean', 'median', or 'max'.")

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - probabilities: (B, 1, H, W)
            - uncertainty: (B, 1, H, W) (normalized std dev)
        """
        outputs = [model(x) for model in self.models]
        
        if not outputs:
            return None, None
        
        # Stack: (B, N, H, W)
        stacked = torch.stack(outputs, dim=1).squeeze(2)
        
        # Aggregation
        if self.mode == "max":
            probs = torch.max(stacked, dim=1, keepdim=True)[0]
        elif self.mode == "mean":
            probs = torch.mean(stacked, dim=1, keepdim=True)
        elif self.mode == "median":
            probs = torch.median(stacked, dim=1, keepdim=True)[0]
        elif self.mode == "min":
            probs = torch.min(stacked, dim=1, keepdim=True)[0]
        elif self.mode == "none":
            return stacked, None
            
        # Uncertainty
        N = len(outputs)
        if N > 1:
            std = torch.std(stacked, dim=1, keepdim=True)
            std_max = math.sqrt(0.25 * N / (N - 1))
            uncertainty = torch.clamp(std / std_max, 0.0, 1.0)
        else:
            uncertainty = torch.zeros_like(probs)
        
        return probs, uncertainty

def get_spline_window(window_size: int, power: int = 2) -> np.ndarray:
    """Generates a 2D Hann window for smoothing tile edges."""
    intersection = np.hanning(window_size)
    window_2d = np.outer(intersection, intersection)
    return (window_2d ** power).astype(np.float32)

def fix_lastchunk(iterchunks, s2dim, chunk_size):
    """Adjusts the last chunks to fit within image boundaries."""
    itercontainer = []
    for index_i, index_j in iterchunks:
        if index_i + chunk_size > s2dim[0]:
            index_i = max(s2dim[0] - chunk_size, 0)
        if index_j + chunk_size > s2dim[1]:
            index_j = max(s2dim[1] - chunk_size, 0)
        itercontainer.append((index_i, index_j))
    return list(set(itercontainer))

def define_iteration(dimension: tuple, chunk_size: int, overlap: int = 0):
    """Generates top-left coordinates for sliding window inference."""
    dimy, dimx = dimension
    if chunk_size > max(dimx, dimy):
        return [(0, 0)]
        
    y_step = chunk_size - overlap
    x_step = chunk_size - overlap
    
    iterchunks = list(itertools.product(
        range(0, dimy, y_step), 
        range(0, dimx, x_step)
    ))
    
    return fix_lastchunk(iterchunks, dimension, chunk_size)

# ==============================================================================
# 2. DATASET FOR PARALLEL LOADING
# ==============================================================================

class PatchDataset(Dataset):
    """
    Dataset wrapper to handle image slicing and padding on CPU workers.
    """
    def __init__(self, image: np.ndarray, coords: List[Tuple[int, int]], chunk_size: int, nodata: float = 0):
        self.image = image
        self.coords = coords
        self.chunk_size = chunk_size
        self.nodata = nodata

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row_off, col_off = self.coords[idx]
        
        # Read patch
        patch = self.image[:, row_off : row_off + self.chunk_size, col_off : col_off + self.chunk_size]
        c, h, w = patch.shape

        patch_tensor = torch.from_numpy(patch).float()

        # Apply padding if patch is smaller than chunk_size (edges)
        pad_h = self.chunk_size - h
        pad_w = self.chunk_size - w
        if pad_h > 0 or pad_w > 0:
            patch_tensor = torch.nn.functional.pad(patch_tensor, (0, pad_w, 0, pad_h), "constant", self.nodata)

        # Identify nodata pixels
        mask_nodata = (patch_tensor == self.nodata).all(dim=0)
        
        return patch_tensor, row_off, col_off, h, w, mask_nodata

# ==============================================================================
# 3. LOADING & INFERENCE LOGIC
# ==============================================================================

class TileModel(torch.nn.Module):
    """Wrap an exported model whose graph only accepts 512 x 512 tiles.

    A smaller crop, such as the published 128 x 128 example patch, is padded to
    512 and the output is cropped back, so the caller always gets a map with the
    size of its input. Anything larger must go through ``predict_large``.
    """

    size = 512

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        height, width = x.shape[-2:]
        if (height, width) == (self.size, self.size):
            return self.model(x)
        if height > self.size or width > self.size:
            raise ValueError(
                f"Input is {height} x {width} px and the exported graph takes "
                "512 x 512 tiles. Use predict_large for full scenes."
            )
        padded = torch.nn.functional.pad(
            x, (0, self.size - width, 0, self.size - height), "replicate"
        )
        out = self.model(padded)
        if isinstance(out, tuple):
            return tuple(item[..., :height, :width] for item in out)
        return out[..., :height, :width]


def compiled_model(
    path: pathlib.Path, 
    stac_item: pystac.Item,
    mode: Literal["min", "mean", "median", "max"] = "max",
    *args, **kwargs
):
    """
    Loads .pt2 model(s). Returns a single model or an EnsembleModel.
    Automatically unwraps ExportedProgram if possible.
    """
    model_paths = sorted([
        asset.href for key, asset in stac_item.assets.items() 
        if asset.href.endswith(".pt2")
    ])
    
    if not model_paths:
        raise ValueError("No .pt2 files found in STAC item assets.")
    
    # Helper to load and unwrap
    def load_pt2(p):
        try:
            program = torch.export.load(p)
        except RuntimeError as error:
            if "schema version" not in str(error):
                raise
            raise RuntimeError(
                f"{pathlib.Path(p).name} was exported with another torch version "
                "and torch.export cannot read it here. The operational model uses "
                "torch 2.7; the ablation models use torch 2.5.1. Install the "
                "matching torch or re-export with pipeline/10_export_pt2.py."
            ) from error
        return program.module() if hasattr(program, "module") else program

    if len(model_paths) == 1:
        return TileModel(load_pt2(model_paths[0]))
    else:
        models = [load_pt2(p) for p in model_paths]
        return TileModel(EnsembleModel(*models, mode=mode))


def predict_large(
    image: np.ndarray,
    model: torch.nn.Module,
    chunk_size: int = 512,
    overlap: int = 128,     
    batch_size: int = 16, 
    num_workers: int = 8,    # Recommended: 8-16
    device: str = "cuda",
    nodata: float = 0.0
) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Optimized inference for large images using Dynamic Batching and Gaussian Blending.
    """
    
    if image.ndim != 3:
        raise ValueError(f"Input image must be (C, H, W). Received {image.shape}")
    
    # --- 1. Robust Model Unwrapping ---
    # Fix for torch.export.load() returning an ExportedProgram container
    if hasattr(model, "module") and callable(model.module):
        try:
            unpacked = model.module()
            if isinstance(unpacked, torch.nn.Module):
                model = unpacked
        except Exception:
            pass

    # --- 2. Setup Model ---
    try:
        model.eval()
        for p in model.parameters(): p.requires_grad = False
    except: pass
    
    # Only move to device if it's a standard Module (ExportedProgram handles device internally or via input)
    if isinstance(model, torch.nn.Module):
        model = model.to(device)

    bands, height, width = image.shape

    # --- 3. Check Signature (Ensemble vs Single) ---
    # Dummy pass (batch=2 to respect dynamic shapes)
    dummy = torch.randn(2, bands, chunk_size, chunk_size).to(device)
    try:
        with torch.no_grad():
            out = model(dummy)
    except RuntimeError as error:
        raise RuntimeError(
            f"The model rejected a {chunk_size} x {chunk_size} tile. The "
            "exported graphs take 512 x 512 tiles, so keep chunk_size=512: "
            f"{error}"
        ) from error
    is_ensemble = isinstance(out, tuple) and len(out) == 2

    # --- 4. Initialize Buffers (Accumulators) ---
    out_probs = np.zeros((1, height, width), dtype=np.float32)
    count_map = np.zeros((1, height, width), dtype=np.float32)
    out_uncert = np.zeros((1, height, width), dtype=np.float32) if is_ensemble else None

    # --- 5. Prepare Spline Window ---
    window_spline = get_spline_window(chunk_size, power=2)
    window_tensor = torch.from_numpy(window_spline).to(device)

    # --- 6. DataLoader Setup ---
    coords = define_iteration((height, width), chunk_size, overlap)
    dataset = PatchDataset(image, coords, chunk_size, nodata)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available()
    )

    # --- 7. Inference Loop ---
    for batch in tqdm(loader, desc=f"Inference (Batch {batch_size})"):
        patches, r_offs, c_offs, h_actuals, w_actuals, nodata_masks = batch
        
        # Move inputs to GPU
        patches = patches.to(device, non_blocking=True)
        nodata_masks = nodata_masks.to(device, non_blocking=True) # (B, H, W)

        # Forward Pass
        with torch.no_grad():
            if is_ensemble:
                probs, uncert = model(patches)
            else:
                probs = model(patches)
                uncert = None

        # Ensure correct dimensions (B, C, H, W)
        if probs.ndim == 3: probs = probs.unsqueeze(1)
        if is_ensemble and uncert.ndim == 3: uncert = uncert.unsqueeze(1)

        # Prepare weights for batch
        B = patches.size(0)
        batch_weights = window_tensor.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Zero out weights where input was nodata
        batch_weights[nodata_masks.unsqueeze(1)] = 0.0

        # Apply weights
        probs_weighted = probs * batch_weights
        if is_ensemble:
            uncert_weighted = uncert * batch_weights

        # Move to CPU
        probs_cpu = probs_weighted.cpu().numpy()
        weights_cpu = batch_weights.cpu().numpy()
        if is_ensemble:
            uncert_cpu = uncert_weighted.cpu().numpy()

        # Accumulate in global map
        for i in range(B):
            r, c = r_offs[i].item(), c_offs[i].item()
            h, w = h_actuals[i].item(), w_actuals[i].item()

            # Slice valid regions
            valid_probs = probs_cpu[i, :, :h, :w]
            valid_weights = weights_cpu[i, :, :h, :w]
            
            out_probs[:, r:r+h, c:c+w] += valid_probs
            count_map[:, r:r+h, c:c+w] += valid_weights

            if is_ensemble:
                valid_uncert = uncert_cpu[i, :, :h, :w]
                out_uncert[:, r:r+h, c:c+w] += valid_uncert

    # --- 8. Normalization ---
    mask_zero = (count_map == 0)
    count_map[mask_zero] = 1.0 # Prevent div/0
    
    out_probs /= count_map
    out_probs[mask_zero] = nodata

    if is_ensemble:
        out_uncert /= count_map
        out_uncert[mask_zero] = nodata
        return out_probs, out_uncert
    
    return out_probs