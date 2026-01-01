"""
GPU Management for AMD ROCm
===========================

Handles all the finicky ROCm/HIP configuration that we've debugged many times:
- Environment variable setup
- GPU detection and isolation
- Memory management
- Multi-GPU handling (especially iGPU vs discrete)
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    index: int
    name: str
    memory_mb: int
    is_igpu: bool = False
    hip_device_id: int = 0


class GPUManager:
    """
    Manages GPU detection, selection, and environment configuration for ROCm.
    
    Usage:
        gpu = GPUManager()
        gpu.setup_for_training(prefer_discrete=True)
        device = gpu.get_device()
    """
    
    # Environment variables that need to be set BEFORE importing torch
    ENV_VARS = {
        "TOKENIZERS_PARALLELISM": "false",  # Avoid fork deadlocks
        "HIP_VISIBLE_DEVICES": None,  # Set dynamically
        "CUDA_VISIBLE_DEVICES": None,  # ROCm honors this too
        "HSA_OVERRIDE_GFX_VERSION": None,  # For architecture mismatches
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",  # Memory allocation
    }
    
    # Known iGPU identifiers
    IGPU_PATTERNS = [
        "Radeon Graphics",  # Generic APU name
        "Vega",  # Older APU
        "Raphael",  # Zen 4 APU
        "Phoenix",  # Zen 4 mobile APU
        "Rembrandt",  # Zen 3+ APU
        "Cezanne",  # Zen 3 APU
        "Renoir",  # Zen 2 APU
    ]
    
    def __init__(self):
        self.gpus: List[GPUInfo] = []
        self.selected_gpu: Optional[GPUInfo] = None
        self._detect_gpus()
    
    def _detect_gpus(self):
        """Detect available GPUs using rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                capture_output=True, text=True
            )
            # Parse rocm-smi output (simplified)
            # In practice, this is complex - fall back to torch detection
        except FileNotFoundError:
            pass
        
        # Use torch for detection (after env setup)
        self._env_setup_minimal()
        
    def _env_setup_minimal(self):
        """Minimal env setup for GPU detection."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def detect_with_torch(self) -> List[GPUInfo]:
        """Detect GPUs using PyTorch (must be called after torch import)."""
        import torch
        
        self.gpus = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            
            is_igpu = any(pattern.lower() in name.lower() for pattern in self.IGPU_PATTERNS)
            
            gpu = GPUInfo(
                index=i,
                name=name,
                memory_mb=props.total_memory // (1024 * 1024),
                is_igpu=is_igpu,
                hip_device_id=i,
            )
            self.gpus.append(gpu)
        
        return self.gpus
    
    def setup_environment(
        self, 
        gpu_index: Optional[int] = None,
        prefer_discrete: bool = True,
        gfx_version_override: Optional[str] = None,
    ):
        """
        Set up environment variables for training.
        
        MUST be called BEFORE importing torch!
        
        Args:
            gpu_index: Specific GPU index to use. If None, auto-select.
            prefer_discrete: If True and gpu_index is None, prefer discrete GPU over iGPU.
            gfx_version_override: Override HSA_OVERRIDE_GFX_VERSION (e.g., "11.0.0" for RDNA3).
        """
        # Set standard vars
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # GPU selection
        if gpu_index is not None:
            visible = str(gpu_index)
        elif prefer_discrete:
            # Default to GPU 0 (usually discrete), isolate from iGPU
            visible = "0"
        else:
            # Use all GPUs
            visible = None
        
        if visible is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = visible
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
            print(f"🔧 GPU isolation: HIP_VISIBLE_DEVICES={visible}")
        
        # GFX version override for architecture mismatches
        if gfx_version_override:
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = gfx_version_override
            print(f"🔧 GFX override: HSA_OVERRIDE_GFX_VERSION={gfx_version_override}")
        
        # Memory allocation strategy
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    
    def get_device_map(self) -> Dict[str, Any]:
        """
        Get the device_map for model loading.
        
        Returns device_map={"": 0} for explicit single-GPU placement,
        which is more reliable on ROCm than device_map="auto".
        """
        return {"": 0}
    
    def get_device(self) -> str:
        """Get the device string for tensor operations."""
        return "cuda:0"
    
    def clear_memory(self, aggressive: bool = False):
        """
        Clear GPU memory cache.
        
        Args:
            aggressive: If True, use more aggressive cleanup methods
        """
        import torch
        import gc
        
        # Python garbage collection first
        gc.collect()
        
        # PyTorch cache clearing
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if aggressive:
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Force another gc pass
            gc.collect()
            torch.cuda.empty_cache()
        
        print("🧹 GPU memory cleared")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics in GB."""
        import torch
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2),
        }
    
    def print_status(self):
        """Print GPU status information."""
        import torch
        
        print("\n" + "="*60)
        print("🎮 GPU STATUS")
        print("="*60)
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"ROCm/CUDA available: {torch.cuda.is_available()}")
        print(f"Device count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        
        # Show relevant env vars
        print("\nEnvironment:")
        for var in ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"]:
            val = os.environ.get(var, "(not set)")
            print(f"  {var}={val}")
        
        print("="*60 + "\n")


# Convenience function for quick setup
def setup_rocm_environment(
    gpu_index: int = 0,
    gfx_override: Optional[str] = None,
) -> GPUManager:
    """
    Quick setup for ROCm training environment.
    
    Call this BEFORE importing torch!
    
    Args:
        gpu_index: Which GPU to use (0 = first discrete GPU)
        gfx_override: GFX version override if needed
        
    Returns:
        Configured GPUManager instance
    """
    manager = GPUManager()
    manager.setup_environment(
        gpu_index=gpu_index,
        prefer_discrete=True,
        gfx_version_override=gfx_override,
    )
    return manager


def force_gpu_cleanup():
    """
    Force aggressive GPU memory cleanup.
    
    This is useful when previous training runs didn't release memory properly.
    Can be called from command line: python -m harness.gpu --cleanup
    """
    import gc
    
    # Set minimal env
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("🔄 Force GPU cleanup starting...")
    
    # Import torch after env setup
    import torch
    
    if not torch.cuda.is_available():
        print("❌ No GPU available")
        return
    
    # Get initial state
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated_before = torch.cuda.memory_allocated() / 1e9
    
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Total VRAM: {total:.2f} GB")
    print(f"   Allocated (before): {allocated_before:.2f} GB")
    
    # Aggressive cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    allocated_after = torch.cuda.memory_allocated() / 1e9
    print(f"   Allocated (after): {allocated_after:.2f} GB")
    
    if allocated_after < allocated_before:
        freed = allocated_before - allocated_after
        print(f"   ✅ Freed {freed:.2f} GB")
    else:
        print("   ⚠️  Memory may be held by another process")
        print("   Try: pkill -9 -f python  # Kill all Python processes")
    
    print("🧹 Cleanup complete!")


if __name__ == "__main__":
    import sys
    
    if "--cleanup" in sys.argv or "-c" in sys.argv:
        force_gpu_cleanup()
    elif "--status" in sys.argv or "-s" in sys.argv:
        # Show status
        manager = GPUManager()
        manager.setup_environment(gpu_index=0)
        import torch
        manager.detect_with_torch()
        manager.print_status()
        
        # Show memory
        stats = manager.get_memory_stats()
        print("Memory Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print("GPU Management Utilities")
        print("Usage:")
        print("  python -m harness.gpu --status   # Show GPU info")
        print("  python -m harness.gpu --cleanup  # Force memory cleanup")
