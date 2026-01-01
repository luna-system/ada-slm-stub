"""
Eigenvalue Monitoring Callback
==============================

Track spectral health of attention matrices during training.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import torch
from transformers import TrainerCallback


# Golden ratio for phi-proximity calculations
PHI = 1.618033988749895


def extract_eigenvalues_from_attention(
    model, 
    tokenizer, 
    prompt: str, 
    device: str = "cuda:0",
    debug: bool = False,
) -> List[np.ndarray]:
    """
    Extract eigenvalues from attention matrices for a given prompt.
    
    Args:
        model: The model (must support output_attentions)
        tokenizer: Tokenizer for the model
        prompt: Text prompt to analyze
        device: Device to run on
        debug: Print debug info
        
    Returns:
        List of eigenvalue arrays, one per attention head
    """
    was_training = model.training
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )
    
    # Restore training mode
    if was_training:
        model.train()
    
    all_eigenvalues = []
    
    if outputs.attentions is None:
        if debug:
            print(f"   ⚠️ outputs.attentions is None")
            print(f"   Model type: {type(model).__name__}")
            # Check if model has config
            if hasattr(model, 'config'):
                attn_impl = getattr(model.config, '_attn_implementation', 'unknown')
                print(f"   Attention impl: {attn_impl}")
        return all_eigenvalues
    
    if debug:
        print(f"   📊 Got {len(outputs.attentions)} attention layers")
        if outputs.attentions:
            print(f"   First layer shape: {outputs.attentions[0].shape}")
    
    for layer_idx, attention in enumerate(outputs.attentions):
        # Convert to float32 for numpy.linalg compatibility (eigvals doesn't support float16)
        attn_matrix = attention[0].float().cpu().numpy()
        
        for head_idx in range(attn_matrix.shape[0]):
            head_attn = attn_matrix[head_idx]
            
            try:
                eigenvalues = np.linalg.eigvals(head_attn)
                magnitudes = np.abs(eigenvalues)
                magnitudes = np.sort(magnitudes)[::-1]
                all_eigenvalues.append(magnitudes)
            except Exception as e:
                # Log first failure for debugging
                if layer_idx == 0 and head_idx == 0:
                    print(f"   ⚠️ Eigenvalue extraction failed: {e}")
                continue
    
    if debug:
        print(f"   📊 Extracted {len(all_eigenvalues)} eigenvalue sets")
    
    return all_eigenvalues


def compute_spectral_metrics(all_eigenvalues: List[np.ndarray]) -> Dict:
    """
    Compute spectral health metrics from eigenvalues.
    
    Returns:
        Dict with spectral_entropy, phi_proximity, dominant_ratio
    """
    if not all_eigenvalues:
        return {"spectral_entropy": 0.0, "phi_proximity": 0.0, "dominant_ratio": 0.0}
    
    entropies = []
    dominant_ratios = []
    phi_proximities = []
    
    for magnitudes in all_eigenvalues:
        magnitudes = magnitudes[magnitudes > 1e-10]
        if len(magnitudes) < 2:
            continue
            
        # Spectral entropy
        probs = magnitudes / magnitudes.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
        
        # Dominant eigenvalue ratio
        dominant_ratios.append(magnitudes[0] / magnitudes.sum())
        
        # Phi proximity (how close consecutive ratios are to golden ratio)
        for i in range(len(magnitudes) - 1):
            if magnitudes[i+1] > 1e-10:
                ratio = magnitudes[i] / magnitudes[i+1]
                error = abs(ratio - PHI) / PHI
                phi_proximities.append(max(0, 1 - error))
    
    return {
        "spectral_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "phi_proximity": float(np.max(phi_proximities)) if phi_proximities else 0.0,
        "dominant_ratio": float(np.mean(dominant_ratios)) if dominant_ratios else 0.0,
    }


class EigenvalueMonitorCallback(TrainerCallback):
    """
    Monitor eigenvalues during training.
    
    Logs spectral metrics at regular intervals and provides
    visual health indicators during training.
    """
    
    def __init__(
        self, 
        tokenizer, 
        probe_prompts: List[str],
        sample_interval: int = 50, 
        device: str = "cuda:0",
        log_file: str = "eigenvalue_log.jsonl",
    ):
        self.tokenizer = tokenizer
        self.probe_prompts = probe_prompts
        self.sample_interval = sample_interval
        self.device = device
        self.log_file = Path(log_file)
        self.history: List[Dict] = []
        
        # Initialize log file with header
        with open(self.log_file, "w") as f:
            header = {
                "type": "header",
                "timestamp": datetime.now().isoformat(),
                "probe_prompts": probe_prompts,
                "sample_interval": sample_interval,
            }
            f.write(json.dumps(header) + "\n")
        
        print(f"\n📊 Eigenvalue monitoring enabled!")
        print(f"   Sampling every {sample_interval} steps")
        print(f"   Log file: {self.log_file}")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        if state.global_step % self.sample_interval != 0 or state.global_step == 0:
            return
        
        all_metrics = []
        errors = []
        debug_first = state.global_step == self.sample_interval  # Debug on first sample
        
        for prompt in self.probe_prompts:
            try:
                eigenvalues = extract_eigenvalues_from_attention(
                    model, self.tokenizer, prompt, self.device, debug=debug_first
                )
                metrics = compute_spectral_metrics(eigenvalues)
                all_metrics.append(metrics)
            except Exception as e:
                errors.append(str(e))
                continue
        
        # Log errors on first failure
        if errors and not all_metrics and state.global_step == self.sample_interval:
            print(f"   ⚠️ Eigenvalue extraction errors: {errors[:3]}")
        
        if not all_metrics:
            return
        
        # Average across probes
        avg_metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat(),
            "spectral_entropy": np.mean([m["spectral_entropy"] for m in all_metrics]),
            "phi_proximity": np.mean([m["phi_proximity"] for m in all_metrics]),
            "dominant_ratio": np.mean([m["dominant_ratio"] for m in all_metrics]),
            "loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
        }
        
        self.history.append(avg_metrics)
        
        # Log to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(avg_metrics) + "\n")
        
        # Print health indicator
        entropy = avg_metrics["spectral_entropy"]
        if entropy > 7.0:
            status = "🟢 HEALTHY"
        elif entropy > 6.5:
            status = "🟡 DRIFTING"
        else:
            status = "🔴 WARNING"
        
        bar_len = int(min(entropy, 8) / 8 * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        
        print(f"   📊 Step {state.global_step:5d} | {status} | entropy={entropy:.3f} [{bar}] | dom={avg_metrics['dominant_ratio']:.3f} | loss={avg_metrics['loss']:.4f}")
    
    def get_history(self) -> List[Dict]:
        """Get the full history of eigenvalue metrics."""
        return self.history
    
    def save_history(self, path: str):
        """Save history to JSON file."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Print summary and trends at end of training."""
        if not self.history:
            return
        
        print("\n" + "="*70)
        print("📊 EIGENVALUE TRAINING SUMMARY")
        print("="*70)
        
        # Compute trends (first half vs second half)
        mid = len(self.history) // 2
        first_half = self.history[:mid] if mid > 0 else []
        second_half = self.history[mid:] if mid > 0 else self.history
        
        if first_half and second_half:
            entropy_first = np.mean([m["spectral_entropy"] for m in first_half])
            entropy_second = np.mean([m["spectral_entropy"] for m in second_half])
            entropy_trend = entropy_second - entropy_first
            
            phi_first = np.mean([m["phi_proximity"] for m in first_half])
            phi_second = np.mean([m["phi_proximity"] for m in second_half])
            phi_trend = phi_second - phi_first
            
            dom_first = np.mean([m["dominant_ratio"] for m in first_half])
            dom_second = np.mean([m["dominant_ratio"] for m in second_half])
            dom_trend = dom_second - dom_first
            
            print(f"\n   Trends (first half → second half):")
            print(f"   - Spectral entropy: {entropy_trend:+.4f} ({'↑ improving' if entropy_trend > 0 else '↓ declining'})")
            print(f"   - φ-proximity:      {phi_trend:+.4f} ({'↑ improving' if phi_trend > 0 else '↓ declining'})")
            print(f"   - Dominant ratio:   {dom_trend:+.4f} ({'↓ improving' if dom_trend < 0 else '↑ concentrating'})")
        
        # Final metrics
        final = self.history[-1]
        print(f"\n   Final metrics (step {final['step']}):")
        print(f"   - Spectral entropy: {final['spectral_entropy']:.4f}")
        print(f"   - φ-proximity:      {final['phi_proximity']:.4f}")
        print(f"   - Dominant ratio:   {final['dominant_ratio']:.4f}")
        
        # Health assessment
        entropy = final['spectral_entropy']
        if entropy > 7.0:
            assessment = "🟢 HEALTHY - Good spectral distribution"
        elif entropy > 6.5:
            assessment = "🟡 ACCEPTABLE - Slightly concentrated"
        else:
            assessment = "🔴 WARNING - May need attention rebalancing"
        
        print(f"\n   Assessment: {assessment}")
        print("="*70)
