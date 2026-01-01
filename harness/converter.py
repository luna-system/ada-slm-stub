"""
Model Converter Module
======================

Convert trained LoRA adapters to various deployment formats.

Supported formats:
- Ollama (via GGUF conversion)
- HuggingFace Hub (direct upload)
- Merged safetensors (full model)
"""

import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(
    base_model: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: str = "float16",
) -> Path:
    """
    Merge LoRA adapter weights into base model.
    
    Args:
        base_model: HuggingFace model ID or path
        adapter_path: Path to LoRA adapter checkpoint
        output_path: Where to save merged model
        torch_dtype: Data type for merged model
        
    Returns:
        Path to merged model directory
    """
    print(f"🔄 Merging LoRA weights...")
    print(f"   Base: {base_model}")
    print(f"   Adapter: {adapter_path}")
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype_map.get(torch_dtype, torch.float16),
        device_map="cpu",  # Merge on CPU to save VRAM
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Load and merge LoRA
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    # Save merged model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"   ✅ Merged model saved to: {output_dir}")
    return output_dir


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
    llama_cpp_path: Optional[str] = None,
) -> Path:
    """
    Convert HuggingFace model to GGUF format for Ollama.
    
    Args:
        model_path: Path to HuggingFace model directory
        output_path: Where to save GGUF file
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16)
        llama_cpp_path: Path to llama.cpp repo (auto-detected if None)
        
    Returns:
        Path to GGUF file
    """
    print(f"🔄 Converting to GGUF ({quantization})...")
    
    # Find llama.cpp
    if llama_cpp_path is None:
        # Common locations
        candidates = [
            Path.home() / "llama.cpp",
            Path.home() / "Code" / "llama.cpp",
            Path("/opt/llama.cpp"),
        ]
        for p in candidates:
            if (p / "convert_hf_to_gguf.py").exists():
                llama_cpp_path = str(p)
                break
    
    if llama_cpp_path is None:
        raise RuntimeError(
            "llama.cpp not found! Please clone it:\n"
            "  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
            "  cd ~/llama.cpp && pip install -r requirements.txt"
        )
    
    llama_cpp = Path(llama_cpp_path)
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    
    if not convert_script.exists():
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert to f16 GGUF
    f16_path = output_dir / "model-f16.gguf"
    print(f"   Converting to f16 GGUF...")
    
    result = subprocess.run(
        ["python", str(convert_script), str(model_path), "--outfile", str(f16_path)],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"   ❌ Conversion failed: {result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
    
    # Step 2: Quantize if needed
    if quantization != "f16":
        if not quantize_bin.exists():
            # Try alternative location
            quantize_bin = llama_cpp / "llama-quantize"
            if not quantize_bin.exists():
                print(f"   ⚠️ llama-quantize not found, keeping f16")
                shutil.move(str(f16_path), output_path)
                return Path(output_path)
        
        print(f"   Quantizing to {quantization}...")
        result = subprocess.run(
            [str(quantize_bin), str(f16_path), output_path, quantization.upper()],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"   ❌ Quantization failed: {result.stderr}")
            # Fall back to f16
            shutil.move(str(f16_path), output_path)
        else:
            f16_path.unlink()  # Clean up f16 file
    else:
        shutil.move(str(f16_path), output_path)
    
    print(f"   ✅ GGUF saved to: {output_path}")
    return Path(output_path)


def create_ollama_modelfile(
    gguf_path: str,
    model_name: str,
    output_path: str,
    system_prompt: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> Path:
    """
    Create an Ollama Modelfile for the converted model.
    
    Args:
        gguf_path: Path to GGUF file
        model_name: Name for the Ollama model
        output_path: Where to save the Modelfile
        system_prompt: Optional system prompt to bake in
        parameters: Optional Ollama parameters (temperature, etc.)
        
    Returns:
        Path to Modelfile
    """
    print(f"📝 Creating Ollama Modelfile...")
    
    lines = [f"FROM {gguf_path}"]
    
    if system_prompt:
        # Escape quotes in system prompt
        escaped = system_prompt.replace('"', '\\"')
        lines.append(f'SYSTEM "{escaped}"')
    
    if parameters:
        for key, value in parameters.items():
            lines.append(f"PARAMETER {key} {value}")
    
    # Add default parameters for consciousness models
    default_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
    
    for key, value in default_params.items():
        if parameters is None or key not in parameters:
            lines.append(f"PARAMETER {key} {value}")
    
    content = "\n".join(lines) + "\n"
    
    output_file = Path(output_path)
    output_file.write_text(content)
    
    print(f"   ✅ Modelfile saved to: {output_file}")
    print(f"\n   To create the Ollama model:")
    print(f"   ollama create {model_name} -f {output_file}")
    
    return output_file


def register_with_ollama(
    modelfile_path: str,
    model_name: str,
) -> bool:
    """
    Register the model with Ollama.
    
    Args:
        modelfile_path: Path to Modelfile
        model_name: Name for the Ollama model
        
    Returns:
        True if successful
    """
    print(f"🚀 Registering with Ollama as '{model_name}'...")
    
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"   ❌ Registration failed: {result.stderr}")
        return False
    
    print(f"   ✅ Model registered! Run with: ollama run {model_name}")
    return True


class ModelConverter:
    """
    High-level converter for training outputs.
    
    Usage:
        converter = ModelConverter(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path="ada-slm-v5e-antithesis/checkpoint-5000",
            output_name="ada-slm-v5e",
        )
        converter.to_ollama(quantization="q4_k_m", register=True)
    """
    
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        output_name: str,
        output_dir: str = "exports",
    ):
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.output_name = output_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._merged_path: Optional[Path] = None
    
    def merge(self, torch_dtype: str = "float16") -> "ModelConverter":
        """Merge LoRA weights into base model."""
        merged_dir = self.output_dir / f"{self.output_name}-merged"
        self._merged_path = merge_lora_weights(
            self.base_model,
            str(self.adapter_path),
            str(merged_dir),
            torch_dtype,
        )
        return self
    
    def to_gguf(
        self,
        quantization: str = "q4_k_m",
        llama_cpp_path: Optional[str] = None,
    ) -> Path:
        """Convert to GGUF format."""
        if self._merged_path is None:
            self.merge()
        
        gguf_path = self.output_dir / f"{self.output_name}-{quantization}.gguf"
        return convert_to_gguf(
            str(self._merged_path),
            str(gguf_path),
            quantization,
            llama_cpp_path,
        )
    
    def to_ollama(
        self,
        quantization: str = "q4_k_m",
        system_prompt: Optional[str] = None,
        parameters: Optional[dict] = None,
        register: bool = False,
        llama_cpp_path: Optional[str] = None,
    ) -> Path:
        """
        Full pipeline: merge → GGUF → Modelfile → (optional) register.
        
        Args:
            quantization: GGUF quantization type
            system_prompt: System prompt to bake in
            parameters: Ollama parameters
            register: Whether to register with Ollama
            llama_cpp_path: Path to llama.cpp
            
        Returns:
            Path to Modelfile
        """
        print(f"\n{'='*60}")
        print(f"🎯 Converting {self.output_name} to Ollama")
        print(f"{'='*60}\n")
        
        # Step 1: GGUF conversion
        gguf_path = self.to_gguf(quantization, llama_cpp_path)
        
        # Step 2: Create Modelfile
        modelfile_path = self.output_dir / f"{self.output_name}.Modelfile"
        create_ollama_modelfile(
            str(gguf_path),
            self.output_name,
            str(modelfile_path),
            system_prompt,
            parameters,
        )
        
        # Step 3: Register if requested
        if register:
            register_with_ollama(str(modelfile_path), self.output_name)
        
        print(f"\n{'='*60}")
        print(f"✅ Conversion complete!")
        print(f"{'='*60}")
        
        return modelfile_path
    
    def to_huggingface(
        self,
        repo_id: str,
        private: bool = True,
        commit_message: str = "Upload model",
    ) -> str:
        """
        Upload to HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repo (e.g., "luna-system/ada-slm-v5e")
            private: Whether to make repo private
            commit_message: Commit message
            
        Returns:
            URL to uploaded model
        """
        if self._merged_path is None:
            self.merge()
        
        print(f"🚀 Uploading to HuggingFace: {repo_id}")
        
        from huggingface_hub import HfApi
        
        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        
        api.upload_folder(
            folder_path=str(self._merged_path),
            repo_id=repo_id,
            commit_message=commit_message,
        )
        
        url = f"https://huggingface.co/{repo_id}"
        print(f"   ✅ Uploaded to: {url}")
        return url
