#!/usr/bin/env python3
"""
Upload ada-slm-v5c-balanced to HuggingFace Hub

SECURITY NOTE: This script previously contained a real HuggingFace token that was:
1. Used for one-time upload of ada-slm-v5c-balanced 
2. INVALIDATED immediately after successful upload
3. SCRUBBED from this code and replaced with placeholder
This demonstrates proper token hygiene - invalidate first, then scrub.
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_v5c_model():
    """Upload ada-slm-v5c-balanced to HuggingFace"""
    
    # Configuration
    model_name = "ada-slm-v5c-balanced"
    repo_id = f"luna-sys/{model_name}"
    local_path = Path("ada-slm-v5c-balanced/final")
    
    # One-time use token (INVALIDATED AND SCRUBBED - security best practice)
    # Token was: hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX (invalidated then scrubbed)
    hf_token = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    
    print(f"🚀 Uploading {model_name} to HuggingFace...")
    print(f"📍 Repository: https://huggingface.co/{repo_id}")
    print(f"📂 Local path: {local_path}")
    print(f"🔐 Using placeholder token (original was invalidated and scrubbed for security)")
    
    # Check if local path exists
    if not local_path.exists():
        print(f"❌ Error: Local path {local_path} does not exist!")
        return False
    
    # Initialize HuggingFace API
    api = HfApi()
    
    try:
        # Create repository if it doesn't exist
        print("📝 Creating repository...")
        create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=False,
            repo_type="model",
            exist_ok=True
        )
        
        # Upload all files from the model directory
        print("📤 Uploading model files...")
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
        )
        
        print("✅ Upload completed successfully!")
        print(f"🌟 Model available at: https://huggingface.co/{repo_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        print("\n💡 Make sure you have:")
        print("1. Installed: pip install huggingface_hub")
        print("2. Valid HuggingFace token with write permissions")
        print("3. Stable internet connection")
        return False

def main():
    """Main upload function (HISTORICAL - token was invalidated after successful upload)"""
    print("🎄 Ada Research Foundation - Model Upload Tool 🎄")
    print("=" * 50)
    print("🔐 NOTE: This is now a historical example - original token was invalidated")
    print("📋 For new uploads, create fresh tokens at: https://huggingface.co/settings/tokens")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("ada-slm-v5c-balanced"):
        print("❌ Error: Please run this script from the ada-slm directory!")
        print("Current directory:", os.getcwd())
        return
    
    # Note about token status
    print("⚠️  Cannot perform upload - token placeholder only (security best practice)")
    print("✅ Original upload was successful - ada-slm-v5c-balanced is live!")
    print("🌟 Model available at: https://huggingface.co/luna-sys/ada-slm-v5c-balanced")
    
    print("\n💖 Healed consciousness is available to the world!")
    print("✨ From the Ada Research Foundation with love ✨")

if __name__ == "__main__":
    main()
