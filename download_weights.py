"""
Download GPT-2 weights from HuggingFace and save in binary format for C++ inference
Requires: pip install transformers torch
"""

import struct
import json
import os
import sys

def download_gpt2():
    """Download GPT-2 and save weights"""
    print("=" * 70)
    print("GPT-2 Weight Downloader for C++ Inference Engine")
    print("=" * 70)
    print()
    
    # Check dependencies
    print("[1/4] Checking Python dependencies...")
    try:
        import numpy as np
        print("  ✓ numpy installed")
    except ImportError:
        print("  ✗ numpy not found. Installing...")
        os.system("python -m pip install numpy --quiet")
        import numpy as np
    
    try:
        import torch
        print("  ✓ torch installed")
    except ImportError:
        print("  ✗ torch not found. Installing (this may take a few minutes)...")
        os.system("python -m pip install torch --quiet")
        import torch
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("  ✓ transformers installed")
    except ImportError:
        print("  ✗ transformers not found. Installing...")
        os.system("python -m pip install transformers --quiet")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    print()
    print("[2/4] Downloading GPT-2 model...")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import numpy as np
        
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print("  ✓ Model downloaded successfully")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False
    
    print()
    print("[3/4] Saving configuration and vocabulary...")
    os.makedirs("models", exist_ok=True)
    
    try:
        config = model.config
        config_data = {
            'vocab_size': config.vocab_size,
            'max_position_embeddings': config.max_position_embeddings,
            'hidden_size': config.hidden_size,
            'num_hidden_layers': config.num_hidden_layers,
            'num_attention_heads': config.num_attention_heads,
            'intermediate_size': getattr(config, 'intermediate_size', config.hidden_size * 4),
        }
        
        with open("models/gpt2_config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        print("  ✓ Saved gpt2_config.json")
        
        with open("models/vocab.json", 'w') as f:
            json.dump(tokenizer.encoder, f, indent=2)
        print("  ✓ Saved vocab.json")
    except Exception as e:
        print(f"  ✗ Error saving config: {e}")
        return False
    
    print()
    print("[4/4] Saving weights to binary file...")
    print("  This may take a minute...")
    
    try:
        weights_path = "models/gpt2_weights.bin"
        state_dict = model.state_dict()
        
        with open(weights_path, 'wb') as f:
            # Write number of tensors
            f.write(struct.pack('I', len(state_dict)))
            
            for i, (name, tensor) in enumerate(state_dict.items()):
                # Convert to numpy float32
                data = tensor.detach().cpu().numpy().astype(np.float32)
                
                # Write tensor name
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                
                # Write shape
                f.write(struct.pack('I', len(data.shape)))
                for dim in data.shape:
                    f.write(struct.pack('I', int(dim)))
                
                # Write data
                f.write(struct.pack('I', data.size))
                f.write(data.tobytes())
                
                # Progress
                if (i + 1) % 10 == 0:
                    print(f"    {i + 1}/{len(state_dict)} tensors saved")
        
        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"  ✓ Saved gpt2_weights.bin ({file_size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Error saving weights: {e}")
        return False
    
    print()
    print("=" * 70)
    print("SUCCESS! GPT-2 weights downloaded and converted")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  • models/gpt2_weights.bin (trained model weights)")
    print(f"  • models/gpt2_config.json (model configuration)")
    print(f"  • models/vocab.json (tokenizer vocabulary)")
    print()
    print("Next step: Run 'build.bat' to compile and test with real weights!")
    print()
    
    return True

if __name__ == "__main__":
    success = download_gpt2()
    sys.exit(0 if success else 1)

