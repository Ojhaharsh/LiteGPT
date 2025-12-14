"""
Download GPT-2 weights from HuggingFace and convert to binary format for C++ inference
"""

import struct
import json
import os
import numpy as np

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not found. Install with: pip install transformers")

def download_and_convert_gpt2(output_dir="models"):
    """
    Download GPT-2 from HuggingFace and save weights in binary format
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library required")
        print("Install with: pip install transformers torch")
        return False
    
    print("Downloading GPT-2 model from HuggingFace...")
    
    # Download model and tokenizer
    model_name = "gpt2"  # Using small GPT-2
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model config
    config = model.config
    model_config = {
        'vocab_size': config.vocab_size,
        'max_position_embeddings': config.max_position_embeddings,
        'hidden_size': config.hidden_size,
        'num_hidden_layers': config.num_hidden_layers,
        'num_attention_heads': config.num_attention_heads,
        'intermediate_size': config.intermediate_size,
        'hidden_dropout_prob': config.hidden_dropout_prob,
        'attention_probs_dropout_prob': config.attention_probs_dropout_prob,
    }
    
    # Save config as JSON
    config_path = os.path.join(output_dir, "gpt2_config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Save tokenizer vocab
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump(tokenizer.encoder, f)
    print(f"Saved tokenizer vocab to {vocab_path}")
    
    # Save tokenizer merges
    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, 'w') as f:
        f.write('#version: 0.2\n')
        for merge in tokenizer.bpe_ranks.keys():
            f.write(f"{merge[0]} {merge[1]}\n")
    print(f"Saved tokenizer merges to {merges_path}")
    
    # Save weights in binary format
    weights_path = os.path.join(output_dir, "gpt2_weights.bin")
    with open(weights_path, 'wb') as f:
        # Write header with counts
        num_tensors = len(list(model.state_dict().items()))
        f.write(struct.pack('I', num_tensors))  # Number of tensors
        
        print(f"\nSaving {num_tensors} weight tensors to {weights_path}...")
        
        for name, tensor in model.state_dict().items():
            # Convert to numpy and float32
            np_tensor = tensor.detach().cpu().numpy().astype(np.float32)
            
            # Write tensor name (length-prefixed string)
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write tensor shape
            shape = np_tensor.shape
            f.write(struct.pack('I', len(shape)))  # Number of dimensions
            for dim in shape:
                f.write(struct.pack('I', dim))
            
            # Write tensor data
            num_elements = np_tensor.size
            f.write(struct.pack('I', num_elements))
            f.write(np_tensor.astype(np.float32).tobytes())
            
            print(f"  ✓ {name}: shape={shape}")
    
    print(f"\n✓ Successfully saved weights to {weights_path}")
    print(f"\nModel Config:")
    print(f"  Vocab size: {model_config['vocab_size']}")
    print(f"  Hidden size: {model_config['hidden_size']}")
    print(f"  Num layers: {model_config['num_hidden_layers']}")
    print(f"  Num heads: {model_config['num_attention_heads']}")
    print(f"  Max position embeddings: {model_config['max_position_embeddings']}")
    
    return True

if __name__ == "__main__":
    download_and_convert_gpt2()
