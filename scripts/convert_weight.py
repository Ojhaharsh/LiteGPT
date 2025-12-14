import torch
from transformers import GPT2Model
import json
import struct
import numpy as np

def quantize_tensor(tensor, bits=8):
    """
    Simple absmax quantization to int8 or int4.
    For int4, pack two int4 into one int8.
    """
    max_val = torch.max(torch.abs(tensor)).item()
    if max_val == 0:
        scale = 1.0
    else:
        if bits == 8:
            scale = 127.0 / max_val
        elif bits == 4:
            scale = 7.0 / max_val  # -8 to 7
        else:
            raise ValueError("Unsupported bits")

    quantized = torch.clamp(torch.round(tensor * scale), min=-(2**(bits-1)), max=2**(bits-1)-1).to(torch.int8)

    if bits == 4:
        # Pack two int4 into one int8
        flat = quantized.flatten()
        if len(flat) % 2 != 0:
            flat = torch.cat([flat, torch.tensor([0], dtype=torch.int8)])  # pad
        packed = []
        for i in range(0, len(flat), 2):
            val = ((flat[i] & 0xF) | ((flat[i+1] & 0xF) << 4))
            packed.append(val)
        quantized = torch.tensor(packed, dtype=torch.int8)

    return quantized, scale

def save_quantized_weights(model_name="gpt2", output_file="gpt2_quant.bin", config_file="gpt2_quant_config.json", bits=4):
    model = GPT2Model.from_pretrained(model_name)
    state_dict = model.state_dict()

    config = model.config.__dict__
    config['quant_bits'] = bits

    scales = {}
    quantized_weights = {}

    for name, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            quant, scale = quantize_tensor(tensor, bits)
            quantized_weights[name] = quant
            scales[name] = scale

    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f)

    # Save weights in binary, matching existing format
    with open(output_file, 'wb') as f:
        # Number of tensors: quantized + scales
        num_tensors = len(quantized_weights) + len(scales)
        f.write(struct.pack('I', num_tensors))

        for name, quant in quantized_weights.items():
            quant_np = quant.numpy()
            # Write name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            # Shape
            shape = quant_np.shape
            f.write(struct.pack('I', len(shape)))
            for s in shape:
                f.write(struct.pack('I', s))
            # Num elements
            num_elements = np.prod(shape)
            f.write(struct.pack('I', num_elements))
            # Data
            f.write(quant_np.tobytes())

        for name, scale in scales.items():
            # Write scale as float tensor
            scale_tensor = np.array([scale], dtype=np.float32)
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            shape = scale_tensor.shape
            f.write(struct.pack('I', len(shape)))
            for s in shape:
                f.write(struct.pack('I', s))
            num_elements = np.prod(shape)
            f.write(struct.pack('I', num_elements))
            f.write(scale_tensor.tobytes())

if __name__ == "__main__":
    print("Starting weight conversion...")
    save_quantized_weights(bits=4)
    print("Conversion complete.")
