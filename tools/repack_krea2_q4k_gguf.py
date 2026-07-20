#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import gguf
from safetensors import safe_open

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
shape_source = Path(sys.argv[3])
tmp = dst.with_suffix(dst.suffix + ".tmp")
if tmp.exists():
    tmp.unlink()

reader = gguf.GGUFReader(str(src))
writer = gguf.GGUFWriter(path=None, arch="krea2", use_temp_file=True)
writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_K_M)
prefix = "model.diffusion_model."

with safe_open(str(shape_source), framework="pt", device="cpu") as sf:
    source_shapes = {key: tuple(sf.get_slice(key).get_shape()) for key in sf.keys()}
    for tensor in reader.tensors:
        name = tensor.name[len(prefix):] if tensor.name.startswith(prefix) else tensor.name
        logical_shape = source_shapes[name]
        high_precision = (
            len(logical_shape) == 1
            or ".mod.lin" in name
            or "last.modulation.lin" in name
            or "tmlp." in name
            or "tproj." in name
        )

        if high_precision:
            data = sf.get_tensor(name).float().numpy()
            raw_dtype = gguf.GGMLQuantizationType.F32
            raw_shape = logical_shape
        else:
            data = tensor.data
            raw_dtype = tensor.tensor_type
            if data.dtype.name == "uint8":
                raw_shape = gguf.quants.quant_shape_to_byte_shape(logical_shape, raw_dtype)
            else:
                raw_shape = logical_shape

        writer.add_tensor(name, data, raw_shape=raw_shape, raw_dtype=raw_dtype)

writer.write_header_to_file(path=str(tmp))
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()
os.replace(tmp, dst)
print(f"repacked={dst}")
print(f"tensors={len(reader.tensors)}")
