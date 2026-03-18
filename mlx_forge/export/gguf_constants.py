"""GGUF format constants and tensor type definitions.

Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

# Magic number for GGUF files
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian

# GGUF version
GGUF_VERSION = 3

# Metadata value types
class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

# Tensor types (quantization formats)
class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9

# Type sizes in bytes per element
GGML_TYPE_SIZE = {
    GGMLType.F32: 4,
    GGMLType.F16: 2,
    GGMLType.Q4_0: 0.5 + 2/32,  # block size 32
    GGMLType.Q8_0: 1 + 2/32,    # block size 32
}

# Standard metadata keys
METADATA_KEYS = {
    "general.architecture": "general.architecture",
    "general.name": "general.name",
    "general.file_type": "general.file_type",
}
