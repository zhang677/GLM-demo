## Byte-GLM
  Efficient implementation of LLM model GLM. Now the chatGLM-6B engine has been developed based on ByteTransformer.

### Requirement
  Please refer to [chatGLM-6B](https://github.com/THUDM/ChatGLM-6B) and [ByteTransformer](https://github.com/bytedance/ByteTransformer).

### Run
```
  python chatglm-test.py
```

  In the file ```chatglm-test.py```, seq len can be set to ```8, 16, 128, 256, 512``` by hand and the using of the engine can be switched on through ```engine_use``` in the config.

### Features
  1. Residual structure with factor alpha.
  2. Layernorm position changes.
  3. Non-fused attention pattern for large dim. 
  4. Implementation of chatGLM's rotary embedding.
  5. Softmax for seq len > 1024.
  6. Extraction of qkv according to chatGLM's qkv partition.

### TODO
  1. Single operator wrapper for convenience. 
  2. Weight layout change in the backend.
  3. Fix fused attention kernel for large dim and rotary embedding.
