### Supported graphics cards

| MODULE / FILE | SAGE | ARCHITECTURES / FEATURES |
| :--- | :---: | :--- |
| **fp4attn_cuda** | v3 | FP4 ACCEL + sm_120 + Blackwell Native |
| **fp4quant_cuda** | v3 | FP4 ACCEL + sm_120 + Blackwell Native |
| **_fused** | v2 | sm_80, sm_89, sm_90, sm_100, sm_120 + Blackwell Native |
| **_qattn_sm80** | v2 | sm_80, **sm_86 (RTX 30xx)** |
| **_qattn_sm89** | v2 | sm_89, sm_100, sm_120 + Blackwell Native |
| **_qattn_sm90** | v2 | sm_90 |

> **Note:** > * **sm_120** = RTX 50xx (Blackwell) 
> * **sm_89** = RTX 40xx (Ada Lovelace) 
> * **sm_80 / sm_86** = A100 / RTX 30xx (Ampere) 
> * **sm_90** = H100 (Hopper)