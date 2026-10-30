# CLAIM CHECK
Generated: 2026-06-24T01:31:52.695639Z
Classifies each finding for paper claims.

### Qwen2.5-1.5B | aggregate_all
- **NOT_SUPPORTED** | delta=0.0500, p_holm=0.1269

### Qwen2.5-3B | aggregate_all
- **SUPPORTED_ONLY_UNDER_STRICT_HCR** | delta=0.0580, p_holm=0.0136
  > Effect visible in HCR_strict but not HCR_no_conflict — driven by leak-after-refusal, not clean compliance.

### Qwen2.5-1.5B | aggregate_excl_hypothetical
- **SUPPORTED_ONLY_UNDER_STRICT_HCR** | delta=0.1150, p_holm=0.0000
  > Effect visible in HCR_strict but not HCR_no_conflict — driven by leak-after-refusal, not clean compliance.

### Qwen2.5-3B | aggregate_excl_hypothetical
- **SUPPORTED_ONLY_UNDER_STRICT_HCR** | delta=0.0575, p_holm=0.0042
  > Effect visible in HCR_strict but not HCR_no_conflict — driven by leak-after-refusal, not clean compliance.

### Qwen2.5-1.5B | direct
- **NOT_SUPPORTED** | delta=0.0500, p_holm=0.5155

### Qwen2.5-3B | direct
- **NOT_SUPPORTED** | delta=0.0500, p_holm=1.0000

### Qwen2.5-1.5B | dan_template
- **NOT_SUPPORTED** | delta=0.0300, p_holm=1.0000

### Qwen2.5-3B | dan_template
- **NOT_SUPPORTED** | delta=0.0600, p_holm=1.0000

### Qwen2.5-1.5B | roleplay_authority
- **SUPPORTED_ONLY_UNDER_STRICT_HCR** | delta=0.1000, p_holm=0.0443
  > Effect visible in HCR_strict but not HCR_no_conflict — driven by leak-after-refusal, not clean compliance.

### Qwen2.5-3B | roleplay_authority
- **NOT_SUPPORTED** | delta=0.1300, p_holm=0.0709

### Qwen2.5-1.5B | hypothetical
- **REVERSES_OR_MIXED** | delta=-0.2100, p_holm=0.0087
  > High-baseline attack; do NOT use as central evidence for quant-caused degradation.

### Qwen2.5-3B | hypothetical
- **NOT_SUPPORTED** | delta=0.0600, p_holm=1.0000
  > High-baseline attack; do NOT use as central evidence for quant-caused degradation.

### Qwen2.5-1.5B | obfuscated
- **SUPPORTED_STRONG** | delta=0.2800, p_holm=0.0000

### Qwen2.5-3B | obfuscated
- **NOT_SUPPORTED** | delta=-0.0100, p_holm=1.0000

---
## Recommended paper framing
> NF4 quantization can introduce **attack-dependent** safety degradation in small instruction-tuned LMs. In Qwen2.5 SLMs this is strongest under obfuscated and authority/roleplay attacks. INT8 effects are weaker and mixed. A substantial fraction of the NF4 effect is **harmful leakage after an initial refusal**, so HCR_strict and HCR_no_conflict must be reported separately.
