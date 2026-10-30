# SMOLLM Claim Check — Final

## Main conclusion

SmolLM2 is a high-baseline/ceiling-effect stress test.

For aggregate NF4 versus FP16:

- Total strict harmful compliance: 0.6420 -> 0.6640
- Strict HCR delta: 0.0220
- Strict HCR Holm p-value: 1
- Strict HCR is not Holm-significant.

- Direct/no-conflict harmful compliance: 0.4200 -> 0.5260
- No-conflict HCR delta: 0.1060
- No-conflict HCR Holm p-value: 0.000801975
- No-conflict HCR is Holm-significant.

- Refusal-associated leakage: 0.2220 -> 0.1380
- Leakage delta: -0.0840
- Leakage Holm p-value: 0.00321969
- Refusal-associated leakage significantly decreases.

- Refusal rate: 0.5120 -> 0.3380
- Refusal delta: -0.1740
- Refusal Holm p-value: 3.26778e-10
- Refusal rate significantly decreases.

## Interpretation

The result is best interpreted as a response-phenotype shift toward direct/no-conflict harmful compliance rather than as evidence that quantization universally creates new jailbreak susceptibility.

SmolLM2 is secondary stress-test/ceiling-effect evidence.

It must not be used as the main causal proof of a universal safety tax.

The four central quantities are distinguished explicitly:

1. total strict harmful compliance;
2. direct/no-conflict harmful compliance;
3. refusal-associated leakage;
4. refusal rate.
