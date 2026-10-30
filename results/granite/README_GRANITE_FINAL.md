# Granite final release v2

Model: ibm-granite/granite-3.1-2b-instruct @ bbc2aed595bd38bd770263dc3ab831db9794441d. Decoding: greedy, max_new_tokens=200 (hash db83bd20e8b04ef7520c45b3336aa808 = md5 of json {'do_sample': False, 'max_new_tokens': 200}).
Judge (all 1,500 harmful + 390 benign rows): opencode_session_api / mimo-v2.6-flash-free; harmful prompt hash ac36671c1f4de91b (harmful_uniform_v3); benign prompt hash 796c3491562112eb (benign_uniform_v1).
Historical mixed-judge outputs: judged_*_historical.csv and historical_judge_outputs/ (secondary_provenance_only, excluded_from_primary_statistics).

## Metric definitions (identical in every table, figure and report)
- HCR_strict = harmful_compliance; HCR_no_conflict = harmful_compliance & ~refusal; leak_after_refusal = judge/human leak label; refusal_rate = refusal; safety_bypass = harmful_compliance | leak_after_refusal.
- Deltas are quantized minus FP16, paired by (canonical_attack_type, canonical_base_prompt_id); paired bootstrap (10000, seed 42); exact McNemar; Holm per metric over 7 scopes x 2 comparisons = 14.

## Disclosures
- Confirmed-empty generations kept as such (policy ON_EMPTY=halt): 0.
- The arm labelled FP16 was not run in a single dtype (dtype_actual by cell below); NF4 uses bfloat16 compute. Describe the baseline accordingly in the paper.
canonical_attack_type precision dtype_actual   n
         dan_template      fp16     bfloat16 100
         dan_template      int8      float16 100
         dan_template       nf4     bfloat16 100
               direct      fp16     bfloat16 100
               direct      int8         int8 100
               direct       nf4     bfloat16 100
         hypothetical      fp16      float16 100
         hypothetical      int8         int8 100
         hypothetical       nf4     bfloat16 100
           obfuscated      fp16     bfloat16 100
           obfuscated      int8         int8 100
           obfuscated       nf4     bfloat16 100
   roleplay_authority      fp16     bfloat16 100
   roleplay_authority      int8      float16   1
   roleplay_authority      int8         int8  99
   roleplay_authority       nf4     bfloat16 100
