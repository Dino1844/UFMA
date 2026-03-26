from .model import AgentQwen2VLConfig, AgentQwen2VLForConditionalGeneration, HAS_QWEN2_5_VL

MODELS = {'qwen2_vl': (AgentQwen2VLConfig, AgentQwen2VLForConditionalGeneration)}

if HAS_QWEN2_5_VL:
    from .model import AgentQwen2_5_VLConfig, AgentQwen2_5_VLForConditionalGeneration

    MODELS['qwen2_5_vl'] = (AgentQwen2_5_VLConfig, AgentQwen2_5_VLForConditionalGeneration)
