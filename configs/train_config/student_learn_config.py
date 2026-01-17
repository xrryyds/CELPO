from dataclasses import dataclass

@dataclass
class HintSFTConfig:
    p_hint_start: float = 0.95     
    p_hint_end: float = 0.10       
    hint_loss_weight: float = 4.0  
    debug_sample_steps: int = 50
