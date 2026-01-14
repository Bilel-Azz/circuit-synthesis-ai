# Models module - V1 (supervisé)
from .encoder import ImpedanceEncoder
from .decoder import TransformerDecoder
from .model import CircuitTransformer

# Models module - V2 (avec reconstruction loss)
from .forward_model import ForwardModel
from .model_v2 import CircuitTransformerV2
