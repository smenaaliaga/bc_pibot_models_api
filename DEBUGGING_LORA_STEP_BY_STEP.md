"""
PASO 1: RUN DIAGNOSTIC
=======================

Ejecuta estos scripts en orden para identificar el problema:

1. Diagnosticar estructura de archivos:
   python tests/diagnose_lora_mismatch.py

2. Comparar training vs endpoint:
   python tests/compare_training_vs_endpoint.py


PASO 2: COMPARTIR CÓDIGO DE TRAINING
======================================

Por favor compartir:

1. El código donde CREAS el modelo con LoRA durante training:
   - Cómo cargas el encoder base?
   - Cómo aplicas LoRA? (get_peft_model con qué config?)
   - Qué target_modules usas? (CRÍTICO!)

2. El código donde GUARDAS:
   - El encoder: encoder.save_pretrained(...)?
   - Los LoRA weights: model.save_pretrained(...)?
   - Las heads: torch.save(...)?

Ejemplo template:

# ============ TRAINING CODE ============
from peft import get_peft_model, LoraConfig
from sentence_transformers import SentenceTransformer
import torch

# 1. Cargar encoder base
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
base_model = encoder[0].auto_model

# 2. Aplicar LoRA (¿CUÁL ES TU CONFIG EXACTA?)
lora_config = LoraConfig(
    r=????,           # ← Qué valor?
    lora_alpha=????,  # ← Qué valor?
    target_modules=????,  # ← ESTO ES CRÍTICO! Qué módulos?
    lora_dropout=????,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
peft_model = get_peft_model(base_model, lora_config)

# 3. Entrenar (...)
# peft_model.train()
# ... tuning ...

# 4. Guardar (¿CÓMO LO GUARDAS?)
# Opción A: Merge y guardar
# peft_model.merge_and_unload()
# encoder[0].auto_model.save_pretrained(...)

# Opción B: Guardar con LoRA separado
# peft_model.save_pretrained(...)
# LoRA weights en: encoder/lora_weights/adapter_model.bin

# Opción C: Guardar manualmente
# torch.save(peft_model.state_dict(), ...)


PASO 3: MAIN ISSUES TO FIX
===========================

Sospecha: target_modules mismatch

Audit checklist:
- [ ] En train_config.json hay target_modules?
- [ ] El endpoint usa los mismos target_modules?
- [ ] Durante training, los módulos eran ["query", "key", "value"]?
- [ ] O eran ["q_proj", "v_proj", "k_proj"]?
- [ ] O ["attention.self.query", ...]?

La solución es: GUARDAR target_modules en train_config.json
y LEERLO en el endpoint, no hardcodearlo.
"""

# Print instructions
if __name__ == "__main__":
    print(__doc__)
