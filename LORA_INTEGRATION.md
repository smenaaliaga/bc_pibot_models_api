# Guía de integración: Soporte LoRA para Router Multitarea

## Resumen

El endpoint ahora soporta modelos SetFit multitarea entrenados con **LoRA** (Low-Rank Adaptation), manteniendo compatibilidad completa con modelos antiguos sin LoRA.

---

## Estructura de artefactos

### Modelo sin LoRA (backward compatible)

```text
router_artifacts/
├── encoder/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── ...
├── heads.pt
├── id2label.json
├── label2id.json
└── train_config.json
```

### Modelo con LoRA (nuevo)

```text
router_artifacts/
├── encoder/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── lora_weights/             ← Nuevo
│   │   └── adapter_model.bin
│   │   └── adapter_config.json   ← Importante
│   └── ...
├── heads.pt
├── id2label.json
├── label2id.json
└── train_config.json                ← Incluye claves LoRA
```

---

## Configuración en `train_config.json`

### Ejemplo con LoRA

```json
{
  "max_length": 24,
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "target_modules": ["query", "key", "value"]
}
```

**⚠️ CRÍTICO**: El campo `target_modules` DEBE coincidir exactamente con lo usado durante training. Si no lo especificas, el endpoint asume `["query", "key", "value"]`.

### Ejemplo sin LoRA (backward compatible)

```json
{
  "max_length": 24
}
```

O explícitamente:

```json
{
  "max_length": 24,
  "use_lora": false
}
```

---

## Parámetros LoRA

| Parámetro | Tipo | Descripción | Default | ⚠️ Critical |
|-----------|------|-------------|---------|-----------|
| `use_lora` | `bool` | Habilita carga de LoRA weights | `false` | ❌ |
| `lora_r` | `int` | Rank de las matrices LoRA | `8` | ❌ |
| `lora_alpha` | `int` | Scaling factor de LoRA | `16` | ❌ |
| `lora_dropout` | `float` | Dropout en capas LoRA | `0.1` | ❌ |
| `target_modules` | `list[str]` | Módulos a adaptar con LoRA | `["query", "key", "value"]` | 🔴 **SÍ** |

### ⚠️ IMPORTANTE: target_modules

Si durante training usaste:
- `["query", "key", "value"]` → Guardar eso
- `["q_proj", "v_proj", "k_proj"]` → Guardar eso
- `["attention.self.query", ...]` → Guardar eso

**Si no coincide, los LoRA weights se aplican a los módulos equivocados y obtendrás resultados completamente diferentes.**

---

## Troubleshooting: Resultados diferentes entre Training e Inferencia

### Síntoma: Misma query, etiquetas completamente distintas

**Causas más probables (en orden de probabilidad):**

#### 1. 🔴 `target_modules` mismatch (90% de los casos)

Durante training probablemente usaste módulos diferentes:

```python
# TRAINING - Ejemplo de config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # ← DIFERENTE del default
    ...
)
```

**Endpoint asume:** `["query", "key", "value"]`  
**Training usó:** `["q_proj", "v_proj"]`

**Resultado:** Los pesos LoRA se aplican a los módulos equivocados.

**SOLUCIÓN:**
```json
{
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "v_proj"],
  "max_length": 24
}
```

#### 2. 🟡 Encoder base cambió

El modelo base descargado de HF tiene versión diferente:

```python
# TRAINING
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2@revision_A")

# ENDPOINT (descarga revisión diferente)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # @revision_B
```

**SOLUCIÓN:** Fixear revisión en train_config.json:
```json
{
  "encoder_model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "encoder_revision": "main",
  "use_lora": true,
  ...
}
```

#### 3. 🟡 LoRA weights guardados incorrectamente

No se guardaron con `peft.save_pretrained()`:

```python
# ❌ INCORRECTO - Guarda solo los pesos base
torch.save(base_model.state_dict(), "weights.pt")

# ✅ CORRECTO - Guarda los pesos LoRA
peft_model.save_pretrained("encoder/lora_weights")
```

**SOLUCIÓN:** Re-entrenar con:
```python
peft_model.save_pretrained("path/encoder/lora_weights")
```

#### 4. 🟡 Normalización de embeddings diferente

Durante training normalizabas embeddings, en endpoint no:

```python
# TRAINING
embedding = encoder.encode(text, normalize_embeddings=True)

# ENDPOINT
embedding = encoder.encode(text, normalize_embeddings=False)
```

**SOLUCIÓN:** Verificar y hacer consistente en train_config.json:
```json
{
  "normalize_embeddings": false,
  ...
}
```

#### 5. 🟡 peft version mismatch

```
Training: peft==0.6.2
Endpoint: peft==0.8.0
```

**SOLUCIÓN:** Fixear peft a versión específica:
```bash
pip install peft==0.8.0  # Ambos lados
```

---

## Diagnóstico paso a paso

### 1. Verificar estructura de archivos

```bash
python tests/diagnose_lora_mismatch.py
```

### 2. Comparar training vs endpoint

```bash
python tests/compare_training_vs_endpoint.py
```

### 3. Si el embedding es diferente

El problema está en la carga de LoRA o encoder base. Revisar:
- `target_modules` en train_config.json
- Encoder model_name exacto
- peft version

### 4. Si el embedding es igual pero predicción es diferente

El problema está en las heads. Revisar:
- Los pesos de las heads se guardaron correctamente
- El mapeo id2label es correcto

---

## Preparar artefactos con LoRA (Correctamente)

### 1. Entrenar modelo con LoRA

```python
from peft import get_peft_model, LoraConfig
from sentence_transformers import SentenceTransformer
import torch

# Configuración EXACTA que debes guardar en train_config.json
LORA_CONFIG_DICT = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["query", "key", "value"],  # CRÍTICO!
}

# Cargar encoder base
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
base_model = encoder[0].auto_model

# Aplicar LoRA
lora_config = LoraConfig(
    r=LORA_CONFIG_DICT["r"],
    lora_alpha=LORA_CONFIG_DICT["lora_alpha"],
    lora_dropout=LORA_CONFIG_DICT["lora_dropout"],
    bias="none",
    task_type="FEATURE_EXTRACTION",
    target_modules=LORA_CONFIG_DICT["target_modules"],
)

peft_model = get_peft_model(base_model, lora_config)

# ... entrenar ...
# peft_model.train()
# ... fine-tuning con datos ...

# Guardar encoder (sin LoRA aplicado)
encoder.save_pretrained("artifacts/encoder")

# Guardar LoRA weights
peft_model.save_pretrained("artifacts/encoder/lora_weights")

# IMPORTANTE: Guardar la configuración exacta
import json
config = {
    "max_length": 24,
    "use_lora": True,
    "lora_r": LORA_CONFIG_DICT["r"],
    "lora_alpha": LORA_CONFIG_DICT["lora_alpha"],
    "lora_dropout": LORA_CONFIG_DICT["lora_dropout"],
    "target_modules": LORA_CONFIG_DICT["target_modules"],  # CRÍTICO!
    "normalize_embeddings": False,
    "encoder_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "encoder_revision": "main",
}

with open("artifacts/train_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

### 2. Verificar que LoRA se guardó correctamente

```bash
ls artifacts/encoder/lora_weights/
# Debe tener:
# - adapter_config.json
# - adapter_model.bin

cat artifacts/encoder/lora_weights/adapter_config.json
# Verifica que target_modules coincida
```

### 3. Guardar cabezas multitarea

```python
heads_state = {
    "macro": {
        "weight": macro_head.weight.data,
        "bias": macro_head.bias.data,
    },
    "intent": {
        "weight": intent_head.weight.data,
        "bias": intent_head.bias.data,
    },
    "context": {
        "weight": context_head.weight.data,
        "bias": context_head.bias.data,
    },
}

torch.save(heads_state, "artifacts/heads.pt")
```

### 4. Subir a Hugging Face Hub

```bash
huggingface-cli upload BCCh/pibot-intent-router \
  artifacts/ \
  --repo-type model
```

---

## Logs para verificar LoRA carga correcta

### Carga correcta con LoRA

```
INFO - Applying LoRA to encoder: r=16, alpha=32, dropout=0.10, target_modules=['query', 'key', 'value']
INFO - Successfully loaded LoRA weights from model_cache/router_artifacts/encoder/lora_weights/adapter_model.bin
INFO - RouterBundle loaded from HF repo 'BCCh/pibot-intent-router' on device 'cpu'
```

### Carga sin LoRA (backward compatible)

```
INFO - LoRA not enabled in train_config. Loading encoder without LoRA.
INFO - RouterBundle loaded from HF repo 'BCCh/pibot-intent-router' on device 'cpu'
```

### Problema: target_modules incorrecto

Resultarán en predicciones completamente diferentes. Revisar logs y comparar training vs endpoint.

---

## Testing

### Ejecutar tests de LoRA

```bash
pytest tests/test_lora_router.py -v
```

### Comparar training vs endpoint

```bash
python tests/compare_training_vs_endpoint.py
```

### Diagnosticar artefactos

```bash
python tests/diagnose_lora_mismatch.py
```

---

## Soporte

Si los resultados siguen siendo diferentes:

1. ✅ Ejecutar `diagnose_lora_mismatch.py` - Verificar estructura
2. ✅ Ejecutar `compare_training_vs_endpoint.py` - Ver dónde difiere
3. ✅ Compartir output de ambos scripts
4. ✅ Compartir código de training con LoRA config exacto
5. ✅ Verificar `train_config.json` tiene `target_modules` correcto
