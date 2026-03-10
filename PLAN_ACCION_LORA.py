#!/usr/bin/env python
"""
🚀 PLAN DE ACCIÓN: Encontrar y corregir diferencia de resultados entre Training e Inferencia

El problema más probable: target_modules incorrecto en LoRA
"""

import sys

STEPS = """

╔════════════════════════════════════════════════════════════════════════════╗
║          DIFERENCIA ENTRE TRAINING E INFERENCIA CON LoRA                  ║
║                                                                            ║
║ Síntoma: Misma query, etiquetas completamente distintas                  ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ PASO 1: Verificar artefactos (2 minutos) ─────────────────────────────────┐
│                                                                             │
│ $ python tests/diagnose_lora_mismatch.py                                  │
│                                                                             │
│ ✅ Verifica:                                                              │
│   • Estructura de archivos                                                │
│   • train_config.json tiene todos los parámetros?                          │
│   • adapter_model.bin existe?                                             │
│   • adapter_config.json es válido?                                        │
│                                                                             │
│ 📌 Si falta algo → Re-entrenar y subir artefactos a HF                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ PASO 2: Comparar Training vs Endpoint (3 minutos) ───────────────────────┐
│                                                                             │
│ $ python tests/compare_training_vs_endpoint.py                            │
│                                                                             │
│ ✅ Verifica:                                                              │
│   • ¿Son iguales los embeddings? (cosine similarity > 0.95)               │
│   • ¿Las cabezas producen las mismas predicciones?                        │
│                                                                             │
│ 📍 Resultados posibles:                                                   │
│                                                                             │
│   Caso A: Embeddings DIFERENTES (< 0.95)                                  │
│   → PROBLEMA EN LoRA ENCODER                                              │
│   → Ver PASO 3                                                             │
│                                                                             │
│   Caso B: Embeddings IGUALES pero predicciones DIFERENTES                │
│   → PROBLEMA EN HEADS                                                      │
│   → Verificar: heads.pt, id2label.json, mapeo labels                      │
│                                                                             │
│   Caso C: TODO IGUAL                                                      │
│   → No hay problema, revisar código de training                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ PASO 3: Detectar target_modules correcto (5 minutos) ─────────────────────┐
│                                                                             │
│ $ python tests/find_target_modules.py                                     │
│                                                                             │
│ ✅ Muestra:                                                               │
│   • Todos los módulos disponibles en el encoder                            │
│   • Candidatos más probables para LoRA                                     │
│   • Intenta aplicar LoRA con cada candidato                               │
│                                                                             │
│ 🔴 PROBLEMA MÁS PROBABLE:                                                │
│    El endpoint usa target_modules = ["query", "key", "value"]            │
│    Pero tu training usó módulos DIFERENTES                                │
│                                                                             │
│    Ejemplo:                                                               │
│    • Training: ["q_proj", "v_proj", "k_proj"]                            │
│    • Endpoint: ["query", "key", "value"]                                 │
│    → ¡Incompatibles! Los pesos se aplican mal                            │
│                                                                             │
│ ✅ SOLUCIÓN:                                                              │
│    1. Identificar qué usaste en training                                   │
│    2. Guardar en train_config.json:                                       │
│       {                                                                    │
│         "use_lora": true,                                                 │
│         "lora_r": 16,                                                     │
│         "lora_alpha": 32,                                                 │
│         "target_modules": ["q_proj", "v_proj", "k_proj"],   ← ¡CRÍTICO!  │
│         ...                                                               │
│       }                                                                    │
│    3. Subir a HF Hub                                                      │
│    4. Test nuevamente                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ PASO 4: Verificar código de Training ────────────────────────────────────┐
│                                                                             │
│ Compartir este snippet exacto (reemplazar ???):                           │
│                                                                             │
│ # ============== CONFIG DE TRAINING ==================                    │
│ from peft import get_peft_model, LoraConfig                              │
│                                                                             │
│ lora_config = LoraConfig(                                                 │
│     r=????,              # Qué valor usaste?                              │
│     lora_alpha=????,     # Qué valor?                                     │
│     target_modules=????, # CRÍTICO: Qué módulos?                         │
│     lora_dropout=????,   # Qué valor?                                     │
│     task_type="FEATURE_EXTRACTION",                                      │
│     bias="none",                                                          │
│ )                                                                          │
│                                                                             │
│ # ============== GUARDAR CONFIG ========================                  │
│ train_config = {                                                          │
│     "use_lora": True,                                                     │
│     "lora_r": ???,       # Mismo valor que arriba                         │
│     "lora_alpha": ???,   # Mismo valor que arriba                         │
│     "lora_dropout": ???, # Mismo valor que arriba                         │
│     "target_modules": ???, # Mismo valor que arriba ¡¡¡                  │
│ }                                                                          │
│                                                                             │
│ # ============== GUARDAR ARTEFACTOS ====================                  │
│ peft_model.save_pretrained("artifacts/encoder/lora_weights")             │
│ encoder.save_pretrained("artifacts/encoder")                              │
│ # ❌ NO busques merge_and_unload() a menos que guardes en otro lugar      │
│                                                                             │
│ # ============== SUBIR A HF ============================                  │
│ huggingface-cli upload BCCh/pibot-intent-router artifacts/               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ PASO 5: Quick Test después de corregir ──────────────────────────────────┐
│                                                                             │
│ 1. Iniciar servidor:                                                       │
│    $ uvicorn app.main:app --reload                                        │
│                                                                             │
│ 2. En otra terminal:                                                       │
│    $ python tests/compare_training_vs_endpoint.py                         │
│                                                                             │
│ 3. Verificar logs del servidor para ver si LoRA se carga:                 │
│    INFO - Applying LoRA to encoder: r=X, alpha=Y, target_modules=[...]   │
│                                                                             │
│ 4. Comparar resultados con training                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║                         CHECKLIST DE VALIDACIÓN                           ║
╚════════════════════════════════════════════════════════════════════════════╝

Antes de re-subir a HF, verificar:

□ train_config.json tiene:
  □ "use_lora": true
  □ "lora_r": (número, ej: 16)
  □ "lora_alpha": (número, ej: 32)
  □ "target_modules": (lista, ej: ["query", "key", "value"])
  □ "lora_dropout": (float, ej: 0.1)

□ Estructura de archivos:
  □ encoder/config.json
  □ encoder/model.safetensors
  □ encoder/lora_weights/adapter_model.bin
  □ encoder/lora_weights/adapter_config.json
  □ heads.pt
  □ id2label.json
  □ train_config.json

□ Código de training:
  □ Guardó con peft_model.save_pretrained()
  □ No hizo merge_and_unload() (o lo guardó aparte)
  □ target_modules coincide con train_config.json

□ Resultados después de corregir:
  □ Embeddings similares (cosine > 0.95)
  □ Predicciones coinciden (macro, intent, context)
  □ Pesos y confianzas cerrados


╔════════════════════════════════════════════════════════════════════════════╗
║                      CONTACTO / SOPORTE                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Si después de estos pasos sigue sin funcionar:

1. Ejecutar los 3 scripts de diagnóstico
2. Compartir output de los scripts
3. Compartir código exacto de training con LoRA config
4. Verificar adapter_config.json en HF (¿tiene target_modules?)
"""

print(STEPS)
print("\n" + "=" * 80)
print("✅ Comenzar por PASO 1: $ python tests/diagnose_lora_mismatch.py")
print("=" * 80 + "\n")
