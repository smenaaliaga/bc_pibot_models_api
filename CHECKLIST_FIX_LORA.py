#!/usr/bin/env python
"""
CHECKLIST: Cambios para que LoRA funcione consistentemente

Tu problema: target_modules = ["query", "value"] en training,
pero el endpoint espera otro valor diferente.

Resultado: Etiquetas completamente distintas.

SOLUCIÓN: Hacer target_modules configurable y guardar en train_config.json
"""

CHECKLIST = """
╔════════════════════════════════════════════════════════════════════════════╗
║         CHECKLIST: Fix LoRA - 5 cambios simples                           ║
╚════════════════════════════════════════════════════════════════════════════╝

□ CAMBIO 1: Actualizar TrainConfig dataclass
  Archivo: src/config/schema.py
  ─────────────────────────────────────────────────────────────────────────
  from dataclasses import dataclass, field
  from typing import List
  
  @dataclass
  class TrainConfig:
      ...
      use_lora: bool = False
      lora_r: int = 8
      lora_alpha: int = 16
      lora_dropout: float = 0.1
      target_modules: List[str] = field(         # ← NUEVA LÍNEA
          default_factory=lambda: ["query", "value"]
      )
  ─────────────────────────────────────────────────────────────────────────

□ CAMBIO 2: Agregar argumento en CLI
  Archivo: main.py
  ─────────────────────────────────────────────────────────────────────────
  En build_parser(), dentro de train_parser:
  
  train_parser.add_argument(
      "--lora-target-modules",
      nargs="+",
      default=["query", "value"],
      help="Target modules for LoRA adaptation (e.g., query value key)"
  )
  ─────────────────────────────────────────────────────────────────────────

□ CAMBIO 3: Pasar target_modules en train_command()
  Archivo: main.py
  ─────────────────────────────────────────────────────────────────────────
  En train_command():
  
  train_config = TrainConfig(
      ...
      use_lora=args.use_lora,
      lora_r=args.lora_r,
      lora_alpha=args.lora_alpha,
      lora_dropout=args.lora_dropout,
      target_modules=args.lora_target_modules,  # ← AGREGAR
  )
  
  encoder = SharedEncoder(
      model_name=train_config.model_name,
      device=train_config.device,
      use_lora=train_config.use_lora,
      lora_r=train_config.lora_r,
      lora_alpha=train_config.lora_alpha,
      lora_dropout=train_config.lora_dropout,
      target_modules=train_config.target_modules,  # ← AGREGAR
  )
  ─────────────────────────────────────────────────────────────────────────

□ CAMBIO 4: Actualizar SharedEncoder (reemplazar class)
  Archivo: src/model/encoder.py
  ─────────────────────────────────────────────────────────────────────────
  Copiar clase desde: SHARED_ENCODER_FIXED.py
  
  Cambios principales:
  • Agregar parámetro target_modules
  • Pasar target_modules a LoraConfig
  • Usar load_adapter() en lugar de PeftModel.from_pretrained()
  ─────────────────────────────────────────────────────────────────────────

□ CAMBIO 5: Actualizar load_artifacts()
  Archivo: src/serialization/artifacts.py
  ─────────────────────────────────────────────────────────────────────────
  def load_artifacts(artifact_dir: Path, device: str = "cpu"):
      with open(artifact_dir / "train_config.json", "r", encoding="utf-8") as file:
          train_config = json.load(file)
      
      use_lora = train_config.get("use_lora", False)
      lora_r = train_config.get("lora_r", 8)
      lora_alpha = train_config.get("lora_alpha", 16)
      lora_dropout = train_config.get("lora_dropout", 0.1)
      target_modules = train_config.get("target_modules", ["query", "value"])  # ← AGREGAR
      
      encoder = SharedEncoder.load(
          str(artifact_dir / "encoder"),
          device=device,
          use_lora=use_lora,
          lora_r=lora_r,
          lora_alpha=lora_alpha,
          lora_dropout=lora_dropout,
          target_modules=target_modules,  # ← AGREGAR
      )
  ─────────────────────────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════════════════════════

DESPUÉS DE CAMBIOS: Entrenar
────────────────────────────────────────────────────────────────────────────

# Entrenar con target_modules explícitos:
python main.py train --use-lora --lora-target-modules query value

# O con otros módulos si los necesitas:
python main.py train --use-lora --lora-target-modules q_proj v_proj

# Verificar que train_config.json tiene target_modules:
cat models/artifacts/train_config.json | grep target_modules

═══════════════════════════════════════════════════════════════════════════════

VERIFICACIÓN: Comparar training vs endpoint
─────────────────────────────────────────────────────────────────────────────

# En terminal 1: Servidor de training corriendo
python -m pytest tests/compare_training_vs_endpoint.py

# Ahora los embeddings deben ser iguales (cosine similarity > 0.95)
# Las predicciones deben coincidir

═══════════════════════════════════════════════════════════════════════════════

NOTAS IMPORTANTES
─────────────────────────────────────────────────────────────────────────────

1. target_modules ["query", "value"] es lo que tienes AHORA
   Si quieres otros, actualiza en argparse y vuelve a entrenar

2. Una vez entrenes con target_modules correcto, train_config.json lo guardará
   
3. Cuando el endpoint cargue, leerá train_config.json y usará el MISMO target_modules
   
4. Resultado: Embeddings idénticos → etiquetas idénticas ✅

═══════════════════════════════════════════════════════════════════════════════

ARCHIVOS A MODIFICAR (RESUMEN)
─────────────────────────────────────────────────────────────────────────────

✏️  src/config/schema.py        - Agregar field target_modules
✏️  main.py                      - Agregar argumento y pasar a SharedEncoder
✏️  src/model/encoder.py         - Reemplazar con SHARED_ENCODER_FIXED.py
✏️  src/serialization/artifacts  - Leer y pasar target_modules

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(CHECKLIST)
