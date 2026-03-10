"""
Script para detectar los target_modules correctos en el encoder.

Útil para identificar qué módulos estaban siendo adaptados con LoRA durante training.
"""

import json
from pathlib import Path
import sys

try:
    import torch
except ImportError:
    print("❌ Requiere torch")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Requiere peft y sentence-transformers")
    sys.exit(1)


def analyze_model_structure(encoder_dir: str = "model_cache/router_artifacts/encoder"):
    """Analizar estructura del modelo para encontrar módulos válidos para LoRA."""
    
    encoder_path = Path(encoder_dir)
    print("🔍 Analizando estructura del encoder para encontrar target_modules válidos")
    print("=" * 80)
    
    # Cargar el encoder
    print(f"\nCargando encoder desde: {encoder_path}")
    encoder = SentenceTransformer(str(encoder_path), device="cpu")
    base_model = encoder[0].auto_model
    
    print(f"Modelo: {base_model.__class__.__name__}")
    print(f"Config: {base_model.config}")
    
    # Obtener todos los módulos del modelo
    print(f"\n📋 Módulos disponibles en el modelo:")
    print("-" * 80)
    
    all_modules = {}
    for name, module in base_model.named_modules():
        module_type = module.__class__.__name__
        if module_type not in all_modules:
            all_modules[module_type] = []
        all_modules[module_type].append(name)
    
    for module_type, names in sorted(all_modules.items()):
        print(f"\n{module_type}: ({len(names)} modules)")
        for name in sorted(names)[:5]:  # Mostrar solo los primeros 5
            print(f"  - {name}")
        if len(names) > 5:
            print(f"  ... y {len(names) - 5} más")
    
    # Buscar módulos de atención (comunes para LoRA)
    print(f"\n🎯 Módulos de atención encontrados (comunes para LoRA):")
    print("-" * 80)
    
    attention_modules = []
    for name, module in base_model.named_modules():
        if any(x in name.lower() for x in ["attention", "self", "query", "key", "value", "q_proj", "k_proj", "v_proj"]):
            attention_modules.append(name)
    
    if attention_modules:
        for name in sorted(attention_modules)[:20]:
            print(f"  - {name}")
        if len(attention_modules) > 20:
            print(f"  ... y {len(attention_modules) - 20} más")
    else:
        print("  ❌ No se encontraron módulos de atención")
    
    # Inferir target_modules más probables
    print(f"\n💡 Target modules más probables (por orden):")
    print("-" * 80)
    
    candidates = [
        (["query", "key", "value"], "Estándar (ModifyAttnHead)"),
        (["q_proj", "v_proj", "k_proj"], "Transformer estándar"),
        (["q_proj", "v_proj"], "Solo Q-V"),
        (["attention.self.query", "attention.self.value", "attention.self.key"], "Full path"),
    ]
    
    # Filtrar solo los que existen en el modelo
    valid_candidates = []
    for target_modules, desc in candidates:
        # Verificar si alguno existe
        exists = any(
            any(mod in name for name in attention_modules)
            for mod in target_modules
        )
        if exists or not attention_modules:  # Si no hay info, mostrar todos
            valid_candidates.append((target_modules, desc))
    
    for i, (target_modules, desc) in enumerate(valid_candidates, 1):
        print(f"{i}. {target_modules}")
        print(f"   Descripción: {desc}")
    
    # Probar aplicar LoRA con diferentes configs
    print(f"\n🧪 Probando aplicar LoRA con diferentes target_modules:")
    print("-" * 80)
    
    for i, (target_modules, desc) in enumerate(valid_candidates, 1):
        try:
            # Reload model
            base_model = encoder[0].auto_model
            
            # Crear config LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules=target_modules,
            )
            
            # Intentar aplicar
            peft_model = get_peft_model(base_model, lora_config)
            
            # Contar parámetros LoRA
            lora_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in peft_model.base_model.parameters())
            
            print(f"✅ Opción {i}: {target_modules}")
            print(f"   Parámetros LoRA: {lora_params:,} ({100.0 * lora_params / total_params:.2f}%)")
            
        except Exception as e:
            print(f"❌ Opción {i}: {target_modules}")
            print(f"   Error: {e}")
    
    # Recomendación
    print(f"\n📝 RECOMENDACIÓN:")
    print("-" * 80)
    print(f"""
Si los resultados en endpoint son muy diferentes del training:

1. Verificar en tu código de training exactamente qué target_modules usaste
2. Guardar en train_config.json:
   {{
       "target_modules": {valid_candidates[0][0] if valid_candidates else ["query", "key", "value"]},
       "use_lora": true,
       ...
   }}
3. Re-subirlo a HF Hub
4. Ejecutar endpoint nuevamente

Si no sabes qué target_modules usaste, probar las opciones listadas arriba
en orden hasta que los resultados coincidan con training.
    """)


if __name__ == "__main__":
    encoder_dir = sys.argv[1] if len(sys.argv) > 1 else "model_cache/router_artifacts/encoder"
    analyze_model_structure(encoder_dir)
