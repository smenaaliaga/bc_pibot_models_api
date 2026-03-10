"""
Script de diagnóstico para verificar si los LoRA weights se cargan correctamente.

Compara:
- Estado de los weights en training vs en el endpoint
- Configuración de LoRA en training vs en carga
- Diferencias en la salida del modelo
"""

import json
from pathlib import Path
import sys

try:
    import torch
    import numpy as np
except ImportError:
    print("❌ Requiere torch instalado")
    sys.exit(1)


def diagnose_lora_mismatch(artifacts_dir: str = "model_cache/router_artifacts", encoder_repo_id: str = None):
    """Diagnosticar por qué los LoRA weights dan resultados diferentes."""
    
    artifacts_path = Path(artifacts_dir)
    print("🔍 Diagnóstico de LoRA - Verificar carga correcta")
    print("=" * 80)
    
    # 1. Verificar estructura de archivos
    print("\n1️⃣  ESTRUCTURA DE ARCHIVOS")
    print("-" * 80)
    
    required_files = {
        "train_config": artifacts_path / "train_config.json",
        "encoder_config": artifacts_path / "encoder" / "config.json",
        "lora_weights_dir": artifacts_path / "encoder" / "lora_weights",
        "lora_adapter_config": artifacts_path / "encoder" / "lora_weights" / "adapter_config.json",
        "lora_weights": artifacts_path / "encoder" / "lora_weights" / "adapter_model.bin",
        "heads": artifacts_path / "heads.pt",
    }
    
    for name, file_path in required_files.items():
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {file_path}")
    
    # 2. Leer train_config
    print("\n2️⃣  CONFIGURACIÓN DE TRAINING (train_config.json)")
    print("-" * 80)
    
    try:
        with open(artifacts_path / "train_config.json") as f:
            train_config = json.load(f)
        
        print(json.dumps(train_config, indent=2))
        
        # Extraer parámetros LoRA
        use_lora = train_config.get("use_lora", False)
        if not use_lora:
            print("\n⚠️  use_lora=false. LoRA NO está habilitado.")
            return
        
        print(f"\n✅ LoRA habilitado:")
        print(f"   - lora_r: {train_config.get('lora_r', 8)}")
        print(f"   - lora_alpha: {train_config.get('lora_alpha', 16)}")
        print(f"   - lora_dropout: {train_config.get('lora_dropout', 0.1)}")
        print(f"   - target_modules: {train_config.get('target_modules', 'NO ESPECIFICADO')}")
        
    except Exception as e:
        print(f"❌ Error leyendo train_config: {e}")
        return
    
    # 3. Verificar adapter_config en lora_weights
    print("\n3️⃣  CONFIGURACIÓN DE LoRA (adapter_config.json)")
    print("-" * 80)
    
    try:
        lora_config_path = artifacts_path / "encoder" / "lora_weights" / "adapter_config.json"
        with open(lora_config_path) as f:
            lora_config = json.load(f)
        
        print(json.dumps(lora_config, indent=2))
        
        # Comparar con train_config
        print("\n⚠️  VERIFICACIÓN DE CONSISTENCIA:")
        
        lora_r_config = lora_config.get("r")
        lora_r_train = train_config.get("lora_r", 8)
        
        lora_alpha_config = lora_config.get("lora_alpha")
        lora_alpha_train = train_config.get("lora_alpha", 16)
        
        target_modules_config = lora_config.get("target_modules", [])
        target_modules_train = train_config.get("target_modules", ["query", "key", "value"])
        
        print(f"   r: {lora_r_config} vs train_config: {lora_r_train} - {'✅' if lora_r_config == lora_r_train else '❌'}")
        print(f"   lora_alpha: {lora_alpha_config} vs train_config: {lora_alpha_train} - {'✅' if lora_alpha_config == lora_alpha_train else '❌'}")
        print(f"   target_modules: {target_modules_config}")
        print(f"   target_modules vs train_config: {target_modules_train or 'NO ESPECIFICADO'} - {'✅' if target_modules_config == target_modules_train else '❌'}")
        
    except Exception as e:
        print(f"❌ Error leyendo adapter_config.json: {e}")
        print("\n⚠️  PROBABLE CAUSA: LoRA weights fueron salvados con librería vieja de peft")
        print("   Solución: Re-entrene y guarde con peft>=0.7")
    
    # 4. Verificar tamaño de weights
    print("\n4️⃣  TAMAÑO DE WEIGHTS")
    print("-" * 80)
    
    try:
        lora_weights_path = artifacts_path / "encoder" / "lora_weights" / "adapter_model.bin"
        if lora_weights_path.exists():
            size_mb = lora_weights_path.stat().st_size / (1024 * 1024)
            print(f"   adapter_model.bin: {size_mb:.2f} MB")
            
            # Cargar y verificar estructura
            lora_state = torch.load(lora_weights_path, map_location="cpu")
            print(f"   Claves en adapter_model.bin: {list(lora_state.keys())[:10]}")  # Primeras 10
            print(f"   Total de claves: {len(lora_state)}")
            
            # Ejemplo de shape
            if lora_state:
                first_key = list(lora_state.keys())[0]
                first_tensor = lora_state[first_key]
                print(f"   Ejemplo tensor ({first_key}): {first_tensor.shape}")
        else:
            print(f"   ❌ adapter_model.bin no encontrado en {lora_weights_path}")
            
    except Exception as e:
        print(f"   ❌ Error verificando weights: {e}")
    
    # 5. Verificar encoder base
    print("\n5️⃣  VERIFICACIÓN DEL ENCODER BASE")
    print("-" * 80)
    
    try:
        encoder_config_path = artifacts_path / "encoder" / "config.json"
        with open(encoder_config_path) as f:
            encoder_config = json.load(f)
        
        model_name = encoder_config.get("_name_or_path", "DESCONOCIDO")
        print(f"   Model: {model_name}")
        print(f"   Hidden size: {encoder_config.get('hidden_size', 'N/A')}")
        print(f"   Num heads: {encoder_config.get('num_attention_heads', 'N/A')}")
        print(f"   Num layers: {encoder_config.get('num_hidden_layers', 'N/A')}")
        
        print(f"\n   ℹ️  Si cambió el modelo base, todo fallará")
        print(f"   El encoder base debe ser EXACTAMENTE el mismo que en training")
        
    except Exception as e:
        print(f"   ❌ Error leyendo encoder config: {e}")
    
    # 6. Recomendaciones
    print("\n6️⃣  RECOMENDACIONES")
    print("-" * 80)
    
    print("""
¿Los LoRA weights dan resultados diferentes?

Causas más comunes:

1. ❌ target_modules incorrecto
   → El endpoint usa ["query", "key", "value"]
   → En training probablemente fue diferente
   → Solución: Verificar qué target_modules se usó en training
   → Guardar en train_config.json

2. ❌ Encoder base cambió
   → Versión diferente de sentence-transformers
   → Modelo descargado de HF pero con revisión distinta
   → Solución: Verificar modelo_name en encoder/config.json coincida
   → Usar revision="main" en train_config.json

3. ❌ LoRA weights guardados mal
   → Durante training: save_pretrained() desde peft model
   → En HF: asegurar que adapter_config.json sea válido
   → Solución: Re-entrenar y verificar adapter_config.json

4. ❌ Modo eval() incorrecto
   → El encoder debe estar en .eval() para inference
   → Dropout debe estar deshabilitado
   → Se está fijando en router.py pero podría fallar

5. ❌ Device/dtype mismatch
   → Entrenar en float32, cargar en float16 (o viceversa)
   → Solución: Forzar float32
    """)
    
    print("\n" + "=" * 80)
    print("✅ Diagnóstico completado. Revisa los puntos ❌ anteriores.")
    print("\nPróximo paso: Ejecutar comparación con test_lora_inference.py")


if __name__ == "__main__":
    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else "model_cache/router_artifacts"
    diagnose_lora_mismatch(artifacts_dir)
