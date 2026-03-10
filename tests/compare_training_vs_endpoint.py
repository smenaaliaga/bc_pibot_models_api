"""
Script de comparación: Training vs Endpoint Inference

Carga el modelo exactamente como en training y lo compara con el endpoint.
"""

import json
from pathlib import Path
from typing import Dict, Any
import sys

import torch
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Requirere sentence-transformers")
    sys.exit(1)

try:
    from peft import PeftModel, LoraConfig, get_peft_model
except ImportError:
    print("❌ Requiere peft")
    sys.exit(1)


def compare_inference_modes(
    artifacts_dir: str = "model_cache/router_artifacts",
    test_text: str = "cual fue el ultimo imacec",
):
    """Comparar predicciones entre modo training vs endpoint."""
    
    artifacts_path = Path(artifacts_dir)
    device = "cpu"
    
    print("🔬 COMPARACIÓN: Training Mode vs Endpoint Mode")
    print("=" * 80)
    
    # ========== CARGAR CONFIG ==========
    with open(artifacts_path / "train_config.json") as f:
        train_config = json.load(f)
    
    with open(artifacts_path / "id2label.json") as f:
        id2label_raw = json.load(f)
    
    use_lora = train_config.get("use_lora", False)
    print(f"\n📋 LoRA enabled: {use_lora}")
    
    # ========== MÉTODO 1: TRAINING MODE (Como lo haces en training) ==========
    print(f"\n1️⃣  MODO TRAINING (con LoRA aplicado directamente)")
    print("-" * 80)
    
    try:
        # Cargar encoder base
        encoder_training = SentenceTransformer(
            str(artifacts_path / "encoder"),
            device=device,
        )
        
        # Si usa LoRA, aplicar durante training
        if use_lora:
            print("   ℹ️  Aplicando LoRA como en training...")
            
            lora_r = int(train_config.get("lora_r", 8))
            lora_alpha = int(train_config.get("lora_alpha", 16))
            lora_dropout = float(train_config.get("lora_dropout", 0.1))
            target_modules = train_config.get("target_modules", ["query", "key", "value"])
            
            print(f"       r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            print(f"       target_modules={target_modules}")
            
            # Obtener modelo base
            base_model = encoder_training[0].auto_model
            
            # Crear config LoRA (como en training)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules=target_modules,
            )
            
            # Aplicar LoRA
            peft_model = get_peft_model(base_model, lora_config)
            
            # Cargar pesos LoRA guardados
            lora_weights_path = artifacts_path / "encoder" / "lora_weights" / "adapter_model.bin"
            if lora_weights_path.exists():
                print(f"   ✅ Cargando LoRA weights desde {lora_weights_path.name}")
                lora_state = torch.load(lora_weights_path, map_location=device)
                peft_model.load_state_dict(lora_state, strict=False)
                print(f"      Claves cargadas: {len(lora_state)}")
            else:
                print(f"   ⚠️  No se encontró {lora_weights_path}")
            
            # IMPORTANTE: El modelo de training probablemente tiene LoRA en eval mode
            peft_model.eval()
            encoder_training[0].auto_model = peft_model
        
        encoder_training.eval()
        
        # Hacer embedding (como en training)
        with torch.no_grad():
            embedding_training = encoder_training.encode(
                [test_text],
                convert_to_tensor=True,
                normalize_embeddings=False,  # ⚠️ Critical!
            )
        
        print(f"   ✅ Embedding shape: {embedding_training.shape}")
        print(f"   ✅ Embedding stats: min={embedding_training.min().item():.4f}, max={embedding_training.max().item():.4f}, mean={embedding_training.mean().item():.4f}")
        
    except Exception as e:
        print(f"   ❌ Error en training mode: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== MÉTODO 2: ENDPOINT MODE (Como lo hace el endpoint) ==========
    print(f"\n2️⃣  MODO ENDPOINT (como en router.py)")
    print("-" * 80)
    
    try:
        # Cargar encoder nuevamente (simular fresh load)
        from app.model.router import _apply_lora_to_encoder
        
        encoder_endpoint = SentenceTransformer(
            str(artifacts_path / "encoder"),
            device=device,
        )
        
        # Aplicar LoRA (como en endpoint)
        encoder_endpoint = _apply_lora_to_encoder(
            encoder=encoder_endpoint,
            train_config=train_config,
            artifact_dir=artifacts_path,
            device=device,
        )
        
        encoder_endpoint.eval()
        
        # Hacer embedding (como en endpoint)
        with torch.no_grad():
            embedding_endpoint = encoder_endpoint.encode(
                [test_text],
                convert_to_tensor=True,
                normalize_embeddings=False,
            )
        
        print(f"   ✅ Embedding shape: {embedding_endpoint.shape}")
        print(f"   ✅ Embedding stats: min={embedding_endpoint.min().item():.4f}, max={embedding_endpoint.max().item():.4f}, mean={embedding_endpoint.mean().item():.4f}")
        
    except Exception as e:
        print(f"   ❌ Error en endpoint mode: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== COMPARACIÓN DE EMBEDDINGS ==========
    print(f"\n3️⃣  COMPARACIÓN DE EMBEDDINGS")
    print("-" * 80)
    
    # Convertir a numpy para comparación
    emb_train_np = embedding_training.cpu().numpy().flatten()
    emb_endpoint_np = embedding_endpoint.cpu().numpy().flatten()
    
    # Diferencias
    diff = np.abs(emb_train_np - emb_endpoint_np)
    print(f"   Diferencia absoluta:")
    print(f"     - Mean: {diff.mean():.6f}")
    print(f"     - Max: {diff.max():.6f}")
    print(f"     - Min: {diff.min():.6f}")
    
    # Cosine similarity
    cos_sim = np.dot(emb_train_np, emb_endpoint_np) / (
        np.linalg.norm(emb_train_np) * np.linalg.norm(emb_endpoint_np)
    )
    print(f"   Cosine similarity: {cos_sim:.6f} (1.0 = idénticos)")
    
    if cos_sim < 0.95:
        print(f"\n   🔴 PROBLEMA DETECTADO: Embeddings muy diferentes!")
        print(f"      Esto indica que LoRA no se aplica igual")
    else:
        print(f"\n   ✅ Embeddings son similares")
    
    # ========== COMPARACIÓN DE HEADS ==========
    print(f"\n4️⃣  COMPARACIÓN DE HEADS (multitask)")
    print("-" * 80)
    
    try:
        # Cargar heads como en endpoint
        heads_path = artifacts_path / "heads.pt"
        heads_state = torch.load(heads_path, map_location=device)
        
        print(f"   Heads guardados: {list(heads_state.keys())}")
        
        # Hacer predicción con embedding
        with torch.no_grad():
            for task in ("macro", "intent", "context"):
                if task not in heads_state:
                    print(f"   ⚠️  Task '{task}' no encontrada en heads")
                    continue
                
                task_state = heads_state[task]
                weight = task_state.get("weight")
                bias = task_state.get("bias", torch.zeros(weight.shape[0]))
                
                if weight is None:
                    print(f"   ⚠️  No hay weight para {task}")
                    continue
                
                # Predicción training mode
                logits_train = torch.matmul(
                    embedding_training.to(device),
                    weight.to(device).transpose(0, 1)
                ) + bias.to(device)
                probs_train = torch.softmax(logits_train.squeeze(0), dim=-1)
                pred_train = probs_train.argmax().item()
                conf_train = probs_train[pred_train].item()
                
                # Predicción endpoint mode
                logits_endpoint = torch.matmul(
                    embedding_endpoint.to(device),
                    weight.to(device).transpose(0, 1)
                ) + bias.to(device)
                probs_endpoint = torch.softmax(logits_endpoint.squeeze(0), dim=-1)
                pred_endpoint = probs_endpoint.argmax().item()
                conf_endpoint = probs_endpoint[pred_endpoint].item()
                
                match = "✅" if pred_train == pred_endpoint else "❌"
                print(f"   {match} {task}:")
                print(f"      Training:  {pred_train} (conf: {conf_train:.4f})")
                print(f"      Endpoint:  {pred_endpoint} (conf: {conf_endpoint:.4f})")
                
                if abs(conf_train - conf_endpoint) > 0.05:
                    print(f"      ⚠️  Diferencia de confianza: {abs(conf_train - conf_endpoint):.4f}")
                
    except Exception as e:
        print(f"   ❌ Error comparando heads: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== RECOMENDACIONES ==========
    print("\n" + "=" * 80)
    print("🔧 RECOMENDACIONES")
    print("=" * 80)
    
    if cos_sim < 0.95:
        print("""
El problema está en la aplicación de LoRA.

Causas comunes:

1. 🔴 target_modules diferentes
   - En training: usaste otros módulos (ej: ["q_proj", "v_proj"])
   - En endpoint: se fija ["query", "key", "value"]
   - SOLUCIÓN: Guardar target_modules en train_config.json
   
2. 🔴 LoRA configuration incompleta
   - Faltan parámetros que no están en train_config.json
   - SOLUCIÓN: Exportar config completo desde peft durante training
   
3. 🔴 Versión de peft diferente
   - Cambió entre training e inferencia
   - SOLUCIÓN: Usar peft>=0.7 en ambos lados
        """)
    else:
        print("""
✅ Los embeddings son correctos.

El problema podría estar en:
1. Las cabezas (heads.pt)
2. Cómo se combina el resultado de cabezas
3. El mapeo id2label
        """)


if __name__ == "__main__":
    compare_inference_modes()
