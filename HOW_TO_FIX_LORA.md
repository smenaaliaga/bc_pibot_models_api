"""
GUÍA: Cómo integrar el SharedEncoder corregido

El problema es que tu target_modules NO es consistente entre training e inferencia.

CAMBIOS NECESARIOS:
===================

1. En train_config.json: Agregar target_modules
2. En main.py: Pasar target_modules cuando creas SharedEncoder
3. En artifacts.py: Guardar y cargar target_modules desde train_config.json
4. En evaluate_command: Pasar target_modules al cargar

PASO A PASO
===========
"""

# ============================================================================
# CAMBIO 1: main.py
# ============================================================================
# En train_command(), actualizar creación de train_config:

"""
ANTES:
    train_config = TrainConfig(
        ...
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

DESPUÉS: Agregar target_modules a ArgParse y TrainConfig

ARGPARSE: En build_parser(), agregar a train_parser:
"""
# train_parser.add_argument(
#     "--lora-target-modules",
#     nargs="+",
#     default=["query", "value"],
#     help="Target modules for LoRA (e.g., query value key)"
# )

"""
TRAININGCONFIG: En TrainConfig dataclass, agregar:
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])

En train_command():
    train_config = TrainConfig(
        ...
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,  # ← NUEVO
    )

Cuando creas el encoder:
    encoder = SharedEncoder(
        model_name=train_config.model_name,
        device=train_config.device,
        use_lora=train_config.use_lora,
        lora_r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
        target_modules=train_config.target_modules,  # ← NUEVO
    )
"""

# ============================================================================
# CAMBIO 2: artifacts.py (save_artifacts y load_artifacts)
# ============================================================================

"""
save_artifacts() ya funciona bien porque usa asdict(train_config)
que ahora incluye target_modules.

PERO en load_artifacts(), necesitas pasar target_modules a SharedEncoder.load():
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import torch

from src.model.encoder import SharedEncoder
from src.model.multitask_model import MultiTaskClassifier


def load_artifacts(artifact_dir: Path, device: str = "cpu"):
    # Cargar configuración de entrenamiento
    with open(artifact_dir / "train_config.json", "r", encoding="utf-8") as file:
        train_config = json.load(file)
    
    use_lora = train_config.get("use_lora", False)
    lora_r = train_config.get("lora_r", 8)
    lora_alpha = train_config.get("lora_alpha", 16)
    lora_dropout = train_config.get("lora_dropout", 0.1)
    target_modules = train_config.get("target_modules", ["query", "value"])  # ← NUEVO
    
    # Cargar encoder con LoRA config CONSISTENTE
    encoder = SharedEncoder.load(
        str(artifact_dir / "encoder"),
        device=device,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,  # ← PASO LOS MISMOS target_modules
    )

    with open(artifact_dir / "label2id.json", "r", encoding="utf-8") as file:
        label2id = json.load(file)

    num_classes_by_task = {task: len(task_map) for task, task_map in label2id.items()}
    embedding_dim = encoder.encode(["_probe_"], convert_to_tensor=True).shape[-1]
    multitask_model = MultiTaskClassifier(
        embedding_dim=embedding_dim,
        num_classes_by_task=num_classes_by_task
    )
    multitask_model.load_state_dict(
        torch.load(artifact_dir / "heads.pt", map_location=device)
    )
    multitask_model.to(device)
    multitask_model.eval()

    with open(artifact_dir / "id2label.json", "r", encoding="utf-8") as file:
        raw_id2label = json.load(file)
    id2label = {
        task: {int(index): label for index, label in task_map.items()}
        for task, task_map in raw_id2label.items()
    }

    return encoder, multitask_model, label2id, id2label


# ============================================================================
# CAMBIO 3: main.py (evaluate_command)
# ============================================================================

"""
En evaluate_command(), cuando cargues el encoder directamente (sin load_artifacts):

ANTES:
    encoder = SharedEncoder.load(str(Path(args.artifact_dir) / "encoder"), device=args.device)

DESPUÉS: Leer train_config y pasar parámetros
"""

def evaluate_command(args) -> None:
    from src.config.schema import TrainConfig
    from src.train.trainer import MultiTaskTrainer
    from src.data.io_csv import load_task_data, build_label_maps
    from src.data.datasets import split_by_task
    from src.train.trainer import build_datasets_from_frames
    
    data_config_params = {
        'data_dir': Path(args.data_dir),
        'macro_file': args.macro_file,
        'intent_file': args.intent_file,
        'context_file': args.context_file,
        'text_col': args.text_col,
        'label_col': args.label_col,
        'val_size': args.val_size,
        'test_size': args.test_size,
        'random_state': args.seed,
    }
    data_config = DataConfig(**data_config_params)
    
    # Cargar train_config para obtener todos los parámetros
    artifact_dir = Path(args.artifact_dir)
    with open(artifact_dir / "train_config.json", "r", encoding="utf-8") as file:
        train_config_dict = json.load(file)
    
    # Crear encoder con TODOS los parámetros LoRA
    encoder = SharedEncoder.load(
        str(artifact_dir / "encoder"),
        device=args.device,
        use_lora=train_config_dict.get("use_lora", False),
        lora_r=train_config_dict.get("lora_r", 8),
        lora_alpha=train_config_dict.get("lora_alpha", 16),
        lora_dropout=train_config_dict.get("lora_dropout", 0.1),
        target_modules=train_config_dict.get("target_modules", ["query", "value"]),  # ← NUEVO
    )
    
    # Resto del código igual...


# ============================================================================
# SUMMARY: Cambios mínimos para que funcione
# ============================================================================

"""
1. Actualizar SharedEncoder (ya está en SHARED_ENCODER_FIXED.py)
   - Aceptar target_modules como parámetro
   - Usarlo consistentemente en save() y load()

2. En su TrainConfig dataclass:
   - Agregar campo: target_modules: List[str]

3. En main.py argparse:
   - Agregar --lora-target-modules

4. En train_command():
   - Pasar target_modules a SharedEncoder()

5. En load_artifacts():
   - Leer target_modules de train_config.json
   - Pasarlo a SharedEncoder.load()

6. En evaluate_command():
   - Leer target_modules de train_config.json
   - Pasarlo a SharedEncoder.load()

Resultado: train_config.json incluirá:
{
  "use_lora": true,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1,
  "target_modules": ["query", "value"],  # ← AHORA SÍ ESTÁ
}

Y todos los lugares de carga usarán el MISMO target_modules.

TESTING
=======
Después de cambios:

1. Entrenar:
   python main.py train --use-lora --lora-target-modules query value

2. Verificar train_config.json:
   cat models/artifacts/train_config.json
   # Debe incluir "target_modules": ["query", "value"]

3. Cargar y comparar:
   python tests/compare_training_vs_endpoint.py
   # Debe dar embeddings iguales (cosine > 0.95)
   # Debe dar predicciones iguales
"""
