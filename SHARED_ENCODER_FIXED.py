"""
SharedEncoder corregido: LoRA consistente entre training e inferencia.

Cambios principales:
1. target_modules se lee desde train_config.json
2. Se guarda correctamente con peft.save_pretrained()
3. Se carga sin conflictos
"""

from pathlib import Path
from typing import List, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging


class SharedEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,  # ← NUEVO
    ):
        hf_logging.set_verbosity_error()
        self.device = device
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["query", "value"]  # ← Default explícito
        
        self.model = SentenceTransformer(model_name, device=device)
        self.model.to(device)
        
        if use_lora:
            self._apply_lora()

    def _apply_lora(self) -> None:
        """Aplicar LoRA al transformer subyacente del modelo."""
        # Obtener el transformer dentro de SentenceTransformer
        transformer_module = self.model[0]
        
        if not hasattr(transformer_module, 'auto_model'):
            raise RuntimeError("Transformer module no encontrado en SentenceTransformer")
        
        base_model = transformer_module.auto_model
        
        # Configurar LoRA (DEBE coincidir exactamente entre training e inferencia)
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,  # ← CRÍTICO: debe ser consistente
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=None,
        )
        
        # Aplicar LoRA al modelo
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()
        
        # Reemplazar el modelo base con la versión LoRA
        transformer_module.auto_model = peft_model

    def encode(self, texts: List[str], convert_to_tensor: bool = True) -> torch.Tensor:
        """Encode texts (inference mode)."""
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(texts, convert_to_tensor=convert_to_tensor)

    def encode_for_training(self, texts: List[str]) -> torch.Tensor:
        """Encode texts (training mode, con gradientes)."""
        tokenized = self.model.tokenize(texts)
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        output = self.model(tokenized)
        return output["sentence_embedding"]

    def save(self, path: str) -> None:
        """Guardar encoder + LoRA weights (si aplica)."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        if self.use_lora:
            # Obtener el modelo PEFT y guardar LoRA weights
            transformer_module = self.model[0]
            if hasattr(transformer_module.auto_model, 'save_pretrained'):
                # Guardar solo los pesos LoRA en lora_weights/
                lora_path = path_obj / "lora_weights"
                transformer_module.auto_model.save_pretrained(str(lora_path))
                print(f"LoRA weights guardados en: {lora_path}")
        
        # Guardar el modelo base de SentenceTransformer
        self.model.save(str(path_obj))
        print(f"Encoder guardado en: {path_obj}")

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,  # ← NUEVO
    ):
        """Cargar encoder con LoRA weights (si aplica)."""
        hf_logging.set_verbosity_error()
        
        path_obj = Path(path)
        
        # Crear instancia nuevo
        instance = cls.__new__(cls)
        instance.device = device
        instance.use_lora = use_lora
        instance.lora_r = lora_r
        instance.lora_alpha = lora_alpha
        instance.lora_dropout = lora_dropout
        instance.target_modules = target_modules or ["query", "value"]
        
        # Cargar modelo base
        instance.model = SentenceTransformer(str(path_obj), device=device)
        instance.model.to(device)
        
        if use_lora:
            # Obtener el transformer
            transformer_module = instance.model[0]
            base_model = transformer_module.auto_model
            
            # Aplicar configuración de LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=instance.target_modules,  # ← USO EL MISMO target_modules
                lora_dropout=lora_dropout,
                bias="none",
                task_type=None,
            )
            peft_model = get_peft_model(base_model, lora_config)
            
            # Cargar LoRA weights si existen
            lora_path = path_obj / "lora_weights"
            if lora_path.exists():
                try:
                    # Cargar los pesos LoRA que se guardaron con save_pretrained
                    peft_model.load_adapter(str(lora_path), adapter_name="default")
                    print(f"LoRA weights cargados desde: {lora_path}")
                except Exception as e:
                    print(f"Advertencia: No se pudieron cargar pesos de LoRA: {e}")
            else:
                print(f"Advertencia: No se encontraron LoRA weights en {lora_path}")
            
            # Reemplazar el modelo base
            transformer_module.auto_model = peft_model
            peft_model.print_trainable_parameters()
        
        instance.model.eval()
        return instance
