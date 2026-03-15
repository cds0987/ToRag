from .basemodel import BaseModel
from transformers import AutoTokenizer, AutoModel
import warnings
from .utils import get_device
from typing import List, Union, Optional
import torch        
class jinaitokenencode(BaseModel):
    def __init__(
        self,
        model_name, **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = get_device()
        self.model.to(self.device)

        warnings.filterwarnings("ignore")

        # flags to prevent repeated prints
        self._error_printed = False
        self._task_warning_printed = False

    def tokenencode(self, texts: Union[str, List[str]], **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        try:
            max_length = kwargs.get("max_length", 512)
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            task = kwargs.get("task", "retrieval.query")

            try:
                task_id = self.model._adaptation_map[task]

                adapter_mask = torch.full(
                    (len(texts),),
                    task_id,
                    dtype=torch.int32,
                    device=self.device
                )

            except Exception:
                if not self._task_warning_printed:
                    print(f"Task is not supported for lager v3 model.")
                    self._task_warning_printed = True
                adapter_mask = None

            with torch.inference_mode():
                model_output = (
                    self.model(**inputs, adapter_mask=adapter_mask)
                    if adapter_mask is not None
                    else self.model(**inputs)
                )

            embeddings = model_output[0].float().cpu().numpy()

        except Exception as e:
            if not self._error_printed:
                print(f"Error encoding documents: {e}")
                self._error_printed = True
            embeddings = None

        return embeddings
        