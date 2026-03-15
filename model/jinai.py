from .basemodel import BaseModel
from transformers import AutoTokenizer, AutoModel
import warnings
from .utils import get_device
from typing import List, Union, Optional
import torch
import numpy as np 
from tqdm.auto import tqdm       
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

        batch_size = kwargs.get("batch_size", 32)
        max_length = kwargs.get("max_length", 512)
        task = kwargs.get("task", "retrieval.query")
        show_progress = kwargs.get("show_progress", True)

        all_embeddings = []

        try:

            try:
                task_id = self.model._adaptation_map[task]
            except Exception:
                if not self._task_warning_printed:
                    print("Task is not supported for lager v3 model.")
                    self._task_warning_printed = True
                task_id = None

            iterator = range(0, len(texts), batch_size)

            if show_progress:
                iterator = tqdm(
                    iterator,
                    desc="Encoding",
                    unit="batch",
                    colour="green",
                    dynamic_ncols=True,
                    total=(len(texts) + batch_size - 1) // batch_size
                )

            for start in iterator:

                batch_texts = texts[start:start + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                adapter_mask = None
                if task_id is not None:
                    adapter_mask = torch.full(
                        (len(batch_texts),),
                        task_id,
                        dtype=torch.int32,
                        device=self.device
                    )

                with torch.inference_mode():

                    model_output = (
                        self.model(**inputs, adapter_mask=adapter_mask)
                        if adapter_mask is not None
                        else self.model(**inputs)
                    )

                embeddings = model_output[0].float().cpu().numpy()

                all_embeddings.append(embeddings)

            embeddings = np.vstack(all_embeddings)

        except Exception as e:

            if not self._error_printed:
                print(f"Error encoding documents: {e}")
                self._error_printed = True

            embeddings = None

        return embeddings

        