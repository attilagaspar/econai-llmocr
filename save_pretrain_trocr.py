from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
processor.save_pretrained("trocr_finetuned")