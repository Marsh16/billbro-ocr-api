from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

model_name = "mychen76/invoice-and-receipts_donut_v1" 
tokenizer = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.to("cuda" if torch.cuda.is_available() else "cpu")