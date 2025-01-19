import torch
import base64
from PIL import Image
import numpy as np
import io

def generateTextInImage(processor,model,input_image,task_prompt="<s_receipt>"):
    pixel_values = processor(input_image, return_tensors="pt").pixel_values
    print ("input pixel_values: ",pixel_values.shape)
    task_prompt = "<s_receipt>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    outputs = model.generate(pixel_values.to(device),
                               decoder_input_ids=decoder_input_ids.to(device),
                               max_length=model.decoder.config.max_position_embeddings,
                               early_stopping=True,
                               pad_token_id=processor.tokenizer.pad_token_id,
                               eos_token_id=processor.tokenizer.eos_token_id,
                               use_cache=True,
                               num_beams=1,
                               bad_words_ids=[[processor.tokenizer.unk_token_id]],
                               return_dict_in_generate=True,
                               output_scores=True,)
    return outputs
    
def generateOutputXML(processor,model, input_image, task_start="<s_receipt>"):
    import re
    outputs=generateTextInImage(processor,model,input_image,task_prompt=task_start)
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip() 
    return sequence

def generateOutputJson(processor,model, input_image, task_start="<s_receipt>",task_end="</s_receipt>"):
    image_data_bytes = base64.b64decode(input_image)
    image = Image.open(io.BytesIO(image_data_bytes))
    receipt_image_array = np.array(image.convert('RGB'))
    xml = generateOutputXML(processor,model, receipt_image_array,task_start=task_start,task_end=task_end)
    result=processor.token2json(xml)
    print(":vampire:",result)
    return result