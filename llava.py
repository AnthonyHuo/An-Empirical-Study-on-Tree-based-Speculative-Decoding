import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
from llava_model import LlavaForConditionalGeneration,GraphInferenceEngineTG
from torch.nn.functional import softmax
from transformers.cache_utils import Cache
import torch
from utils import get_sampling_logits, _make_causal_mask
def prepare_inputs_for_generation(input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif 32000 in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )
        return model_inputs
# Load the model and processor
# model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
target_model =  GraphInferenceEngineTG(max_length=1024, model_name_or_path = "llava-hf/llava-1.5-7b-hf", dtype = torch.float16, device="cuda:0")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Load the JSON file
with open('/home/zhuominc/Sequoia_mingxiaohuo/instruct150k_subset.json', 'r') as file:
    data = json.load(file)

# Iterate over the first 200 labels
responses = []
for entry in tqdm(data[:2], desc="Processing labels"):
    # Construct the prompt
    prompt = f"USER: {entry['conversations'][0]['value']} ASSISTANT:"
    # prompt = entry['conversations'][0]['value']
    print(prompt)
    image_id = entry['id']  # Extract image ID
    image_path = f"/home/zhuominc/Sequoia_mingxiaohuo/train2014/COCO_train2014_{image_id.zfill(12)}.jpg"
    image = Image.open(image_path)

    # Process the prompt and image
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # # Generate response
    # generate_ids = model.generate(**inputs, max_new_tokens=64)
    # output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(output)
    inputs.update( {
    "pixel_values": inputs['pixel_values'].to('cuda:0'),
    "input_ids": inputs['input_ids'].to('cuda:0')
    })
    position_ids = torch.arange(1024).to('cuda:0').unsqueeze(0)
    storage_ids = torch.arange(1024).to('cuda:0')
    attn_mask = _make_causal_mask((1024, 1024), target_model.dtype, target_model.device)
    start_length = 0
    inner_decoding_step = 0
    while inner_decoding_step < 64:
        if inner_decoding_step == 0:
            inputs_embeds, attention_mask, position_id,storage_id = target_model.inference_image(input_ids = inputs['input_ids'], storage_ids=storage_ids[:start_length], pixel_values=inputs['pixel_values'],
                                                    position_ids = position_ids[..., :start_length], 
                                                    attn_mask=attn_mask[:start_length, :start_length][None, None, :, :])
            start_length = inputs_embeds.shape[1]
            # inputs['attn_mask'] = attn_mask[:start_length, :start_length][None, None, :, :]
            logits = target_model.inference(input_ids = None, inputs_embeds = inputs_embeds, storage_ids=storage_ids[:start_length], pixel_values=inputs['pixel_values'],
                                                    position_ids = position_ids[..., :start_length], 
                                                    attn_mask=attn_mask[:start_length, :start_length][None, None, :, :])[0][-1]
        # output_logits = output[0][-1]
        else:
            logits = target_model.inference(input_ids = inputs['input_ids'], inputs_embeds = None, storage_ids=storage_ids[start_length + inner_decoding_step-1 : start_length + inner_decoding_step],
                                                    position_ids = position_ids[..., start_length + inner_decoding_step-1 : start_length + inner_decoding_step], 
                                                    attn_mask=attn_mask[start_length + inner_decoding_step-1 : start_length + inner_decoding_step, :start_length + inner_decoding_step][None, None, :, :])[0][-1]
        p = softmax(logits / 0.01, dim=-1)
        new_token = p.multinomial(num_samples=1).unsqueeze(0)
        inner_decoding_step += 1
        inputs['input_ids'] = new_token
        # # Concatenate the tensors to get a tensor of shape [1, 29]
        # inputs['attn_mask']  = attn_mask[start_length + i + 1 -1 : start_length + i + 1, :start_length + i + 1][None, None, :, :]

        # new_position_id = 603 + i
        # new_position_id = torch.tensor(new_position_id).to(torch.long)
        # inputs['position_ids'] = new_position_id.unsqueeze(0).unsqueeze(0)
        # inputs['storage_ids'] = new_position_id.unsqueeze(0)
        output = processor.batch_decode(new_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output)
    target_model.clear_kv()
    # Store or display the output
    responses.append(output)

# You might want to save or further process `responses` depending on your needs
print(responses)
