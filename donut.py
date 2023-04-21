import os
from PIL import Image
import json
from pathlib import Path
from transformers import DonutProcessor
import re

import torch
from transformers import VisionEncoderDecoderModel, TrainingArguments, Trainer
from datasets import Dataset




training_set_path = Path("receipts")
output_dir = "training_checkpoints"
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")



images = []
text = []
for file in os.listdir(training_set_path):
    if '.jpg' in file:
        images_open = Image.open(os.path.join(training_set_path, file)).convert("RGB")
        images.append(images_open)
    if '.json' in file:
        with (training_set_path / file).open(mode='r') as j:
            jsons = json.load(j)
            text.append(jsons)

dic = {'image': images,
       'text': text}
dataset = Dataset.from_dict(dic)

new_special_tokens = []  # new tokens which will be added to the tokenizer
task_start_token = "<s>"  # start of task token
eos_token = "</s>"  # eos token of tokenizer


def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                        fr"<s_{k}>"
                        + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj


def preprocess_documents_for_donut(sample):
    image = sample['image']
    text = sample["text"]
    d_doc = task_start_token + json2token(text) + eos_token
    # convert all images to RGB
    return {"image": image, "text": d_doc}


proc_dataset = dataset.map(preprocess_documents_for_donut)


# add new special tokens to tokenizer
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})

# we update some settings which differ from pretraining; namely the size of the images + no rotation required
# resizing the image to smaller sizes from [1920, 2560] to [960,1280]
processor.feature_extractor.size = [720,960] # should be (width, height)
processor.feature_extractor.do_align_long_axis = False


def transform_and_tokenize(sample, processor=processor, split="train", max_length=512, ignore_id=-100):
    # create tensor from image
    try:
        pixel_values = processor(
            sample["image"], random_padding=split == "train", return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}

    # tokenize document
    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token
    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}


# need at least 32-64GB of RAM to run this
processed_dataset = proc_dataset.map(transform_and_tokenize,remove_columns=["image","text"])

# Train test split
final_dataset = processed_dataset.train_test_split(test_size=0.1)


# Resize embedding layer to match vocabulary size
model.decoder.resize_token_embeddings(len(processor.tokenizer))
# Adjust our image size and output sequence lengths
model.config.encoder.image_size = processor.feature_extractor.size[::-1]  # (height, width)
model.config.decoder.max_length = len(max(final_dataset['train']["labels"], key=len))

# Add task token for decoder to start
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]



def train(data, m):

    NUM_TRAIN_EPOCHS = 20
    LEARNING_RATE = 4e-5
    os.environ["WANDB_DISABLED"] = 'true'

    for param in model.base_model.parameters():
        if param.requires_grad:
            param.requires_grad =True


    training_args = TrainingArguments(output_dir=output_dir,
                                      num_train_epochs=NUM_TRAIN_EPOCHS,
                                      # max_steps=1500,
                                      logging_strategy="epoch",
                                      save_total_limit=1,
                                      learning_rate=LEARNING_RATE,
                                      evaluation_strategy="epoch",
                                      save_strategy="epoch",
                                      # eval_steps=100,
                                      load_best_model_at_end=True,  ####### set to false; as untrained model might seem to perform best acc. to f1
                                        )

    # Initialize our Trainer
    trainer = Trainer(
        model=m,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"]
    )
    trainer.train()

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

train(final_dataset, model)


############################################################### EVAL ####################################################
processor = DonutProcessor.from_pretrained(output_dir)
model = VisionEncoderDecoderModel.from_pretrained(output_dir)

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load random document image from the test set
test_sample = final_dataset["train"][1]
def run_prediction(sample, model=model, processor=processor):
    # prepare inputs
    pixel_values = torch.tensor(sample["pixel_values"]).unsqueeze(0)
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # run inference
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = prediction.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    prediction = re.sub(r"<.*?>", "", prediction, count=1).strip()  # remove first task start token
    prediction = processor.token2json(prediction)

    # load reference target
    target = processor.token2json(test_sample["target_sequence"])
    return prediction, target

prediction, target = run_prediction(test_sample)
print(f"Reference:\n {target}")
print(f"Prediction:\n {prediction}")
