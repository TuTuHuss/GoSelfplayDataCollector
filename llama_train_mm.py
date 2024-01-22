import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from einops.layers.torch import Rearrange

import evaluate
from transformers import TrainingArguments, Trainer, LlamaTokenizer, LlamaForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_int8_training
from transformers import TrainerCallback

random.seed(0)


def preprocess_data(tokenizer, prompt, img, transform):
    # Prompt before the image.
    pre_prompt = tokenizer.encode(
        'Here is a board of Go game: <img>',
        return_tensors="pt",
        add_special_tokens=True  # bos token should be added.
    )
    # Prompt after the image.
    post_prompt = tokenizer.encode(
        '<\\img> Here is the corresponding explanation: ',
        return_tensors='pt',
        add_special_tokens=False  # bos token should not be added.
    )

    expl = tokenizer.encode(
        prompt + tokenizer.eos_token,
        return_tensors='pt',
        add_special_tokens=False  # bos token should not be added.
    )

    visual_preprocessed = transform(img)

    return {
        'image': visual_preprocessed,
        'pre_prompt': pre_prompt,
        'post_prompt': post_prompt,
        'expl': expl
    }


class GoMMDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_data = self.data[index]
        img, sentence = item_data['board'], item_data['expl']

        # Tokenize and embed mm data.
        inputs = preprocess_data(self.tokenizer, prompt=sentence,
                                 img=img, transform=self.transform)

        return inputs


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            outputs = model(**inputs)
            return outputs.loss, outputs

        outputs = model(**inputs)
        return outputs.loss


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


def compute_metrics(pred):
    rouge = evaluate.load('rouge')

    # Convert token ids into string.
    labels_ids = pred.label_ids[..., 1:]
    pred_ids = pred.predictions[0][..., :-1]
    for id, pred in enumerate(pred_ids):
        pred_ids[id][labels_ids[id] == -100] = 2
        pred_ids[id][pred_ids[id] == -100] = 2
        labels_ids[id][labels_ids[id] == -100] = 2
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Calculate metrics.
    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )

    acc_count = 0
    for pred, label in zip(pred_str, label_str):
        if pred == label:
            acc_count += 1

    res_dict = {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
        "acc": round(acc_count / len(label_str), 4)
    }

    return res_dict


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def prepare_dataset(tokenizer):
    with open(f'dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    train_data = data[:-int(len(data) / 5)]
    test_data = data[-int(len(data) / 5):]

    return GoMMDataset(train_data, tokenizer), GoMMDataset(test_data, tokenizer)


def collate_fn(batch):
    bsz = len(batch)
    vision_t = (IMG_SIZE // PATCH_SIZE) * (IMG_SIZE // PATCH_SIZE)

    seq_lens = [batch[i]['pre_prompt'].shape[1] + batch[i]['post_prompt'].shape[1]
                + batch[i]['expl'].shape[1] + vision_t for i in range(bsz)]
    max_len = max(seq_lens)

    res = {'pre_prompt': [], 'post_prompt': [], 'expl': [], 'image': [], 'labels': []}
    for i in range(bsz):
        if seq_lens[i] < max_len:
            paddings = batch[i]['pre_prompt'].new_full((1, max_len - seq_lens[i]), 0)
            res['pre_prompt'].append(torch.cat([paddings, batch[i]['pre_prompt']], dim=1))
        else:
            res['pre_prompt'].append(batch[i]['pre_prompt'])
        res['post_prompt'].append(batch[i]['post_prompt'])
        res['expl'].append(batch[i]['expl'])
        res['image'].append(batch[i]['image'])

        label_mask = res['expl'][-1].new_full(
            (res['pre_prompt'][-1].shape[1] + vision_t + res['post_prompt'][-1].shape[1],), -100
        )
        res['labels'].append(torch.cat([label_mask, res['expl'][-1].squeeze(0)], dim=0))
    res['labels'] = torch.stack(res['labels'], dim=0)
    res['image'] = torch.stack(res['image'], dim=0)

    labels = res.pop('labels')
    return {'inputs': res, 'labels': labels}


class VisionLanguageModel(nn.Module):
    def __init__(self, language_model, lm_tokenizer, vision_model=None, patch_size=32, img_size=224):
        super(VisionLanguageModel, self).__init__()
        self.language_model = language_model
        self.tokenizer = lm_tokenizer
        self.patch_size = patch_size
        self.img_size = img_size

        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_size * patch_size, self.language_model.config.hidden_size)
            )

    def forward(self, inputs, labels):
        bsz = inputs['image'].shape[0]
        vision_embeds = self.vision_model(inputs['image'])
        embedded_inputs = []
        for i in range(bsz):
            # language_model: peft model; language_model.model: LlamaForCausalLM; language_model.model: LlamaModel
            pre_embed = self.language_model.model.model.embed_tokens(inputs['pre_prompt'][i]).squeeze(0)
            post_embed = self.language_model.model.model.embed_tokens(inputs['post_prompt'][i]).squeeze(0)
            expl_embed = self.language_model.model.model.embed_tokens(inputs['expl'][i]).squeeze(0)

            embedded_inputs.append(torch.cat([pre_embed, vision_embeds[i], post_embed, expl_embed], dim=0))

        model_inputs = {
            'inputs_embeds': torch.stack(embedded_inputs, dim=0),
            'labels': labels
        }
        return self.language_model(**model_inputs)


if __name__ == '__main__':
    # Initialize base model.
    base_model = '/mnt/nfs/whl/LLM/llama-2-7b-hf'
    MODE = 'mm'
    IMG_SIZE = 224
    PATCH_SIZE = 32

    # Initialize base model.
    tokenizer = LlamaTokenizer.from_pretrained(base_model, trust_remote_code=True)
    llama_model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, revision='main',
                                                   device_map='auto')

    # Initialize lora model.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1
    )
    llama_model = get_peft_model(llama_model, peft_config)

    model = VisionLanguageModel(language_model=llama_model, lm_tokenizer=tokenizer,
                                img_size=IMG_SIZE, patch_size=PATCH_SIZE).cuda()
    print(model)
    llama_model.print_trainable_parameters()

    # Prepare dataset.
    train_dataset, test_dataset = prepare_dataset(tokenizer)

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=f'./output_{MODE}',
        num_train_epochs=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to=None,
        remove_unused_columns=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        group_by_length=False,
        dataloader_pin_memory=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        tf32=True
    )
    trainer = ModifiedTrainer(
        model=model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Resume from the checkpoint
    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.train()
