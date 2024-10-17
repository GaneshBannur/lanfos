from typing import List
import open_clip
import torch
from torch import nn
from PIL import Image
import warnings

class SigLIPInference(nn.Module):
    def __init__(self, model_name, pretraining_dataset, weight_precision, torch_device="cuda"):
        super().__init__()
        self.model, _, self.process = open_clip.create_model_and_transforms(
            model_name,     # Example: "ViT-B-16"
            pretrained=pretraining_dataset,     # Example: "laion2b_s34b_b88k"
            precision=weight_precision,
            device=torch_device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.torch_device = torch_device
        self.weight_precision = weight_precision

    def preprocess_image_list(self, image_list: List[Image.Image]):
        '''
        image_list must be a list of PIL images in order to align with how OpenCLIP does training
        '''
        preprocessed_images = [self.process(image) for image in image_list]
        batched_image_tensor = torch.stack(preprocessed_images, dim=0).to(self.torch_device)
        return batched_image_tensor

    def encode_image(self, batched_image_tensor):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            image_features = self.model.encode_image(batched_image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def check_context_len_overflow(self, text_list):
        for text in text_list:
            text_tensor = self.tokenizer(text, context_length=self.tokenizer.context_length+1)[0]
            # An EOS token is always inserted at the end so if there is no PAD token at text_tensor[-2] it means that
            # the text was cut off since context_length was 1 more than the model's limit and the text was at least as
            # long as this since it didn't have to be padded
            if text_tensor[-2]!=self.tokenizer.tokenizer.pad_token_id:
                warnings.warn(f"The following input exceeded the context length: {text}")

    def encode_text(self, text_list):
        self.check_context_len_overflow(text_list)
        text_tensor = self.tokenizer(text_list).to(self.torch_device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            text_features = self.model.encode_text(text_tensor)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def get_relevancy(self, image_features, text_features):
        '''
        Expects features to be two dimensional with shape [N, embedding_dim]
        '''
        if self.weight_precision=="fp16":
            image_features = image_features.half()
            text_features = text_features.half()

        image_features = image_features.to(self.torch_device)
        text_features = text_features.to(self.torch_device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = torch.mm(image_features, text_features.T)
            logits = logits*self.model.logit_scale.exp() + self.model.logit_bias
            relevancies = torch.sigmoid(logits)
        
        # Having everything in float rather than half is easier downstream
        relevancies = relevancies.to(torch.float32)
        return relevancies
