from typing import ForwardRef, Tuple
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.runtime.pipe.module import LayerSpec, PipelineModule
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import model_type_to_module_name
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
from transformers.models.gptj.modeling_gptj import GPTJBlock
from transformers import GPTNeoForSequenceClassification
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from torch import nn
import torch
from copy import deepcopy
from functools import partial

from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss


def get_embed_dim(config):
    if hasattr(config, 'hidden_size'):
        return config.hidden_size
    if hasattr(config, 'n_embd'):
        return config.n_embd

def get_num_layers(config):
    if hasattr(config, 'num_layers'):
        return config.num_layers
    if hasattr(config, 'n_layer'):
        return config.n_layer

def get_embed_dropout(config):
    if hasattr(config, 'embd_pdrop'):
        return config.embd_pdrop
    if hasattr(config, 'embed_dropout'):
        return config.embed_dropout

def get_attn_mask(attention_mask):
    attention_mask = attention_mask[:, None, None, :]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    # attention_mask = attention_mask.to(
        # dtype=self.dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * -10000.0

    return attention_mask

def get_loss_fn(model, labels):
    # if labels is not None:
    if model.config.problem_type is None:
        if model.num_labels == 1:
            model.config.problem_type = "regression"
        elif model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            model.config.problem_type = "single_label_classification"
        else:
            model.config.problem_type = "multi_label_classification"

    loss_fct = None
    if model.config.problem_type == "regression":
        loss_fct = (lambda logits, labels: MSELoss()(logits.squeeze(), labels.squeeze(
        ))) if model.num_labels == 1 else lambda logits, labels: MSELoss()(logits, labels)
    elif model.config.problem_type == "single_label_classification":
        # loss_fct = lambda logits, labels: CrossEntropyLoss()(logits.view(-1, model.num_labels), labels.view(-1))
        # loss_fct = partial(CrossEntropyLoss(), )
        loss_fct = CrossEntropyLoss()
    elif model.config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
    return loss_fct

# def deepcopy_layer(src_module, trg_module):
#     src_param = list(src_module.parameters())
#     trg_param = list(trg_module.parameters())

#     assert len(src_param) == len(trg_param)

def concat_outputs(hidden_states, attention_mask):
    return torch.concat([attention_mask.unsqueeze(0).transpose(-2,-1), hidden_states.transpose(0,-1)])

def split_outputs(out_tensor):
    return out_tensor[1:].transpose(0,-1), out_tensor[0].transpose(-2,-1).squeeze()

class DummyModule(nn.Module):

    device = "cpu"

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = get_embed_dim(config)

    def to(self, device):
        self.device = device
        return super().to(device)

def move_data(outputs, device):
    if outputs[0].device == device: return outputs
    outputs = list(outputs)
    # print(outputs)
    for i in range(len(outputs)):
        # if type(outputs[i]) is torch.Tensor:
        outputs[i] = outputs[i].to(device)
    outputs = tuple(outputs)
    return outputs

class GPTEmbedding(DummyModule):
    def __init__(self, config, model=None):
        super().__init__(config)

        self.config = config

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        if config.model_type != "gptj":
            print("max_position_embeddings")
            self.wpe = nn.Embedding(
                config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(get_embed_dropout(config))

        if model:
            print("GPTEmbedding copy_weights")
            self.copy_weights(model)

    def to(self, device):
        self.device = device
        self.wte = self.wte.to(device)
        if hasattr(self, 'wpe'):
            self.wpe = self.wpe.to(device)
        return self

    def copy_weights(self, model):
        # super().__init__()
        self.wte.load_state_dict(model.transformer.wte.state_dict())
        if hasattr(self, 'wpe'):
            self.wpe.load_state_dict(model.transformer.wpe.state_dict())
        self.dtype = model.dtype
        self.drop.load_state_dict(model.transformer.drop.state_dict())
        return self

    # def forward(
    #     self,
    #     input_ids,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     # head_mask=None,
    # ):
    def forward(self, args):

        input_ids, attention_mask = args
        # input_ids = input_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        position_ids = None

        # input_ids = args.get('input_ids', None)
        # attention_mask = args.get('attention_mask', None)
        # token_type_ids = args.get('token_type_ids', None)
        # position_ids = args.get('position_ids', None)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        device = input_ids.device  # if input_ids is not None else inputs_embeds.device

        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids.view(-1, input_shape[-1])
        # if position_ids is not None:
        #     position_ids = position_ids.view(-1, input_shape[-1])

        past_length = 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        # if attention_mask is not None:
        #     if batch_size <= 0:
        #         raise ValueError("batch_size has to be defined and > 0")
        #     attention_mask = attention_mask.view(batch_size, -1)
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask[:, None, None, :]

        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and -10000.0 for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     # attention_mask = attention_mask.to(
        #     #     dtype=self.dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * -10000.0


        # attn_mask = attention_mask.clone()
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        # head_mask = get_head_mask(head_mask, self.config.n_layer)

        # if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if hasattr(self, 'wpe'):
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

        # if token_type_ids is not None:
        #     token_type_embeds = self.wte(token_type_ids)
        #     hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        # head_mask = [None] * get_num_layers(self.config)

        # attention_mask.require_grad = True
        # attention_mask.grad_fn = hidden_states.grad_fn
        # print(hidden_states.shape, attention_mask.shape, attention_mask.dtype)
        # cat_outputs = torch.concat([torch.squeeze(attn_mask).unsqueeze(0).transpose(-2,-1), hidden_states.transpose(0,-1)])
        # print(input_ids, attention_mask, head_mask)
        return hidden_states, input_ids, attention_mask# , head_mask
        # return concat_outputs(hidden_states, attention_mask)

    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    #     encoder_hidden_states=None,
    #     encoder_attention_mask=None,
    # ):

    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    #     elif input_ids is not None:
    #         input_shape = input_ids.size()
    #         input_ids = input_ids.view(-1, input_shape[-1])
    #         batch_size = input_ids.shape[0]
    #     elif inputs_embeds is not None:
    #         input_shape = inputs_embeds.size()[:-1]
    #         batch_size = inputs_embeds.shape[0]
    #     else:
    #         raise ValueError("You have to specify either input_ids or inputs_embeds")

    #     device = input_ids.device if input_ids is not None else inputs_embeds.device

    #     if token_type_ids is not None:
    #         token_type_ids = token_type_ids.view(-1, input_shape[-1])
    #     if position_ids is not None:
    #         position_ids = position_ids.view(-1, input_shape[-1])

        
    #     past_length = 0
    #     # past_key_values = tuple([None] * len(self.h))

    #     if position_ids is None:
    #         position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    #         position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    #     # GPT2Attention mask.
    #     if attention_mask is not None:
    #         if batch_size <= 0:
    #             raise ValueError("batch_size has to be defined and > 0")
    #         attention_mask = attention_mask.view(batch_size, -1)
    #         # We create a 3D attention mask from a 2D tensor mask.
    #         # Sizes are [batch_size, 1, 1, to_seq_length]
    #         # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    #         # this attention mask is more simple than the triangular masking of causal attention
    #         # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    #         attention_mask = attention_mask[:, None, None, :]

    #         # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    #         # masked positions, this operation will create a tensor which is 0.0 for
    #         # positions we want to attend and -10000.0 for masked positions.
    #         # Since we are adding it to the raw scores before the softmax, this is
    #         # effectively the same as removing these entirely.
    #         # attention_mask = attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    #         # attention_mask = (1.0 - attention_mask) * -10000.0

    #     # If a 2D or 3D attention mask is provided for the cross-attention
    #     # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    #     if self.config.add_cross_attention and encoder_hidden_states is not None:
    #         encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
    #         encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
    #         if encoder_attention_mask is None:
    #             encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
    #         encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    #     else:
    #         encoder_attention_mask = None

    #     # Prepare head mask if needed
    #     # 1.0 in head_mask indicate we keep the head
    #     # attention_probs has shape bsz x n_heads x N x N
    #     # head_mask has shape n_layer x batch x n_heads x N x N
    #     # head_mask = self.get_head_mask(head_mask, self.config.n_layer)
    #     head_mask = [None] * get_num_layers(self.config)

    #     if inputs_embeds is None:
    #         inputs_embeds = self.wte(input_ids)
    #     position_embeds = self.wpe(position_ids)
    #     hidden_states = inputs_embeds + position_embeds

    #     if token_type_ids is not None:
    #         token_type_embeds = self.wte(token_type_ids)
    #         hidden_states = hidden_states + token_type_embeds

    #     hidden_states = self.drop(hidden_states)
    #     print(input_ids, attention_mask, head_mask)
    #     return hidden_states, input_ids, attention_mask, head_mask


class GPTBlock(DummyModule):
    def __init__(self, config, model=None, layer_idx=None):
        super().__init__(config)

        # print(config)

        if config.model_type == "gptj":
            self.block = GPTJBlock(config)
        elif config.model_type == "gpt2":
            self.block = GPT2Block(config, layer_idx)
        elif config.model_type == "gpt_neo":
            self.block = GPTNeoBlock(config, layer_idx)
        else:
            raise NotImplementedError()
        
        self.layer_idx = layer_idx
        if model:
            print(f"GPTBlock copy_weights {layer_idx}")
            self.copy_weights(model, layer_idx)

    def to(self, device):
        self.device = device
        self.block = self.block.to(device)
        return self

    def forward(self, args):
        hidden_states, input_ids, attention_mask = args
        # hidden_states = hidden_states.to(self.device)
        # input_ids = input_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # hidden_states, attn_mask = split_outputs(args)
        # attention_mask = get_attn_mask(attn_mask)

        # print(hidden_states.shape, attention_mask.shape)
    
        # head_mask = args['head_mask']
        outputs = self.block(
            hidden_states,
            attention_mask=attention_mask,
            # head_mask=head_mask[self.layer_idx],
        )
        # print(self.layer_idx, outputs[0])
        # attention_mask.require_grad = True
        # print(input_ids, attention_mask, head_mask)
        return outputs[0], input_ids, attention_mask#, head_mask
        # if self.layer_idx == 11: exit()
        # return concat_outputs(outputs[0], attn_mask)

    def copy_weights(self, model, layer_idx):
        # super().__init__()
        self.block.load_state_dict(model.transformer.h[layer_idx].state_dict())
        self.dtype = model.dtype

        assert self.block.named_parameters()
        return 


class LMOutput():
    logits = None
    attention_mask = None

    def __init__(self, pooled_logits, attention_mask) -> None:
        self.logits = pooled_logits
        self.attention_mask = attention_mask


class GPTOutput(DummyModule):
    def __init__(self, config, task_type, model=None):
        super().__init__(config)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # self.task_type = task_type
        # if task_type.lower() == "classification":
        self.num_labels = config.num_labels
        self.score = nn.Linear(self.embed_dim, self.num_labels, bias=False)

        if model:
            print("GPTOutput copy_weights")
            self.copy_weights(model)

    def to(self, device):
        self.device = device
        self.ln_f = self.ln_f.to(device)
        self.score = self.score.to(device)
        return self

    def forward(self, args):
        hidden_states, input_ids, attention_mask = args
        # hidden_states = hidden_states.to(self.device)
        # input_ids = input_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # hidden_states, attention_mask = split_outputs(args)
        # hidden_states = args['hidden_states']
        hidden_states = self.ln_f(hidden_states)
        input_shape = input_ids.size()
        output_shape = input_shape + (hidden_states.size(-1),)
        hidden_states = hidden_states.view(*output_shape)
        # print(hidden_states.shape)
        output = self.score(hidden_states)
        # print(output.shape)

        batch_size, _ = input_ids.shape[:2]

        # sequence_lengths = torch.eq(attention_mask, 0.0).sum(dim=1) - 1
        # sequence_lengths = sequence_lengths.squeeze()
        # print(sequence_lengths.shape, attention_mask, sequence_lengths)
        # sequence_lengths = sequence_lengths.squeeze()
        # sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1

        # assert (
        #     self.config.pad_token_id is not None or batch_size == 1
        # ), "Cannot handle batch sizes > 1 if no padding token is defined."
        # if self.config.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     if input_ids is not None:
        #         sequence_lengths = torch.eq(attention_mask, 0.0).sum(dim=1) - 1
        #         # print(sequence_lengths.shape, attention_mask, sequence_lengths)
        #         # sequence_lengths = sequence_lengths.squeeze()
        #         # sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        #     else:
        #         sequence_lengths = -1

        sequence_lengths = torch.ne(input_ids, 50256).sum(-1) - 1

        pooled_logits = output[range(batch_size), sequence_lengths]
        # print(pooled_logits)
        # print(pooled_logits.shape, sequence_lengths, output)
        return hidden_states, pooled_logits

    def copy_weights(self, model):
        self.ln_f.load_state_dict(model.transformer.ln_f.state_dict())

        # if self.task_type.lower() == "classification":
        self.score.load_state_dict(model.score.state_dict())

        self.dtype = model.dtype

        return self

# class GPTNeoPipe(GPTNeoForSequenceClassification):
#     def __init__(self, path):
#         self.model = GPTNeoForSequenceClassification.from_pretrained(path)
#         self.exec_map = (0, self.num_layers)
#         self.num_layers = get_num_layers(self.model.config) + 2

#     def forward_layers(self, args):
#         outputs = args
#         for idx in range(*self.exec_map):
#             # print(outputs[0].shape, outputs[1].shape)
#             if idx == 0:
#                 outputs = self.embed(outputs)
#             elif idx == self.num_layers -1:
#                 outputs = self.out(outputs)
#             else:
#                 outputs = self.h[idx-1](outputs)
#         return outputs[0] if idx != self.num_layers -1 else outputs[1] # HACK since 0 is always hidden states


class GPTModelPipe(nn.Module):
    def __init__(self, config, task_type, model=None):
        super().__init__()

        self.config = config
        self.embed = GPTEmbedding(config, model)
        self.h = nn.ModuleList([GPTBlock(config, model, i)
                               for i in range(get_num_layers(config))])
        self.out = GPTOutput(config, task_type, model)

        # for param in self.parameters():
        #     param.requires_grad = True
        #     # param.retain_grad()

        # if model:
        #     self.copy_weights(model)
        self.num_layers = get_num_layers(config) + 2

        # self.layers = self.to_layers()

        self.exec_map = (0, self.num_layers)

        self.model_parallel = False

        self.first_device = "cpu"
        self.last_device = "cpu"
        # self.device = "cuda:0"

    def copy_weights(self, model):
        # super().__init__()
        self.embed.copy_weights(model)
        # self.h.load_state_dict(model.transformer.h.state_dict())
        for idx in range(len(self.h)):
            self.h[idx].copy_weights(model, idx)
        # self.h = nn.ModuleList([GPTBlock(self.config).copy_weights(model, idx) for idx in range(self.config.n_layer)])
        self.out.copy_weights(model)

        return self

    def to(self, device):
        # self.embed.device  = device
        # for idx in range(len(self.h)):
        #     self.h[idx].device  = device
        # self.out.device = device

        for idx in range(*self.exec_map):
            if idx == 0:
                self.embed.device  = device
                self.embed = self.embed.to(device)
            elif idx == self.num_layers -1:
                self.out.device = device
                self.out = self.out.to(device)
            else:
                self.h[idx-1].device  = device
                self.h[idx-1].device  = self.h[idx-1].to(device)

        return self #super().to(device)

    def eval(self):
        self.embed.eval()
        for idx in range(len(self.h)):
            self.h[idx].eval()
        self.out.eval()

    def to_layers(self):
        # for h in self.h:
        #     print("to_layers", type(h), dir(h))
        return [
            self.embed,
            *self.h,
            self.out,
        ]

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.embed = self.embed.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # out to last
        self.out = self.out.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.embed = self.embed.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.out = self.out.to("cpu")
        torch.cuda.empty_cache()

    def forward_layers(self, args, output_hidden_states=False):
        outputs = args
        for idx in range(*self.exec_map):
            print(idx)
            if idx == 0:
                outputs = self.embed(outputs)
            elif idx == self.num_layers -1:
                outputs = self.out(outputs)
            else:
                outputs = self.h[idx-1](outputs)
        return outputs[0] if idx != self.num_layers -1 else outputs[1] # HACK since 0 is always hidden states

    def forward(self, args, output_hidden_states=False):
        all_hidden_states = ()
        
        outputs = self.embed(args)
        print(outputs[0].device, self.model_parallel)
        for block in self.h:
            if output_hidden_states:
                all_hidden_states += (outputs[0], )
            if self.model_parallel:
                torch.cuda.set_device(outputs[0].device)
                outputs = move_data(outputs, block.device)
            print(block.device, outputs[0].device)
            outputs = block(outputs)
            # print(hidden_states)
        # all_hidden_states += (outputs[0], )
        if self.model_parallel:
            torch.cuda.set_device(outputs[0].device)
            outputs = move_data(outputs, self.out.device)
        print(block.device, outputs[0].device)
        outputs = self.out(outputs)
        # exit()
        
        if output_hidden_states: 
            all_hidden_states += (outputs[0], )
        return outputs[1], all_hidden_states

        # if config.model_type == "gptj":
        #     self.h = nn.ModuleList([nn.ModuleList([GPTJBlock(config) for _ in range(config.n_layer)])])
        # if config.model_type == "gpt2":
        #     self.h = nn.ModuleList([nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])])
        # if config.model_type == "gpt_neo":
        #     self.h = nn.ModuleList([nn.ModuleList([GPTNeoBlock(config) for _ in range(config.n_layer)])])
        # self.models = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])


# class GPT2ModelPipe(PipelineModule, torch.nn.Module):
#     """GPT2Model adapted for pipeline parallelism.

#     The largest change is flattening the GPTModel class so we can express it as a
#     sequence of layers including embedding, transformer layers, and output.
#     """

#     def __init__(self, config, task_type, loss_fn, num_stages=None, model=None, topology=None,
#                  partition_method='parameters',
#                  activation_checkpoint_interval=0,
#                  activation_checkpoint_func=checkpointing.checkpoint):
#         self.config = config
#         self.task_type = task_type
#         self.__topology__ = topology

#         self.specs = []
#         self.init_specs(model)
#         super().__init__(layers=self.specs,
#                          loss_fn=loss_fn,
#                          num_stages=num_stages,
#                          topology=topology,
#                          activation_checkpoint_interval=activation_checkpoint_interval,
#                          partition_method=partition_method,
#                          activation_checkpoint_func=activation_checkpoint_func
#                          )

#     def init_specs(self, model=None):
#         self.specs = []
#         # Embedding layer
#         # input will be (input_ids, attention_mask)

#         self.specs.append(LayerSpec(GPTEmbedding, self.config, model))

#         # Transformer layers
#         for i in range(get_num_layers(self.config)):
#             self.specs.append(LayerSpec(GPTBlock, self.config, model, i))

#         self.specs.append(LayerSpec(GPTOutput, self.config, self.task_type, model))

        