from transformers import AutoModel, AutoModelForMaskedLM, T5ForConditionalGeneration

model = model = T5ForConditionalGeneration.from_pretrained("t5-small")

print(model)

# for name, child in model.base_model.named_children():
#     print(name, child.__class__)