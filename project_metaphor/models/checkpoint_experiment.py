from transformers import AutoModel
from transformers import AutoTokenizer

without_csr_scm = './model_config/distilbert-base-uncased-single-hghl-split0-batch32-2022-12-30_14-55-43_best_scm_without_csr_32/'
tokenizer = AutoTokenizer.from_pretrained(without_csr_scm)
model = AutoModel.from_pretrained(without_csr_scm)
text = "So, that money, which had raced forcibly far upstream of pieces and their implementations, retreated."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

csk = "Reasons:  speaker is being chased by a bear, speaker is being chased by the police,  speaker is being chased by a criminal"
encoded_input_csk = tokenizer(csk, return_tensors='pt')
output_csk = model(**encoded_input_csk)

