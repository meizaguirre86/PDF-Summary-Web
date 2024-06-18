from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


source_lang = "deu_Latn"
target_lang = "eng_Latn"

def translate_text(input,source_lang, target_lang):
    model_id = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang, max_length = 400)
    output = translator(text)
    return output[0]["translation_text"]

def detect_source_language(input):
    model_id = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model_id)
    return pipe(input, top_k=1, truncation=True)



text = "Hola, qué tal estás? Estamos trabajando en la oficina de Araia aunque vamos a movernos a final de año a una nueva oficina. No estamos seguro cuándo, pero parece ser que será a finales de año."

#source_lang = detect_source_lang(text)
#print(source_lang)

print(translate_text(text, source_lang, target_lang))