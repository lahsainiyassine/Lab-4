import warnings
from transformers import pipeline

# Masquer les avertissements mineurs de dépréciation
warnings.filterwarnings("ignore")

print("=" * 65)
print("LAB 4 : CLASSIFICATION DE TOKENS ET GÉNÉRATION DE TEXTE")
print("=" * 65)

# ============================================================
# ÉTAPE 2 : Reconnaissance d'entités nommées (NER)
# ============================================================
print("\n--- ÉTAPE 2 : RECONNAISSANCE D'ENTITÉS NOMMÉES (NER) ---")

ner_pipeline = pipeline(
    task="ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"  # Équivalent recommandé et robuste de grouped_entities=True
)

text_ner = "Zara Venn established NovaCore Dynamics in London."
ner_results = ner_pipeline(text_ner)
print(f"Phrase : '{text_ner}'")
print("Entités identifiées :")
for ent in ner_results:
    print(f"  [{ent['entity_group']}] '{ent['word']}' (Score : {ent['score']:.4f})")

# Test en lot
batch_ner_texts = [
    "Apple CEO Tim Cook visited Paris yesterday.",
    "Google announced Gemini at their Mountain View headquarters."
]
batch_ner_results = ner_pipeline(batch_ner_texts)
print("\nTest en lot (Batch NER) :")
for txt, ents in zip(batch_ner_texts, batch_ner_results):
    detected = ", ".join([f"{e['word']} ({e['entity_group']})" for e in ents])
    print(f"  '{txt}' -> [{detected}]")

# ============================================================
# ÉTAPE 3 : Étiquetage morphosyntaxique (Part-of-Speech Tagging)
# ============================================================
print("\n--- ÉTAPE 3 : ÉTIQUETAGE GRAMMATICAL (POS TAGGING) ---")

pos_pipeline = pipeline(
    task="token-classification",
    model="vblagoje/bert-english-uncased-finetuned-pos",
    aggregation_strategy="simple"
)

sentence_pos = "Zara Venn established NovaCore Dynamics in London."
pos_results = pos_pipeline(sentence_pos)
print(f"Phrase : '{sentence_pos}'")
print("Catégories grammaticales :")
for token in pos_results:
    print(f"  Token : '{token['word']:<12}' | POS : {token['entity_group']:<6} | Confiance : {token['score']:.4f}")

# ============================================================
# ÉTAPE 4 : Question–Réponse (Extractive vs Abstractive)
# ============================================================
print("\n--- ÉTAPE 4 : QUESTION-RÉPONSE (QA) ---")

context_qa = (
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "covering parts of Brazil, Peru, and Colombia."
)
question_qa = "Which countries does the Amazon rainforest cover?"

# 4.1 Extractive QA
qa_extractive = pipeline(
    task="question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)
ans_extractive = qa_extractive(question=question_qa, context=context_qa)
print(f"4.1 QA Extractive (DistilBERT Squad) :")
print(f"  Question : '{question_qa}'")
print(f"  Réponse extraite : '{ans_extractive['answer']}' (Score: {ans_extractive['score']:.4f})")
print(f"  Localisation     : Début = {ans_extractive['start']}, Fin = {ans_extractive['end']}")
print(f"  Vérification     : '{context_qa[ans_extractive['start']:ans_extractive['end']]}'")

# 4.2 Abstractive QA
qa_abstractive = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small"
)
input_t5 = f"question: {question_qa} context: {context_qa}"
ans_abstractive = qa_abstractive(input_t5)[0]["generated_text"]
print(f"\n4.2 QA Abstractive (T5 HotpotQA) :")
print(f"  Réponse générée  : '{ans_abstractive}'")

# ============================================================
# ÉTAPE 5 : Résumé de texte (Summarization)
# ============================================================
print("\n--- ÉTAPE 5 : RÉSUMÉ DE TEXTE (SUMMARIZATION) ---")

summarizer = pipeline(
    task="summarization",
    model="cnicu/t5-small-booksum"
)

text_to_summarize = (
    'The Amazon rainforest, often referred to as the "lungs of the Earth," is one '
    'of the most biologically diverse regions in the world. Spanning over nine '
    'countries in South America, the majority of the forest lies in Brazil. '
    'It is home to an estimated 390 billion individual trees, divided into 16,000 '
    'different species. The rainforest plays a critical role in regulating the global '
    'climate by absorbing vast amounts of carbon dioxide and producing oxygen.'
)

summary_default = summarizer(text_to_summarize)[0]["summary_text"]
summary_short = summarizer(text_to_summarize, max_length=35, min_length=15)[0]["summary_text"]

print(f"Texte source ({len(text_to_summarize.split())} mots)")
print(f"Résumé standard : '{summary_default}'")
print(f"Résumé contrôlé : '{summary_short}'")

# ============================================================
# ÉTAPE 6 : Traduction automatique (Translation)
# ============================================================
print("\n--- ÉTAPE 6 : TRADUCTION AUTOMATIQUE (EN -> FR) ---")

translator = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-fr"
)

sentence_en = "The rainforest helps regulate the Earth's climate."
translation_res = translator(sentence_en)[0]["translation_text"]
print(f"Source (EN) : '{sentence_en}'")
print(f"Cible  (FR) : '{translation_res}'")

# ============================================================
# ÉTAPE 7 : Modélisation et génération de texte (Autoregressive)
# ============================================================
print("\n--- ÉTAPE 7 : GÉNÉRATION DE TEXTE AUTORÉGRESSIVE (DistilGPT2) ---")

generator = pipeline(
    task="text-generation",
    model="distilgpt2"
)

prompt = "Once upon a time,"
generated_outputs = generator(
    prompt,
    max_length=30,
    num_return_sequences=3,
    temperature=0.8,
    do_sample=True,
    pad_token_id=generator.tokenizer.eos_token_id
)

print(f"Prompt de départ : '{prompt}'")
for idx, out in enumerate(generated_outputs, 1):
    print(f"  Variante {idx} : {out['generated_text'].strip()}")