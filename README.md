Étape 2 — Reconnaissance d'entités nommées (NER)
Modèle :dslim/bert-base-NER

Objectif : Segmenter et classifier les entités textuelles en catégories standards (personnes, organisations, lieux).

Fonctionnement : Chaque sous-mot reçoit une étiquette au format IOB (ex. B-PER, I-PER). L'activation de aggregation_strategy="simple"(ou grouped_entities=True) regroupe les sous-tokens contigus pour restituer des entités complètes (ex. Zara Venn en tant que PER, NovaCore Dynamics en ORG, London en LOC).

Constat : Le regroupement automatique est indispensable pour éviter une fragmentation artificielle des noms propres composites.

Étape 3 — Étiquetage morphosyntaxique (Étiquetage des parties du discours)
Modèle :vblagoje/bert-english-uncased-finetuned-pos

Objectif : Assigner à chaque unité lexicale son rôle syntaxique et grammatical dans l'énoncé.

Résultats : Les catégories grammaticales majeures sont étiquetées avec un score de confiance élevé (ex. PROPN pour les noms propres, VERB pour les formes verbales, ADP pour les prépositions).

Spécificité : Contrairement aux approches par règles, l'encodeur contextuel BERT désambiguïse la catégorie d'un mot selon sa position et ses voisins immédiats.

Étape 4 — Question-Réponse : Extractif vs Abstractif
Deux paradigmes distincts ont été mis en œuvre sur un même contexte :

4.1 Approche extractive ( distilbert-base-cased-distilled-squad) :

Mécanisme : Le modèle prédit la distribution de probabilité des indices de début ( start) et de fin ( end) dans le texte source pour délimiter le segment réponse ( "Brazil, Peru, and Colombia" ).

Avantage : Fidélité stricte au document source, risque d'hallucination nul.

Limite : Incapable de formuler une réponse synthétique si l'information est morcelée.

4.2 Approche Abstractive ( fangyuan/hotpotqa_abstractive/T5) :

Mécanisme : Génération séquence-à-séquence conditionnée par un prompt structuré ( question: ... context: ...).

Avantage : Production d'une phrase réalisée complète, fluide et syntaxiquement autonome.

Limite : Dépendance forte au formatage textuel en entrée et latence de génération supérieure.

Étape 5 — Résumé de texte
Modèle :cnicu/t5-small-booksum

Objectif : Condenser un paragraphe dense sur l'écosystème amazonien en conservant les faits essentiels (superficie, biodiversité, rôle régulateur du climat).

Contrôle du décodage : L'ajustement des paramètres max_length=35et min_length=15permet de calibrer la concision du résumé généré sans tronquer les propositions principales.

Étape 6 — Traduction automatique (Traduction)
Modèle : Helsinki-NLP/opus-mt-en-fr(architecture MarianMT)

Objectif : Traduire du texte anglais vers le français en conservant le registre technique.

Constat : La traduction obtenue ( "La forêt tropicale aide à réguler le climat de la Terre." ) préserve fidèlement la syntaxe et la sémantique de la phrase d'origine. Les modèles Helsinki-NLP offrent un excellent compromis entre légèreté en mémoire et rigueur terminologique.

Étape 7 — Modélisation autorégressive et génération créative
Modèle :distilgpt2

Mécanisme : Prédiction itérative du token suivant à partir du prompt initial ( "Once upon a time," ).

Paramètres d'échantillonnage :

max_length=30: limite supérieure du budget de jetons générés.

num_return_sequences=3: production en parallèle de variantes distinctes.

temperature=0.8et do_sample=True: équilibre entre cohérence globale et diversité lexicale, entraînant les répétitions cycliques du décodage glouton ( greedy ).

Synthèse comparative et discussion (Étape 8)
1. Classification de tokens vs Transformation de texte
Les modèles de classification de tokens (NER, PoS) conservent la longueur d'origine du document et enrichissent chaque mot d'une méta-information discrète.

Les modèles génératifs et encodeur-décodeur (T5, Marian, GPT-2) réécrivent ou projettent l'entrée vers une nouvelle séquence, introduisant de la flexibilité mais aussi un coût de calcul accru.

2. Risques d'hallucination et de fiabilité
En contexte critique (analyse juridique, médicale ou financière), le Question-Answering extractif doit être préféré à l'abtractif car il interdit l'invention d'éléments factuels hors contexte.

Les modèles de génération autorégressive purs (comme GPT-2) exigent un encadrement strict par température et pénalité de répétition pour éviter la dérive thématique.

3. Contraintes de déploiement
Sensibilité linguistique : Les modèles testés en NER et PoS sont calibrés pour la langue anglaise ; un passage vers le français nécessite des poids multilingues (ex. XLM-RoBERTa, CamemBERT).

Dimensionnement mémoire : L'inférence CPU reste viable pour des modèles de taille petite ou moyenne (DistilBERT, T5-small, DistilGPT-2), mais le passage à l'échelle sur des corpus nécessite du calcul par lot ( batching ) et une accélération matérielle (CUDA/GPU).



https://github.com/user-attachments/assets/02aae1c3-0269-4b8e-8eae-a7ce3b6e4a07

