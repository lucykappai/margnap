# margnap
AI text classification model, using RoBERTa-uncased tokeniser, trained on the [`acmc/multi_domain_ai_human_text`](https://huggingface.co/datasets/acmc/multi_domain_ai_human_text) dataset. 

# Dataset

## Data Wrangling

# Tokenizer

# Validation

# Test Cases

# Overview of work 

To build up my ML skills, I chose to train a simple classification model on detecting AI vs human text across multiple genres of writing and multiple families of LLM. To get started quickly, I checked Hugging Face for simple 'AI vs. Human' text datasets. Initially wary of a lack of dataset cards and readme's, I decided to quickly go ahead with a large dataset with multiple different types of LLM content: [`ahmadreza13/human-vs-Ai-generated-dataset`](https://huggingface.co/datasets/ahmadreza13/human-vs-Ai-generated-dataset). I immediately noticed that (i) the data was imbalanced ,with ~58% human written and 42% AI-written text (with further heavy weighting toward GPT4, and less toward e.g. Claude Opus 3 or Gemini 1.5 Pro), and (ii) the entirety of the human-written content was from Wikipedia, as this will have obvious stylistic differences from both AI-written and other human-written text. Furthermore, with no detailed data sourcing, there was no guarantee the Wikipedia content was not AI generated.



After tokenising with [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased), I chose to explore some properties of the dataset. Wikipedia articles are obviously much longer in length than other genres of writing like restaurant reviews, and the AI-written content seemed to span things like code, explanations, and letters, so this seemed like a sensible place to start. Indeed, whilst the human-written content was typically less than the max length of the model (<0.01%), AI written content ranged from 2.62% (unspecified Claude model) to 58.95% (Claude Opus 3). 

On the 10% sized test dataset, the trained model achieved an accuracy of 99.90%, a FPR of 0.1046%, and a FNR of 0.0938%. Though I like to imagine I am a skilled data scientist, I am not hubristic enough to believe these results! Of course, the model is confounding the genres of writing in the different sources of data. Indeed, when testing on an OOD dataset of 50%/50% human-written vs. Llama-3.1-8B-written CNN articles ([`ilyasoulk/ai-vs-human-meta-llama-Llama-3.1-8B-Instruct-CNN`](https://huggingface.co/datasets/ilyasoulk/ai-vs-human-meta-llama-Llama-3.1-8B-Instruct-CNN)), the accuracy falls to 50.0400%; random chance. 

