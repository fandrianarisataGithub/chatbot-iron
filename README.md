# Chatbot pour Analyse et Réponse Automatique

Bienvenue dans ce projet de chatbot basé sur l'IA ! Ce chatbot utilise un modèle pré-entraîné et fine-tuné sur des données spécifiques pour répondre aux questions ou demandes dans le cadre d'une analyse de texte. Il a été conçu pour fournir des réponses automatiques dans le contexte d'interactions simples et spécifiques.

## Table des Matières
- [Présentation](#présentation)
- [Technologies](#technologies)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du Projet](#structure-du-projet)
- [Contribuer](#contribuer)
- [Licence](#licence)

## Présentation

Ce projet utilise un modèle d'apprentissage profond pour analyser des textes et répondre à des requêtes. Le modèle a été fine-tuné avec des données spécifiques pour fournir des réponses pertinentes et contextuelles. Il peut être utilisé pour automatiser des tâches comme la réponse aux questions fréquentes, la génération de résumés ou même pour des applications plus avancées de dialogue.

**Fonctionnalités principales :**
- Réponse automatique basée sur l'analyse de texte.
- Prise en charge de divers types de prompts pour générer des réponses cohérentes.
- Fine-tuning sur des datasets spécifiques pour améliorer la pertinence des réponses.

## Technologies

Ce projet repose sur les technologies suivantes :
- **Transformers** (Hugging Face) : pour le modèle pré-entraîné et le fine-tuning.
- **PyTorch** : pour la gestion du modèle et des opérations de deep learning.
- **Hugging Face Trainer** : pour la gestion de l'entraînement du modèle.

## Installation

### Prérequis

Avant de commencer, assurez-vous d'avoir Python 3.7+ et les bibliothèques suivantes installées :

1. **Clonez ce dépôt :**
    ```bash
    git clone https://github.com/votre-username/chatbot-ia.git
    cd chatbot-ia
    ```

2. **Installez les dépendances :**
    Nous vous recommandons d'utiliser un environnement virtuel (venv ou conda) pour éviter les conflits de dépendances.
    ```bash
    pip install -r requirements.txt
    ```

### Dépendances

Les principales bibliothèques utilisées dans ce projet sont :
- `transformers`
- `torch`
- `datasets`

## Utilisation

### 1. **Lancer le modèle**

Une fois l'installation terminée, vous pouvez démarrer une session pour interagir avec le chatbot.

1. **Charger le modèle fine-tuné et tokenizer :**

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Chargez le tokenizer et le modèle
    tokenizer = AutoTokenizer.from_pretrained('path_to_your_model')
    model = AutoModelForCausalLM.from_pretrained('path_to_your_model')

    # Fonction pour générer des réponses
    def generate_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # Exemple de prompt
    print(generate_response("Hello, how are you?"))
    ```

2. **Testez le chatbot avec un prompt :**

    Vous pouvez tester différents prompts pour voir comment le modèle répond à différentes requêtes. Par exemple :
    - `"What's the weather like today?"`
    - `"Tell me a joke."`

### 2. **Entraîner le modèle**

Si vous souhaitez fine-tuner le modèle sur un autre jeu de données, vous pouvez exécuter le script suivant :

```python
from transformers import Trainer, TrainingArguments

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Lancer l'entraînement
trainer.train()
