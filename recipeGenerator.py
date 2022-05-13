# SETUP
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

file = open("recipes.json")
dataDict = json.loads(file.read())
file.close()

# STOPWORDS
STOP_WORD_TITLE="[TITTEL]\n"
STOP_WORD_ATTRIBUTES="\n\n[ATTRIBUTTER]\n"
STOP_WORD_DESCRIPTION="\n\n[INTRODUKSJON]\n"
STOP_WORD_NUMBER_OF_PEOPLE="\n\n[ANTALL PORSJONER]\n"
STOP_WORD_TIME="\n\n[TID]\n"
STOP_WORD_DIFFICULTY="\n\n[VANSKLIGHETSGRAD]\n"
STOP_WORD_INGREDIENTS="\n\n[INGREDIENSER]\n* "
STOP_WORD_STEPS="\n\n[STEG]\n- "


# CONVERT RECIPE OBJECT TO STRING
def recipeToString(recipe):
    title = recipe["title"]
    attributes = recipe["attributes"]
    description = recipe["description"]
    nPeople = recipe["numberOfPeople"]
    time = recipe["time"]
    difficulty = recipe["difficulty"]
    ingredients = recipe["ingredients"]
    steps = recipe["steps"]

    stringyfiedRecipe = STOP_WORD_TITLE + title
    stringyfiedRecipe += STOP_WORD_ATTRIBUTES + " - ".join(attributes)
    stringyfiedRecipe += STOP_WORD_DESCRIPTION + description
    stringyfiedRecipe += STOP_WORD_NUMBER_OF_PEOPLE + nPeople
    stringyfiedRecipe += STOP_WORD_TIME + time
    stringyfiedRecipe += STOP_WORD_DIFFICULTY + difficulty
    stringyfiedRecipe += STOP_WORD_INGREDIENTS + "\n* ".join(ingredients)
    stringyfiedRecipe += STOP_WORD_STEPS + "\n- ".join(steps)

    return stringyfiedRecipe

stringyfiedData = np.array([recipeToString(recipe) for recipe in dataDict])

print(f"Number of recipes: {len(stringyfiedData)}")
print(stringyfiedData[0])

recipeLengths = [len(recipe) for recipe in stringyfiedData]

plt.hist(recipeLengths, bins=50)
plt.show()

MAX_RECIPE_LENGTH = 1800

# FILTER OUT TOO LARGE RECIPES
filteredDataset = [recipe for recipe in stringyfiedData if len(recipe) <= MAX_RECIPE_LENGTH]

print(f"Number of filtered recipes: {len(filteredDataset)}")
print(f"Number of recipes removed: {len(stringyfiedData)- len(filteredDataset)}")

# VECTORIZE RECIPES
END_SIGN="~"

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    char_level=True,
    filters='',
    lower=False,
    split=""
)

tokenizer.fit_on_texts([END_SIGN])
tokenizer.fit_on_texts(filteredDataset)


# + 1 since index 0 is reserved
VOCABULARY_SIZE = len(tokenizer.word_counts) + 1 

print(f"Vocabulary Size: {VOCABULARY_SIZE}")

vectorizedData = tokenizer.texts_to_sequences(filteredDataset)

def vectorizedSequenceToText(vectorizedSeq):
    return tokenizer.sequences_to_texts([vectorizedSeq])[0]

#
# Two different to make sure that all end with atleast one end sign
#
vectorizedDataWithoutStops = tf.keras.preprocessing.sequence.pad_sequences(
    vectorizedData,
    padding="post",
    truncating="post",
    maxlen=MAX_RECIPE_LENGTH-1,
    value=tokenizer.texts_to_sequences([END_SIGN])[0]
)

vectorizedDataPadded = tf.keras.preprocessing.sequence.pad_sequences(
    vectorizedDataWithoutStops,
    padding="post",
    truncating="post",
    maxlen=MAX_RECIPE_LENGTH+1,
    value=tokenizer.texts_to_sequences([END_SIGN])[0]
)

#CONVERT TO TENSOR FLOW DATASET, AND SPLIT INTO INPUT AND TARGET
tfDataset = tf.data.Dataset.from_tensor_slices(vectorizedDataPadded)

def splitInputAndTarget(recipe):
    return recipe[:-1], recipe[1:]

tfDatasetTargeted = tfDataset.map(splitInputAndTarget)

#SPLIT INTO BATCHES
BATCH_SIZE=64
SHUFFLE_BUFFER_SIZE=1000

tfDatasetTrain = tfDatasetTargeted.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

#BUILD MODEL
def buildModel(vocabSize, embeddingDim, rnnUnits, batchSize):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(
        input_dim=vocabSize,
        output_dim=embeddingDim,
        batch_input_shape=[batchSize, None]
    ))
    model.add(tf.keras.layers.GRU(
        units=rnnUnits,
        activation="tanh",
        recurrent_activation="sigmoid",
        return_sequences=True,
        stateful=True,
    ))
    model.add(tf.keras.layers.Dense(vocabSize))

    return model

#DEFINE LOSS AND OPTIMIZER
model = buildModel(
    vocabSize=VOCABULARY_SIZE,
    embeddingDim=256,
    rnnUnits=750,
    batchSize=BATCH_SIZE
)

def loss(labels, logits):
    entropy = tf.keras.losses.sparse_categorical_crossentropy(
      y_true=labels,
      y_pred=logits,
      from_logits=True
    )
    return entropy

adamOptimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=adamOptimizer,
    loss=loss
)

earlyStopping = tf.keras.callbacks.EarlyStopping(
    patience=5,
    monitor='loss',
    restore_best_weights=True,
    verbose=1
)

#STORE CHECKPOINTS
checkpointDirectory = 'tmp/checkpoints'
os.makedirs(checkpointDirectory, exist_ok=True)
checkpointPrefix = os.path.join(checkpointDirectory, 'ckpt_{epoch}')
checkpointCallback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointPrefix,
    save_weights_only=True
)

#RUN MODEL
EPOCHS = 10
INITIAL_EPOCH = 0
STEPS_PER_EPOCH = 1500

def render_training_history(training_history):
    loss = training_history.history['loss']
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss, label='Training set')
    plt.legend()
    plt.grid(linestyle='--', linewidth=1, alpha=0.5)
    plt.show()

history = model.fit(
    x=tfDatasetTrain,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    initial_epoch=INITIAL_EPOCH,
    callbacks=[
        checkpointCallback,
        earlyStopping
    ]
)
modelName = 'recipeGenerationRaw.h5'
model.save(modelName, save_format='h5')
render_training_history(history)

#BUILD MODEL AGAIN WITH DIFFERENT BATCH SIZE TO BE ABLE TO USE CUSTOM INPUT LENGTHS
SIMPLIFIED_BATCH_SIZE=1

simplifiedModel = buildModel(vocabSize=VOCABULARY_SIZE,
    embeddingDim=256,
    rnnUnits=750,
    batchSize=SIMPLIFIED_BATCH_SIZE)
simplifiedModel.load_weights(tf.train.latest_checkpoint(checkpointDirectory))
simplifiedModel.build(tf.TensorShape([SIMPLIFIED_BATCH_SIZE, None]))

#GENERATE RECIPE
def generateText(model, startString, numGenerate=1000, temperature=1.0):
  startString = STOP_WORD_TITLE + startString
  inputIndices = np.array(tokenizer.texts_to_sequences([startString]))
  generatedText = []
  model.reset_states()
  
  for charIndex in range(numGenerate):
    predictions = model(inputIndices)
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / temperature
    predictedId = tf.random.categorical(
            predictions,
            num_samples=1
        )[-1, 0].numpy()
    inputIndices = tf.expand_dims([predictedId], 0)
    nextCharacter = tokenizer.sequences_to_texts(inputIndices.numpy())[0]
    if nextCharacter == "~":
        break
    generatedText.append(nextCharacter)
  
  return (startString + ''.join(generatedText))

#Change temperature to make output more or less random. 1.2 seems to work well.
print(generateText(simplifiedModel, startString="K", numGenerate=1800, temperature=1.2))


# Perform tests on model
import nltk
import random
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

def getRandomLetter():
    return chr(random.randint(ord('A'), ord('Z')))

def generateRandomRecipes(n=10):
    recipes = []
    for i in range(n):
        recipes.append(generateText(simplifiedModel, startString=getRandomLetter(), numGenerate=1800, temperature=1.2))
        print(f"{len(recipes)} out of {n} generated...")
    return recipes

referenceRecipes = filteredDataset[:10]
generatedRecipes = generateRandomRecipes()

def bleu(references, generated):
    splitReferences = [ref.split() for ref in references]
    splitGenerated = [gen.split() for gen in generated]
    cc = SmoothingFunction()    
    return corpus_bleu(splitReferences, splitGenerated, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method4)

print(f"BLEU Score: {bleu(referenceRecipes, generatedRecipes)}")

from rouge import Rouge

def rouge(generated, references):
    rouge = Rouge()
    return rouge.get_scores(generated, references, avg=True)

print(rouge(generatedRecipes, referenceRecipes))

