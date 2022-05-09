import json
from pyrsistent import v
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

#
# WILL PROBABLY NEED MORE DATA
#
file = open("recipes.json")
dataDict = json.loads(file.read())
file.close()

#
# MIGHT ADD FILTERING AT A LATER STAGE
# 


# STOPWORDS
STOP_WORD_TITLE="📖 "
STOP_WORD_ATTRIBUTES="\n\n📝\n"
STOP_WORD_DESCRIPTION="\n\n🎧\n"
STOP_WORD_NUMBER_OF_PEOPLE="\n\n👨‍🔧\n"
STOP_WORD_TIME="\n\n⏰\n"
STOP_WORD_DIFFICULTY="\n\n😓\n"
STOP_WORD_INGREDIENTS="\n\n🥗\n"
STOP_WORD_STEPS="\n\n👨‍🍳\n"

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
    stringyfiedRecipe += STOP_WORD_ATTRIBUTES + attributes
    stringyfiedRecipe += STOP_WORD_DESCRIPTION + description
    stringyfiedRecipe += STOP_WORD_NUMBER_OF_PEOPLE + nPeople + " personer"
    stringyfiedRecipe += STOP_WORD_TIME + time
    stringyfiedRecipe += STOP_WORD_DIFFICULTY + difficulty
    stringyfiedRecipe += STOP_WORD_INGREDIENTS + "\n".join(ingredients)
    stringyfiedRecipe += STOP_WORD_STEPS + "\n".join(steps)

    return stringyfiedRecipe

stringyfiedData = [recipeToString(recipe) for recipe in dataDict]

# recipeLengths = [len(recipe) for recipe in stringyfiedData]
# print(f"Longest recipe: {max(recipeLengths)}")
# 2459
# Can get most recipes if max length is 2000
MAX_RECIPE_LENGTH = 2000

filteredDataset = [recipe for recipe in stringyfiedData if len(recipe) <= MAX_RECIPE_LENGTH]

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

tfDataset = tf.data.Dataset.from_tensor_slices(vectorizedDataPadded)

def splitInputAndTarget(recipe):
    return recipe[:-1], recipe[1:]

tfDatasetTargeted = tfDataset.map(splitInputAndTarget)

BATCH_SIZE=32
SHUFFLE_BUFFER_SIZE=1000

tfDatasetTrain = tfDatasetTargeted.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

def buildModel(vocabSize, embeddingDim, rnnUnits, batchSize):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(
        input_dim=vocabSize,
        output_dim=embeddingDim,
        batch_input_shape=[batchSize, None]
    ))
    model.add(tf.keras.layers.LSTM(
        units=rnnUnits,
        return_sequences=True,
        stateful=True,
        recurrent_initializer=tf.keras.initializers.GlorotNormal()
    ))
    model.add(tf.keras.layers.Dense(vocabSize))

    return model

model = buildModel(
    vocabSize=VOCABULARY_SIZE,
    embeddingDim=256,
    rnnUnits=1024,
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

checkpointDirectory = 'tmp/checkpoints'
os.makedirs(checkpointDirectory, exist_ok=True)
checkpointPrefix = os.path.join(checkpointDirectory, 'ckpt_{epoch}')
checkpointCallback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointPrefix,
    save_weights_only=True
)

EPOCHS = 20
INITIAL_EPOCH = 1
STEPS_PER_EPOCH = 1500

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

def render_training_history(training_history):
    loss = training_history.history['loss']
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss, label='Training set')
    plt.legend()
    plt.grid(linestyle='--', linewidth=1, alpha=0.5)
    plt.show()

render_training_history(history)

