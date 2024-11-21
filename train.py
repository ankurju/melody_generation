import tensorflow.keras as keras
from preprocess import generate_training_sequences
from config import OUTPUT_UNITS,NUM_UNITS,LOSS,LEARNING_RATE,EPOCHS,BATCH_SIZE,SAVE_MODEL_PATH,SEQUENCE_LENGTH


def build_model(output_units, num_units, loss, learning_rate):
    #create the model architecture
    input = keras.layers.Input(shape=(None,output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units,activation="softmax")(x)

    model = keras.Model(input,output)

    #compile the model
    model.compile(loss=loss,optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=["accuracy"])

    model.summary()

    return model

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE):
    #generate the training sequence
    X,y = generate_training_sequences(SEQUENCE_LENGTH)

    #build the network
    model = build_model(output_units,num_units,loss,learning_rate)

    #train the model
    model.fit(X,y,epochs=epochs,batch_size=batch_size)

    #save the model
    model.save(SAVE_MODEL_PATH, save_format='keras')

if __name__ == "__main__":
    train()