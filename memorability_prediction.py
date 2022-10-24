"""
--- UNUSED CODE ---
"""

# train an MLP on layer similarity scores and on concatenated semantic embeddings

class MemMLP(tf.keras.Model):
    def __init__(self, activation, num_layers):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(num_layers//2, activation=activation)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.hidden2 = tf.keras.layers.Dense(num_layers//4, activation=activation)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.mem = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.hidden(inputs)
        x = self.dropout1(x)
        x = self.hidden2(x)
        x = self.dropout2(x)
        x = self.mem(x)
        return x

    def model(self):
        inputs = tf.keras.Input(shape=(1, 9408))
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))

mem_mlp = MemMLP(activation="sigmoid", num_layers=9408)
mem_mlp.model().summary()
opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
mem_mlp.compile(
    optimizer = opt,
    loss = "mse",
    metrics = ["mae"]
)

history = mem_mlp.fit(
    X_train,
    y_train, 
    batch_size = 4,
    shuffle = True,
    epochs = 15, 
    validation_split = 0.2,
)