from tensorflow.keras.callbacks import Callback

class LoggerDePerda(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"🔁 Época {epoch + 1}: perda = {logs.get('loss'):.5f}")