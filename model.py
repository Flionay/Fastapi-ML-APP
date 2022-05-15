import joblib

class ModelApp():
    def __init__(self):
        # load the model
        print("------loading model------")
        self.model = joblib.load('./RF_Boston/MLP.weight')
        # load the scaler
        print("------loading scaler-----")
        self.x_scaler = joblib.load('./RF_Boston/x_scaler')
        self.y_scaler = joblib.load('./RF_Boston/y_scaler')

