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

    def input_process_component(self, input_data):
        '''
        :param data:
        :return:
        模型预测前 数据处理 组件
        '''
        # feature selection
        input_data = input_data[[True, False, False, False, True, True, False, True, False,
                                 False, False, False, True]]

        input_data = self.x_scaler.transform(input_data.reshape((1, -1)))
        return input_data

    def output_process_component(self, ypre):
        '''
        模型预测后，反归一化组件
        :return:
        '''
        out = self.y_scaler.inverse_transform(ypre.reshape((1, -1)))
        return out
