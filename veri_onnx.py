import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader, CalibrationMethod
import numpy as np
import time
import os
import csv

model_path = 'model/cnn.onnx'
quantized_model_path = "model/quantized_model.onnx"
calibration_data_path = 'calibrate_data/'
onnx_model = onnx.load(model_path)

# 预处理函数
def proprocess_func(dataset_folder, height, width, size_limit=0):
    dataset_names = os.listdir(dataset_folder)
    input_data = np.zeros((64,1,300), np.float32)
    if size_limit > 0 and len(dataset_names) >= size_limit:
        batch_filenames = [dataset_names[i] for i in range(size_limit)]
    else:
        batch_filenames = dataset_names
    unconcatenated_batch_data = []
    for dataset_name in batch_filenames:
        # print(image_name)
        dataset_filepath = dataset_folder + '/' + dataset_name
        with open(dataset_filepath, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for i, row in enumerate(csvreader):
                # datalist = row.split()
                for j, data in enumerate(row):
                    if((i>0 and i<64) and j<300):
                        input_data[i-1][0][j] = float(data)
            # for i,line in enumerate(txtfile):
            #     data_list = line.strip().split()  # 去除行尾的换行符，并以空格分隔数据
            #     for j,data in enumerate(data_list):
            #         if(i<64 and j<300):
            #             input_data[i][0][j] = int(data, 2)/63.5-1
        nhwc_data = np.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 1, 2, 3)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(
        unconcatenated_batch_data
    )
    return batch_data


def benchmark(model_path):
    """
    用于测试速度
    """
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((64, 1, 300), np.float32)  # 随便输入一个假数据
    # warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


# DataReader类
class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = ort.InferenceSession(model_path, None)
        (height, _, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = proprocess_func(
            calibration_image_folder, height, width, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)
    # def __init__(self, calibration_data_folder, augmented_model_path):
    #     self.image_folder = calibration_data_folder
    #     self.augmented_model_path = augmented_model_path
    #     self.preprocess_flag = True
    #     self.enum_data_dicts = []
    #     self.datasize = 0
    #
    # def get_next(self):
    #     if self.preprocess_flag:
    #         self.preprocess_flag = False
    #         session = ort.InferenceSession(self.augmented_model_path, None)
    #         width, height = session.get_inputs()[0].shape
    #         nhwc_data_list = proprocess_func(self.image_folder, height, width, size_limit=0)
    #         input_name = session.get_inputs()[0].name
    #         self.datasize = len(nhwc_data_list)
    #         self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
    #     return next(self.enum_data_dicts, None)


if __name__ == '__main__':
    datareader = DataReader(calibration_data_path, model_path)
    #执行静态量化
    quantize_static(
        model_input = model_path,
        model_output=quantized_model_path,
        calibration_data_reader=datareader,
        calibrate_method= CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8
    )
    print("Calibrated and quantized model saved.")
    print("benchmarking fp32 model...")
    benchmark(model_path)
    print("benchmarking int8 model...")
    benchmark(quantized_model_path)

    # verify if correct
    ort_session = ort.InferenceSession(model_path)
    ort_quan_session = ort.InferenceSession(quantized_model_path)

    input_name = ort_session.get_inputs()[0].name
    input_name_quan = ort_quan_session.get_inputs()[0].name

    # warming up
    for i in range(10):
        input_data = np.zeros((64, 1, 300), np.float32)  # 随便输入一个假数据
        input_quan = np.zeros((64, 1, 300), np.float32)
        with open('calibrate_data/NAVLR_rawdata_310.csv', 'r') as csv_file:
            reader = csv.reader(csv_file)
            for i, row in enumerate(reader):
                if(i>0 and i<64):
                    for j, data in enumerate(row):
                        if j < 300:
                            try:
                                data_int = int((float(data) + 1) * 63.5)
                            except ValueError:
                                data_int = 0
                            data_int = max(0, min(data_int, 127))
                            binary_data = format(data_int, '07b')
                            if float(data) < 0:
                                binary_data = '1' + binary_data
                            else:
                                binary_data = '0' + binary_data
                            input_data[i][0][j] = data
                            input_quan[i][0][j] = float(np.int8(binary_data))
        # input_data = np.float32(np.random.randn(64, 1, 300))  # 随便输入一个假数据
        # input_quan = input_data
        output = ort_session.run([], {input_name: input_data})
        output_quan = ort_quan_session.run([], {input_name_quan: input_data})
        print(f"f32 out is\n{output}\n int8 out is\n{output_quan}\n \n\n", output, output_quan)

