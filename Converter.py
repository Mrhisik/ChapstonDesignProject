from rknn.api import RKNN

INPUT_SIZE = 11

if __name__ == '__main__':
    # Create RKNN execution objects
    rknn = RKNN()
    # Configure model input for NPU preprocessing of data input
    # channel_mean_value='0 0 0 255'，In model reasoning, RGB data will be transformed as follows
    # (R - 0)/255, (G - 0)/255, (B - 0)/255。When reasoning, RKNN model will automatically do mean and normalization processing
    # reorder_channel=’0 1 2’ Used to specify whether to adjust the image channel order, set to 0 1 2, that is, do not adjust according to the order of the input image channel.
    # reorder_channel=’2 1 0’ Represents switching channels 0 and 2, and if the input is RGB, it will be adjusted to BGR. If it is BGR, it will be adjusted to RGB.
    #The order of image channels is not adjusted
    rknn.config(target_platform="rk3568", mean_values=[0], std_values=[1])

    # Loading TensorFlow Model
    # tf_pb='digital_gesture.pb' Specify the TensorFlow model to be converted
    # Inputs specify input nodes in the model
    # Outputs specify the output node in the model
    # Input_size_list specifies the size of model input
    print('--> Loading model')
    rknn.load_tflite(model='sleep_moni.tflite')
    print('done')

    # Creating Analytical Pb Model
    # do_quantization=False Specify not to quantify
    # Quantization reduces the size of the model and improves the speed of computation, but it loses accuracy.
    print('--> Building model')
    rknn.build(do_quantization=False)
    print('done')


    # Export and save RkNN model file
    rknn.export_rknn('./sleep_moni.rknn')

    # Release RKNN Context
    rknn.release()
