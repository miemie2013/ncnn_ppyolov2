
'''
（在pycharm的普通终端输入）：
cd build/examples
.\test2_01_conv_ncnn 0 ../../my_tests/my_test32.jpg ../../my_tests/11_pncnn.param ../../my_tests/11_pncnn.bin

.\test2_01_conv_ncnn 1 in0.bin ../../my_tests/11_pncnn.param ../../my_tests/11_pncnn.bin 4 8 8 1 1



(linux)
cd build/examples
./test2_01_conv_ncnn ../../my_tests/my_test32.jpg ../../my_tests/11_pncnn.param ../../my_tests/11_pncnn.bin


（在pycharm的普通终端输入）：
cd my_tests
../tools/pnnx/build/install/bin/pnnx 11.pt inputshape=[1,3,32,32] moduleop=models.common.Focus


(linux)
cd my_tests
../tools/pnnx/build/install/bin/pnnx 11.pt inputshape=[1,3,32,32]


(预测)
cd build/examples
.\test2_01_conv_ncnn ../../my_tests/my_test32.jpg ../../my_tests/11.ncnn.param ../../my_tests/11.ncnn.bin




'''



