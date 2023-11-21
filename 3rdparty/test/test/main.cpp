#include "../runtime/diopirt/diopirt_impl.h"

bool test_diopiGetVersion() {
    std::cout<<diopiGetVersion()<<std::endl;
    return true;
}

bool test_diopiFusedSiluFfnInp() {
    std::cout<<diopiGetVersion()<<std::endl;
    dipu::devapis::setDevice(0);
    dipu::deviceStream_t stream_;
    dipu::devapis::createStream(&stream_);
    diopiContext ctx(stream_);
    int64_t hidden_units_ = 4096;
    int64_t inter_size_ = 11008;
    int64_t temp_size = -1;
    int64_t token_num = 63;
    std::string input0_path = "/nvme/share/share/shenhao/test/data/diopiFusedSiluFfnInp_input0.npy";
    std::string rawoutput0_path = "/nvme/share/share/shenhao/test/data/diopiFusedSiluFfnInp_rawoutput0.npy";
    std::string output0_path = "/nvme/share/share/shenhao/test/data/diopiFusedSiluFfnInp_nvoutput0.npy";
    std::string newoutput0_path = "/nvme/share/share/shenhao/test/data/diopiFusedSiluFfnInp_newoutput0.npy";
    std::string weight1_path = "/nvme/share/share/shenhao/test/data/diopiFusedSiluFfnInp_weight1.npy";
    std::string weight2_path = "/nvme/share/share/shenhao/test/data/diopiFusedSiluFfnInp_weight2.npy";
    std::string weight3_path = "/nvme/share/share/shenhao/test/data/diopiFusedSiluFfnInp_weight3.npy";
    turbomind::Tensor input_tensor = turbomind::Tensor::loadNpy(input0_path, turbomind::MEMORY_GPU);
    diopiTensorHandle_t inout = dipu::diopi_helper::toDiopiTensorHandle(input_tensor);
    turbomind::Tensor weight1_tensor = turbomind::Tensor::loadNpy(weight1_path, turbomind::MEMORY_GPU);
    diopiTensorHandle_t weight1 = dipu::diopi_helper::toDiopiTensorHandle(weight1_tensor);
    std::cout<<"weight1"<<std::endl;
    turbomind::Tensor weight3_tensor = turbomind::Tensor::loadNpy(weight3_path, turbomind::MEMORY_GPU);
    diopiTensorHandle_t weight3 = dipu::diopi_helper::toDiopiTensorHandle(weight3_tensor);
    std::cout<<"weight3"<<std::endl;
    turbomind::Tensor weight2_tensor = turbomind::Tensor::loadNpy(weight2_path, turbomind::MEMORY_GPU);
    diopiTensorHandle_t weight2 = dipu::diopi_helper::toDiopiTensorHandle(weight2_tensor);
    diopiTensorHandle_t workspace;
    diopiFusedSiluFfnInp(&ctx, inout, weight1, weight2, weight3, workspace, &temp_size, 0);
    std::vector<int64_t> shape{temp_size};
    diopiSize_t newshape{shape.data(), 1};
    diopiRequireTensor(&ctx, &workspace, &newshape, nullptr, diopiDtype_t::diopi_dtype_float16, diopiDevice_t::diopi_device);
    std::cout<<"workspace"<<std::endl;
    void* temp;
    diopiGetTensorData(inout, &temp);
    diopiGetTensorData(weight1, &temp);
    diopiGetTensorData(weight2, &temp);
    diopiGetTensorData(weight3, &temp);
    diopiGetTensorData(workspace, &temp);
    diopiFusedSiluFfnInp(&ctx, inout, weight1, weight2, weight3, workspace, &temp_size, 0);
    std::cout<<"diopiFusedSiluFfnInp:"<<temp_size<<std::endl;
    // input_tensor.saveNpy(newoutput0_path);
    return true;
}

#define TEST_FUNC(func, ...)  \
        if (!func(__VA_ARGS__)) { \
            std::cout<<#func<<" FAIL!\n"<<std::endl; \
        } else { \
            std::cout<<#func<<" SUCCESS!\n"<<std::endl; \
        } \

int main() {
    // test test_diopiGetVersion
    // TEST_FUNC(test_diopiGetVersion)
    // test test_diopiFusedSiluFfnInp
    TEST_FUNC(test_diopiFusedSiluFfnInp)



    return 0;
}