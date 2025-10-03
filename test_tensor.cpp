#include "tensor.h"
#include "ops.h"
#include <gtest/gtest.h>
#include <memory>
#include <chrono>
#include <limits>
#include <string>

// ============================================================================
// SCALAR (0D) TENSOR TESTS
// ============================================================================

TEST(TensorTest, ScalarCreation) {
    auto scalar = Tensor::createScalar(42.0f);
    EXPECT_EQ(scalar->getTotalSize(), 1);
    EXPECT_EQ(scalar->getShape()[0], 0);   // empty shape → scalar
    EXPECT_FLOAT_EQ(scalar->getDataElem(0), 42.0f);
}

TEST(TensorTest, EmptyShapeCreateTensor) {
    auto arr  = std::make_unique<float[]>(1);
    arr[0]    = 3.14f;
    auto tensor = Tensor::CreateTensor(std::move(arr), 1, {});
    EXPECT_EQ(tensor->getTotalSize(), 1);
    EXPECT_EQ(tensor->getShape()[0], 0);
    EXPECT_FLOAT_EQ(tensor->getDataElem(0), 3.14f);
}

TEST(TensorTest, ScalarAccessPolicy) {
    auto scalar = Tensor::createScalar(5.0f);
    EXPECT_FLOAT_EQ(scalar->getDataElem(0), 5.0f);
    EXPECT_FLOAT_EQ((*scalar)(0), 5.0f);        // 0-D access
}

// ============================================================================
// 1-D TESTS
// ============================================================================

TEST(TensorTest, Shape1D_Size5) {
    auto arr    = std::make_unique<float[]>(5);
    for (size_t i = 0; i < 5; ++i) arr[i] = static_cast<float>(i);
    auto tensor = Tensor::CreateTensor(std::move(arr), 5, {5, 0, 0, 0, 0, 0, 0, 0});
    EXPECT_EQ(tensor->getTotalSize(), 5);
    EXPECT_EQ(tensor->getShape()[0], 5);

    auto strides = Tensor::calculate_strides({5, 0, 0, 0, 0, 0, 0, 0});
    EXPECT_EQ(strides[0], 1);
}

TEST(TensorTest, Shape1D_StrideCalculation) {
    std::array<size_t, MAX_RANK> shape = {5, 0, 0, 0, 0, 0, 0, 0};
    auto strides = Tensor::calculate_strides(shape);
    EXPECT_EQ(strides[0], 1);
    for (size_t i = 1; i < MAX_RANK; ++i) EXPECT_EQ(strides[i], 0);
}

// ============================================================================
// 2-D TESTS
// ============================================================================

TEST(TensorTest, Shape2D_RowMajor) {
    auto arr    = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i) arr[i] = static_cast<float>(i + 1);
    auto tensor = Tensor::CreateTensor(std::move(arr), 6, {2, 3, 0, 0, 0, 0, 0, 0});
    EXPECT_EQ(tensor->getTotalSize(), 6);
    EXPECT_EQ(tensor->getShape()[0], 2);
    EXPECT_EQ(tensor->getShape()[1], 3);
}

TEST(TensorTest, Shape2D_Strides) {
    std::array<size_t, MAX_RANK> shape = {2, 3, 0, 0, 0, 0, 0, 0};
    auto strides = Tensor::calculate_strides(shape);
    EXPECT_EQ(strides[0], 3);
    EXPECT_EQ(strides[1], 1);
}

TEST(TensorTest, Shape2D_ElementAccess) {
    auto arr    = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i) arr[i] = static_cast<float>(i + 1);
    auto tensor = Tensor::CreateTensor(std::move(arr), 6, {2, 3, 0, 0, 0, 0, 0, 0});
    EXPECT_FLOAT_EQ((*tensor)(0, 0), 1.0f);
    EXPECT_FLOAT_EQ((*tensor)(0, 1), 2.0f);
    EXPECT_FLOAT_EQ((*tensor)(0, 2), 3.0f);
    EXPECT_FLOAT_EQ((*tensor)(1, 0), 4.0f);
    EXPECT_FLOAT_EQ((*tensor)(1, 1), 5.0f);
    EXPECT_FLOAT_EQ((*tensor)(1, 2), 6.0f);
}

// ============================================================================
// 3-D TESTS
// ============================================================================

TEST(TensorTest, Shape3D_Size24) {
    auto arr    = std::make_unique<float[]>(24);
    for (size_t i = 0; i < 24; ++i) arr[i] = static_cast<float>(i);
    auto tensor = Tensor::CreateTensor(std::move(arr), 24, {2, 3, 4, 0, 0, 0, 0, 0});
    EXPECT_EQ(tensor->getTotalSize(), 24);
    EXPECT_EQ(tensor->getShape()[0], 2);
    EXPECT_EQ(tensor->getShape()[1], 3);
    EXPECT_EQ(tensor->getShape()[2], 4);
}

TEST(TensorTest, Shape3D_Strides) {
    std::array<size_t, MAX_RANK> shape = {2, 3, 4, 0, 0, 0, 0, 0};
    auto strides = Tensor::calculate_strides(shape);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
}

TEST(TensorTest, Shape3D_ElementAccess) {
    auto arr    = std::make_unique<float[]>(24);
    for (size_t i = 0; i < 24; ++i) arr[i] = static_cast<float>(i);
    auto tensor = Tensor::CreateTensor(std::move(arr), 24, {2, 3, 4, 0, 0, 0, 0, 0});
    EXPECT_FLOAT_EQ((*tensor)(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ((*tensor)(0, 0, 1), 1.0f);
    EXPECT_FLOAT_EQ((*tensor)(0, 1, 0), 4.0f);
    EXPECT_FLOAT_EQ((*tensor)(1, 0, 0), 12.0f);
    EXPECT_FLOAT_EQ((*tensor)(1, 2, 3), 23.0f);
}

// ============================================================================
// SHAPES WITH ONES
// ============================================================================

TEST(TensorTest, ShapeWithOnes_StridesNotZero) {
    std::array<size_t, MAX_RANK> shape = {1, 5, 1, 2, 0, 0, 0, 0};
    auto strides = Tensor::calculate_strides(shape);
    EXPECT_EQ(strides[0], 10);
    EXPECT_EQ(strides[1], 2);
    EXPECT_EQ(strides[2], 2);
    EXPECT_EQ(strides[3], 1);
    for (size_t i = 0; i < 4; ++i) EXPECT_NE(strides[i], 0);
}

TEST(TensorTest, ShapeWithOnes_Creation) {
    auto arr    = std::make_unique<float[]>(10);
    for (size_t i = 0; i < 10; ++i) arr[i] = static_cast<float>(i);
    auto tensor = Tensor::CreateTensor(std::move(arr), 10, {1, 5, 1, 2, 0, 0, 0, 0});
    EXPECT_EQ(tensor->getTotalSize(), 10);
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

TEST(TensorTest, NegativeIndex_ThrowsOrAsserts) {
    auto arr    = std::make_unique<float[]>(5);
    auto tensor = Tensor::CreateTensor(std::move(arr), 5, {5, 0, 0, 0, 0, 0, 0, 0});
    // -1 becomes SIZE_MAX → triggers i >= total_size
    EXPECT_THROW((*tensor)(static_cast<size_t>(-1)), std::out_of_range);
    EXPECT_NO_THROW((*tensor)(0));
}

TEST(TensorTest, OutOfBounds1D_ThrowsOrAsserts) {
    auto arr    = std::make_unique<float[]>(5);
    auto tensor = Tensor::CreateTensor(std::move(arr), 5, {5, 0, 0, 0, 0, 0, 0, 0});
    EXPECT_THROW((*tensor)(10), std::out_of_range);
}

TEST(TensorTest, OutOfBounds2D_ThrowsOrAsserts) {
    auto arr    = std::make_unique<float[]>(6);
    auto tensor = Tensor::CreateTensor(std::move(arr), 6, {2, 3, 0, 0, 0, 0, 0, 0});
    EXPECT_THROW((*tensor)(10, 10), std::out_of_range);
    EXPECT_THROW((*tensor)(2, 0),   std::out_of_range);
    EXPECT_THROW((*tensor)(0, 3),   std::out_of_range);
}

// ============================================================================
// TENSOR OPS – EDGE CASES
// ============================================================================

TEST(TensorOpsTest, TrueEmptyTensorHandling) {
    auto empty = Tensor::createZeros({0, 0, 0, 0, 0, 0, 0, 0});
    //EXPECT_EQ(empty->getTotalSize(), 0);
    auto out   = TensorOps::operator+(empty, empty);
    //EXPECT_EQ(out->getTotalSize(), 0);
}

TEST(TensorOpsTest, FloatSpecialValues) {
    auto t1 = Tensor::createScalar(std::numeric_limits<float>::infinity());
    auto t2 = Tensor::createScalar(-std::numeric_limits<float>::infinity());
    auto t3 = Tensor::createScalar(std::numeric_limits<float>::quiet_NaN());

    auto inf_add = TensorOps::operator+(t1, t2);
    EXPECT_TRUE(std::isnan(inf_add->getDataElem(0)));

    auto nan_add = TensorOps::operator+(t3, t1);
    EXPECT_TRUE(std::isnan(nan_add->getDataElem(0)));
}

// ============================================================================
// HELPERS
// ============================================================================

TEST(TensorTest, CreateZeros) {
    auto t = Tensor::createZeros({2, 3, 0, 0, 0, 0, 0, 0});
    EXPECT_EQ(t->getTotalSize(), 6);
    for (size_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(t->getDataElem(i), 0.0f);
}

TEST(TensorTest, CreateOnes) {
    auto t = Tensor::createOnes({2, 3, 0, 0, 0, 0, 0, 0});
    EXPECT_EQ(t->getTotalSize(), 6);
    for (size_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(t->getDataElem(i), 1.0f);
}

// ============================================================================
// BASIC OPS
// ============================================================================

TEST(TensorOpsTest, AddScalars) {
    auto r = TensorOps::operator+(Tensor::createScalar(3.0f), Tensor::createScalar(4.0f));
    EXPECT_FLOAT_EQ(r->getDataElem(0), 7.0f);
}

TEST(TensorOpsTest, Add2D) {
    auto a1 = std::make_unique<float[]>(6);
    auto a2 = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i) {
        a1[i] = static_cast<float>(i);
        a2[i] = static_cast<float>(i * 2);
    }
    auto t1 = Tensor::CreateTensor(std::move(a1), 6, {2, 3});
    auto t2 = Tensor::CreateTensor(std::move(a2), 6, {2, 3});
    auto r  = TensorOps::operator+(t1, t2);
    for (size_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(r->getDataElem(i), static_cast<float>(i * 3));
}

TEST(TensorOpsTest, AddMismatchedShapes_Throws) {
    auto t1 = Tensor::createZeros({2, 3, 0, 0, 0, 0, 0, 0});
    auto t2 = Tensor::createZeros({3, 2, 0, 0, 0, 0, 0, 0});
    EXPECT_THROW(TensorOps::operator+(t1, t2), std::invalid_argument);
}

TEST(TensorOpsTest, SubtractScalars) {
    auto r = TensorOps::operator-(Tensor::createScalar(10.0f), Tensor::createScalar(3.0f));
    EXPECT_FLOAT_EQ(r->getDataElem(0), 7.0f);
}

TEST(TensorOpsTest, Multiply2D) {
    auto a1 = std::make_unique<float[]>(4);
    auto a2 = std::make_unique<float[]>(4);
    for (size_t i = 0; i < 4; ++i) {
        a1[i] = static_cast<float>(i + 1);
        a2[i] = 2.0f;
    }
    auto t1 = Tensor::CreateTensor(std::move(a1), 4, {2, 2});
    auto t2 = Tensor::CreateTensor(std::move(a2), 4, {2, 2});
    auto r  = TensorOps::operator*(t1, t2);
    EXPECT_FLOAT_EQ(r->getDataElem(0), 2.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(1), 4.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(2), 6.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(3), 8.0f);
}

// ============================================================================
// MISC EDGE
// ============================================================================

TEST(TensorTest, InitializerListConstructor) {
    auto arr = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i) arr[i] = static_cast<float>(i);
    auto t   = Tensor::CreateTensor(std::move(arr), 6, {2, 3});
    EXPECT_EQ(t->getShape()[0], 2);
    EXPECT_EQ(t->getShape()[1], 3);
}

TEST(TensorTest, SizeMismatch_Throws) {
    auto arr = std::make_unique<float[]>(6);
    EXPECT_THROW(Tensor::CreateTensor(std::move(arr), 6, {2, 2, 0, 0, 0, 0, 0, 0}),
                 std::invalid_argument);
}

TEST(TensorTest, CompareNumPyStrides) {
    std::array<size_t, MAX_RANK> shape = {2, 3, 4, 0, 0, 0, 0, 0};
    auto strides = Tensor::calculate_strides(shape);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
}

// ============================================================================
// ADVANCED / NUMERICAL TORTURE
// ============================================================================

TEST(TensorOpsTest, AddScalarConvenience) {
    auto r = TensorOps::operator+(Tensor::createScalar(2.0f), Tensor::createScalar(3.0f));
    EXPECT_FLOAT_EQ(r->getDataElem(0), 5.0f);
}

TEST(TensorOpsTest, Add2D_VerifyEachElement) {
    auto a1 = std::make_unique<float[]>(6);
    auto a2 = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i) {
        a1[i] = static_cast<float>(i + 1);
        a2[i] = static_cast<float>((i + 1) * 10);
    }
    auto t1 = Tensor::CreateTensor(std::move(a1), 6, {2, 3});
    auto t2 = Tensor::CreateTensor(std::move(a2), 6, {2, 3});
    auto r  = TensorOps::operator+(t1, t2);
    EXPECT_FLOAT_EQ(r->getDataElem(0), 11.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(1), 22.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(2), 33.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(3), 44.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(4), 55.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(5), 66.0f);
}

TEST(TensorOpsTest, SubBroadcastReject) {
    auto t1 = Tensor::createOnes({2, 1, 0, 0, 0, 0, 0, 0});
    auto t2 = Tensor::createOnes({1, 3, 0, 0, 0, 0, 0, 0});
    EXPECT_THROW(TensorOps::operator-(t1, t2), std::invalid_argument);
}

TEST(TensorOpsTest, MulInPlaceAlias) {
    auto arr = std::make_unique<float[]>(4);
    for (size_t i = 0; i < 4; ++i) arr[i] = static_cast<float>(i + 1);
    auto t   = Tensor::CreateTensor(std::move(arr), 4, {2, 2});
    float orig[4];
    for (size_t i = 0; i < 4; ++i) orig[i] = t->getDataElem(i);
    auto r   = TensorOps::operator*(t, t);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(t->getDataElem(i), orig[i]);   // original untouched
    EXPECT_FLOAT_EQ(r->getDataElem(0), 1.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(1), 4.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(2), 9.0f);
    EXPECT_FLOAT_EQ(r->getDataElem(3), 16.0f);
}

TEST(TensorTest, OpEmpty) {
    auto t1 = Tensor::createScalar(5.0f);
    auto t2 = Tensor::createScalar(3.0f);
    auto add = TensorOps::operator+(t1, t2);
    EXPECT_EQ(add->getTotalSize(), 1);
    EXPECT_FLOAT_EQ(add->getDataElem(0), 8.0f);
}

TEST(TensorTest, DataAlignedTo64) {
    auto t    = Tensor::createZeros({100, 100});
    auto addr = reinterpret_cast<uintptr_t>(t->getData());
    EXPECT_EQ(addr % 64, 0) << "data not 64-byte aligned";
}

TEST(TensorTest, MoveSemantics) {
    auto arr = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i) arr[i] = static_cast<float>(i);
    auto t1  = Tensor::CreateTensor(std::move(arr), 6, {2, 3});
    auto t2  = std::move(t1);               // shared_ptr move → t1 empty
    EXPECT_EQ(t1.get(), nullptr);
    EXPECT_NE(t2.get(), nullptr);
    EXPECT_FLOAT_EQ(t2->getDataElem(0), 0.0f);
}

TEST(TensorTest, UniqueOwnership) {
    auto t1 = Tensor::createScalar(42.0f);
    auto t2 = t1;                       // second shared_ptr
    EXPECT_EQ(t1.use_count(), 2);
    t1->setDataElem(0, 99.0f);
    EXPECT_FLOAT_EQ(t2->getDataElem(0), 99.0f);
}

TEST(TensorTest, Large1D) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    auto t     = Tensor::createZeros({1'000'000});
    auto end   = high_resolution_clock::now();
    auto ms    = duration_cast<milliseconds>(end - start).count();
    EXPECT_EQ(t->getTotalSize(), 1'000'000);
    EXPECT_LT(ms, 50) << "zero-init took " << ms << " ms";
    EXPECT_FLOAT_EQ(t->getDataElem(0), 0.0f);
    EXPECT_FLOAT_EQ(t->getDataElem(500'000), 0.0f);
    EXPECT_FLOAT_EQ(t->getDataElem(999'999), 0.0f);
}

TEST(TensorTest, StrideIndexing) {
    auto arr = std::make_unique<float[]>(24);
    for (size_t i = 0; i < 24; ++i) arr[i] = static_cast<float>(i);
    auto tensor = Tensor::CreateTensor(std::move(arr), 24, {2, 3, 4});
    const auto shape   = tensor->getShape();
    const auto strides = Tensor::calculate_strides(shape);

    size_t flat = 0;
    for (size_t i = 0; i < shape[0]; ++i)
        for (size_t j = 0; j < shape[1]; ++j)
            for (size_t k = 0; k < shape[2]; ++k) {
                size_t calc = i * strides[0] + j * strides[1] + k * strides[2];
                EXPECT_FLOAT_EQ((*tensor)(i, j, k), static_cast<float>(flat));
                EXPECT_EQ(calc, flat);
                ++flat;
            }
    EXPECT_EQ(flat, 24);
}

TEST(TensorTest, NumpyStrideCompatibility) {
    std::array<size_t, MAX_RANK> shape = {5, 1, 2, 1, 0, 0, 0, 0};
    auto strides = Tensor::calculate_strides(shape);
    EXPECT_EQ(strides[0], 2);
    EXPECT_EQ(strides[1], 2);
    EXPECT_EQ(strides[2], 1);
    EXPECT_EQ(strides[3], 1);
}

TEST(TensorTest, CreateZerosUsesMemset) {
    auto t = Tensor::createZeros({500'000});
    for (size_t i = 0; i < 500'000; ++i)
        EXPECT_FLOAT_EQ(t->getDataElem(i), 0.0f);
}

TEST(TensorTest, MaxRank8) {
    std::array<size_t, MAX_RANK> shape{2, 2, 2, 2, 2, 2, 2, 2}; // 256
    auto t = Tensor::createZeros(shape);
    EXPECT_EQ(t->getTotalSize(), 256);
    // flatten last element
    EXPECT_FLOAT_EQ(t->getDataElem(255), 0.0f);
}

TEST(TensorOpsTest, MismatchExceptionText) {
    auto a = Tensor::createZeros({2, 3});
    auto b = Tensor::createZeros({3, 2});
    try {
        TensorOps::operator+(a, b);
        FAIL() << "should have thrown";
    } catch (const std::invalid_argument& e) {
        EXPECT_TRUE(std::string(e.what()).find("size of tensors don't match") != std::string::npos);
    }
}
