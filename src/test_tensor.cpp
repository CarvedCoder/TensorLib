#include "../include/tensor.h"
#include "../include/ops.h"
#include <gtest/gtest.h>
#include <memory>
#include <chrono>
#include <limits>
#include <string>
#include <vector>
#include <cmath>
// SCALAR (0D) TENSOR TESTS

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
    std::vector<size_t> shape = {5};
    auto tensor = Tensor::CreateTensor(std::move(arr), 5, shape);
    EXPECT_EQ(tensor->getTotalSize(), 5);
    EXPECT_EQ(tensor->getShape()[0], 5);

    auto strides = Tensor::calculateStrides({5, 0, 0, 0, 0, 0, 0, 0});
    EXPECT_EQ(strides[0], 1);
}

TEST(TensorTest, Shape1D_StrideCalculation) {
    std::array<size_t, MAX_RANK> shape = {5, 0, 0, 0, 0, 0, 0, 0};
    auto strides = Tensor::calculateStrides(shape);
    EXPECT_EQ(strides[0], 1);
    for (size_t i = 1; i < MAX_RANK; ++i) EXPECT_EQ(strides[i], 0);
}

// ============================================================================
// 2-D TESTS
// ============================================================================

TEST(TensorTest, Shape2D_RowMajor) {
    auto arr    = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i) arr[i] = static_cast<float>(i + 1);
    auto tensor = Tensor::CreateTensor(std::move(arr), 6, {2, 3});
    EXPECT_EQ(tensor->getTotalSize(), 6);
    EXPECT_EQ(tensor->getShape()[0], 2);
    EXPECT_EQ(tensor->getShape()[1], 3);
}

TEST(TensorTest, Shape2D_Strides) {
    std::array<size_t, MAX_RANK> shape = {2, 3, 0, 0, 0, 0, 0, 0};
    auto strides = Tensor::calculateStrides(shape);
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
    auto strides = Tensor::calculateStrides(shape);
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
    auto strides = Tensor::calculateStrides(shape);
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
    auto strides = Tensor::calculateStrides(shape);
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
    const auto strides = Tensor::calculateStrides(shape);

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
    auto strides = Tensor::calculateStrides(shape);
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
        EXPECT_TRUE(std::string(e.what()).find("shape of tensors don't match for add operator") != std::string::npos);
    }
}
// ============================================================================
// TRANSPOSE 2D TESTS
// ============================================================================

TEST(TensorOpsTest, Transpose2D_Square) {
    auto arr = std::make_unique<float[]>(9);
    for (size_t i = 0; i < 9; ++i) arr[i] = static_cast<float>(i);
    // [[0,1,2], [3,4,5], [6,7,8]]
    auto t = Tensor::CreateTensor(std::move(arr), 9, {3, 3});
    auto t_T = TensorOps::transpose2D(t);

    EXPECT_EQ(t_T->getShape()[0], 3);
    EXPECT_EQ(t_T->getShape()[1], 3);
    EXPECT_FLOAT_EQ((*t_T)(0, 0), 0.0f);  // [0,0] unchanged
    EXPECT_FLOAT_EQ((*t_T)(0, 1), 3.0f);  // [0,1] was [1,0]
    EXPECT_FLOAT_EQ((*t_T)(0, 2), 6.0f);  // [0,2] was [2,0]
    EXPECT_FLOAT_EQ((*t_T)(1, 0), 1.0f);  // [1,0] was [0,1]
    EXPECT_FLOAT_EQ((*t_T)(1, 1), 4.0f);  // [1,1] unchanged
    EXPECT_FLOAT_EQ((*t_T)(2, 2), 8.0f);  // [2,2] unchanged
}

TEST(TensorOpsTest, Transpose2D_Rectangular) {
    auto arr = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i) arr[i] = static_cast<float>(i + 1);
    // [[1,2,3], [4,5,6]]  (2x3)
    auto t = Tensor::CreateTensor(std::move(arr), 6, {2, 3});
    auto t_T = TensorOps::transpose2D(t);

    EXPECT_EQ(t_T->getShape()[0], 3);
    EXPECT_EQ(t_T->getShape()[1], 2);
    EXPECT_FLOAT_EQ((*t_T)(0, 0), 1.0f);  // [0,0] from [0,0]
    EXPECT_FLOAT_EQ((*t_T)(0, 1), 4.0f);  // [0,1] from [1,0]
    EXPECT_FLOAT_EQ((*t_T)(1, 0), 2.0f);  // [1,0] from [0,1]
    EXPECT_FLOAT_EQ((*t_T)(1, 1), 5.0f);  // [1,1] from [1,1]
    EXPECT_FLOAT_EQ((*t_T)(2, 0), 3.0f);  // [2,0] from [0,2]
    EXPECT_FLOAT_EQ((*t_T)(2, 1), 6.0f);  // [2,1] from [1,2]
}

TEST(TensorOpsTest, Transpose2D_SingleRow) {
    auto arr = std::make_unique<float[]>(4);
    for (size_t i = 0; i < 4; ++i) arr[i] = static_cast<float>(i);
    // [[0,1,2,3]]  (1x4)
    auto t = Tensor::CreateTensor(std::move(arr), 4, {1, 4});
    auto t_T = TensorOps::transpose2D(t);

    EXPECT_EQ(t_T->getShape()[0], 4);
    EXPECT_EQ(t_T->getShape()[1], 1);
    EXPECT_FLOAT_EQ((*t_T)(0, 0), 0.0f);
    EXPECT_FLOAT_EQ((*t_T)(1, 0), 1.0f);
    EXPECT_FLOAT_EQ((*t_T)(2, 0), 2.0f);
    EXPECT_FLOAT_EQ((*t_T)(3, 0), 3.0f);
}

TEST(TensorOpsTest, Transpose2D_SingleColumn) {
    auto arr = std::make_unique<float[]>(4);
    for (size_t i = 0; i < 4; ++i) arr[i] = static_cast<float>(i);
    // [[0], [1], [2], [3]]  (4x1)
    auto t = Tensor::CreateTensor(std::move(arr), 4, {4, 1});
    auto t_T = TensorOps::transpose2D(t);

    EXPECT_EQ(t_T->getShape()[0], 1);
    EXPECT_EQ(t_T->getShape()[1], 4);
    EXPECT_FLOAT_EQ((*t_T)(0, 0), 0.0f);
    EXPECT_FLOAT_EQ((*t_T)(0, 1), 1.0f);
    EXPECT_FLOAT_EQ((*t_T)(0, 2), 2.0f);
    EXPECT_FLOAT_EQ((*t_T)(0, 3), 3.0f);
}

TEST(TensorOpsTest, Transpose2D_DoubleTranspose) {
    auto arr = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i) arr[i] = static_cast<float>(i);
    auto t = Tensor::CreateTensor(std::move(arr), 6, {2, 3});
    auto t_T = TensorOps::transpose2D(t);
    auto t_T_T = TensorOps::transpose2D(t_T);

    EXPECT_EQ(t_T_T->getShape()[0], 2);
    EXPECT_EQ(t_T_T->getShape()[1], 3);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(t_T_T->getDataElem(i), t->getDataElem(i));
    }
}

TEST(TensorOpsTest, Transpose2D_3DTensor_Throws) {
    auto arr = std::make_unique<float[]>(8);
    auto t = Tensor::CreateTensor(std::move(arr), 8, {2, 2, 2});
    EXPECT_THROW(TensorOps::transpose2D(t), std::invalid_argument);
}

TEST(TensorOpsTest, Transpose2D_1DTensor_Throws) {
    auto arr = std::make_unique<float[]>(5);
    auto t = Tensor::CreateTensor(std::move(arr), 5, {5});
    EXPECT_THROW(TensorOps::transpose2D(t), std::invalid_argument);
}

// ============================================================================
// MATMUL TESTS
// ============================================================================

TEST(TensorOpsTest, Matmul_2x2_Times_2x2) {
    auto a1 = std::make_unique<float[]>(4);
    auto a2 = std::make_unique<float[]>(4);
    // A = [[1,2], [3,4]]
    a1[0] = 1.0f; a1[1] = 2.0f; a1[2] = 3.0f; a1[3] = 4.0f;
    // B = [[5,6], [7,8]]
    a2[0] = 5.0f; a2[1] = 6.0f; a2[2] = 7.0f; a2[3] = 8.0f;

    auto A = Tensor::CreateTensor(std::move(a1), 4, {2, 2});
    auto B = Tensor::CreateTensor(std::move(a2), 4, {2, 2});
    auto C = TensorOps::matmul(A, B);

    // C = [[19,22], [43,50]]
    EXPECT_EQ(C->getShape()[0], 2);
    EXPECT_EQ(C->getShape()[1], 2);
    EXPECT_FLOAT_EQ((*C)(0, 0), 19.0f);  // 1*5 + 2*7
    EXPECT_FLOAT_EQ((*C)(0, 1), 22.0f);  // 1*6 + 2*8
    EXPECT_FLOAT_EQ((*C)(1, 0), 43.0f);  // 3*5 + 4*7
    EXPECT_FLOAT_EQ((*C)(1, 1), 50.0f);  // 3*6 + 4*8
}

TEST(TensorOpsTest, Matmul_2x3_Times_3x2) {
    auto a1 = std::make_unique<float[]>(6);
    auto a2 = std::make_unique<float[]>(6);
    // A = [[1,2,3], [4,5,6]]  (2x3)
    for (size_t i = 0; i < 6; ++i) a1[i] = static_cast<float>(i + 1);
    // B = [[7,8], [9,10], [11,12]]  (3x2)
    for (size_t i = 0; i < 6; ++i) a2[i] = static_cast<float>(i + 7);

    auto A = Tensor::CreateTensor(std::move(a1), 6, {2, 3});
    auto B = Tensor::CreateTensor(std::move(a2), 6, {3, 2});
    auto C = TensorOps::matmul(A, B);

    EXPECT_EQ(C->getShape()[0], 2);
    EXPECT_EQ(C->getShape()[1], 2);
    EXPECT_FLOAT_EQ((*C)(0, 0), 58.0f);   // 1*7 + 2*9 + 3*11
    EXPECT_FLOAT_EQ((*C)(0, 1), 64.0f);   // 1*8 + 2*10 + 3*12
    EXPECT_FLOAT_EQ((*C)(1, 0), 139.0f);  // 4*7 + 5*9 + 6*11
    EXPECT_FLOAT_EQ((*C)(1, 1), 154.0f);  // 4*8 + 5*10 + 6*12
}

TEST(TensorOpsTest, Matmul_3x4_Times_4x2) {
    auto a1 = std::make_unique<float[]>(12);
    auto a2 = std::make_unique<float[]>(8);
    for (size_t i = 0; i < 12; ++i) a1[i] = static_cast<float>(i);
    for (size_t i = 0; i < 8; ++i) a2[i] = 1.0f;

    auto A = Tensor::CreateTensor(std::move(a1), 12, {3, 4});
    auto B = Tensor::CreateTensor(std::move(a2), 8, {4, 2});
    auto C = TensorOps::matmul(A, B);

    EXPECT_EQ(C->getShape()[0], 3);
    EXPECT_EQ(C->getShape()[1], 2);
    // Each row of C sums elements of corresponding row in A
    EXPECT_FLOAT_EQ((*C)(0, 0), 6.0f);   // 0+1+2+3
    EXPECT_FLOAT_EQ((*C)(0, 1), 6.0f);
    EXPECT_FLOAT_EQ((*C)(1, 0), 22.0f);  // 4+5+6+7
    EXPECT_FLOAT_EQ((*C)(1, 1), 22.0f);
    EXPECT_FLOAT_EQ((*C)(2, 0), 38.0f);  // 8+9+10+11
    EXPECT_FLOAT_EQ((*C)(2, 1), 38.0f);
}

TEST(TensorOpsTest, Matmul_Identity) {
    auto a1 = std::make_unique<float[]>(4);
    auto a2 = std::make_unique<float[]>(4);
    // A = [[2,3], [4,5]]
    a1[0] = 2.0f; a1[1] = 3.0f; a1[2] = 4.0f; a1[3] = 5.0f;
    // I = [[1,0], [0,1]]
    a2[0] = 1.0f; a2[1] = 0.0f; a2[2] = 0.0f; a2[3] = 1.0f;

    auto A = Tensor::CreateTensor(std::move(a1), 4, {2, 2});
    auto I = Tensor::CreateTensor(std::move(a2), 4, {2, 2});
    auto C = TensorOps::matmul(A, I);

    EXPECT_FLOAT_EQ((*C)(0, 0), 2.0f);
    EXPECT_FLOAT_EQ((*C)(0, 1), 3.0f);
    EXPECT_FLOAT_EQ((*C)(1, 0), 4.0f);
    EXPECT_FLOAT_EQ((*C)(1, 1), 5.0f);
}

TEST(TensorOpsTest, Matmul_MatrixVector) {
    auto a1 = std::make_unique<float[]>(6);
    auto a2 = std::make_unique<float[]>(3);
    // A = [[1,2,3], [4,5,6]]  (2x3)
    for (size_t i = 0; i < 6; ++i) a1[i] = static_cast<float>(i + 1);
    // v = [[1], [2], [3]]  (3x1)
    a2[0] = 1.0f; a2[1] = 2.0f; a2[2] = 3.0f;

    auto A = Tensor::CreateTensor(std::move(a1), 6, {2, 3});
    auto v = Tensor::CreateTensor(std::move(a2), 3, {3, 1});
    auto result = TensorOps::matmul(A, v);

    EXPECT_EQ(result->getShape()[0], 2);
    EXPECT_EQ(result->getShape()[1], 1);
    EXPECT_FLOAT_EQ((*result)(0, 0), 14.0f);  // 1*1 + 2*2 + 3*3
    EXPECT_FLOAT_EQ((*result)(1, 0), 32.0f);  // 4*1 + 5*2 + 6*3
}

TEST(TensorOpsTest, Matmul_VectorMatrix) {
    auto a1 = std::make_unique<float[]>(3);
    auto a2 = std::make_unique<float[]>(6);
    // v = [[1,2,3]]  (1x3)
    a1[0] = 1.0f; a1[1] = 2.0f; a1[2] = 3.0f;
    // B = [[1,2], [3,4], [5,6]]  (3x2)
    for (size_t i = 0; i < 6; ++i) a2[i] = static_cast<float>(i + 1);

    auto v = Tensor::CreateTensor(std::move(a1), 3, {1, 3});
    auto B = Tensor::CreateTensor(std::move(a2), 6, {3, 2});
    auto result = TensorOps::matmul(v, B);

    EXPECT_EQ(result->getShape()[0], 1);
    EXPECT_EQ(result->getShape()[1], 2);
    EXPECT_FLOAT_EQ((*result)(0, 0), 22.0f);  // 1*1 + 2*3 + 3*5
    EXPECT_FLOAT_EQ((*result)(0, 1), 28.0f);  // 1*2 + 2*4 + 3*6
}

TEST(TensorOpsTest, Matmul_ZeroMatrix) {
    auto A = Tensor::createZeros({2, 3});
    auto B = Tensor::createOnes({3, 2});
    auto C = TensorOps::matmul(A, B);

    EXPECT_EQ(C->getShape()[0], 2);
    EXPECT_EQ(C->getShape()[1], 2);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(C->getDataElem(i), 0.0f);
    }
}

TEST(TensorOpsTest, Matmul_InnerDimensionMismatch_Throws) {
    auto A = Tensor::createZeros({2, 3});
    auto B = Tensor::createZeros({4, 2});  // 3 != 4
    EXPECT_THROW(TensorOps::matmul(A, B), std::invalid_argument);
}

TEST(TensorOpsTest, Matmul_3DTensor_Throws) {
    auto arr1 = std::make_unique<float[]>(8);
    auto arr2 = std::make_unique<float[]>(8);
    auto A = Tensor::CreateTensor(std::move(arr1), 8, {2, 2, 2});
    auto B = Tensor::CreateTensor(std::move(arr2), 8, {2, 2, 2});
    EXPECT_THROW(TensorOps::matmul(A, B), std::invalid_argument);
}

TEST(TensorOpsTest, Matmul_1DTensor_Throws) {
    auto arr1 = std::make_unique<float[]>(3);
    auto arr2 = std::make_unique<float[]>(3);
    auto A = Tensor::CreateTensor(std::move(arr1), 3, {3});
    auto B = Tensor::CreateTensor(std::move(arr2), 3, {3});
    EXPECT_THROW(TensorOps::matmul(A, B), std::invalid_argument);
}

TEST(TensorOpsTest, Matmul_LargeMatrix) {
    auto A = Tensor::createOnes({100, 50});
    auto B = Tensor::createOnes({50, 75});
    auto C = TensorOps::matmul(A, B);

    EXPECT_EQ(C->getShape()[0], 100);
    EXPECT_EQ(C->getShape()[1], 75);
    // Each element should be 50 (sum of 50 ones)
    EXPECT_FLOAT_EQ((*C)(0, 0), 50.0f);
    EXPECT_FLOAT_EQ((*C)(50, 37), 50.0f);
    EXPECT_FLOAT_EQ((*C)(99, 74), 50.0f);
}

TEST(TensorOpsTest, Matmul_NonCommutative) {
    auto a1 = std::make_unique<float[]>(6);
    auto a2 = std::make_unique<float[]>(6);
    // A = [[1,2,3], [4,5,6]]  (2x3)
    for (size_t i = 0; i < 6; ++i) a1[i] = static_cast<float>(i + 1);
    // B = [[1,2], [3,4], [5,6]]  (3x2)
    for (size_t i = 0; i < 6; ++i) a2[i] = static_cast<float>(i + 1);

    auto A = Tensor::CreateTensor(std::move(a1), 6, {2, 3});
    auto B = Tensor::CreateTensor(std::move(a2), 6, {3, 2});

    auto AB = TensorOps::matmul(A, B);  // 2x2
    auto BA = TensorOps::matmul(B, A);  // 3x3

    EXPECT_EQ(AB->getShape()[0], 2);
    EXPECT_EQ(AB->getShape()[1], 2);
    EXPECT_EQ(BA->getShape()[0], 3);
    EXPECT_EQ(BA->getShape()[1], 3);

    // Verify they're different
    EXPECT_NE(AB->getTotalSize(), BA->getTotalSize());
}

TEST(TensorOpsTest, Matmul_SingleElement) {
    auto a1 = std::make_unique<float[]>(1);
    auto a2 = std::make_unique<float[]>(1);
    a1[0] = 3.0f;
    a2[0] = 7.0f;

    auto A = Tensor::CreateTensor(std::move(a1), 1, {1, 1});
    auto B = Tensor::CreateTensor(std::move(a2), 1, {1, 1});
    auto C = TensorOps::matmul(A, B);

    EXPECT_EQ(C->getShape()[0], 1);
    EXPECT_EQ(C->getShape()[1], 1);
    EXPECT_FLOAT_EQ((*C)(0, 0), 21.0f);
}

TEST(TensorOpsTest, Matmul_NegativeValues) {
    auto a1 = std::make_unique<float[]>(4);
    auto a2 = std::make_unique<float[]>(4);
    a1[0] = -1.0f; a1[1] = 2.0f; a1[2] = -3.0f; a1[3] = 4.0f;
    a2[0] = 1.0f; a2[1] = -2.0f; a2[2] = 3.0f; a2[3] = -4.0f;

    auto A = Tensor::CreateTensor(std::move(a1), 4, {2, 2});
    auto B = Tensor::CreateTensor(std::move(a2), 4, {2, 2});
    auto C = TensorOps::matmul(A, B);

    EXPECT_FLOAT_EQ((*C)(0, 0), 5.0f);    // -1*1 + 2*3
    EXPECT_FLOAT_EQ((*C)(0, 1), -6.0f);   // -1*(-2) + 2*(-4)
    EXPECT_FLOAT_EQ((*C)(1, 0), 9.0f);    // -3*1 + 4*3
    EXPECT_FLOAT_EQ((*C)(1, 1), -10.0f);  // -3*(-2) + 4*(-4)
}
// ============================================================================
// INTENSIVE NUMERICAL CALCULATIONS
// ============================================================================

TEST(TensorIntensiveTest, ChainedOperations_PreservesAccuracy) {
    // Test numerical stability through multiple operations
    auto t1 = Tensor::createOnes({100, 100});
    auto t2 = Tensor::createOnes({100, 100});

    // (((t1 + t2) * t1) - t2) should equal t1
    auto r1 = TensorOps::operator+(t1, t2);  // 2.0
    auto r2 = TensorOps::operator*(r1, t1);  // 2.0
    auto r3 = TensorOps::operator-(r2, t2);  // 1.0

    for (size_t i = 0; i < 10000; ++i) {
        EXPECT_FLOAT_EQ(r3->getDataElem(i), 1.0f);
    }
}

TEST(TensorIntensiveTest, MassiveElementWiseMultiplication) {
    auto arr1 = std::make_unique<float[]>(100000);
    auto arr2 = std::make_unique<float[]>(100000);

    for (size_t i = 0; i < 100000; ++i) {
        arr1[i] = static_cast<float>(i % 1000) / 1000.0f;
        arr2[i] = static_cast<float>(i % 500) / 500.0f;
    }

    auto t1 = Tensor::CreateTensor(std::move(arr1), 100000, {100, 1000});
    auto t2 = Tensor::CreateTensor(std::move(arr2), 100000, {100, 1000});

    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    auto result = TensorOps::operator*(t1, t2);
    auto end = high_resolution_clock::now();
    auto ms = duration_cast<milliseconds>(end - start).count();

    EXPECT_LT(ms, 100) << "100k element multiplication took " << ms << " ms";
    EXPECT_EQ(result->getTotalSize(), 100000);
}

TEST(TensorIntensiveTest, HighDimensionalCalculation_8D) {
    // 2^8 = 256 elements across 8 dimensions
    std::array<size_t, MAX_RANK> shape{2, 2, 2, 2, 2, 2, 2, 2};
    auto t1 = Tensor::createOnes(shape);
    auto t2 = Tensor::createOnes(shape);

    for (size_t i = 0; i < 256; ++i) {
        t1->setDataElem(i, static_cast<float>(i));
        t2->setDataElem(i, static_cast<float>(i * 2));
    }

    auto add_result = TensorOps::operator+(t1, t2);
    auto mul_result = TensorOps::operator*(t1, t2);

    for (size_t i = 0; i < 256; ++i) {
        EXPECT_FLOAT_EQ(add_result->getDataElem(i), static_cast<float>(i * 3));
        EXPECT_FLOAT_EQ(mul_result->getDataElem(i), static_cast<float>(i * i * 2));
    }
}

TEST(TensorIntensiveTest, RepetitiveOperations_MemoryStability) {
    // Ensure no memory corruption or leaks through repeated ops
    auto t1 = Tensor::createOnes({50, 50});
    auto t2 = Tensor::createOnes({50, 50});

    for (int iter = 0; iter < 1000; ++iter) {
        auto result = TensorOps::operator+(t1, t2);
        EXPECT_FLOAT_EQ(result->getDataElem(0), 2.0f);
        EXPECT_FLOAT_EQ(result->getDataElem(2499), 2.0f);
    }
}

TEST(TensorIntensiveTest, AlternatingOpsPattern) {
    auto t = Tensor::createScalar(1.0f);
    auto one = Tensor::createScalar(1.0f);

    // ((((1 + 1) * 1) - 1) + 1) * 1 - 1 ... repeat 100 times
    for (int i = 0; i < 100; ++i) {
        t = TensorOps::operator+(t, one);
        t = TensorOps::operator*(t, one);
        t = TensorOps::operator-(t, one);
    }

    EXPECT_FLOAT_EQ(t->getDataElem(0), 1.0f);
}

// ============================================================================
// FLOATING POINT EDGE CASES
// ============================================================================

TEST(TensorEdgeCaseTest, DenormalNumbers) {
    float denorm = std::numeric_limits<float>::denorm_min();
    auto t1 = Tensor::createScalar(denorm);
    auto t2 = Tensor::createScalar(denorm);

    auto result = TensorOps::operator+(t1, t2);
    EXPECT_GT(result->getDataElem(0), 0.0f);
    EXPECT_LT(result->getDataElem(0), std::numeric_limits<float>::min());
}

TEST(TensorEdgeCaseTest, MaxFloatValues) {
    float max_val = std::numeric_limits<float>::max();
    auto t1 = Tensor::createScalar(max_val);
    auto t2 = Tensor::createScalar(max_val);

    auto result = TensorOps::operator+(t1, t2);
    EXPECT_TRUE(std::isinf(result->getDataElem(0)));
}

TEST(TensorEdgeCaseTest, NegativeZeroHandling) {
    auto t1 = Tensor::createScalar(-0.0f);
    auto t2 = Tensor::createScalar(0.0f);

    auto result = TensorOps::operator+(t1, t2);
    EXPECT_FLOAT_EQ(result->getDataElem(0), 0.0f);
}

TEST(TensorEdgeCaseTest, InfinityArithmetic) {
    auto inf_pos = Tensor::createScalar(std::numeric_limits<float>::infinity());
    auto inf_neg = Tensor::createScalar(-std::numeric_limits<float>::infinity());
    auto finite = Tensor::createScalar(42.0f);

    // inf + finite = inf
    auto r1 = TensorOps::operator+(inf_pos, finite);
    EXPECT_TRUE(std::isinf(r1->getDataElem(0)));
    EXPECT_GT(r1->getDataElem(0), 0.0f);

    // inf * 0 = nan
    auto zero = Tensor::createScalar(0.0f);
    auto r2 = TensorOps::operator*(inf_pos, zero);
    EXPECT_TRUE(std::isnan(r2->getDataElem(0)));

    // inf * -1
    auto neg_one = Tensor::createScalar(-1.0f);
    auto r3 = TensorOps::operator*(inf_pos, neg_one);
    EXPECT_TRUE(std::isinf(r3->getDataElem(0)));
    EXPECT_LT(r3->getDataElem(0), 0.0f);
}

TEST(TensorEdgeCaseTest, NaNPropagation) {
    auto nan_t = Tensor::createScalar(std::numeric_limits<float>::quiet_NaN());
    auto normal = Tensor::createScalar(5.0f);

    // All operations with NaN should produce NaN
    auto add_r = TensorOps::operator+(nan_t, normal);
    auto sub_r = TensorOps::operator-(nan_t, normal);
    auto mul_r = TensorOps::operator*(nan_t, normal);

    EXPECT_TRUE(std::isnan(add_r->getDataElem(0)));
    EXPECT_TRUE(std::isnan(sub_r->getDataElem(0)));
    EXPECT_TRUE(std::isnan(mul_r->getDataElem(0)));
}

TEST(TensorEdgeCaseTest, VerySmallDifferences) {
    // Test precision near epsilon
    float eps = std::numeric_limits<float>::epsilon();
    auto t1 = Tensor::createScalar(1.0f);
    auto t2 = Tensor::createScalar(1.0f + eps);

    auto result = TensorOps::operator-(t2, t1);
    EXPECT_GT(result->getDataElem(0), 0.0f);
    EXPECT_LE(result->getDataElem(0), eps * 2.0f);
}

TEST(TensorEdgeCaseTest, CatastrophicCancellation) {
    // Large numbers that nearly cancel
    auto t1 = Tensor::createScalar(1e20f);
    auto t2 = Tensor::createScalar(-1e20f + 1.0f);

    auto result = TensorOps::operator+(t1, t2);
    // Due to floating point limitations, result may not be exactly 1.0
    EXPECT_LT(result->getDataElem(0), 2.0f);
}

// ============================================================================
// MEMORY AND BOUNDARY TESTS
// ============================================================================

TEST(TensorEdgeCaseTest, SingleDimensionSizes) {
    // Test edge dimensions: 1, 2, prime numbers, powers of 2
    std::vector<size_t> sizes = {1, 2, 3, 5, 7, 11, 16, 32, 64, 127, 128, 256, 1000};

    for (auto size : sizes) {
        auto t = Tensor::createZeros({size, 0, 0, 0, 0, 0, 0, 0});
        EXPECT_EQ(t->getTotalSize(), size);
        EXPECT_FLOAT_EQ(t->getDataElem(0), 0.0f);
        EXPECT_FLOAT_EQ(t->getDataElem(size - 1), 0.0f);
    }
}

TEST(TensorEdgeCaseTest, AllDimensionsOne) {
    std::array<size_t, MAX_RANK> shape{1, 1, 1, 1, 1, 1, 1, 1};
    auto t = Tensor::createScalar(42.0f);
    auto zeros = Tensor::createZeros(shape);

    EXPECT_EQ(zeros->getTotalSize(), 1);
    EXPECT_FLOAT_EQ(zeros->getDataElem(0), 0.0f);
}

TEST(TensorEdgeCaseTest, LargePrimeDimensions) {
    // 101 * 103 = 10403 (both prime)
    auto t = Tensor::createZeros({101, 103, 0, 0, 0, 0, 0, 0});
    EXPECT_EQ(t->getTotalSize(), 10403);

    auto strides = Tensor::calculateStrides({101, 103, 0, 0, 0, 0, 0, 0});
    EXPECT_EQ(strides[0], 103);
    EXPECT_EQ(strides[1], 1);
}

TEST(TensorEdgeCaseTest, BoundaryIndexAccess2D) {
    auto arr = std::make_unique<float[]>(12);
    for (size_t i = 0; i < 12; ++i) arr[i] = static_cast<float>(i);
    auto t = Tensor::CreateTensor(std::move(arr), 12, {3, 4});

    // Corner elements
    EXPECT_FLOAT_EQ((*t)(0, 0), 0.0f);
    EXPECT_FLOAT_EQ((*t)(0, 3), 3.0f);
    EXPECT_FLOAT_EQ((*t)(2, 0), 8.0f);
    EXPECT_FLOAT_EQ((*t)(2, 3), 11.0f);

    // Just beyond boundary should throw
    EXPECT_THROW((*t)(3, 0), std::out_of_range);
    EXPECT_THROW((*t)(0, 4), std::out_of_range);
    EXPECT_THROW((*t)(3, 4), std::out_of_range);
}

TEST(TensorEdgeCaseTest, BoundaryIndexAccess3D) {
    auto arr = std::make_unique<float[]>(24);
    for (size_t i = 0; i < 24; ++i) arr[i] = static_cast<float>(i);
    auto t = Tensor::CreateTensor(std::move(arr), 24, {2, 3, 4});

    // All corners
    EXPECT_FLOAT_EQ((*t)(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ((*t)(0, 0, 3), 3.0f);
    EXPECT_FLOAT_EQ((*t)(0, 2, 0), 8.0f);
    EXPECT_FLOAT_EQ((*t)(0, 2, 3), 11.0f);
    EXPECT_FLOAT_EQ((*t)(1, 0, 0), 12.0f);
    EXPECT_FLOAT_EQ((*t)(1, 2, 3), 23.0f);

    // Boundary violations
    EXPECT_THROW((*t)(2, 0, 0), std::out_of_range);
    EXPECT_THROW((*t)(0, 3, 0), std::out_of_range);
    EXPECT_THROW((*t)(0, 0, 4), std::out_of_range);
}

// ============================================================================
// STRIDE AND INDEXING EDGE CASES
// ============================================================================

TEST(TensorEdgeCaseTest, NonContiguousStrides_AllOnes) {
    std::array<size_t, MAX_RANK> shape{1, 1, 1, 1, 1, 1, 1, 1};
    auto strides = Tensor::calculateStrides(shape);

    // All strides should be 1
    for (size_t i = 0; i < MAX_RANK; ++i) {
        EXPECT_EQ(strides[i], 1);
    }
}

TEST(TensorEdgeCaseTest, MixedStridePattern) {
    // Shape with varying dimensions
    std::array<size_t, MAX_RANK> shape{10, 1, 5, 1, 2, 0, 0, 0};
    auto strides = Tensor::calculateStrides(shape);

    EXPECT_EQ(strides[0], 10);  // 1*5*1*2
    EXPECT_EQ(strides[1], 10);  // 5*1*2
    EXPECT_EQ(strides[2], 2);   // 1*2
    EXPECT_EQ(strides[3], 2);   // 2
    EXPECT_EQ(strides[4], 1);   // 1
}

TEST(TensorEdgeCaseTest, LastDimensionLarge) {
    std::array<size_t, MAX_RANK> shape{2, 2, 10000, 0, 0, 0, 0, 0};
    auto strides = Tensor::calculateStrides(shape);

    EXPECT_EQ(strides[0], 20000);
    EXPECT_EQ(strides[1], 10000);
    EXPECT_EQ(strides[2], 1);
}

// ============================================================================
// OPERATION COMMUTATIVITY AND PROPERTIES
// ============================================================================

TEST(TensorPropertiesTest, AdditionCommutativity) {
    auto arr1 = std::make_unique<float[]>(100);
    auto arr2 = std::make_unique<float[]>(100);

    for (size_t i = 0; i < 100; ++i) {
        arr1[i] = static_cast<float>(i) * 0.5f;
        arr2[i] = static_cast<float>(i) * 1.5f;
    }

    auto t1 = Tensor::CreateTensor(std::move(arr1), 100, {10, 10});
    auto t2 = Tensor::CreateTensor(std::move(arr2), 100, {10, 10});

    auto r1 = TensorOps::operator+(t1, t2);
    auto r2 = TensorOps::operator+(t2, t1);

    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(r1->getDataElem(i), r2->getDataElem(i));
    }
}

TEST(TensorPropertiesTest, MultiplicationCommutativity) {
    auto t1 = Tensor::createOnes({50, 0, 0, 0, 0, 0, 0, 0});
    auto t2 = Tensor::createOnes({50, 0, 0, 0, 0, 0, 0, 0});

    for (size_t i = 0; i < 50; ++i) {
        t1->setDataElem(i, static_cast<float>(i));
        t2->setDataElem(i, static_cast<float>(i + 1));
    }

    auto r1 = TensorOps::operator*(t1, t2);
    auto r2 = TensorOps::operator*(t2, t1);

    for (size_t i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(r1->getDataElem(i), r2->getDataElem(i));
    }
}

TEST(TensorPropertiesTest, AdditionAssociativity) {
    auto t1 = Tensor::createScalar(1.5f);
    auto t2 = Tensor::createScalar(2.5f);
    auto t3 = Tensor::createScalar(3.5f);

    // (t1 + t2) + t3
    auto r1 = TensorOps::operator+(TensorOps::operator+(t1, t2), t3);

    // t1 + (t2 + t3)
    auto r2 = TensorOps::operator+(t1, TensorOps::operator+(t2, t3));

    EXPECT_FLOAT_EQ(r1->getDataElem(0), r2->getDataElem(0));
    EXPECT_FLOAT_EQ(r1->getDataElem(0), 7.5f);
}

TEST(TensorPropertiesTest, DistributiveProperty) {
    auto a = Tensor::createScalar(2.0f);
    auto b = Tensor::createScalar(3.0f);
    auto c = Tensor::createScalar(4.0f);

    // a * (b + c) = a*b + a*c
    auto lhs = TensorOps::operator*(a, TensorOps::operator+(b, c));
    auto rhs = TensorOps::operator+(TensorOps::operator*(a, b),
                                     TensorOps::operator*(a, c));

    EXPECT_FLOAT_EQ(lhs->getDataElem(0), rhs->getDataElem(0));
    EXPECT_FLOAT_EQ(lhs->getDataElem(0), 14.0f);
}

// ============================================================================
// ERROR CONDITION STRESS TESTS
// ============================================================================

TEST(TensorErrorTest, MultipleShapeMismatches) {
    std::vector<std::array<size_t, MAX_RANK>> shapes = {
        {2, 3, 0, 0, 0, 0, 0, 0},
        {3, 2, 0, 0, 0, 0, 0, 0},
        {2, 4, 0, 0, 0, 0, 0, 0},
        {1, 6, 0, 0, 0, 0, 0, 0},
        {6, 0, 0, 0, 0, 0, 0, 0}
    };

    auto base = Tensor::createZeros(shapes[0]);

    for (size_t i = 1; i < shapes.size(); ++i) {
        auto t = Tensor::createZeros(shapes[i]);
        EXPECT_THROW(TensorOps::operator+(base, t), std::invalid_argument);
        EXPECT_THROW(TensorOps::operator-(base, t), std::invalid_argument);
        EXPECT_THROW(TensorOps::operator*(base, t), std::invalid_argument);
    }
}

TEST(TensorErrorTest, ExtremeIndexValues) {
    auto t = Tensor::createZeros({10, 10, 0, 0, 0, 0, 0, 0});

    // Maximum size_t should wrap and cause out of range
    EXPECT_THROW((*t)(SIZE_MAX, 0), std::out_of_range);
    EXPECT_THROW((*t)(0, SIZE_MAX), std::out_of_range);

    // Large but not maximum
    EXPECT_THROW((*t)(1000000, 0), std::out_of_range);
    EXPECT_THROW((*t)(0, 1000000), std::out_of_range);
}

// ============================================================================
// PERFORMANCE REGRESSION TESTS
// ============================================================================

TEST(TensorPerformanceTest, Sequential1MAccess) {
    auto t = Tensor::createZeros({1000, 1000, 0, 0, 0, 0, 0, 0});

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < 1000000; ++i) {
        t->setDataElem(i, static_cast<float>(i));
    }

    auto end = high_resolution_clock::now();
    auto ms = duration_cast<milliseconds>(end - start).count();

    EXPECT_LT(ms, 50) << "1M sequential writes took " << ms << " ms";
}

TEST(TensorPerformanceTest, Random2DAccess) {
    auto t = Tensor::createZeros({1000, 1000, 0, 0, 0, 0, 0, 0});

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < 1000; i += 10) {
        for (size_t j = 0; j < 1000; j += 10) {
            (*t)(i, j) = static_cast<float>(i + j);
        }
    }

    auto end = high_resolution_clock::now();
    auto ms = duration_cast<milliseconds>(end - start).count();

    EXPECT_LT(ms, 10) << "10k random 2D accesses took " << ms << " ms";
}
// ============================================================================
// ADDITIONAL EXTREME EDGE CASES
// ============================================================================

TEST(TensorExtremeTest, AlternatingSignsLargeScale) {
    auto arr = std::make_unique<float[]>(10000);
    for (size_t i = 0; i < 10000; ++i) {
        arr[i] = (i % 2 == 0) ? 1e10f : -1e10f;
    }
    auto t = Tensor::CreateTensor(std::move(arr), 10000, {100, 100});

    auto result = TensorOps::operator+(t, t);
    for (size_t i = 0; i < 10000; ++i) {
        float expected = (i % 2 == 0) ? 2e10f : -2e10f;
        EXPECT_FLOAT_EQ(result->getDataElem(i), expected);
    }
}

TEST(TensorExtremeTest, SubtractionResultingInZeros) {
    auto arr = std::make_unique<float[]>(1000);
    for (size_t i = 0; i < 1000; ++i) {
        arr[i] = static_cast<float>(i) * 0.123456789f;
    }
    auto t1 = Tensor::CreateTensor(std::move(arr), 1000, {10, 100});

    auto arr2 = std::make_unique<float[]>(1000);
    for (size_t i = 0; i < 1000; ++i) {
        arr2[i] = static_cast<float>(i) * 0.123456789f;
    }
    auto t2 = Tensor::CreateTensor(std::move(arr2), 1000, {10, 100});

    auto result = TensorOps::operator-(t1, t2);
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_NEAR(result->getDataElem(i), 0.0f, 1e-6f);
    }
}

TEST(TensorExtremeTest, MultiplicationByZeroMassive) {
    auto t1 = Tensor::createOnes({500, 500, 0, 0, 0, 0, 0, 0});
    auto t2 = Tensor::createZeros({500, 500, 0, 0, 0, 0, 0, 0});

    for (size_t i = 0; i < 250000; ++i) {
        t1->setDataElem(i, static_cast<float>(i) * 1.23f);
    }

    auto result = TensorOps::operator*(t1, t2);
    for (size_t i = 0; i < 250000; ++i) {
        EXPECT_FLOAT_EQ(result->getDataElem(i), 0.0f);
    }
}

TEST(TensorExtremeTest, FractionalAccumulation) {
    // Test accumulation of many small fractions
    auto t_base = Tensor::createZeros({100, 0, 0, 0, 0, 0, 0, 0});
    auto t_small = Tensor::createOnes({100, 0, 0, 0, 0, 0, 0, 0});

    for (size_t i = 0; i < 100; ++i) {
        t_small->setDataElem(i, 0.01f);
    }

    auto result = t_base;
    for (int i = 0; i < 100; ++i) {
        result = TensorOps::operator+(result, t_small);
    }

    // Should be exactly 1.0 after 100 iterations
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_NEAR(result->getDataElem(i), 1.0f, 1e-5f);
    }
}

TEST(TensorExtremeTest, MixedMagnitudeOperations) {
    auto arr1 = std::make_unique<float[]>(1000);
    auto arr2 = std::make_unique<float[]>(1000);

    for (size_t i = 0; i < 1000; ++i) {
        arr1[i] = 1e20f;  // Very large
        arr2[i] = 1e-20f; // Very small
    }

    auto t1 = Tensor::CreateTensor(std::move(arr1), 1000, {10, 100});
    auto t2 = Tensor::CreateTensor(std::move(arr2), 1000, {10, 100});

    auto add_result = TensorOps::operator+(t1, t2);
    auto mul_result = TensorOps::operator*(t1, t2);

    // Addition should be dominated by large values
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(add_result->getDataElem(i), 1e20f);
    }

    // Multiplication should produce 1.0
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(mul_result->getDataElem(i), 1.0f);
    }
}

TEST(TensorExtremeTest, PrecisionLossInChain) {
    // Test catastrophic cancellation - demonstrates precision limits
    auto t = Tensor::createScalar(1.0f);
    auto large = Tensor::createScalar(1e15f);

    // When adding 1.0 to 1e15, the 1.0 is lost due to float precision
    // float has ~7 significant digits, so 1e15 + 1 = 1e15
    auto r1 = TensorOps::operator+(t, large);
    EXPECT_FLOAT_EQ(r1->getDataElem(0), 1e15f); // 1.0 was lost

    auto r2 = TensorOps::operator-(r1, large);
    EXPECT_FLOAT_EQ(r2->getDataElem(0), 0.0f);  // Results in 0, not 1

    // This demonstrates why order matters in floating-point arithmetic
    // Better approach: add small numbers first, then large
    auto small = Tensor::createScalar(1.0f);
    auto small2 = Tensor::createScalar(1e-5f);
    auto r3 = TensorOps::operator+(small, small2);
    auto r4 = TensorOps::operator+(r3, large);
    auto r5 = TensorOps::operator-(r4, large);

    // Still loses precision but we can verify the behavior
    EXPECT_GE(r5->getDataElem(0), 0.0f);
}

TEST(TensorExtremeTest, SymmetricOperations) {
    auto arr = std::make_unique<float[]>(100);
    for (size_t i = 0; i < 100; ++i) {
        arr[i] = static_cast<float>(i) - 50.0f; // -50 to 49
    }
    auto t = Tensor::CreateTensor(std::move(arr), 100, {10, 10});

    // t * t should be all positive
    auto squared = TensorOps::operator*(t, t);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_GE(squared->getDataElem(i), 0.0f);
        float expected = (static_cast<float>(i) - 50.0f) * (static_cast<float>(i) - 50.0f);
        EXPECT_FLOAT_EQ(squared->getDataElem(i), expected);
    }
}

TEST(TensorExtremeTest, NonUniformShape_7D) {
    // 7D tensor with varying dimensions
    std::array<size_t, MAX_RANK> shape{3, 1, 2, 1, 4, 2, 1, 0};
    auto t = Tensor::createOnes(shape);

    size_t expected_size = 3 * 1 * 2 * 1 * 4 * 2 * 1;
    EXPECT_EQ(t->getTotalSize(), expected_size);

    auto strides = Tensor::calculateStrides(shape);
    EXPECT_EQ(strides[0], 16);
    EXPECT_EQ(strides[1], 16);
    EXPECT_EQ(strides[2], 8);
    EXPECT_EQ(strides[3], 8);
    EXPECT_EQ(strides[4], 2);
    EXPECT_EQ(strides[5], 1);
    EXPECT_EQ(strides[6], 1);
}

TEST(TensorExtremeTest, OperationChainWithTemporaries) {
    // Test that temporary results don't cause issues
    auto t1 = Tensor::createScalar(2.0f);
    auto t2 = Tensor::createScalar(3.0f);
    auto t3 = Tensor::createScalar(5.0f);
    auto t4 = Tensor::createScalar(7.0f);

    // Complex expression: ((t1 + t2) * (t3 - t4)) + t1
    // = ((2+3) * (5-7)) + 2 = (5 * -2) + 2 = -10 + 2 = -8
    auto temp1 = TensorOps::operator+(t1, t2);
    auto temp2 = TensorOps::operator-(t3, t4);
    auto temp3 = TensorOps::operator*(temp1, temp2);
    auto result = TensorOps::operator+(temp3, t1);

    EXPECT_FLOAT_EQ(result->getDataElem(0), -8.0f);
}

TEST(TensorExtremeTest, SameShapeDifferentData_1000Ops) {
    auto t1 = Tensor::createOnes({10, 10, 0, 0, 0, 0, 0, 0});
    auto t2 = Tensor::createZeros({10, 10, 0, 0, 0, 0, 0, 0});

    for (size_t i = 0; i < 100; ++i) {
        t2->setDataElem(i, static_cast<float>(i) * 0.1f);
    }

    // Perform 1000 operations
    for (int op = 0; op < 1000; ++op) {
        auto temp = TensorOps::operator+(t1, t2);
        EXPECT_FLOAT_EQ(temp->getDataElem(0), 1.0f);
        EXPECT_FLOAT_EQ(temp->getDataElem(50), 6.0f);
        EXPECT_FLOAT_EQ(temp->getDataElem(99), 10.9f);
    }
}
<<<<<<< HEAD
=======
TEST(TensorOpsTest,CalcLoss) {
    auto t1 = Tensor::createRandTensor({20,10});
    auto t2 = Tensor::createRandTensor({20,10});
    auto result = TensorOps::calcCost(t1,t2);
    EXPECT_TRUE(result);
}

TEST(TensorOpsTest,CalcLossSizeMismatch) {
    auto t1 = Tensor::createRandTensor({3,4});
    auto t2 = Tensor::createRandTensor({4,3});
    EXPECT_THROW(TensorOps::calcCost(t1,t2),std::invalid_argument);
}
>>>>>>> 5748cdc (some changes)
