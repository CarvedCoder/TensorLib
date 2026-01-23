#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tensorlib/ops/ops.h>
#include <tensorlib/tensor/tensor.h>
#include <utility>
#include <vector>
// ============================================================================
// SCALAR (RANK-0) TESTS
// ============================================================================

TEST(ScalarTest, CreationAndInvariants) {
    auto scalar = Tensor::createScalar(42.0f);

    EXPECT_EQ(scalar.getRank(), 0);
    EXPECT_EQ(scalar.getTotalSize(), 1);
    EXPECT_EQ(scalar.getShape().size(), 0); // span length is 0 for scalars
    EXPECT_FLOAT_EQ(scalar.getDataPtr()[0], 42.0f);
}

TEST(ScalarTest, ZeroAccess) {
    auto scalar = Tensor::createScalar(5.0f);
    EXPECT_FLOAT_EQ(scalar(0), 5.0f);
}

TEST(ScalarTest, Operations) {
    auto s1 = Tensor::createScalar(3.0f);
    auto s2 = Tensor::createScalar(4.0f);

    auto add = TensorOps::operator+(s1, s2);
    auto sub = TensorOps::operator-(s1, s2);
    auto mul = TensorOps::operator*(s1, s2);

    EXPECT_FLOAT_EQ(add(0), 7.0f);
    EXPECT_FLOAT_EQ(sub(0), -1.0f);
    EXPECT_FLOAT_EQ(mul(0), 12.0f);
}

// ============================================================================
// 1-D TESTS
// ============================================================================

TEST(Tensor1DTest, CreationAndInvariants) {
    auto arr = std::make_unique<float[]>(5);
    for (size_t i = 0; i < 5; ++i)
        arr[i] = static_cast<float>(i);

    auto tensor = Tensor::createTensor(std::move(arr), 5, {5});

    EXPECT_EQ(tensor.getRank(), 1);
    EXPECT_EQ(tensor.getTotalSize(), 5);
    EXPECT_EQ(tensor.getShape().size(), 1);
    EXPECT_EQ(tensor.getShape()[0], 5);
}

TEST(Tensor1DTest, IndexingBehavior) {
    auto tensor = Tensor::createOnes({10});

    for (size_t i = 0; i < 10; ++i) {
        tensor.setDataElem(i, static_cast<float>(i * 2));
    }

    EXPECT_FLOAT_EQ(tensor(0), 0.0f);
    EXPECT_FLOAT_EQ(tensor(5), 10.0f);
    EXPECT_FLOAT_EQ(tensor(9), 18.0f);
}

TEST(Tensor1DTest, BoundaryAccess) {
    auto tensor = Tensor::createZeros({5});

    EXPECT_NO_THROW(tensor(0));
    EXPECT_NO_THROW(tensor(4));
    EXPECT_THROW(tensor(5), std::out_of_range);
    EXPECT_THROW(tensor(100), std::out_of_range);
}

// ============================================================================
// 2-D TESTS
// ============================================================================

TEST(Tensor2DTest, CreationAndInvariants) {
    auto arr = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i)
        arr[i] = static_cast<float>(i + 1);

    auto tensor = Tensor::createTensor(std::move(arr), 6, {2, 3});

    EXPECT_EQ(tensor.getRank(), 2);
    EXPECT_EQ(tensor.getTotalSize(), 6);
    EXPECT_EQ(tensor.getShape().size(), 2);
    EXPECT_EQ(tensor.getShape()[0], 2);
    EXPECT_EQ(tensor.getShape()[1], 3);
}

TEST(Tensor2DTest, RowMajorIndexing) {
    auto arr = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i)
        arr[i] = static_cast<float>(i + 1);

    auto tensor = Tensor::createTensor(std::move(arr), 6, {2, 3});

    // Row-major layout: [[1,2,3], [4,5,6]]
    EXPECT_FLOAT_EQ(tensor(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(tensor(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(tensor(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(tensor(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(tensor(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(tensor(1, 2), 6.0f);
}

TEST(Tensor2DTest, BoundaryAccess) {
    auto tensor = Tensor::createZeros({3, 4});

    // Valid corners
    EXPECT_NO_THROW(tensor(0, 0));
    EXPECT_NO_THROW(tensor(0, 3));
    EXPECT_NO_THROW(tensor(2, 0));
    EXPECT_NO_THROW(tensor(2, 3));

    // Invalid access
    EXPECT_THROW(tensor(3, 0), std::out_of_range);
    EXPECT_THROW(tensor(0, 4), std::out_of_range);
    EXPECT_THROW(tensor(3, 4), std::out_of_range);
}

TEST(Tensor2DTest, StridesViaIndexing) {
    auto arr = std::make_unique<float[]>(12);
    for (size_t i = 0; i < 12; ++i)
        arr[i] = static_cast<float>(i);

    auto tensor = Tensor::createTensor(std::move(arr), 12, {3, 4});

    // Verify row-major stride behavior
    // Element at (i,j) should be at index i*4 + j
    EXPECT_FLOAT_EQ(tensor(0, 0), 0.0f);  // index 0
    EXPECT_FLOAT_EQ(tensor(0, 3), 3.0f);  // index 3
    EXPECT_FLOAT_EQ(tensor(1, 0), 4.0f);  // index 4
    EXPECT_FLOAT_EQ(tensor(2, 3), 11.0f); // index 11
}

// ============================================================================
// 3-D AND HIGH-RANK TESTS
// ============================================================================

TEST(Tensor3DTest, CreationAndInvariants) {
    auto arr = std::make_unique<float[]>(24);
    for (size_t i = 0; i < 24; ++i)
        arr[i] = static_cast<float>(i);

    auto tensor = Tensor::createTensor(std::move(arr), 24, {2, 3, 4});

    EXPECT_EQ(tensor.getRank(), 3);
    EXPECT_EQ(tensor.getTotalSize(), 24);
    EXPECT_EQ(tensor.getShape().size(), 3);
    EXPECT_EQ(tensor.getShape()[0], 2);
    EXPECT_EQ(tensor.getShape()[1], 3);
    EXPECT_EQ(tensor.getShape()[2], 4);
}

TEST(Tensor3DTest, RowMajorIndexing) {
    auto arr = std::make_unique<float[]>(24);
    for (size_t i = 0; i < 24; ++i)
        arr[i] = static_cast<float>(i);

    auto tensor = Tensor::createTensor(std::move(arr), 24, {2, 3, 4});

    EXPECT_FLOAT_EQ(tensor(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(tensor(0, 0, 3), 3.0f);
    EXPECT_FLOAT_EQ(tensor(0, 2, 0), 8.0f);
    EXPECT_FLOAT_EQ(tensor(1, 0, 0), 12.0f);
    EXPECT_FLOAT_EQ(tensor(1, 2, 3), 23.0f);
}

TEST(HighRankTest, MaxRank8Tensor) {
    std::array<size_t, 8> shape{2, 2, 2, 2, 2, 2, 2, 2};
    auto tensor = Tensor::createZeros(shape);

    EXPECT_EQ(tensor.getRank(), 8);
    EXPECT_EQ(tensor.getTotalSize(), 256);
    EXPECT_EQ(tensor.getShape().size(), 8);

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(tensor.getShape()[i], 2);
    }
}

TEST(HighRankTest, NonUniformDimensions) {
    std::array<size_t, 8> shape{3, 1, 2, 1, 4, 2, 1, 0};
    auto tensor = Tensor::createOnes(shape);

    EXPECT_EQ(tensor.getRank(), 7);
    EXPECT_EQ(tensor.getTotalSize(), 48); // 3*1*2*1*4*2*1
}

// ============================================================================
// FACTORY FUNCTION TESTS
// ============================================================================

TEST(FactoryTest, CreateZeros) {
    auto tensor = Tensor::createZeros({10, 20});

    EXPECT_EQ(tensor.getTotalSize(), 200);
    for (size_t i = 0; i < 200; ++i) {
        EXPECT_FLOAT_EQ(tensor.getDataPtr()[i], 0.0f);
    }
}

TEST(FactoryTest, CreateOnes) {
    auto tensor = Tensor::createOnes({5, 5});

    EXPECT_EQ(tensor.getTotalSize(), 25);
    for (size_t i = 0; i < 25; ++i) {
        EXPECT_FLOAT_EQ(tensor.getDataPtr()[i], 1.0f);
    }
}

TEST(FactoryTest, CreateRandomHe) {
    auto tensor = Tensor::createRandTensor({10, 10}, InitType::He);

    EXPECT_EQ(tensor.getTotalSize(), 100);

    // Verify values are distributed (not all zeros or ones)
    bool has_different_values = false;
    float first = tensor.getDataPtr()[0];
    for (size_t i = 1; i < 100; ++i) {
        if (std::abs(tensor.getDataPtr()[i] - first) > 1e-6f) {
            has_different_values = true;
            break;
        }
    }
    EXPECT_TRUE(has_different_values);
}

// ============================================================================
// SHAPE VALIDATION TESTS
// ============================================================================

TEST(ShapeValidationTest, SizeMismatch) {
    auto arr = std::make_unique<float[]>(6);

    // Shape implies 4 elements, but we have 6
    EXPECT_THROW(Tensor::createTensor(std::move(arr), 6, {2, 2}),
                 std::invalid_argument);
}

TEST(ShapeValidationTest, ExcessiveRank) {
    auto arr = std::make_unique<float[]>(1);
    std::vector<size_t> shape(10, 1); // 10 dimensions

    // Should handle gracefully (MAX_RANK is 8)
    EXPECT_THROW(Tensor::createTensor(std::move(arr), 1, shape),
                 std::invalid_argument);
}

// ============================================================================
// ELEMENTWISE OPERATION TESTS
// ============================================================================

TEST(ElementwiseOpsTest, Addition) {
    auto t1 = Tensor::createOnes({10, 10});
    auto t2 = Tensor::createOnes({10, 10});

    for (size_t i = 0; i < 100; ++i) {
        t1.setDataElem(i, static_cast<float>(i));
        t2.setDataElem(i, static_cast<float>(i * 2));
    }

    auto result = TensorOps::operator+(t1, t2);

    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(result.getDataPtr()[i], static_cast<float>(i * 3));
    }
}

TEST(ElementwiseOpsTest, Subtraction) {
    auto t1 = Tensor::createScalar(10.0f);
    auto t2 = Tensor::createScalar(3.0f);

    auto result = TensorOps::operator-(t1, t2);
    EXPECT_FLOAT_EQ(result(0), 7.0f);
}

TEST(ElementwiseOpsTest, Multiplication) {
    auto t1 = Tensor::createOnes({4, 4});
    auto t2 = Tensor::createOnes({4, 4});

    for (size_t i = 0; i < 16; ++i) {
        t1.setDataElem(i, static_cast<float>(i + 1));
        t2.setDataElem(i, 2.0f);
    }

    auto result = TensorOps::operator*(t1, t2);

    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(result.getDataPtr()[i],
                        static_cast<float>((i + 1) * 2));
    }
}

TEST(ElementwiseOpsTest, ShapeMismatch) {
    auto t1 = Tensor::createZeros({2, 3});
    auto t2 = Tensor::createZeros({3, 2});

    EXPECT_THROW(TensorOps::operator+(t1, t2), std::invalid_argument);
    EXPECT_THROW(TensorOps::operator-(t1, t2), std::invalid_argument);
    EXPECT_THROW(TensorOps::operator*(t1, t2), std::invalid_argument);
}

// ============================================================================
// TRANSPOSE TESTS
// ============================================================================

TEST(TransposeTest, Square2x2) {
    auto arr = std::make_unique<float[]>(4);
    arr[0] = 1.0f;
    arr[1] = 2.0f;
    arr[2] = 3.0f;
    arr[3] = 4.0f;

    auto tensor = Tensor::createTensor(std::move(arr), 4, {2, 2});
    auto transposed = TensorOps::transpose2D(tensor);

    EXPECT_EQ(transposed.getShape()[0], 2);
    EXPECT_EQ(transposed.getShape()[1], 2);

    EXPECT_FLOAT_EQ(transposed(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(transposed(0, 1), 3.0f);
    EXPECT_FLOAT_EQ(transposed(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(transposed(1, 1), 4.0f);
}

TEST(TransposeTest, Rectangular) {
    auto arr = std::make_unique<float[]>(6);
    for (size_t i = 0; i < 6; ++i)
        arr[i] = static_cast<float>(i + 1);

    auto tensor = Tensor::createTensor(std::move(arr), 6, {2, 3});
    auto transposed = TensorOps::transpose2D(tensor);

    EXPECT_EQ(transposed.getShape()[0], 3);
    EXPECT_EQ(transposed.getShape()[1], 2);

    // Original: [[1,2,3], [4,5,6]]
    // Transposed: [[1,4], [2,5], [3,6]]
    EXPECT_FLOAT_EQ(transposed(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(transposed(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(transposed(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(transposed(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(transposed(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(transposed(2, 1), 6.0f);
}

TEST(TransposeTest, DoubleTranspose) {
    auto tensor = Tensor::createOnes({3, 4});
    for (size_t i = 0; i < 12; ++i)
        tensor.setDataElem(i, static_cast<float>(i));

    auto t1 = TensorOps::transpose2D(tensor);
    auto t2 = TensorOps::transpose2D(t1);

    EXPECT_EQ(t2.getShape()[0], 3);
    EXPECT_EQ(t2.getShape()[1], 4);

    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(t2.getDataPtr()[i], tensor.getDataPtr()[i]);
    }
}

TEST(TransposeTest, InvalidRank) {
    auto tensor1d = Tensor::createZeros({5});
    auto tensor3d = Tensor::createZeros({2, 2, 2});

    EXPECT_THROW(TensorOps::transpose2D(tensor1d), std::invalid_argument);
    EXPECT_THROW(TensorOps::transpose2D(tensor3d), std::invalid_argument);
}

// ============================================================================
// MATMUL TESTS
// ============================================================================

TEST(MatMulTest, Basic2x2) {
    auto arr_a = std::make_unique<float[]>(4);
    auto arr_b = std::make_unique<float[]>(4);

    // A = [[1,2], [3,4]]
    arr_a[0] = 1.0f;
    arr_a[1] = 2.0f;
    arr_a[2] = 3.0f;
    arr_a[3] = 4.0f;

    // B = [[5,6], [7,8]]
    arr_b[0] = 5.0f;
    arr_b[1] = 6.0f;
    arr_b[2] = 7.0f;
    arr_b[3] = 8.0f;

    auto A = Tensor::createTensor(std::move(arr_a), 4, {2, 2});
    auto B = Tensor::createTensor(std::move(arr_b), 4, {2, 2});

    auto C = TensorOps::matmul(A, B);

    // C = [[19,22], [43,50]]
    EXPECT_EQ(C.getShape()[0], 2);
    EXPECT_EQ(C.getShape()[1], 2);
    EXPECT_FLOAT_EQ(C(0, 0), 19.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 22.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 43.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 50.0f);
}

TEST(MatMulTest, Rectangular) {
    auto arr_a = std::make_unique<float[]>(6);
    auto arr_b = std::make_unique<float[]>(6);

    for (size_t i = 0; i < 6; ++i) {
        arr_a[i] = static_cast<float>(i + 1);
        arr_b[i] = static_cast<float>(i + 7);
    }

    auto A = Tensor::createTensor(std::move(arr_a), 6, {2, 3}); // 2x3
    auto B = Tensor::createTensor(std::move(arr_b), 6, {3, 2}); // 3x2

    auto C = TensorOps::matmul(A, B);

    EXPECT_EQ(C.getShape()[0], 2);
    EXPECT_EQ(C.getShape()[1], 2);
}

TEST(MatMulTest, MatrixVector) {
    auto arr_a = std::make_unique<float[]>(6);
    auto arr_v = std::make_unique<float[]>(3);

    for (size_t i = 0; i < 6; ++i)
        arr_a[i] = static_cast<float>(i + 1);

    arr_v[0] = 1.0f;
    arr_v[1] = 2.0f;
    arr_v[2] = 3.0f;

    auto A = Tensor::createTensor(std::move(arr_a), 6, {2, 3});
    auto v = Tensor::createTensor(std::move(arr_v), 3, {3, 1});

    auto result = TensorOps::matmul(A, v);

    EXPECT_EQ(result.getShape()[0], 2);
    EXPECT_EQ(result.getShape()[1], 1);
    EXPECT_FLOAT_EQ(result(0, 0), 14.0f); // 1*1 + 2*2 + 3*3
    EXPECT_FLOAT_EQ(result(1, 0), 32.0f); // 4*1 + 5*2 + 6*3
}

TEST(MatMulTest, DimensionMismatch) {
    auto A = Tensor::createZeros({2, 3});
    auto B = Tensor::createZeros({4, 2}); // 3 != 4

    EXPECT_THROW(TensorOps::matmul(A, B), std::invalid_argument);
}

TEST(MatMulTest, InvalidRank) {
    auto tensor1d = Tensor::createZeros({5});
    auto tensor2d = Tensor::createZeros({5, 5});

    EXPECT_THROW(TensorOps::matmul(tensor1d, tensor2d), std::invalid_argument);
}

// ============================================================================
// FLOATING POINT EDGE CASES
// ============================================================================

TEST(FloatEdgeCaseTest, Infinity) {
    auto inf_pos = Tensor::createScalar(std::numeric_limits<float>::infinity());
    auto inf_neg =
        Tensor::createScalar(-std::numeric_limits<float>::infinity());
    auto finite = Tensor::createScalar(42.0f);

    auto r1 = TensorOps::operator+(inf_pos, finite);
    EXPECT_TRUE(std::isinf(r1(0)));
    EXPECT_GT(r1(0), 0.0f);

    auto zero = Tensor::createScalar(0.0f);
    auto r2 = TensorOps::operator*(inf_pos, zero);
    EXPECT_TRUE(std::isnan(r2(0)));
}

TEST(FloatEdgeCaseTest, NaNPropagation) {
    auto nan_t = Tensor::createScalar(std::numeric_limits<float>::quiet_NaN());
    auto normal = Tensor::createScalar(5.0f);

    auto add_r = TensorOps::operator+(nan_t, normal);
    auto sub_r = TensorOps::operator-(nan_t, normal);
    auto mul_r = TensorOps::operator*(nan_t, normal);

    EXPECT_TRUE(std::isnan(add_r(0)));
    EXPECT_TRUE(std::isnan(sub_r(0)));
    EXPECT_TRUE(std::isnan(mul_r(0)));
}

TEST(FloatEdgeCaseTest, VerySmallValues) {
    float eps = std::numeric_limits<float>::epsilon();
    auto t1 = Tensor::createScalar(1.0f);
    auto t2 = Tensor::createScalar(1.0f + eps);

    auto result = TensorOps::operator-(t2, t1);
    EXPECT_GT(result(0), 0.0f);
    EXPECT_LE(result(0), eps * 2.0f);
}

// ============================================================================
// NUMERICAL STABILITY TESTS
// ============================================================================

TEST(NumericalStabilityTest, ChainedOperations) {
    auto t1 = Tensor::createScalar(2.0f);
    auto t2 = Tensor::createScalar(3.0f);
    auto t3 = Tensor::createScalar(5.0f);

    // (t1 + t2) * t1 - t2 = 5 * 2 - 3 = 7
    auto r1 = TensorOps::operator+(t1, t2);
    auto r2 = TensorOps::operator*(r1, t1);
    auto r3 = TensorOps::operator-(r2, t2);

    EXPECT_FLOAT_EQ(r3(0), 7.0f);
}

TEST(NumericalStabilityTest, MixedMagnitudes) {
    auto large = Tensor::createScalar(1e10f);
    auto small = Tensor::createScalar(1e-10f);

    auto sum = TensorOps::operator+(large, small);

    // Due to float precision, small value is lost
    EXPECT_FLOAT_EQ(sum(0), 1e10f);
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================

TEST(PropertyTest, AdditionCommutativity) {
    auto t1 = Tensor::createOnes({10, 10});
    auto t2 = Tensor::createOnes({10, 10});

    for (size_t i = 0; i < 100; ++i) {
        t1.setDataElem(i, static_cast<float>(i) * 0.5f);
        t2.setDataElem(i, static_cast<float>(i) * 1.5f);
    }

    auto r1 = TensorOps::operator+(t1, t2);
    auto r2 = TensorOps::operator+(t2, t1);

    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(r1.getDataPtr()[i], r2.getDataPtr()[i]);
    }
}

TEST(PropertyTest, AdditionAssociativity) {
    auto t1 = Tensor::createScalar(1.5f);
    auto t2 = Tensor::createScalar(2.5f);
    auto t3 = Tensor::createScalar(3.5f);

    auto r1 = TensorOps::operator+(TensorOps::operator+(t1, t2), t3);
    auto r2 = TensorOps::operator+(t1, TensorOps::operator+(t2, t3));

    EXPECT_FLOAT_EQ(r1(0), r2(0));
    EXPECT_FLOAT_EQ(r1(0), 7.5f);
}

TEST(PropertyTest, MultiplicationIdentity) {
    auto t = Tensor::createOnes({3, 3});
    for (size_t i = 0; i < 9; ++i)
        t.setDataElem(i, static_cast<float>(i));

    auto identity = Tensor::createZeros({3, 3});
    identity.setDataElem(0, 1.0f);
    identity.setDataElem(4, 1.0f);
    identity.setDataElem(8, 1.0f);

    auto result = TensorOps::matmul(t, identity);

    for (size_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(result.getDataPtr()[i], t.getDataPtr()[i]);
    }
}
