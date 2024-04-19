#include <gtest/gtest.h>
#include "tensor.h"

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Any setup needed before each test
    }

    void TearDown() override {
        // Any cleanup needed after each test
    }
};

// Test initialization with integers using an initializer list
TEST_F(TensorTest, IntegerInitializationWithInitializerList) {
    Tensor<int> tensorInt({1, 1}, I32);
    int valueInt = tensorInt[{0, 0}];
    ASSERT_EQ(valueInt, 0);

    Tensor<long long> tensorLong({1, 1}, I64);
    long long valueLong = tensorLong[{0, 0}];
    ASSERT_EQ(valueLong, 0LL);
}

// Test initialization with integers using a vector
TEST_F(TensorTest, IntegerInitializationWithVector) {
    Tensor<int> tensorInt(std::vector<int>{1, 1}, I32);
    int valueInt = tensorInt[std::vector<int>{0, 0}];
    ASSERT_EQ(valueInt, 0);

    Tensor<long long> tensorLong(std::vector<int>{1, 1}, I64);
    long long valueLong = tensorLong[std::vector<int>{0, 0}];
    ASSERT_EQ(valueLong, 0LL);
}

// Test initialization with floating points using an initializer list
TEST_F(TensorTest, FloatingPointInitializationWithInitializerList) {
    Tensor<float> tensorFloat({1, 1}, FP32);
    float valueFloat = tensorFloat[{0, 0}];
    ASSERT_FLOAT_EQ(valueFloat, 0.0f);

    Tensor<double> tensorDouble({1, 1}, FP64);
    double valueDouble = tensorDouble[{0, 0}];
    ASSERT_DOUBLE_EQ(valueDouble, 0.0);
}

// Test initialization with floating points using a vector
TEST_F(TensorTest, FloatingPointInitializationWithVector) {
    Tensor<float> tensorFloat(std::vector<int>{1, 1}, FP32);
    float valueFloat = tensorFloat[std::vector<int>{0, 0}];
    ASSERT_FLOAT_EQ(valueFloat, 0.0f);

    Tensor<double> tensorDouble(std::vector<int>{1, 1}, FP64);
    double valueDouble = tensorDouble[std::vector<int>{0, 0}];
    ASSERT_DOUBLE_EQ(valueDouble, 0.0);
}

// Test initialization with complex numbers using an initializer list
TEST_F(TensorTest, ComplexNumberInitializationWithInitializerList) {
    Tensor<Complex<float>> tensorCFP32({1, 1}, CFP32);
    Complex<float> valueCFP32 = tensorCFP32[{0, 0}];
    ASSERT_FLOAT_EQ(valueCFP32.real, 0.0f);
    ASSERT_FLOAT_EQ(valueCFP32.imag, 0.0f);

    Tensor<Complex<double>> tensorCFP64({1, 1}, CFP64);
    Complex<double> valueCFP64 = tensorCFP64[{0, 0}];
    ASSERT_DOUBLE_EQ(valueCFP64.real, 0.0);
    ASSERT_DOUBLE_EQ(valueCFP64.imag, 0.0);
}

// Test initialization with complex numbers using a vector
TEST_F(TensorTest, ComplexNumberInitializationWithVector) {
    Tensor<Complex<float>> tensorCFP32(std::vector<int>{1, 1}, CFP32);
    Complex<float> valueCFP32 = tensorCFP32[std::vector<int>{0, 0}];
    ASSERT_FLOAT_EQ(valueCFP32.real, 0.0f);
    ASSERT_FLOAT_EQ(valueCFP32.imag, 0.0f);

    Tensor<Complex<double>> tensorCFP64(std::vector<int>{1, 1}, CFP64);
    Complex<double> valueCFP64 = tensorCFP64[std::vector<int>{0, 0}];
    ASSERT_DOUBLE_EQ(valueCFP64.real, 0.0);
    ASSERT_DOUBLE_EQ(valueCFP64.imag, 0.0);
}

class TensorIndexingTest : public ::testing::Test {
protected:
    Tensor<int> tensorInt3D{{3, 3, 3}, I32};
    Tensor<Complex<float>> tensorComplex2D{{3, 3}, CFP32};

    void SetUp() override {
        int value = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    tensorInt3D[{i, j, k}] = ++value;
                }
            }
        }

        int count = 1;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                tensorComplex2D[{i, j}] = Complex<float>(count * 0.1f, count * 0.1f);
                ++count;
            }
        }
    }
};

// Test integer tensor indexing with initializer list
TEST_F(TensorIndexingTest, IndexWithInitializerListInt) {
    int expectedValue = tensorInt3D[{1, 1, 1}];
    ASSERT_EQ(expectedValue, 14);
}

// Test integer tensor indexing with vector
TEST_F(TensorIndexingTest, IndexWithVectorInt) {
    std::vector<int> indices = {1, 1, 1};
    int expectedValue = tensorInt3D[indices];
    ASSERT_EQ(expectedValue, 14);
}

// Test complex tensor indexing with initializer list
TEST_F(TensorIndexingTest, IndexWithInitializerListComplex) {
    Complex<float> expectedValue = tensorComplex2D[{2, 2}];
    ASSERT_FLOAT_EQ(expectedValue.real, 0.9f);
    ASSERT_FLOAT_EQ(expectedValue.imag, 0.9f);
}

// Test complex tensor indexing with vector
TEST_F(TensorIndexingTest, IndexWithVectorComplex) {
    std::vector<int> indices = {2, 2};
    Complex<float> expectedValue = tensorComplex2D[indices];
    ASSERT_FLOAT_EQ(expectedValue.real, 0.9f);
    ASSERT_FLOAT_EQ(expectedValue.imag, 0.9f);
}

// Test indexing with wrong number of indices
TEST_F(TensorIndexingTest, IndexWithWrongNumberOfIndices) {
    std::vector<int> indicesTooMany = {0, 1, 2, 3};
    std::initializer_list<int> indicesTooFew = {1};

    EXPECT_THROW(tensorInt3D[indicesTooMany], std::invalid_argument);
    EXPECT_THROW(tensorComplex2D[indicesTooFew], std::invalid_argument);
}

// Test const versions of indexing operators
TEST_F(TensorIndexingTest, ConstIndexing) {
    const Tensor<int>& constIntTensor = tensorInt3D;
    const Tensor<Complex<float>>& constComplexTensor = tensorComplex2D;

    int expectedIntValue = constIntTensor[{2, 2, 2}];
    Complex<float> retrievedComplexValue = constComplexTensor[{2, 2}];
    float expectedComplexReal = 0.9f;
    float expectedComplexImag = 0.9f;

    ASSERT_EQ(expectedIntValue, 27);
    ASSERT_FLOAT_EQ(retrievedComplexValue.real, expectedComplexReal);
    ASSERT_FLOAT_EQ(retrievedComplexValue.imag, expectedComplexImag);
}

class TensorSlicingTest : public ::testing::Test {
protected:
    Tensor<int> tensorInt2D;

    void SetUp() override {
        tensorInt2D = Tensor<int>({5, 5}, I32);
        int value = 1;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                tensorInt2D[{i, j}] = value++;
            }
        }
    }
};

// Test full tensor slicing with default slice
TEST_F(TensorSlicingTest, DefaultFullSlice) {
    Tensor<int> sliced = tensorInt2D[Slice()];
    int firstElement = sliced[{0, 0}];
    int lastElement = sliced[{4, 4}];
    int numElements = sliced.getShape()[0] * sliced.getShape()[1];
    
    ASSERT_EQ(firstElement, 1);
    ASSERT_EQ(lastElement, 25);
    ASSERT_EQ(numElements, 25);
}

// Test specific slice with start, stop, and step
TEST_F(TensorSlicingTest, SpecificSlice) {
    Tensor<int> sliced = tensorInt2D[Slice(1, 4, 2)];
    int firstElement = sliced[{0, 0}];  // First row, first column in the slice
    int secondElement = sliced[{1, 1}]; // Second row, second column in the slice
    int numElements = sliced.getShape()[0] * sliced.getShape()[1];

    ASSERT_EQ(firstElement, 6);  // Value at position (1, 0) in the original tensor
    ASSERT_EQ(secondElement, 17); // Value at position (3, 1) in the original tensor
    ASSERT_EQ(numElements, 10);
}

// Test slicing with negative indices
TEST_F(TensorSlicingTest, NegativeIndicesSlice) {
    Tensor<int> sliced = tensorInt2D[Slice(-4, -1)];
    int firstElement = sliced[{0, 0}];
    int lastElement = sliced[{2, 4}];
    int numElements = sliced.getShape()[0] * sliced.getShape()[1];

    ASSERT_EQ(firstElement, 6);
    ASSERT_EQ(lastElement, 20);
    ASSERT_EQ(numElements, 15);
}

// Test slicing from the start with specific stop and step
TEST_F(TensorSlicingTest, FromStartSlice) {
    Tensor<int> sliced = tensorInt2D[Slice::fromStart(3)];
    int firstElement = sliced[{0, 0}];
    int lastElement = sliced[{2, 4}];
    int numElements = sliced.getShape()[0] * sliced.getShape()[1];

    ASSERT_EQ(firstElement, 1);
    ASSERT_EQ(lastElement, 15);
    ASSERT_EQ(numElements, 15);
}

// Test slicing a single element
TEST_F(TensorSlicingTest, SingleElementSlice) {
    Tensor<int> sliced = tensorInt2D[Slice::single(2)];
    int firstElement = sliced[{0, 0}];
    int lastElement = sliced[{0, 4}];
    int numElements = sliced.getShape()[0] * sliced.getShape()[1];

    ASSERT_EQ(firstElement, 11);
    ASSERT_EQ(lastElement, 15);
    ASSERT_EQ(numElements, 5);
}

TEST_F(TensorSlicingTest, OutOfRangeSlice) {
    Tensor<int> sliced = tensorInt2D[Slice(0, 6)];
    int numElements = sliced.getShape()[0] * sliced.getShape()[1];
    ASSERT_EQ(numElements, 25);

    sliced = tensorInt2D[Slice(-10, 3)];
    int firstElement = sliced[{0, 0}];
    int lastElement = sliced[{2, 4}];
    ASSERT_EQ(firstElement, 1);
    ASSERT_EQ(lastElement, 15);
}

// Test invalid slice (step zero)
TEST_F(TensorSlicingTest, InvalidStepSlice) {
    EXPECT_THROW(tensorInt2D[Slice(1, 3, 0)], std::invalid_argument);
}

// Test slice requiring the tensor to be contiguous
TEST_F(TensorSlicingTest, RequiresContiguity) {
    Tensor<int> nonContiguousTensor(std::vector<int>{3, 3}, I32);
    nonContiguousTensor = nonContiguousTensor.transpose(); // Make the tensor non-contiguous
    EXPECT_THROW(nonContiguousTensor[Slice()], std::runtime_error);
}

class TensorConcatTest : public ::testing::Test {
protected:
    Tensor<int> tensor2D;
    Tensor<int> tensor2D_other;
    Tensor<int> tensor3D;

    void SetUp() override {
        // Setting up a 2x2 tensor
        tensor2D = Tensor<int>({2, 2}, I32);
        int value = 1;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                tensor2D[{i, j}] = value++;
            }
        }

        // Setting up another 2x2 tensor for concatenation tests
        tensor2D_other = Tensor<int>({2, 2}, I32);
        value = 5;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                tensor2D_other[{i, j}] = value++;
            }
        }

        // Setting up a 2x2x2 tensor for 3D concatenation tests
        tensor3D = Tensor<int>({2, 2, 2}, I32);
        value = 1;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    tensor3D[{i, j, k}] = value++;
                }
            }
        }
    }
};

TEST_F(TensorConcatTest, ConcatAlongFirstDimension) {
    Tensor<int> result = tensor2D.concat(tensor2D_other, 0);
    ASSERT_EQ(result.getShape()[0], 4);
    ASSERT_EQ(result.getShape()[1], 2);
    int val = result[{3, 1}];
    ASSERT_EQ(val, 8);  // Check last element in the concatenated result
}

TEST_F(TensorConcatTest, ConcatAlongInnerDimension) {
    Tensor<int> result = tensor2D.concat(tensor2D_other, 1);
    ASSERT_EQ(result.getShape()[0], 2);
    ASSERT_EQ(result.getShape()[1], 4);
    int val = result[{1, 3}];
    ASSERT_EQ(val, 8);  // Check last element in the concatenated result
}

TEST_F(TensorConcatTest, ConcatAlongLastDimension) {
    Tensor<int> result = tensor3D.concat(tensor3D, 2);
    ASSERT_EQ(result.getShape()[2], 4); // Depth doubled
    int val = result[{1, 1, 3}];
    ASSERT_EQ(val, 8);  // Check an element in the concatenated result
}

TEST_F(TensorConcatTest, NonContiguousTensorConcat) {
    Tensor<int> nonContiguousTensor = tensor2D.permute({1, 0});  // Make it non-contiguous
    EXPECT_THROW(nonContiguousTensor.concat(tensor2D, 0), std::runtime_error);
}

TEST_F(TensorConcatTest, MismatchedDimensions) {
    EXPECT_THROW(tensor2D.concat(tensor3D, 0), std::invalid_argument);
}

TEST_F(TensorConcatTest, MismatchedShapes) {
    Tensor<int> mismatchedTensor = Tensor<int>({3, 2}, I32);  // Different first dimension
    EXPECT_THROW(tensor2D.concat(mismatchedTensor, 1), std::invalid_argument);
}

TEST_F(TensorConcatTest, OutOfBoundsAxis) {
    EXPECT_THROW(tensor2D.concat(tensor2D, 3), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}