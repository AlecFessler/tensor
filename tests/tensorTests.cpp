#include <complex>

#include <gtest/gtest.h>
#include "tensor.h"

class TensorTest : public ::testing::Test {};

// Test initialization with integers using an initializer list
TEST_F(TensorTest, IntegerInitializationWithInitializerList) {
    Tensor<int> tensorInt({1, 1});
    int valueInt = tensorInt[{0, 0}];
    ASSERT_EQ(valueInt, 0);

    Tensor<long> tensorLong({1, 1});
    long valueLong = tensorLong[{0, 0}];
    ASSERT_EQ(valueLong, 0L);
}

// Test initialization with integers using a vector
TEST_F(TensorTest, IntegerInitializationWithVector) {
    Tensor<int> tensorInt(std::vector<int>{1, 1});
    int valueInt = tensorInt[std::vector<int>{0, 0}];
    ASSERT_EQ(valueInt, 0);

    Tensor<long> tensorLong(std::vector<int>{1, 1});
    long long valueLong = tensorLong[std::vector<int>{0, 0}];
    ASSERT_EQ(valueLong, 0L);
}

// Test initialization with floating points using an initializer list
TEST_F(TensorTest, FloatingPointInitializationWithInitializerList) {
    Tensor<float> tensorFloat({1, 1});
    float valueFloat = tensorFloat[{0, 0}];
    ASSERT_FLOAT_EQ(valueFloat, 0.0f);

    Tensor<double> tensorDouble({1, 1});
    double valueDouble = tensorDouble[{0, 0}];
    ASSERT_DOUBLE_EQ(valueDouble, 0.0);
}

// Test initialization with floating points using a vector
TEST_F(TensorTest, FloatingPointInitializationWithVector) {
    Tensor<float> tensorFloat(std::vector<int>{1, 1});
    float valueFloat = tensorFloat[std::vector<int>{0, 0}];
    ASSERT_FLOAT_EQ(valueFloat, 0.0f);

    Tensor<double> tensorDouble(std::vector<int>{1, 1});
    double valueDouble = tensorDouble[std::vector<int>{0, 0}];
    ASSERT_DOUBLE_EQ(valueDouble, 0.0);
}

// Test initialization with complex numbers using an initializer list
TEST_F(TensorTest, ComplexNumberInitializationWithInitializerList) {
    Tensor<std::complex<float>> tensorC64({1, 1});
    std::complex<float> valueC64 = tensorC64[{0, 0}];
    ASSERT_FLOAT_EQ(valueC64.real(), 0.0f);
    ASSERT_FLOAT_EQ(valueC64.imag(), 0.0f);

    Tensor<std::complex<double>> tensorC128({1, 1});
    std::complex<double> valueC128 = tensorC128[{0, 0}];
    ASSERT_DOUBLE_EQ(valueC128.real(), 0.0);
    ASSERT_DOUBLE_EQ(valueC128.imag(), 0.0);
}

// Test initialization with complex numbers using a vector
TEST_F(TensorTest, ComplexNumberInitializationWithVector) {
    Tensor<std::complex<float>> tensorC64(std::vector<int>{1, 1});
    std::complex<float> valueC64 = tensorC64[std::vector<int>{0, 0}];
    ASSERT_FLOAT_EQ(valueC64.real(), 0.0f);
    ASSERT_FLOAT_EQ(valueC64.imag(), 0.0f);

    Tensor<std::complex<double>> tensorC128(std::vector<int>{1, 1});
    std::complex<double> valueC128 = tensorC128[std::vector<int>{0, 0}];
    ASSERT_DOUBLE_EQ(valueC128.real(), 0.0);
    ASSERT_DOUBLE_EQ(valueC128.imag(), 0.0);
}

TEST_F(TensorTest, FromFileNpy) {
    std::string filePath = "test_data.npy";
    std::vector<int> expectedShape = {2, 3, 4};
    d_type expectedDtype = d_type::FP64;

    // Create a sample tensor and save it to a .npy file
    Tensor<double> originalTensor(expectedShape);
    for (int i = 0; i < expectedShape[0]; ++i) {
        for (int j = 0; j < expectedShape[1]; ++j) {
            for (int k = 0; k < expectedShape[2]; ++k) {
                double value = static_cast<double>(i * expectedShape[1] * expectedShape[2] + j * expectedShape[2] + k);
                originalTensor[{i, j, k}] = value;
            }
        }
    }
    originalTensor.toFile(filePath);

    // Load the tensor from the .npy file
    Tensor<double> loadedTensor = Tensor<double>::fromFile(filePath);

    // Check if the loaded tensor has the expected shape, dtype, and values
    std::vector<int> loadedShape = loadedTensor.getShape();
    EXPECT_EQ(loadedShape, expectedShape);
    EXPECT_EQ(loadedTensor.getDtype(), expectedDtype);
    for (int i = 0; i < expectedShape[0]; ++i) {
        for (int j = 0; j < expectedShape[1]; ++j) {
            for (int k = 0; k < expectedShape[2]; ++k) {
                double expectedValue = static_cast<double>(i * expectedShape[1] * expectedShape[2] + j * expectedShape[2] + k);
                double loadedValue = loadedTensor[{i, j, k}];
                EXPECT_EQ(loadedValue, expectedValue);
            }
        }
    }
}

TEST_F(TensorTest, ToFileNpy) {
    std::string filePath = "test_data.npy";
    std::vector<int> expectedShape = {2, 3, 4};
    d_type expectedDtype = d_type::FP64;

    // Create a sample tensor
    Tensor<double> tensor(expectedShape);
    for (int i = 0; i < expectedShape[0]; ++i) {
        for (int j = 0; j < expectedShape[1]; ++j) {
            for (int k = 0; k < expectedShape[2]; ++k) {
                double value = static_cast<double>(i * expectedShape[1] * expectedShape[2] + j * expectedShape[2] + k);
                tensor[{i, j, k}] = value;
            }
        }
    }

    // Save the tensor to a .npy file
    tensor.toFile(filePath);

    // Load the tensor from the .npy file
    Tensor<double> loadedTensor = Tensor<double>::fromFile(filePath);

    // Check if the loaded tensor has the expected shape, dtype, and values
    std::vector<int> loadedShape = loadedTensor.getShape();
    EXPECT_EQ(loadedShape, expectedShape);
    EXPECT_EQ(loadedTensor.getDtype(), expectedDtype);
    for (int i = 0; i < expectedShape[0]; ++i) {
        for (int j = 0; j < expectedShape[1]; ++j) {
            for (int k = 0; k < expectedShape[2]; ++k) {
                double expectedValue = static_cast<double>(i * expectedShape[1] * expectedShape[2] + j * expectedShape[2] + k);
                double loadedValue = loadedTensor[{i, j, k}];
                EXPECT_EQ(loadedValue, expectedValue);
            }
        }
    }
}

TEST_F(TensorTest, arange) {
    Tensor<int> tensor = Tensor<int>::arange(0, 10, 1);
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(tensor[{i}], i);
    }

    Tensor<int> tensorStep = Tensor<int>::arange(0, 10, 2);
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(tensorStep[{i}], i * 2);
    }
}

TEST_F(TensorTest, full) {
    Tensor<int> tensor = Tensor<int>::full({2, 2}, 5);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int result = tensor[{i, j}];
            ASSERT_EQ(result, 5);
        }
    }

    Tensor<float> tensorFloat = Tensor<float>::full({2, 2}, 3.14f);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float result = tensorFloat[{i, j}];
            ASSERT_FLOAT_EQ(result, 3.14f);
        }
    }
}

class TensorIndexingTest : public ::testing::Test {
protected:
    Tensor<int> tensorInt3D{{3, 3, 3}};
    Tensor<std::complex<float>> tensorComplex2D{{3, 3}};

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
                tensorComplex2D[{i, j}] = std::complex<float>(count * 0.1f, count * 0.1f);
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
    std::complex<float> expectedValue = tensorComplex2D[{2, 2}];
    ASSERT_FLOAT_EQ(expectedValue.real(), 0.9f);
    ASSERT_FLOAT_EQ(expectedValue.imag(), 0.9f);
}

// Test complex tensor indexing with vector
TEST_F(TensorIndexingTest, IndexWithVectorComplex) {
    std::vector<int> indices = {2, 2};
    std::complex<float> expectedValue = tensorComplex2D[indices];
    ASSERT_FLOAT_EQ(expectedValue.real(), 0.9f);
    ASSERT_FLOAT_EQ(expectedValue.imag(), 0.9f);
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
    const Tensor<std::complex<float>>& constComplexTensor = tensorComplex2D;

    int expectedIntValue = constIntTensor[{2, 2, 2}];
    std::complex<float> retrievedComplexValue = constComplexTensor[{2, 2}];
    float expectedComplexReal = 0.9f;
    float expectedComplexIm = 0.9f;

    ASSERT_EQ(expectedIntValue, 27);
    ASSERT_FLOAT_EQ(retrievedComplexValue.real(), expectedComplexReal);
    ASSERT_FLOAT_EQ(retrievedComplexValue.imag(), expectedComplexIm);
}

class TensorSlicingTest : public ::testing::Test {
protected:
    Tensor<int> tensorInt2D;

    void SetUp() override {
        tensorInt2D = Tensor<int>({5, 5});
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
    Tensor<int> nonContiguousTensor(std::vector<int>{3, 3});
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
        tensor2D = Tensor<int>({2, 2});
        int value = 1;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                tensor2D[{i, j}] = value++;
            }
        }

        // Setting up another 2x2 tensor for concatenation tests
        tensor2D_other = Tensor<int>({2, 2});
        value = 5;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                tensor2D_other[{i, j}] = value++;
            }
        }

        // Setting up a 2x2x2 tensor for 3D concatenation tests
        tensor3D = Tensor<int>({2, 2, 2});
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
    Tensor<int> mismatchedTensor = Tensor<int>({3, 2});  // Different first dimension
    EXPECT_THROW(tensor2D.concat(mismatchedTensor, 1), std::invalid_argument);
}

TEST_F(TensorConcatTest, OutOfBoundsAxis) {
    EXPECT_THROW(tensor2D.concat(tensor2D, 3), std::invalid_argument);
}

class TensorPadTest : public ::testing::Test {
protected:
    Tensor<int> tensor1D;
    Tensor<int> tensor2D;
    Tensor<int> tensor3D;

    void SetUp() override {
        tensor1D = Tensor<int>({5});
        for (int i = 0; i < 5; ++i) {
            tensor1D[{i}] = i + 1;  // 1, 2, 3, 4, 5
        }

        tensor2D = Tensor<int>({3, 3});
        int value = 1;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                tensor2D[{i, j}] = value++;
            }
        }

        tensor3D = Tensor<int>({2, 2, 2});
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

TEST_F(TensorPadTest, ReflectivePadding1D) {
    auto paddedTensor = tensor1D.pad({5, 5}, 0, REFLECTIVE);
    int firstReflectiveValue = paddedTensor[{0}];
    int lastReflectiveValue = paddedTensor[{static_cast<int>(paddedTensor.size()) - 1}];
    int expectedFirstReflectiveValue = tensor1D[{3}];
    int expectedLastReflectiveValue = tensor1D[{1}];
    ASSERT_EQ(firstReflectiveValue, expectedFirstReflectiveValue);
    ASSERT_EQ(lastReflectiveValue, expectedLastReflectiveValue);
}

TEST_F(TensorPadTest, ConstantPaddingFirstAxis) {
    auto paddedTensor = tensor2D.pad({1, 1}, 0, CONSTANT, 0);
    int firstPaddingValue = paddedTensor[{0, 0}];
    int lastPaddingValue = paddedTensor[{4, 2}];
    ASSERT_EQ(firstPaddingValue, 0);
    ASSERT_EQ(lastPaddingValue, 0);
}

TEST_F(TensorPadTest, ReflectivePaddingFirstAxis) {
    auto paddedTensor = tensor2D.pad({1, 1}, 0, REFLECTIVE);
    int firstReflectiveValue = paddedTensor[{0, 0}];
    int lastReflectiveValue = paddedTensor[{4, 0}];
    int expectedFirstReflectiveValue = tensor2D[{1, 0}];
    int expectedLastReflectiveValue = tensor2D[{1, 0}];
    ASSERT_EQ(firstReflectiveValue, expectedFirstReflectiveValue);
    ASSERT_EQ(lastReflectiveValue, expectedLastReflectiveValue);
}

TEST_F(TensorPadTest, ConstantPaddingInnerAxis) {
    auto paddedTensor = tensor3D.pad({1, 1}, 1, CONSTANT, -1);
    int firstInnerPaddingValue = paddedTensor[{0, 0, 0}];
    int lastInnerPaddingValue = paddedTensor[{0, 3, 1}];
    ASSERT_EQ(firstInnerPaddingValue, -1);
    ASSERT_EQ(lastInnerPaddingValue, -1);
}

TEST_F(TensorPadTest, ReflectivePaddingInnerAxis) {
    auto paddedTensor = tensor3D.pad({1, 1}, 1, REFLECTIVE);
    int firstReflectiveValue = paddedTensor[{0, 0, 0}];
    int lastReflectiveValue = paddedTensor[{0, 3, 0}];
    int expectedFirstReflectiveValue = tensor3D[{0, 1, 0}];
    int expectedLastReflectiveValue = tensor3D[{0, 0, 0}]; 
    ASSERT_EQ(firstReflectiveValue, expectedFirstReflectiveValue);
    ASSERT_EQ(lastReflectiveValue, expectedLastReflectiveValue);
}

TEST_F(TensorPadTest, ConstantPaddingLastAxis) {
    auto paddedTensor = tensor3D.pad({2, 2}, 2, CONSTANT, 9);
    int firstLastAxisPaddingValue = paddedTensor[{0, 0, 0}];
    int lastLastAxisPaddingValue = paddedTensor[{0, 0, 5}];
    ASSERT_EQ(firstLastAxisPaddingValue, 9);
    ASSERT_EQ(lastLastAxisPaddingValue, 9);
}

TEST_F(TensorPadTest, ReflectivePaddingLastAxis) {
    auto paddedTensor = tensor3D.pad({2, 2}, 2, REFLECTIVE);
    int firstReflectiveValue = paddedTensor[{0, 0, 0}];
    int lastReflectiveValue = paddedTensor[{0, 0, 5}];
    int expectedFirstReflectiveValue = tensor3D[{0, 0, 0}];
    int expectedLastReflectiveValue = tensor3D[{0, 0, 1}];
    ASSERT_EQ(firstReflectiveValue, expectedFirstReflectiveValue);
    ASSERT_EQ(lastReflectiveValue, expectedLastReflectiveValue);
}

TEST_F(TensorPadTest, NonContiguousTensorException) {
    Tensor<int> nonContiguousTensor = tensor2D.transpose(); // Make it non-contiguous
    EXPECT_THROW(nonContiguousTensor.pad({1, 1}, 1, CONSTANT, 0), std::runtime_error);
}

TEST_F(TensorPadTest, InvalidAxisException) {
    EXPECT_THROW(tensor2D.pad({1, 1}, -1, CONSTANT, 0), std::invalid_argument);
    EXPECT_THROW(tensor2D.pad({1, 1}, 3, CONSTANT, 0), std::invalid_argument);
}

class TensorFrameTest : public ::testing::Test {
protected:
    Tensor<int> tensor1D;
    Tensor<int> tensor2D;

    void SetUp() override {
        tensor1D = Tensor<int>({10});
        for (int i = 0; i < 10; ++i) {
            tensor1D[{i}] = i + 1;
        }

        tensor2D = Tensor<int>({4, 4});
        int value = 1;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                tensor2D[{i, j}] = value++;
            }
        }
    }
};

TEST_F(TensorFrameTest, FrameWithOverlap) {
    auto framedTensor = tensor1D.frame(5, 2, 0);
    ASSERT_EQ(framedTensor.getShape()[0], 3);
    ASSERT_EQ(framedTensor.getShape()[1], 5);
    int firstFrameValue = framedTensor[{0, 0}];
    int lastFrameValue = framedTensor[{2, 4}];
    ASSERT_EQ(firstFrameValue, 1);
    ASSERT_EQ(lastFrameValue, 9);
    ASSERT_FALSE(framedTensor.isContiguous());
}

TEST_F(TensorFrameTest, FrameWithoutOverlap) {
    auto framedTensor = tensor1D.frame(5, 5, 0);
    ASSERT_EQ(framedTensor.getShape()[0], 2);
    ASSERT_EQ(framedTensor.getShape()[1], 5);
    int firstFrameValue = framedTensor[{0, 0}];
    int lastFrameValue = framedTensor[{1, 4}];
    ASSERT_EQ(firstFrameValue, 1);
    ASSERT_EQ(lastFrameValue, 10);
    ASSERT_TRUE(framedTensor.isContiguous());
}

TEST_F(TensorFrameTest, FrameWithInnerDimension) {
    auto framedTensor = tensor2D.frame(2, 1, 1);
    ASSERT_EQ(framedTensor.getShape()[0], 4);
    ASSERT_EQ(framedTensor.getShape()[1], 3);
    ASSERT_EQ(framedTensor.getShape()[2], 2);
    int firstFrameValue = framedTensor[{0, 0, 0}];
    int lastFrameValue = framedTensor[{3, 2, 1}];
    ASSERT_EQ(firstFrameValue, 1);
    ASSERT_EQ(lastFrameValue, 16);
    ASSERT_FALSE(framedTensor.isContiguous());
}

TEST_F(TensorFrameTest, NonContiguousTensorException) {
    Tensor<int> nonContiguousTensor = tensor2D.transpose(); // Make it non-contiguous
    EXPECT_THROW(nonContiguousTensor.frame(5, 2, 0), std::runtime_error);
}

TEST_F(TensorFrameTest, InvalidAxisException) {
    EXPECT_THROW(tensor1D.frame(5, 2, -1), std::invalid_argument);
    EXPECT_THROW(tensor1D.frame(5, 2, 2), std::invalid_argument);
}

TEST_F(TensorFrameTest, InvalidFrameSizeException) {
    EXPECT_THROW(tensor1D.frame(0, 2, 0), std::invalid_argument);
    EXPECT_THROW(tensor1D.frame(5, 0, 0), std::invalid_argument);
}

TEST_F(TensorFrameTest, InvalidHopLengthException) {
    EXPECT_THROW(tensor1D.frame(5, -1, 0), std::invalid_argument);
}

class TensorDotTest : public ::testing::Test {};

TEST_F(TensorDotTest, DotProductInt32) {
    Tensor<int> tensor1({3});
    Tensor<int> tensor2({3});
    tensor1[{0}] = 1; tensor1[{1}] = 2; tensor1[{2}] = 3;
    tensor2[{0}] = 4; tensor2[{1}] = 5; tensor2[{2}] = 6;
    int result = tensor1.dot(tensor2);
    ASSERT_EQ(result, 32);
}

TEST_F(TensorDotTest, DotProductInt64) {
    Tensor<long> tensor1({3});
    Tensor<long> tensor2({3});
    tensor1[{0}] = 1L; tensor1[{1}] = 2L; tensor1[{2}] = 3L;
    tensor2[{0}] = 4L; tensor2[{1}] = 5L; tensor2[{2}] = 6L;
    long long result = tensor1.dot(tensor2);
    ASSERT_EQ(result, 32L);
}

TEST_F(TensorDotTest, DotProductFloat32) {
    Tensor<float> tensor1({3});
    Tensor<float> tensor2({3});
    tensor1[{0}] = 1.0f; tensor1[{1}] = 2.0f; tensor1[{2}] = 3.0f;
    tensor2[{0}] = 4.0f; tensor2[{1}] = 5.0f; tensor2[{2}] = 6.0f;
    float result = tensor1.dot(tensor2);
    ASSERT_FLOAT_EQ(result, 32.0f);
}

TEST_F(TensorDotTest, DotProductFloat64) {
    Tensor<double> tensor1({3});
    Tensor<double> tensor2({3});
    tensor1[{0}] = 1.0; tensor1[{1}] = 2.0; tensor1[{2}] = 3.0;
    tensor2[{0}] = 4.0; tensor2[{1}] = 5.0; tensor2[{2}] = 6.0;
    double result = tensor1.dot(tensor2);
    ASSERT_DOUBLE_EQ(result, 32.0);
}

TEST_F(TensorDotTest, DotProductComplex32) {
    Tensor<std::complex<float>> tensor1({3});
    Tensor<std::complex<float>> tensor2({3});
    tensor1[{0}] = std::complex<float>(1.0f, 2.0f);
    tensor1[{1}] = std::complex<float>(3.0f, 4.0f);
    tensor1[{2}] = std::complex<float>(5.0f, 6.0f);
    tensor2[{0}] = std::complex<float>(7.0f, 8.0f);
    tensor2[{1}] = std::complex<float>(9.0f, 10.0f);
    tensor2[{2}] = std::complex<float>(11.0f, 12.0f);
    std::complex<float> result = tensor1.dot(tensor2);
    ASSERT_FLOAT_EQ(result.real(), -39.0f);
    ASSERT_FLOAT_EQ(result.imag(), 214.0f);
}

TEST_F(TensorDotTest, DotProductComplex64) {
    Tensor<std::complex<double>> tensor1({3});
    Tensor<std::complex<double>> tensor2({3});
    tensor1[{0}] = std::complex<double>(1.0f, 2.0f);
    tensor1[{1}] = std::complex<double>(3.0f, 4.0f);
    tensor1[{2}] = std::complex<double>(5.0f, 6.0f);
    tensor2[{0}] = std::complex<double>(7.0f, 8.0f);
    tensor2[{1}] = std::complex<double>(9.0f, 10.0f);
    tensor2[{2}] = std::complex<double>(11.0f, 12.0f);
    std::complex<double> result = tensor1.dot(tensor2);
    ASSERT_DOUBLE_EQ(result.real(), -39.0);
    ASSERT_DOUBLE_EQ(result.imag(), 214.0);
}

TEST_F(TensorDotTest, NonContiguousTensorException) {
    Tensor<int> tensor1({3, 3});
    Tensor<int> tensor2({3, 3});
    Tensor<int> nonContiguousTensor1 = tensor1.permute({1, 0});
    Tensor<int> nonContiguousTensor2 = tensor2.permute({1, 0});
    EXPECT_THROW(nonContiguousTensor1.dot(tensor2), std::runtime_error);
    EXPECT_THROW(tensor1.dot(nonContiguousTensor2), std::runtime_error);
}

TEST_F(TensorDotTest, MismatchedShapesException) {
    Tensor<int> tensor1({3});
    Tensor<int> tensor2({4});
    EXPECT_THROW(tensor1.dot(tensor2), std::invalid_argument);
}

class TensorMatMulTest : public ::testing::Test {
protected:
    Tensor<int> tensorInt1, tensorInt2;
    Tensor<float> tensorFloat1, tensorFloat2;
    Tensor<std::complex<float>> tensorComplex1, tensorComplex2;
    Tensor<float> tensorBatch1, tensorBatch2;

    void SetUp() override {
        // Setup integer tensors
        tensorInt1 = Tensor<int>({2, 2});
        tensorInt2 = Tensor<int>({2, 2});
        tensorInt1[{0, 0}] = 1; tensorInt1[{0, 1}] = 2;
        tensorInt1[{1, 0}] = 3; tensorInt1[{1, 1}] = 4;
        tensorInt2[{0, 0}] = 2; tensorInt2[{0, 1}] = 0;
        tensorInt2[{1, 0}] = 1; tensorInt2[{1, 1}] = 2;

        // Setup float tensors
        tensorFloat1 = Tensor<float>({2, 2});
        tensorFloat2 = Tensor<float>({2, 2});
        tensorFloat1[{0, 0}] = 1.5f; tensorFloat1[{0, 1}] = 2.5f;
        tensorFloat1[{1, 0}] = 3.5f; tensorFloat1[{1, 1}] = 4.5f;
        tensorFloat2[{0, 0}] = 2.1f; tensorFloat2[{0, 1}] = 0.1f;
        tensorFloat2[{1, 0}] = 1.1f; tensorFloat2[{1, 1}] = 2.1f;

        // Setup complex number tensors
        tensorComplex1 = Tensor<std::complex<float>>({2, 2});
        tensorComplex2 = Tensor<std::complex<float>>({2, 2});
        tensorComplex1[{0, 0}] = std::complex<float>(1.0f, 2.0f);
        tensorComplex1[{0, 1}] = std::complex<float>(3.0f, 4.0f);
        tensorComplex1[{1, 0}] = std::complex<float>(5.0f, 6.0f);
        tensorComplex1[{1, 1}] = std::complex<float>(7.0f, 8.0f);
        tensorComplex2[{0, 0}] = std::complex<float>(2.0f, 1.0f);
        tensorComplex2[{0, 1}] = std::complex<float>(0.0f, 0.0f);
        tensorComplex2[{1, 0}] = std::complex<float>(1.0f, 1.0f);
        tensorComplex2[{1, 1}] = std::complex<float>(2.0f, 2.0f);

        // Setup batched tensors
        tensorBatch1 = Tensor<float>({2, 2, 2});
        tensorBatch2 = Tensor<float>({2, 2, 2});

        // First batch initialization
        tensorBatch1[{0, 0, 0}] = 1; tensorBatch1[{0, 0, 1}] = 2;
        tensorBatch1[{0, 1, 0}] = 3; tensorBatch1[{0, 1, 1}] = 4;

        tensorBatch2[{0, 0, 0}] = 2; tensorBatch2[{0, 0, 1}] = 0;
        tensorBatch2[{0, 1, 0}] = 1; tensorBatch2[{0, 1, 1}] = 2;

        // Second batch initialization
        tensorBatch1[{1, 0, 0}] = 5; tensorBatch1[{1, 0, 1}] = 6;
        tensorBatch1[{1, 1, 0}] = 7; tensorBatch1[{1, 1, 1}] = 8;

        tensorBatch2[{1, 0, 0}] = 1; tensorBatch2[{1, 0, 1}] = 2;
        tensorBatch2[{1, 1, 0}] = 3; tensorBatch2[{1, 1, 1}] = 4;
    }
};

TEST_F(TensorMatMulTest, MatMulInteger) {
    Tensor<int> result = tensorInt1.matmul(tensorInt2);
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    ASSERT_EQ(res00, 4);
    ASSERT_EQ(res01, 4);
    ASSERT_EQ(res10, 10);
    ASSERT_EQ(res11, 8);
}

TEST_F(TensorMatMulTest, MatMulFloat) {
    Tensor<float> result = tensorFloat1.matmul(tensorFloat2);
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 5.9f, 0.001);
    ASSERT_NEAR(res01, 5.4f, 0.001);
    ASSERT_NEAR(res10, 12.3f, 0.001);
    ASSERT_NEAR(res11, 9.8f, 0.001);
}

TEST_F(TensorMatMulTest, MatMulComplex) {
    Tensor<std::complex<float>> result = tensorComplex1.matmul(tensorComplex2);
    std::complex<float> res00 = result[{0, 0}];
    std::complex<float> res01 = result[{0, 1}];
    std::complex<float> res10 = result[{1, 0}];
    std::complex<float> res11 = result[{1, 1}];
    ASSERT_FLOAT_EQ(res00.real(), -1.0f);
    ASSERT_FLOAT_EQ(res00.imag(), 12.0f);
    ASSERT_FLOAT_EQ(res01.real(), -2.0f);
    ASSERT_FLOAT_EQ(res01.imag(), 14.0f);
    ASSERT_FLOAT_EQ(res10.real(), 3.0f);
    ASSERT_FLOAT_EQ(res10.imag(), 32.0f);
    ASSERT_FLOAT_EQ(res11.real(), -2.0f);
    ASSERT_FLOAT_EQ(res11.imag(), 30.0f);
}

TEST_F(TensorMatMulTest, MatMulWithBatching) {
    Tensor<float> result = tensorBatch1.matmul(tensorBatch2);
    
    float result00 = result[{0, 0, 0}];
    float result01 = result[{0, 0, 1}];
    float result10 = result[{0, 1, 0}];
    float result11 = result[{0, 1, 1}];
    float result20 = result[{1, 0, 0}];
    float result21 = result[{1, 0, 1}];
    float result30 = result[{1, 1, 0}];
    float result31 = result[{1, 1, 1}];

    ASSERT_NEAR(result00, 4.0f, 0.001);
    ASSERT_NEAR(result01, 4.0f, 0.001);
    ASSERT_NEAR(result10, 10.0f, 0.001);
    ASSERT_NEAR(result11, 8.0f, 0.001);
    ASSERT_NEAR(result20, 23.0f, 0.001);
    ASSERT_NEAR(result21, 34.0f, 0.001);
    ASSERT_NEAR(result30, 31.0f, 0.001);
    ASSERT_NEAR(result31, 46.0f, 0.001);
}

class TensorOperationsTest : public ::testing::Test {
protected:
    Tensor<int> tensorInt1, tensorInt2;
    Tensor<float> tensorFloat;
    Tensor<std::complex<float>> tensorComplex1, tensorComplex2;
    Tensor<int> tensorBroadcast1, tensorBroadcast2;

    void SetUp() override {
        // Setup integer tensors
        tensorInt1 = Tensor<int>({2, 2});
        tensorInt2 = Tensor<int>({2, 2});
        tensorInt1[{0, 0}] = 10; tensorInt1[{0, 1}] = 20;
        tensorInt1[{1, 0}] = 30; tensorInt1[{1, 1}] = 40;
        tensorInt2[{0, 0}] = 5; tensorInt2[{0, 1}] = 6;
        tensorInt2[{1, 0}] = 7; tensorInt2[{1, 1}] = 8;

        // Setup floating point tensor
        tensorFloat = Tensor<float>({2, 2});
        tensorFloat[{0, 0}] = 1.5f; tensorFloat[{0, 1}] = 2.5f;
        tensorFloat[{1, 0}] = 3.5f; tensorFloat[{1, 1}] = 4.5f;

        // Setup complex number tensors
        tensorComplex1 = Tensor<std::complex<float>>({2, 2});
        tensorComplex2 = Tensor<std::complex<float>>({2, 2});
        tensorComplex1[{0, 0}] = std::complex<float>(1.0f, 2.0f);
        tensorComplex1[{0, 1}] = std::complex<float>(3.0f, 4.0f);
        tensorComplex1[{1, 0}] = std::complex<float>(5.0f, 6.0f);
        tensorComplex1[{1, 1}] = std::complex<float>(7.0f, 8.0f);
        tensorComplex2[{0, 0}] = std::complex<float>(2.0f, 1.0f);
        tensorComplex2[{0, 1}] = std::complex<float>(1.0f, 1.0f);
        tensorComplex2[{1, 0}] = std::complex<float>(1.0f, 1.0f);
        tensorComplex2[{1, 1}] = std::complex<float>(2.0f, 2.0f);

        // Setup tensors for broadcasting test
        tensorBroadcast1 = Tensor<int>({1, 3});
        tensorBroadcast2 = Tensor<int>({3, 1});
        tensorBroadcast1[{0, 0}] = 2; tensorBroadcast1[{0, 1}] = 3; tensorBroadcast1[{0, 2}] = 4;
        tensorBroadcast2[{0, 0}] = 1; tensorBroadcast2[{1, 0}] = 2; tensorBroadcast2[{2, 0}] = 3;
    }
};

// Addition tests
TEST_F(TensorOperationsTest, AddTensorsInteger) {
    Tensor<int> result = tensorInt1 + tensorInt2;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    ASSERT_EQ(res00, 15);
    ASSERT_EQ(res01, 26);
    ASSERT_EQ(res10, 37);
    ASSERT_EQ(res11, 48);
}

TEST_F(TensorOperationsTest, AddTensorScalarFloat) {
    float scalar = 1.0f;
    Tensor<float> result = tensorFloat + scalar;
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 2.5f, 0.001);
    ASSERT_NEAR(res01, 3.5f, 0.001);
    ASSERT_NEAR(res10, 4.5f, 0.001);
    ASSERT_NEAR(res11, 5.5f, 0.001);
}

TEST_F(TensorOperationsTest, AddTensorsComplex) {
    Tensor<std::complex<float>> result = tensorComplex1 + tensorComplex2;
    std::complex<float> res00 = result[{0, 0}];
    std::complex<float> res01 = result[{0, 1}];
    std::complex<float> res10 = result[{1, 0}];
    std::complex<float> res11 = result[{1, 1}];
    ASSERT_FLOAT_EQ(res00.real(), 3.0f);
    ASSERT_FLOAT_EQ(res00.imag(), 3.0f);
    ASSERT_FLOAT_EQ(res01.real(), 4.0f);
    ASSERT_FLOAT_EQ(res01.imag(), 5.0f);
    ASSERT_FLOAT_EQ(res10.real(), 6.0f);
    ASSERT_FLOAT_EQ(res10.imag(), 7.0f);
    ASSERT_FLOAT_EQ(res11.real(), 9.0f);
    ASSERT_FLOAT_EQ(res11.imag(), 10.0f);
}

TEST_F(TensorOperationsTest, AddTensorsBroadcast) {
    Tensor<int> result = tensorBroadcast1 + tensorBroadcast2;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res02 = result[{0, 2}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    int res12 = result[{1, 2}];
    int res20 = result[{2, 0}];
    int res21 = result[{2, 1}];
    int res22 = result[{2, 2}];
    ASSERT_EQ(res00, 3);
    ASSERT_EQ(res01, 4);
    ASSERT_EQ(res02, 5);
    ASSERT_EQ(res10, 4);
    ASSERT_EQ(res11, 5);
    ASSERT_EQ(res12, 6);
    ASSERT_EQ(res20, 5);
    ASSERT_EQ(res21, 6);
    ASSERT_EQ(res22, 7);
}

// Subtraction tests
TEST_F(TensorOperationsTest, SubtractTensorsInteger) {
    Tensor<int> result = tensorInt1 - tensorInt2;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    ASSERT_EQ(res00, 5);
    ASSERT_EQ(res01, 14);
    ASSERT_EQ(res10, 23);
    ASSERT_EQ(res11, 32);
}

TEST_F(TensorOperationsTest, SubtractTensorScalarFloat) {
    float scalar = 1.0f;
    Tensor<float> result = tensorFloat - scalar;
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 0.5f, 0.001);
    ASSERT_NEAR(res01, 1.5f, 0.001);
    ASSERT_NEAR(res10, 2.5f, 0.001);
    ASSERT_NEAR(res11, 3.5f, 0.001);
}

TEST_F(TensorOperationsTest, SubtractTensorsComplex) {
    Tensor<std::complex<float>> result = tensorComplex1 - tensorComplex2;
    std::complex<float> res00 = result[{0, 0}];
    std::complex<float> res01 = result[{0, 1}];
    std::complex<float> res10 = result[{1, 0}];
    std::complex<float> res11 = result[{1, 1}];
    ASSERT_FLOAT_EQ(res00.real(), -1.0f);
    ASSERT_FLOAT_EQ(res00.imag(), 1.0f);
    ASSERT_FLOAT_EQ(res01.real(), 2.0f);
    ASSERT_FLOAT_EQ(res01.imag(), 3.0f);
    ASSERT_FLOAT_EQ(res10.real(), 4.0f);
    ASSERT_FLOAT_EQ(res10.imag(), 5.0f);
    ASSERT_FLOAT_EQ(res11.real(), 5.0f);
    ASSERT_FLOAT_EQ(res11.imag(), 6.0f);
}

TEST_F(TensorOperationsTest, SubtractTensorsBroadcast) {
    Tensor<int> result = tensorBroadcast1 - tensorBroadcast2;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res02 = result[{0, 2}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    int res12 = result[{1, 2}];
    int res20 = result[{2, 0}];
    int res21 = result[{2, 1}];
    int res22 = result[{2, 2}];
    ASSERT_EQ(res00, 1);
    ASSERT_EQ(res01, 2);
    ASSERT_EQ(res02, 3);
    ASSERT_EQ(res10, 0);
    ASSERT_EQ(res11, 1);
    ASSERT_EQ(res12, 2);
    ASSERT_EQ(res20, -1);
    ASSERT_EQ(res21, 0);
    ASSERT_EQ(res22, 1);
}

// Multiplication tests
TEST_F(TensorOperationsTest, MultiplyTensorsInteger) {
    Tensor<int> result = tensorInt1 * tensorInt2;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    ASSERT_EQ(res00, 50);
    ASSERT_EQ(res01, 120);
    ASSERT_EQ(res10, 210);
    ASSERT_EQ(res11, 320);
}

TEST_F(TensorOperationsTest, MultiplyTensorScalarFloat) {
    float scalar = 2.0f;
    Tensor<float> result = tensorFloat * scalar;
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 3.0f, 0.001);
    ASSERT_NEAR(res01, 5.0f, 0.001);
    ASSERT_NEAR(res10, 7.0f, 0.001);
    ASSERT_NEAR(res11, 9.0f, 0.001);
}

TEST_F(TensorOperationsTest, MultiplyTensorsComplex) {
    Tensor<std::complex<float>> result = tensorComplex1 * tensorComplex2;
    std::complex<float> res00 = result[{0, 0}];
    std::complex<float> res01 = result[{0, 1}];
    std::complex<float> res10 = result[{1, 0}];
    std::complex<float> res11 = result[{1, 1}];
    ASSERT_FLOAT_EQ(res00.real(), 0.0f);
    ASSERT_FLOAT_EQ(res00.imag(), 5.0f);
    ASSERT_FLOAT_EQ(res01.real(), -1.0f);
    ASSERT_FLOAT_EQ(res01.imag(), 7.0f);
    ASSERT_FLOAT_EQ(res10.real(), -1.0f);
    ASSERT_FLOAT_EQ(res10.imag(), 11.0f);
    ASSERT_FLOAT_EQ(res11.real(), -2.0f);
    ASSERT_FLOAT_EQ(res11.imag(), 30.0f);
}

TEST_F(TensorOperationsTest, MultiplyTensorsBroadcast) {
    Tensor<int> result = tensorBroadcast1 * tensorBroadcast2;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res02 = result[{0, 2}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    int res12 = result[{1, 2}];
    int res20 = result[{2, 0}];
    int res21 = result[{2, 1}];
    int res22 = result[{2, 2}];
    ASSERT_EQ(res00, 2);
    ASSERT_EQ(res01, 3);
    ASSERT_EQ(res02, 4);
    ASSERT_EQ(res10, 4);
    ASSERT_EQ(res11, 6);
    ASSERT_EQ(res12, 8);
    ASSERT_EQ(res20, 6);
    ASSERT_EQ(res21, 9);
    ASSERT_EQ(res22, 12);
}

// Division tests
TEST_F(TensorOperationsTest, DivideTensorsInteger) {
    Tensor<int> result = tensorInt1 / tensorInt2;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    ASSERT_EQ(res00, 2);
    ASSERT_EQ(res01, 3);
    ASSERT_EQ(res10, 4);
    ASSERT_EQ(res11, 5);
}

TEST_F(TensorOperationsTest, DivideTensorScalarFloat) {
    float scalar = 2.0f;
    Tensor<float> result = tensorFloat / scalar;
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 0.75f, 0.001);
    ASSERT_NEAR(res01, 1.25f, 0.001);
    ASSERT_NEAR(res10, 1.75f, 0.001);
    ASSERT_NEAR(res11, 2.25f, 0.001);
}

TEST_F(TensorOperationsTest, DivideTensorsComplex) {
    Tensor<std::complex<float>> result = tensorComplex1 / tensorComplex2;
    std::complex<float> res00 = result[{0, 0}];
    std::complex<float> res01 = result[{0, 1}];
    std::complex<float> res10 = result[{1, 0}];
    std::complex<float> res11 = result[{1, 1}];
    ASSERT_FLOAT_EQ(res00.real(), 0.8f);
    ASSERT_FLOAT_EQ(res00.imag(), 0.6f);
    ASSERT_FLOAT_EQ(res01.real(), 3.5f);
    ASSERT_FLOAT_EQ(res01.imag(), 0.5f);
    ASSERT_FLOAT_EQ(res10.real(), 5.5f);
    ASSERT_FLOAT_EQ(res10.imag(), 0.5f);
    ASSERT_FLOAT_EQ(res11.real(), 3.75f);
    ASSERT_FLOAT_EQ(res11.imag(), 0.25f);
}

TEST_F(TensorOperationsTest, DivideTensorsBroadcast) {
    Tensor<int> result = tensorBroadcast1 / tensorBroadcast2;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res02 = result[{0, 2}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    int res12 = result[{1, 2}];
    int res20 = result[{2, 0}];
    int res21 = result[{2, 1}];
    int res22 = result[{2, 2}];
    ASSERT_EQ(res00, 2);
    ASSERT_EQ(res01, 3);
    ASSERT_EQ(res02, 4);
    ASSERT_EQ(res10, 1);
    ASSERT_EQ(res11, 1);
    ASSERT_EQ(res12, 2);
    ASSERT_EQ(res20, 0);
    ASSERT_EQ(res21, 1);
    ASSERT_EQ(res22, 1);
}

// Scalar-Tensor Addition for Integer Tensors
TEST_F(TensorOperationsTest, AddScalarTensorInteger) {
    int scalar = 5;
    Tensor<int> result = scalar + tensorInt1;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    ASSERT_EQ(res00, 15);
    ASSERT_EQ(res01, 25);
    ASSERT_EQ(res10, 35);
    ASSERT_EQ(res11, 45);
}

// Scalar-Tensor Subtraction for Integer Tensors
TEST_F(TensorOperationsTest, SubtractScalarTensorInteger) {
    int scalar = 50;
    Tensor<int> result = scalar - tensorInt1;
    int res00 = result[{0, 0}];
    int res01 = result[{0, 1}];
    int res10 = result[{1, 0}];
    int res11 = result[{1, 1}];
    ASSERT_EQ(res00, 40);
    ASSERT_EQ(res01, 30);
    ASSERT_EQ(res10, 20);
    ASSERT_EQ(res11, 10);
}

// Scalar-Tensor Multiplication for Float Tensors
TEST_F(TensorOperationsTest, MultiplyScalarTensorFloat) {
    float scalar = 2.0f;
    Tensor<float> result = scalar * tensorFloat;
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 3.0f, 0.001);
    ASSERT_NEAR(res01, 5.0f, 0.001);
    ASSERT_NEAR(res10, 7.0f, 0.001);
    ASSERT_NEAR(res11, 9.0f, 0.001);
}

// Scalar-Tensor Division for Float Tensors
TEST_F(TensorOperationsTest, DivideScalarTensorFloat) {
    float scalar = 10.0f;
    Tensor<float> result = scalar / tensorFloat;
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 10.0f / 1.5f, 0.001);
    ASSERT_NEAR(res01, 10.0f / 2.5f, 0.001);
    ASSERT_NEAR(res10, 10.0f / 3.5f, 0.001);
    ASSERT_NEAR(res11, 10.0f / 4.5f, 0.001);
}

// Scalar-Tensor Division with Complex Numbers
TEST_F(TensorOperationsTest, DivideScalarByComplex) {
    std::complex<float> scalar = std::complex<float>(10.0, 0.0);
    Tensor<std::complex<float>> result = scalar / tensorComplex1;
    std::complex<float> res00 = result[{0, 0}];
    std::complex<float> res01 = result[{0, 1}];
    std::complex<float> res10 = result[{1, 0}];
    std::complex<float> res11 = result[{1, 1}];
    ASSERT_NEAR(res00.real(), 2.0, 1e-5);
    ASSERT_NEAR(res00.imag(), -4.0, 1e-5);
    ASSERT_NEAR(res01.real(), 1.2, 1e-5);
    ASSERT_NEAR(res01.imag(), -1.6, 1e-5);
    ASSERT_NEAR(res10.real(), 0.81967213, 1e-5);
    ASSERT_NEAR(res10.imag(), -0.98360656, 1e-5);
    ASSERT_NEAR(res11.real(), 0.61946903, 1e-5);
    ASSERT_NEAR(res11.imag(), -0.7079646, 1e-5);
}


TEST_F(TensorOperationsTest, BinaryOperationsTensorShapeMismatch) {
    Tensor<int> tensorMismatched({2, 3});
    EXPECT_THROW(tensorInt1 - tensorMismatched, std::invalid_argument);
    EXPECT_THROW(tensorInt1 * tensorMismatched, std::invalid_argument);
    EXPECT_THROW(tensorInt1 / tensorMismatched, std::invalid_argument);
}

class TensorUnaryOperationsTest : public ::testing::Test {
protected:
    Tensor<float> tensorFloat;
    Tensor<std::complex<double>> tensorComplex;

    void SetUp() override {
        // Setup floating-point tensor
        tensorFloat = Tensor<float>({2, 2});
        tensorFloat[{0, 0}] = 1.0f; tensorFloat[{0, 1}] = 2.0f;
        tensorFloat[{1, 0}] = 3.0f; tensorFloat[{1, 1}] = 4.0f;

        // Setup complex tensor
        tensorComplex = Tensor<std::complex<double>>({2, 2});
        tensorComplex[{0, 0}] = std::complex<double>(1.0, 2.0);
        tensorComplex[{0, 1}] = std::complex<double>(3.0, 4.0);
        tensorComplex[{1, 0}] = std::complex<double>(5.0, 6.0);
        tensorComplex[{1, 1}] = std::complex<double>(7.0, 8.0);
    }
};

TEST_F(TensorUnaryOperationsTest, PowFloat) {
    float exponent = 2.0f;
    Tensor<float> result = tensorFloat.pow(exponent);
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 1.0f, 1e-6f);
    ASSERT_NEAR(res01, 4.0f, 1e-6f);
    ASSERT_NEAR(res10, 9.0f, 1e-6f);
    ASSERT_NEAR(res11, 16.0f, 1e-6f);
}

TEST_F(TensorUnaryOperationsTest, PowComplex) {
    double exponent = 2.0;
    Tensor<std::complex<double>> result = tensorComplex.pow(exponent);
    std::complex<double> res00 = result[{0, 0}];
    std::complex<double> res01 = result[{0, 1}];
    std::complex<double> res10 = result[{1, 0}];
    std::complex<double> res11 = result[{1, 1}];
    ASSERT_NEAR(res00.real(), -3.0, 1e-6);
    ASSERT_NEAR(res00.imag(), 4.0, 1e-6);
    ASSERT_NEAR(res01.real(), -7.0, 1e-6);
    ASSERT_NEAR(res01.imag(), 24.0, 1e-6);
    ASSERT_NEAR(res10.real(), -11.0, 1e-6);
    ASSERT_NEAR(res10.imag(), 60.0, 1e-6);
    ASSERT_NEAR(res11.real(), -15.0, 1e-6);
    ASSERT_NEAR(res11.imag(), 112.0, 1e-6);
}


TEST_F(TensorUnaryOperationsTest, SqrtFloat) {
    Tensor<float> result = tensorFloat.sqrt();
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 1.0f, 1e-6f);
    ASSERT_NEAR(res01, 1.41421356f, 1e-6f);
    ASSERT_NEAR(res10, 1.73205081f, 1e-6f);
    ASSERT_NEAR(res11, 2.0f, 1e-6f);
}

TEST_F(TensorUnaryOperationsTest, SqrtComplex) {
    Tensor<std::complex<double>> result = tensorComplex.sqrt();
    std::complex<double> res00 = result[{0, 0}];
    std::complex<double> res01 = result[{0, 1}];
    std::complex<double> res10 = result[{1, 0}];
    std::complex<double> res11 = result[{1, 1}];
    ASSERT_NEAR(res00.real(), 1.27201965, 1e-6);
    ASSERT_NEAR(res00.imag(), 0.78615138, 1e-6);
    ASSERT_NEAR(res01.real(), 2.0, 1e-6);
    ASSERT_NEAR(res01.imag(), 1.0, 1e-6);
    ASSERT_NEAR(res10.real(), 2.53083481, 1e-6);
    ASSERT_NEAR(res10.imag(), 1.18537962, 1e-6);
    ASSERT_NEAR(res11.real(), 2.96901885, 1e-6);
    ASSERT_NEAR(res11.imag(), 1.34724642, 1e-6);
}

TEST_F(TensorUnaryOperationsTest, LogFloat) {
    Tensor<float> result = tensorFloat.log();
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 0.0f, 1e-6f);
    ASSERT_NEAR(res01, 0.69314718f, 1e-6f);
    ASSERT_NEAR(res10, 1.09861229f, 1e-6f);
    ASSERT_NEAR(res11, 1.38629436f, 1e-6f);
}

TEST_F(TensorUnaryOperationsTest, LogComplex) {
    Tensor<std::complex<double>> result = tensorComplex.log();
    std::complex<double> res00 = result[{0, 0}];
    std::complex<double> res01 = result[{0, 1}];
    std::complex<double> res10 = result[{1, 0}];
    std::complex<double> res11 = result[{1, 1}];
    ASSERT_NEAR(res00.real(), 0.80471896, 1e-6);
    ASSERT_NEAR(res00.imag(), 1.10714872, 1e-6);
    ASSERT_NEAR(res01.real(), 1.60943791, 1e-6);
    ASSERT_NEAR(res01.imag(), 0.92729522, 1e-6);
    ASSERT_NEAR(res10.real(), 2.05543693, 1e-6);
    ASSERT_NEAR(res10.imag(), 0.87605805, 1e-6);
    ASSERT_NEAR(res11.real(), 2.36369391, 1e-6);
    ASSERT_NEAR(res11.imag(), 0.85196633, 1e-6);
}

TEST_F(TensorUnaryOperationsTest, AbsFloat) {
    Tensor<float> result = tensorFloat.abs();
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 1.0f, 1e-6f);
    ASSERT_NEAR(res01, 2.0f, 1e-6f);
    ASSERT_NEAR(res10, 3.0f, 1e-6f);
    ASSERT_NEAR(res11, 4.0f, 1e-6f);
}

TEST_F(TensorUnaryOperationsTest, AbsComplex) {
    Tensor<std::complex<double>> result = tensorComplex.abs();
    double sqrt2 = std::sqrt(2.0);
    std::complex<double> res00 = result[{0, 0}];
    std::complex<double> res01 = result[{0, 1}];
    std::complex<double> res10 = result[{1, 0}];
    std::complex<double> res11 = result[{1, 1}];
    ASSERT_NEAR(res00.real(), 2.23606797749979, 1e-6);
    ASSERT_NEAR(res00.imag(), 0.0, 1e-6);
    ASSERT_NEAR(res01.real(), 5.0, 1e-6);
    ASSERT_NEAR(res01.imag(), 0.0, 1e-6);
    ASSERT_NEAR(res10.real(), 7.810249675906654, 1e-6);
    ASSERT_NEAR(res10.imag(), 0.0, 1e-6);
    ASSERT_NEAR(res11.real(), 10.63014581273465, 1e-6);
    ASSERT_NEAR(res11.imag(), 0.0, 1e-6);
}

TEST_F(TensorUnaryOperationsTest, ExpFloat) {
    Tensor<float> result = tensorFloat.exp();
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 2.71828183f, 1e-6f);
    ASSERT_NEAR(res01, 7.3890561f, 1e-6f);
    ASSERT_NEAR(res10, 20.08553692f, 1e-6f);
    ASSERT_NEAR(res11, 54.59815003f, 1e-6f);
}

TEST_F(TensorUnaryOperationsTest, ExpComplex) {
    Tensor<std::complex<double>> result = tensorComplex.exp();
    double res00_real = result[{0, 0}].real();
    double res00_imag = result[{0, 0}].imag();
    double res01_real = result[{0, 1}].real();
    double res01_imag = result[{0, 1}].imag();
    double res10_real = result[{1, 0}].real();
    double res10_imag = result[{1, 0}].imag();
    double res11_real = result[{1, 1}].real();
    double res11_imag = result[{1, 1}].imag();
    ASSERT_NEAR(res00_real, -1.13120438, 1e-6);
    ASSERT_NEAR(res00_imag, 2.47172667, 1e-6);
    ASSERT_NEAR(res01_real, -13.12878308, 1e-6);
    ASSERT_NEAR(res01_imag, -15.20078446, 1e-6);
    ASSERT_NEAR(res10_real, 142.50190552, 1e-6);
    ASSERT_NEAR(res10_imag, -41.46893679, 1e-6);
    ASSERT_NEAR(res11_real, -159.56016163, 1e-6);
    ASSERT_NEAR(res11_imag, 1084.96305881, 1e-6);
}

TEST_F(TensorUnaryOperationsTest, SinFloat) {
    Tensor<float> result = tensorFloat.sin();
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 0.84147098f, 1e-6f);
    ASSERT_NEAR(res01, 0.90929743f, 1e-6f);
    ASSERT_NEAR(res10, 0.14112001f, 1e-6f);
    ASSERT_NEAR(res11, -0.7568025f, 1e-6f);
}

TEST_F(TensorUnaryOperationsTest, SinComplex) {
    Tensor<std::complex<double>> result = tensorComplex.sin();
    double res00_real = result[{0, 0}].real();
    double res00_imag = result[{0, 0}].imag();
    double res01_real = result[{0, 1}].real();
    double res01_imag = result[{0, 1}].imag();
    double res10_real = result[{1, 0}].real();
    double res10_imag = result[{1, 0}].imag();
    double res11_real = result[{1, 1}].real();
    double res11_imag = result[{1, 1}].imag();
    ASSERT_NEAR(res00_real, 3.16577851, 1e-6);
    ASSERT_NEAR(res00_imag, 1.95960104, 1e-6);
    ASSERT_NEAR(res01_real, 3.85373804, 1e-6);
    ASSERT_NEAR(res01_imag, -27.01681326, 1e-6);
    ASSERT_NEAR(res10_real, -193.43002006, 1e-6);
    ASSERT_NEAR(res10_imag, 57.21839506, 1e-6);
    ASSERT_NEAR(res11_real, 979.22483461, 1e-6);
    ASSERT_NEAR(res11_imag, 1123.67534681, 1e-6);
}

TEST_F(TensorUnaryOperationsTest, CosFloat) {
    Tensor<float> result = tensorFloat.cos();
    float res00 = result[{0, 0}];
    float res01 = result[{0, 1}];
    float res10 = result[{1, 0}];
    float res11 = result[{1, 1}];
    ASSERT_NEAR(res00, 0.54030231f, 1e-6f);
    ASSERT_NEAR(res01, -0.41614684f, 1e-6f);
    ASSERT_NEAR(res10, -0.9899925f, 1e-6f);
    ASSERT_NEAR(res11, -0.65364362f, 1e-6f);
}

TEST_F(TensorUnaryOperationsTest, CosComplex) {
    Tensor<std::complex<double>> result = tensorComplex.cos();
    double res00_real = result[{0, 0}].real();
    double res00_imag = result[{0, 0}].imag();
    double res01_real = result[{0, 1}].real();
    double res01_imag = result[{0, 1}].imag();
    double res10_real = result[{1, 0}].real();
    double res10_imag = result[{1, 0}].imag();
    double res11_real = result[{1, 1}].real();
    double res11_imag = result[{1, 1}].imag();
    ASSERT_NEAR(res00_real, 2.03272301, 1e-6);
    ASSERT_NEAR(res00_imag, -3.0518978, 1e-6);
    ASSERT_NEAR(res01_real, -27.0349456, 1e-6);
    ASSERT_NEAR(res01_imag, -3.85115333, 1e-6);
    ASSERT_NEAR(res10_real, 57.21909818, 1e-6);
    ASSERT_NEAR(res10_imag, 193.42764312, 1e-6);
    ASSERT_NEAR(res11_real, 1123.67559972, 1e-6);
    ASSERT_NEAR(res11_imag, -979.22461422, 1e-6);
}

class TensorManipulationTests : public ::testing::Test {
protected:
    Tensor<float> tensor3D;
    Tensor<float> tensor2D;
    Tensor<float> tensor1D;

    void SetUp() override {
        tensor3D = Tensor<float>({2, 3, 4});
        tensor2D = Tensor<float>({2, 3});
        tensor1D = Tensor<float>({2});
    }
};

TEST_F(TensorManipulationTests, PermuteValid) {
    std::vector<int> order = {2, 0, 1};
    Tensor<float> permuted = tensor3D.permute(order);
    EXPECT_EQ(permuted.getShape()[0], 4);
    EXPECT_EQ(permuted.getShape()[1], 2);
    EXPECT_EQ(permuted.getShape()[2], 3);
}

TEST_F(TensorManipulationTests, PermuteInvalidOrder) {
    std::vector<int> order = {2, 0, 3};  // Invalid index 3
    EXPECT_THROW(tensor3D.permute(order), std::invalid_argument);
}

TEST_F(TensorManipulationTests, PermuteOrderSizeMismatch) {
    std::vector<int> order = {1, 0};  // Mismatch in order size
    EXPECT_THROW(tensor3D.permute(order), std::invalid_argument);
}

TEST_F(TensorManipulationTests, TransposeValid) {
    Tensor<float> transposed = tensor2D.transpose();
    EXPECT_EQ(transposed.getShape()[0], 3);
    EXPECT_EQ(transposed.getShape()[1], 2);
}

TEST_F(TensorManipulationTests, TransposeInvalid) {
    EXPECT_THROW(tensor1D.transpose(), std::runtime_error);
}

TEST_F(TensorManipulationTests, ReshapeValid) {
    Tensor<float> reshaped = tensor3D.reshape({6, 4});
    EXPECT_EQ(reshaped.getShape()[0], 6);
    EXPECT_EQ(reshaped.getShape()[1], 4);
}

TEST_F(TensorManipulationTests, ReshapeInvalid) {
    EXPECT_THROW(tensor3D.reshape({10, 10}), std::invalid_argument);
}

TEST_F(TensorManipulationTests, SqueezeValid) {
    tensor3D = Tensor<float>({2, 1, 3});
    Tensor<float> squeezed = tensor3D.squeeze(1);
    EXPECT_EQ(squeezed.getShape().size(), 2);
}

TEST_F(TensorManipulationTests, UnsqueezeValid) {
    Tensor<float> unsqueezed = tensor2D.unsqueeze(0);
    EXPECT_EQ(unsqueezed.getShape().size(), 3);
    EXPECT_EQ(unsqueezed.getShape()[0], 1);
}

TEST_F(TensorManipulationTests, SqueezeInvalidAxis) {
    EXPECT_THROW(tensor3D.squeeze(3), std::invalid_argument);
}

TEST_F(TensorManipulationTests, UnsqueezeInvalidAxis) {
    EXPECT_THROW(tensor2D.unsqueeze(3), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}