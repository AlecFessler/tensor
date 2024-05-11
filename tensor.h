// Implementation motivated in part by this blog post http://blog.ezyang.com/2019/05/pytorch-internals/ detailing the internals of PyTorch

#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

#include <cnpy.h>

#include "tensorStorage.h"

template<typename T>
struct is_complex : std::false_type {};

template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};

enum d_type {
    I32, // 32-bit integer
    I64, // 64-bit integer
    FP32, // 32-bit floating point
    FP64, // 64-bit floating point
    C64, // 64-bit complex (two 32-bit floats)
    C128 // 128-bit complex (two 64-bit floats)
};

struct Slice {
    int start = 0;
    int stop = std::numeric_limits<int>::max();  // Use the standard limit
    int step = 1;

    // Default constructor for a slice from start to end of the data structure, effectively makes a copy
    Slice() = default;

    // Constructor for specifying all three parameters
    Slice(int start, int stop, int step) : start(start), stop(stop), step(step) {}

    // Constructor for specifying start and stop, with step defaulting to 1
    Slice(int start, int stop) : start(start), stop(stop) {}

    Slice(int start) : start(start) {}

    // Static factory methods for common slice operations
    static Slice toEnd(int start = 0, int step = 1) { return Slice(start, std::numeric_limits<int>::max(), step); }

    static Slice fromStart(int stop, int step = 1) { return Slice(0, stop, step); }

    static Slice single(int index) {
        if (index < 0) {
            throw std::invalid_argument("Single element slice index must be non-negative.");
        }
        return Slice(index, index + 1, 1); 
    }
};

enum PaddingType {
    CONSTANT,
    REFLECTIVE
};

/**
 * A tensor is a generalization of vectors and matrices to potentially higher dimensions. Tensors can be
 * operated on efficiently using view based operations in many cases, which can provide a variety of views
 * of the underlying data without copying or reordering the data itself. Rather a simple metadata 
 * update is sufficient to provide a new view of the data.
 * 
 * Some methods require contiguity to function correctly. These methods have a note in their docstring to indicate
 * this requirement. Enforcing contiguity can be an expensive operation, so it is not done implicitly in this class.
 * Consumers of this class are responsible for calling .contiguous() when necessary. Contiguity can be checked using
 * the .isContiguous() method.
 * 
 * Many of the methods in this class have an overload that accepts an initializer list, or a vector of indices.
 * The initializer list is generally more convenient and nicer syntax when used external to the class, but requires
 * knowing the number of dimensions at compile time. The vector of indices is more flexible and can be used when the
 * number of dimensions is not known at compile time. The intializer list is converted to a vector of indices
 * internally to call the vector based method.
*/
template <typename T>
class Tensor {
    TensorStorage<T> data;
    std::vector<int> shape;
    std::vector<int> strides;
    d_type dtype;
    size_t offset = 0;
    bool contiguousFlag = true;

    std::vector<int> computeStrides(const std::vector<int>& shape, int& size) const {
        /**
         * Compute the strides of a tensor given its shape
         * 
         * @attention: Size is only passed as a parameter to avoid recomputation in constructors, not relevant elsewhere
         * 
         * @param shape: Shape of the tensor
         * @param size: Size of the tensor
         * @return: Strides of the tensor
        */
        size = 1; // Initialize size to 1
        std::vector<int> strides(shape.size()); // Initialize strides vector to the same size as the shape vector
        for (int i = shape.size() - 1; i >= 0; --i) { // Iterate over the shape vector in reverse order
            strides[i] = size; // Set the stride at index i to the current size
            size *= shape[i]; // Update the size as the product of the current size and the value at index i
        }
        return strides;
    }

    std::vector<int> indexFromOffset(int offset) const {
        /**
         * Compute the indices of a tensor given an offset
         * 
         * @param offset: Offset of the tensor
         * @return: Indices of the tensor
        */
        std::vector<int> idx(shape.size(), 0); // Initialize indices for each dimension.
        for (int i = 0; i < shape.size(); ++i) {
            if (strides[i] != 0) {  // Update index only if the stride is non-zero.
                idx[i] = (offset / strides[i]) % shape[i];  // Calculate the index within the current dimension.
                offset %= strides[i];  // Reduce offset for the next dimension.
            }
        }
        return idx;
    }

    void printRecursive(int dim, int offset, int indent = 0) const {
        /**
         * Recursively print the tensor in a nested, human-readable format
         * 
         * @param dim: Current dimension
         * @param offset: Current offset
         * @param indent: Indentation level
        */
        std::string space(indent, ' ');

        if (dim == shape.size() - 1) {
            std::cout << space << "[";
            for (int i = 0; i < shape[dim]; ++i) {
                std::cout << data[offset + i * strides[dim]];
                if (i < shape[dim] - 1) std::cout << " ";
            }
            std::cout << "]";
        } else {
            std::cout << space << "[\n";
            for (int i = 0; i < shape[dim]; ++i) {
                printRecursive(dim + 1, offset + i * strides[dim], indent + 1);
                if (i < shape[dim] - 1) std::cout << "\n";
            }
            std::cout << "\n" << space << "]";
        }
    }

    std::vector<int> resolveBroadcast(const std::vector<int> shape1, const std::vector<int> shape2) const {
        /**
         * Resolve the broadcast of two shapes
         * 
         * Shapes are broadcastable if for each dimension: 
         *  1. the dimensions are equal
         *  2. one of them is 1
         *  3. one of them is missing (implicit 1)
         * 
         * @param shape1: Shape of the first tensor
         * @param shape2: Shape of the second tensor
         * @return: Resulting shape after broadcast
         * 
         * @throws: std::invalid_argument if the shapes are not broadcastable
        */
        size_t maxDims = std::max(shape1.size(), shape2.size());
        std::vector<int> result(maxDims);

        for (size_t i = 0; i < maxDims; i++) {
            int dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
            int dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::invalid_argument("Shapes are not broadcastable.");
            }
            result[maxDims - 1 - i] = std::max(dim1, dim2);
        }
        return result;
    }
    
    Tensor<T> setStridesForBroadcast(std::vector<int> targetShape) const {
        Tensor<T> result(targetShape);
        std::vector<int> newStrides(targetShape.size(), 0);  // Initialize strides to zero.
        result.data = data;

        int originalIndex = shape.size() - 1;
        for (int i = targetShape.size() - 1; i >= 0; --i) {
            if (originalIndex >= 0) {
                if (shape[originalIndex] == targetShape[i]) {
                    // Non-broadcasted dimension: set the stride normally.
                    newStrides[i] = std::accumulate(shape.begin() + originalIndex + 1, shape.end(), 1, std::multiplies<int>());
                } else if (shape[originalIndex] == 1) {
                    // Broadcasted dimension: keep stride as zero.
                    newStrides[i] = 0;
                }
                --originalIndex;
            } else {
                // Missing dimension in the original tensor: set stride to zero.
                newStrides[i] = 0;
            }
        }

        result.strides = newStrides;
        return result;
    }

    static d_type T_to_dtype() {
        if (std::is_same<T, int>::value) {
            return I32;
        } else if (std::is_same<T, long>::value) {
            return I64;
        } else if (std::is_same<T, float>::value) {
            return FP32;
        } else if (std::is_same<T, double>::value) {
            return FP64;
        } else if (std::is_same<T, std::complex<float>>::value) {
            return C64;
        } else if (std::is_same<T, std::complex<double>>::value) {
            return C128;
        } else {
            throw std::invalid_argument("Unsupported data type.");
        }
    }

public:

    Tensor() : contiguousFlag(true) {
        shape = {0};
        strides = {0};
        dtype = Tensor<T>::T_to_dtype();
    }

    Tensor(std::initializer_list<int> shape) {
        /**
         * Constructor for the Tensor class
         * 
         * @param shape: Shape of the tensor
         * @return: Tensor object
        */
        this->shape.assign(shape.begin(), shape.end());
        int size = 1;
        strides = computeStrides(this->shape, size); // pass size as reference to avoid recomputation
        data.resize(size); // Resize the data vector to the final size
        this->dtype = Tensor<T>::T_to_dtype();
    }

    Tensor(const std::vector<int>& shape) {
        /**
         * Constructor for the Tensor class
         * 
         * @param shape: Shape of the tensor
         * @return: Tensor object
        */
        this->shape = shape;
        int size = 1;
        strides = computeStrides(this->shape, size); // pass size as reference to avoid recomputation
        data.resize(size); // Resize the data vector to the final size
        this->dtype = Tensor<T>::T_to_dtype();
    }
    
    template <typename U>
    Tensor<U> astype() const {
        /**
         * Convert the tensor to a new data type
         * 
         * This method handles conversions between real and complex types:
         * - From real to complex: Real part is set as the real number, imaginary part is set to zero.
         * - From complex to real: The output is the magnitude of the complex number.
         * 
         * @return: Tensor object with the new data type
         */
        Tensor<U> result(shape);

        auto src_it = begin();
        auto dst_it = result.begin();

        if constexpr (is_complex<U>::value && !is_complex<T>::value) {
            // Converting from real to complex
            while (src_it != end()) {
                *dst_it = U(*src_it, 0);  // Convert real to complex with zero imaginary part
                ++src_it;
                ++dst_it;
            }
        } else if constexpr (!is_complex<U>::value && is_complex<T>::value) {
            // Converting from complex to real (using the magnitude of the complex number)
            while (src_it != end()) {
                *dst_it = static_cast<U>(std::abs(*src_it)); // Use std::abs to get the magnitude
                ++src_it;
                ++dst_it;
            }
        } else {
            // Handling other generic conversions (including real to real and complex to complex)
            while (src_it != end()) {
                *dst_it = static_cast<U>(*src_it);
                ++src_it;
                ++dst_it;
            }
        }

        return result;
    }

    Tensor(const Tensor<T>& other) {
        /**
         * Copy constructor for the Tensor class
         * 
         * @param other: Tensor object to copy
         * @return: Tensor object
        */
        shape = other.shape;
        strides = other.strides;
        dtype = other.dtype;
        data = other.data;
        offset = other.offset;
        contiguousFlag = other.contiguousFlag;
    }

    Tensor<T>& operator=(const Tensor<T>& other) {
        /**
         * Copy assignment operator for the Tensor class
         * 
         * @param other: Tensor object to copy
         * @return: Reference to the new Tensor object
        */
        if (this != &other) {
            shape = other.shape;
            strides = other.strides;
            dtype = other.dtype;
            data = other.data;
            offset = other.offset;
            contiguousFlag = other.contiguousFlag;
        }
        return *this;
    }

    Tensor(Tensor<T>&& other) noexcept {
        /**
         * Move constructor for the Tensor class
         * 
         * @param other: Tensor object to move
         * @return: Tensor object
        */
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        dtype = other.dtype;
        data = std::move(other.data);
        offset = other.offset;
        contiguousFlag = other.contiguousFlag;
    }

    Tensor<T>& operator=(Tensor<T>&& other) noexcept {
        /**
         * Move assignment operator for the Tensor class
         * 
         * @param other: Tensor object to move
         * @return: Reference to the new Tensor object
        */
        if (this != &other) {
            shape = std::move(other.shape);
            strides = std::move(other.strides);
            dtype = other.dtype;
            data = std::move(other.data);
            offset = other.offset;
            contiguousFlag = other.contiguousFlag;
        }
        return *this;
    }
    
    class TensorIterator {
    public:
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;
        using iterator_category = std::input_iterator_tag;

        TensorIterator(Tensor<T>& tensor, size_t offset = 0)
            : tensor_(tensor), offset_(offset), index_(tensor_.shape.size(), 0) {}

        TensorIterator& operator=(const TensorIterator& other) {
            if (this != &other) {
                tensor_ = other.tensor_;
                offset_ = other.offset_;
                index_ = other.index_;
            }
            return *this;
        }

        reference operator*() const {
            return tensor_.data[offset_];
        }

        TensorIterator& operator++() {
            int n = tensor_.shape.size();
            for (int i = n - 1; i >= 0; --i) {
                index_[i]++;
                if (index_[i] < tensor_.shape[i]) {
                    break;
                }
                if (i > 0) {
                    index_[i] = 0;
                }
            }

            offset_ = 0;
            for (int i = 0; i < n; ++i) {
                offset_ += index_[i] * tensor_.strides[i];
            }

            return *this;
        }

        TensorIterator operator++(int) {
            TensorIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const TensorIterator& other) const {
            return offset_ == other.offset_;
        }

        bool operator!=(const TensorIterator& other) const {
            return offset_ != other.offset_;
        }

    private:
        Tensor<T>& tensor_;
        size_t offset_;
        std::vector<int> index_;
    };

    TensorIterator begin() { return TensorIterator(*this); }

    TensorIterator end() { return TensorIterator(*this, this->size()); }

    class ConstTensorIterator {
    public:
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;
        using iterator_category = std::input_iterator_tag;

        ConstTensorIterator(const Tensor<T>& tensor, size_t offset = 0)
            : tensor_(tensor), offset_(offset), index_(tensor_.shape.size(), 0) {}

        ConstTensorIterator& operator=(const ConstTensorIterator& other) {
            if (this != &other) {
                tensor_ = other.tensor_;
                offset_ = other.offset_;
                index_ = other.index_;
            }
            return *this;
        }

        reference operator*() const {
            return tensor_.data[offset_];
        }

        ConstTensorIterator& operator++() {
            int n = tensor_.shape.size();
            for (int i = n - 1; i >= 0; --i) {
                index_[i]++;
                if (index_[i] < tensor_.shape[i]) {
                    break;
                }
                if (i > 0) {
                    index_[i] = 0;
                }
            }

            offset_ = 0;
            for (int i = 0; i < n; ++i) {
                offset_ += index_[i] * tensor_.strides[i];
            }

            return *this;
        }

        ConstTensorIterator operator++(int) {
            ConstTensorIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const ConstTensorIterator& other) const {
            return offset_ == other.offset_;
        }

        bool operator!=(const ConstTensorIterator& other) const {
            return offset_ != other.offset_;
        }

    private:
        const Tensor<T>& tensor_;
        size_t offset_;
        std::vector<int> index_;
    };
    
    ConstTensorIterator begin() const { return ConstTensorIterator(*this); }

    ConstTensorIterator end() const { return ConstTensorIterator(*this, this->size()); }

    T& operator[](std::initializer_list<int> indices) {
        /**
         * Alternative to method to call the operator [] with an initializer list
         * 
         * @param indices: List of indices for each dimension
         * @return: Reference to the element at the specified indices
         * @throws: std::invalid_argument if the number of indices does not match the number of dimensions
        */
        return (*this)[std::vector<int>(indices.begin(), indices.end())];
    }

    T& operator[](const std::vector<int>& indices) {
        /**
         * Overloaded operator [] for indexing the tensor
         * 
         * @param indices: List of indices for each dimension
         * @return: Reference to the element at the specified indices
         * @throws: std::invalid_argument if the number of indices does not match the number of dimensions
        */
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices must match the number of dimensions");
        }
        int offset = 0;
        for (int i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides[i]; // Compute the offset as the sum of the product of the index and stride at each dimension
        }
        return data[offset];
    }
    
    const T& operator[](std::initializer_list<int> indices) const {
        /**
         * Alternative to method to call the const operator [] with an initializer list
         * 
         * @param indices: List of indices for each dimension
         * @return: Reference to the element at the specified indices
         * 
         * @throws: std::invalid_argument if the number of indices does not match the number of dimensions
        */
        return (*this)[std::vector<int>(indices.begin(), indices.end())];
    }

    const T& operator[](const std::vector<int>& indices) const {
        /**
         * Overloaded operator [] for indexing the tensor (const version)
         * 
         * @param indices: List of indices for each dimension
         * @return: Reference to the element at the specified indices
         * 
         * @throws: std::invalid_argument if the number of indices does not match the number of dimensions
        */
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices must match the number of dimensions");
        }
        int offset = 0;
        for (int i = 0; i < indices.size(); ++i) {
            offset += indices[i] * strides[i]; // Compute the offset as the sum of the product of the index and stride at each dimension
        }
        return data[offset];
    }

    Tensor<T> operator[](Slice slice) {
        /**
         * Overloaded operator [] for slicing the tensor along the first dimension
         * 
         * @param slice: Slice object specifying the start, stop, and step
         * @return: Sliced tensor
         * 
         * @throws: std::runtime_error if the tensor is not contiguous
        */
        if (!contiguousFlag) {
            throw std::runtime_error("Slicing requires the tensor to be contiguous.");
        }

        int effective_start = slice.start < 0 ? slice.start + shape[0] : slice.start;
        int effective_stop = (slice.stop == std::numeric_limits<int>::max()) ? shape[0] : (slice.stop < 0 ? slice.stop + shape[0] : slice.stop);

        effective_start = std::max(0, std::min(shape[0], effective_start));
        effective_stop = std::max(0, std::min(shape[0], effective_stop));

        if (slice.step == 0) {
            throw std::invalid_argument("Slice step cannot be zero.");
        }
        if ((slice.step > 0 && effective_start > effective_stop) || (slice.step < 0 && effective_start < effective_stop)) {
            return Tensor<T>({0}); // Return an empty tensor
        }

        int num_elements = (effective_stop - effective_start + slice.step - 1) / slice.step;
        std::vector<int> new_shape = shape;
        new_shape[0] = num_elements;

        Tensor<T> result(new_shape);

        int block_size = 1;
        for (int i = 1; i < shape.size(); ++i) {
            block_size *= shape[i];
        }

        int index = 0;
        for (int i = effective_start; slice.step > 0 ? i < effective_stop : i > effective_stop; i += slice.step) {
            std::copy_n(data.begin() + i * block_size, block_size, result.data.begin() + index * block_size);
            index++;
        }

        return result;
    }
    
    Tensor<T> concat(const Tensor<T>& other, int axis) {
        /**
         * Concatenate two tensors along a specified axis
         * 
         * @attention: The tensors are assumed to be contiguous
         * 
         * @param other: Tensor to concatenate with
         * @param axis: Axis along which to concatenate, default is 0
         * @return: Concatenated tensor
         * 
         * @throws: std::runtime_error if the tensors are not contiguous
         * @throws: std::invalid_argument if the dimensions do not match
         * @throws: std::invalid_argument if the shapes do not match along all dimensions except the concatenation axis
         * @throws: std::invalid_argument if the axis is out of bounds
        */
        if (!isContiguous() || !other.isContiguous()) {
            throw std::runtime_error("Concatenation requires both tensors to be contiguous.");
        }
        if (shape.size() != other.shape.size()) {
            throw std::invalid_argument("Tensor dimensions must match.");
        }
        for (int i = 0; i < shape.size(); ++i) {
            if (i != axis && shape[i] != other.shape[i]) {
                throw std::invalid_argument("Shapes must match in all dimensions except the concatenation axis.");
            }
        }
        if (axis < 0 || axis >= shape.size()) {
            throw std::invalid_argument("Axis out of bounds.");
        }

        // Compute the new shape
        std::vector<int> new_shape = shape;
        new_shape[axis] += other.shape[axis];
        Tensor<T> result(new_shape);

        // Calculate the stride needed to jump to the next block in the axis of concatenation
        int block_size = 1;
        for (int i = axis + 1; i < shape.size(); ++i) {
            block_size *= shape[i];
        }

        int total_blocks = 1;
        for (int i = 0; i < axis; ++i) {
            total_blocks *= shape[i];
        }

        // Copy the elements from this tensor
        for (int block = 0; block < total_blocks; ++block) {
            std::copy_n(data.begin() + block * block_size * shape[axis], block_size * shape[axis],
                        result.data.begin() + block * block_size * new_shape[axis]);
        }

        // Copy the elements from the other tensor
        for (int block = 0; block < total_blocks; ++block) {
            std::copy_n(other.data.begin() + block * block_size * other.shape[axis], block_size * other.shape[axis],
                        result.data.begin() + block * block_size * new_shape[axis] + block_size * shape[axis]);
        }

        return result;
    }

    Tensor<T> pad(const std::pair<int, int> padding, int axis, PaddingType padType = CONSTANT, int value = T{}) const {
        /**
         * Pad a tensor along a specified axis
         * 
         * Currently supports constant and reflective padding
         * 
         * @attention: The tensor is assumed to be contiguous
         * 
         * @param padding: Pair of padding values for the start and end of the axis
         * @param axis: Axis along which to pad
         * @param padType: Type of padding, default is CONSTANT
         * @param value: Value to use for padding if padType is CONSTANT
         * @return: Padded tensor
         * 
         * @throws: std::runtime_error if the tensor is not contiguous
         * @throws: std::invalid_argument if the axis is out of bounds
        */
        if (!isContiguous()) {
            throw std::runtime_error("Padding requires the tensor to be contiguous.");
        }
        if (axis < 0 || axis >= shape.size()) {
            throw std::invalid_argument("Axis out of bounds.");
        }

        std::vector<int> new_shape = shape;
        new_shape[axis] += padding.first + padding.second;
        Tensor<T> result(new_shape);

        int block_size = 1;
        for (int i = axis + 1; i < shape.size(); ++i) {
            block_size *= shape[i];
        }

        int total_blocks = 1;
        for (int i = 0; i < axis; ++i) {
            total_blocks *= shape[i];
        }

        // Copy the original data
        for (int block = 0; block < total_blocks; ++block) {
            std::copy_n(data.begin() + block * block_size * shape[axis], block_size * shape[axis],
                        result.data.begin() + block * block_size * new_shape[axis] + padding.first * block_size);
        }

        if (padType == CONSTANT) {
            // Handle constant padding
            for (int block = 0; block < total_blocks; ++block) {
                std::fill_n(result.data.begin() + block * block_size * new_shape[axis], padding.first * block_size, value);
                std::fill_n(result.data.begin() + block * block_size * new_shape[axis] + (padding.first + shape[axis]) * block_size, padding.second * block_size, value);
            }
        } else if (padType == REFLECTIVE) {
            for (int block = 0; block < total_blocks; ++block) {
                // Left padding
                for (int i = 0; i < padding.first; ++i) {
                    int original_idx = padding.first - i;
                    if (original_idx >= shape[axis]) {
                        original_idx = 2 * shape[axis] - 2 - original_idx;
                    }
                    std::copy_n(data.begin() + block * block_size * shape[axis] + original_idx * block_size, block_size,
                                result.data.begin() + block * block_size * new_shape[axis] + i * block_size);
                }

                // Right padding
                for (int i = 0; i < padding.second; ++i) {
                    int original_idx = shape[axis] - 2 - i;
                    if (original_idx < 0) {
                        original_idx = -original_idx;
                    }
                    std::copy_n(data.begin() + block * block_size * shape[axis] + original_idx * block_size, block_size,
                                result.data.begin() + block * block_size * new_shape[axis] + (padding.first + shape[axis] + i) * block_size);
                }
            }
        }

        return result;
    }
        
    Tensor<T> frame(int frameSize, int hopLength, int axis) const {
        /**
         * Frame a tensor along a specified axis
         * 
         * @attention: The tensor is assumed to be contiguous
         * @attention: If the frame size is not equal to the hop length, the tensor will not be contiguous after framing
         * 
         * @param frameSize: Size of each frame
         * @param hopLength: Number of samples of overlap between frames
         * @param axis: Axis along which to frame
         * @return: Framed tensor
         * 
         * @throws: std::runtime_error if the tensor is not contiguous
         * @throws: std::invalid_argument if the axis is out of bounds
         * @throws: std::invalid_argument if the frame size or hop length is invalid
        */
        if (!contiguousFlag) {
            throw std::runtime_error("Framing requires the tensor to be contiguous.");
        }
        if (axis < 0 || axis >= shape.size()) {
            throw std::invalid_argument("Axis out of bounds.");
        }
        if (frameSize <= 0 || hopLength <= 0) {
            throw std::invalid_argument("Frame size and hop length must be positive.");
        }

        // Calculate the number of frames
        int num_frames = (shape[axis] - frameSize) / hopLength + 1;
        if (num_frames <= 0) {
            throw std::invalid_argument("Invalid frame size or hop length for given dimension.");
        }

        std::vector<int> new_shape = shape;
        new_shape[axis] = num_frames; // Set the frame size for the specified axis
        new_shape.push_back(frameSize); // Add a new dimension at the end for the frames

        Tensor<T> result(new_shape);
        result.data = data; // Share the same data

        // Adjust strides
        result.strides = strides; // Copy the existing strides
        int previous_stride = strides[axis];
        result.strides[axis] = hopLength * previous_stride; // Update the stride for the frame axis
        result.strides.push_back(previous_stride); // Add a new stride for the frame axis

        if (!(frameSize == hopLength)) {
            result.contiguousFlag = false; // If frameSize is not equal to hopLength, set isContiguous to false
        }

        return result;
    }
    
    T dot(const Tensor<T>& other) const {
        /**
         * Dot product of two tensors
         * 
         * @attention: The tensors are assumed to be contiguous
         * 
         * @param other: Tensor to compute the dot product with
         * @return: Dot product of the two tensors
         * 
         * @throws: std::runtime_error if the tensors are not contiguous
         * @throws: std::invalid_argument if the shapes do not match
        */
        if (!contiguousFlag || !other.contiguousFlag) {
            throw std::runtime_error("Dot product requires both tensors to be contiguous.");
        }
        if (shape != other.shape) {
            throw std::invalid_argument("Tensor shapes must match.");
        }

        T result = 0;
        for (int i = 0; i < data.getSize(); ++i) {
            result += data[i] * other.data[i];
        }

        return result;
    }

    Tensor<T> matmul(const Tensor<T>& other) const {
        /**
         * Matrix multiplication of two tensors
         * 
         * Handles standard or batched matrix multiplication
         * 
         * @param other: Tensor to multiply with
         * @return: Result of the matrix multiplication
         * 
         * @throws: std::invalid_argument if the dimensions do not match for matrix multiplication
        */
        if (shape.size() < 2 || other.shape.size() < 2 || shape.back() != other.shape[other.shape.size() - 2]) {
            throw std::invalid_argument("Dimension mismatch for matrix multiplication.");
        }

        std::vector<int> result_shape(shape.begin(), shape.end() - 1);
        result_shape.push_back(other.shape.back());

        Tensor<T> result(result_shape);

        // Aggregate all but final 2 dims into a batch, there are batch_size matrices to multiply
        int batch_size = std::accumulate(shape.begin(), shape.end() - 2, 1, std::multiplies<int>());

        // Iterate over every batch
        for (int batch = 0; batch < batch_size; ++batch) {
            std::vector<int> idx_a(shape.size(), 0);
            std::vector<int> idx_b(other.shape.size(), 0);
            std::vector<int> idx_c(result_shape.size(), 0);


            // Map the batch number to the correct index
            int remaining_batch = batch;
            for (int i = 0; i < shape.size() - 2; ++i) {
                idx_a[i] = remaining_batch % shape[i];
                idx_b[i] = remaining_batch % other.shape[i];
                idx_c[i] = remaining_batch % result_shape[i];
                remaining_batch /= shape[i];
            }

            // result_ij = sum_k a_ik * b_kj
            for (int i = 0; i < shape[shape.size() - 2]; ++i) { // Iterate over the rows of the first matrix
                for (int j = 0; j < other.shape.back(); ++j) { // Iterate over the columns of the second matrix
                    T sum = 0;
                    for (int k = 0; k < shape.back(); ++k) { // Iterate over the columns of the first matrix
                        idx_a[shape.size() - 2] = i;
                        idx_a[shape.size() - 1] = k;
                        idx_b[other.shape.size() - 2] = k;
                        idx_b[other.shape.size() - 1] = j;
                        sum += (*this)[idx_a] * other[idx_b]; // Compute the dot product
                    }
                    idx_c[result_shape.size() - 2] = i;
                    idx_c[result_shape.size() - 1] = j;
                    result[idx_c] = sum; // Set the result at the specified indices
                }
            }
        }

        return result;
    }
    
    template <typename BinaryOp>
    Tensor<T> elementWiseOperation(const Tensor<T>& tensor1, const Tensor<T>& tensor2, BinaryOp op) const {
        /**
         * Elementwise operation on two tensors
         * 
         * @param tensor1: First tensor
         * @param tensor2: Second tensor
         * @param op: Binary operation to perform
         * @return: Result of the elementwise operation
        */
        std::vector<int> target_shape = resolveBroadcast(tensor1.shape, tensor2.shape);
        Tensor<T> result(target_shape);
        Tensor<T> tensor1_broadcasted = tensor1.setStridesForBroadcast(target_shape);
        Tensor<T> tensor2_broadcasted = tensor2.setStridesForBroadcast(target_shape);

        auto tensor1_it = tensor1_broadcasted.begin(), tensor1_end = tensor1_broadcasted.end();
        auto tensor2_it = tensor2_broadcasted.begin(), tensor2_end = tensor2_broadcasted.end();
        auto result_it = result.begin();

        while (result_it != result.end()) {
            *result_it = op(*tensor1_it, *tensor2_it);
            ++result_it;

            if (tensor1_it != tensor1_end) {
                ++tensor1_it;
            }
            if (tensor1_it == tensor1_end) { // reset for broadcasting
                tensor1_it = tensor1_broadcasted.begin();
            }
            if (tensor2_it != tensor2_end) {
                ++tensor2_it;
            }
            if (tensor2_it == tensor2_end) { // reset for broadcasting
                tensor2_it = tensor2_broadcasted.begin();
            }
        }

        return result;
    }

    template <typename BinaryOp>
    Tensor<T> elementWiseOperationScalar(const Tensor<T>& tensor, const T& scalar, BinaryOp op) const {
        /**
         * Elementwise operation on a tensor and a scalar
         * 
         * @param tensor: Tensor
         * @param scalar: Scalar
         * @param op: Binary operation to perform
         * @return: Result of the elementwise operation
        */
        Tensor<T> result(tensor.shape);
        auto tensor_it = tensor.begin();
        auto result_it = result.begin();

        while (tensor_it != tensor.end()) {
            *result_it = op(*tensor_it, scalar);
            ++tensor_it;
            ++result_it;
        }

        return result;
    }

    Tensor<T> operator+(const Tensor<T>& other) const {
        /**
         * Elementwise addition of two tensors
         * 
         * @param other: Tensor to add
         * @return: Result of the elementwise addition
         * 
         * @throws: std::invalid_argument if the shapes do not match for broadcasting
        */
        return elementWiseOperation(*this, other, std::plus<T>());
    }

    Tensor<T> operator+(const T& scalar) const {
        /**
         * Elementwise addition of a tensor and a scalar
         * 
         * @param scalar: Scalar to add
         * @return: Result of the elementwise addition
        */
        return elementWiseOperationScalar(*this, scalar, std::plus<T>());
    }

    friend Tensor<T> operator+(const T& scalar, const Tensor<T>& tensor) {
        /**
         * Elementwise addition of a scalar and a tensor
         * 
         * @param scalar: Scalar to add
         * @param tensor: Tensor to add
         * @return: Result of the elementwise addition
        */
        return tensor + scalar;
    }

    Tensor<T> operator-(const Tensor<T>& other) const {
        /**
         * Elementwise subtraction of two tensors
         * 
         * @param other: Tensor to subtract
         * @return: Result of the elementwise subtraction
         * 
         * @throws: std::invalid_argument if the shapes do not match for broadcasting
        */
        return elementWiseOperation(*this, other, std::minus<T>());
    }

    Tensor<T> operator-(const T& scalar) const {
        /**
         * Elementwise subtraction of a tensor and a scalar
         * 
         * @param scalar: Scalar to subtract
         * @return: Result of the elementwise subtraction
        */
        return elementWiseOperationScalar(*this, scalar, std::minus<T>());
    }

    friend Tensor<T> operator-(const T& scalar, const Tensor<T>& tensor) {
        /**
         * Elementwise subtraction of a scalar and a tensor
         * 
         * @param scalar: Scalar to subtract
         * @param tensor: Tensor to subtract
         * @return: Result of the elementwise subtraction
        */
        Tensor<T> result = Tensor<T>::full(tensor.getShape(), scalar);
        return result - tensor;
    }

    Tensor<T> operator*(const Tensor<T>& other) const {
        /**
         * Elementwise multiplication of two tensors
         * 
         * @param other: Tensor to multiply
         * @return: Result of the elementwise multiplication
         * 
         * @throws: std::invalid_argument if the shapes do not match for broadcasting
        */
        return elementWiseOperation(*this, other, std::multiplies<T>());
    }
    
    Tensor<T> operator*(const T& scalar) const {
        /**
         * Elementwise multiplication of a tensor and a scalar
         * 
         * @param scalar: Scalar to multiply
         * @return: Result of the elementwise multiplication
        */
        return elementWiseOperationScalar(*this, scalar, std::multiplies<T>());
    }

    friend Tensor<T> operator*(const T& scalar, const Tensor<T>& tensor) {
        /**
         * Elementwise multiplication of a scalar and a tensor
         * 
         * @param scalar: Scalar to multiply
         * @param tensor: Tensor to multiply
         * @return: Result of the elementwise multiplication
        */
        return tensor * scalar;
    }

    Tensor<T> operator/(const Tensor<T>& other) const {
        /**
         * Elementwise division of two tensors
         * 
         * @param other: Tensor to divide
         * @return: Result of the elementwise division
         * 
         * @throws: std::invalid_argument if the shapes do not match for broadcasting
        */
        return elementWiseOperation(*this, other, std::divides<T>());
    }

    Tensor<T> operator/(const T& scalar) const {
        /**
         * Elementwise division of a tensor and a scalar
         * 
         * @param scalar: Scalar to divide
         * @return: Result of the elementwise division
        */
        return elementWiseOperationScalar(*this, scalar, std::divides<T>());
    }

    friend Tensor<T> operator/(const T& scalar, const Tensor<T>& tensor) {
        /**
         * Elementwise division of a scalar and a tensor
         * 
         * @param scalar: Scalar to divide
         * @param tensor: Tensor to divide
         * @return: Result of the elementwise division
        */
        Tensor<T> result = Tensor<T>::full(tensor.getShape(), scalar);
        return result / tensor;
    }

    template <typename UnaryOp>
    Tensor<T> unaryOperation(const Tensor<T>& tensor, UnaryOp op) const {
            /**
             * Apply a unary operation to a tensor
             * 
             * @param tensor: Tensor to apply the operation to
             * @param op: Unary operation to perform
             * @return: Result of the unary operation
            */
            Tensor<T> result = *this;
            for (auto it = result.begin(), end = result.end(); it != end; ++it) {
                *it = op(*it);
            }
            return result;
        }

    Tensor<T> pow(const T& exponent) const {
        /**
         * Elementwise exponentiation of a tensor
         * 
         * @param exponent: Exponent to raise the tensor to
         * @return: Result of the elementwise exponentiation
        */
        return unaryOperation(*this, [exponent](const T& value) { return std::pow(value, exponent); });
    }

    Tensor<T> sqrt() const {
        /**
         * Elementwise square root of a tensor
         * 
         * @return: Result of the elementwise square root
        */
        return unaryOperation(*this, [](const T& value) { return std::sqrt(value); });
    }

    Tensor<T> log() const {
        /**
         * Elementwise natural logarithm of a tensor
         * 
         * @return: Result of the elementwise natural logarithm
        */
        return unaryOperation(*this, [](const T& value) { return std::log(value); });
    }
    
    Tensor<T> abs() const {
        /**
         * Elementwise absolute value of a tensor
         * 
         * @return: Result of the elementwise absolute value
        */
        return unaryOperation(*this, [](const T& value) { return std::abs(value); });
    }

    Tensor<T> exp() const {
        /**
         * Elementwise exponential of a tensor
         * 
         * @return: Result of the elementwise exponential
        */
        return unaryOperation(*this, [](const T& value) { return std::exp(value); });
    }

    Tensor<T> sin() const {
        /**
         * Elementwise sine of a tensor
         * 
         * @return: Result of the elementwise sine
        */
        return unaryOperation(*this, [](const T& value) { return std::sin(value); });
    }

    Tensor<T> cos() const {
        /**
         * Elementwise cosine of a tensor
         * 
         * @return: Result of the elementwise cosine
        */
        return unaryOperation(*this, [](const T& value) { return std::cos(value); });
    }

    T mean() const {
        /**
         * Mean of all elements in the tensor
         * 
         * @return: Mean of the tensor elements
         * 
         * TODO: write unit tests for this method
        */
        return std::accumulate(begin(), end(), T{}) / static_cast<T>(size());
    }

    Tensor<T> permute(std::initializer_list<int> order) const {
        /**
         * Alternative method to call permute with an initializer list
         * 
         * @param order: New order of dimensions
         * @return: Permuted tensor
         * 
         * @throws: std::invalid_argument if the permutation indices are invalid.
         * @throws: std::invalid_argument if the order size does not match the number of tensor dimensions.
        */
        return permute(std::vector<int>(order.begin(), order.end()));
    }

    Tensor<T> permute(std::vector<int> order) const {
        /**
         * Returns a new tensor that is a permuted view of the same underlying data
         * 
         * @attention: The tensor will not be contiguous after permutation, consider using contiguous()
         *             if subsequent operations require contiguity.
         * 
         * @param order: New order of dimensions.
         * @return: Permuted tensor
         * 
         * @throws: std::invalid_argument if the permutation indices are invalid.
         * @throws: std::invalid_argument if the order size does not match the number of tensor dimensions.
         */
        // Permutation won't do anything if the order is already sorted
        // But checking is mandatory to prevent isContiguous from incorrectly being set to false
        if (std::is_sorted(order.begin(), order.end())) {
            return *this;
        }

        if (order.size() != this->shape.size()) {
            throw std::invalid_argument("Permutation order must match the number of tensor dimensions.");
        }

        std::vector<int> new_shape(order.size()), new_strides(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i] >= this->shape.size()) {
                throw std::invalid_argument("Permutation index out of dimension range.");
            }
            new_shape[i] = this->shape[order[i]];
            new_strides[i] = this->strides[order[i]];
        }

        Tensor<T> result(new_shape);
        result.strides = new_strides;
        result.data = this->data; // Share the same data

        if (contiguousFlag) {
            result.contiguousFlag = false; // If the original tensor is contiguous, the permuted tensor is not
        } else {
            // Check if the permuted tensor is contiguous
            int expected_size = 1;
            auto expected_strides = computeStrides(new_shape, expected_size);
            if (result.strides == expected_strides && result.data.getSize() == expected_size) {
                result.contiguousFlag = true; // If the permuted tensor is contiguous, set isContiguous to true
            }
        }

        return result;
    }

    Tensor<T> transpose() const {
        /**
         * Returns a new tensor that is a transpose of this tensor, specifically swapping the last two dimensions.
         * This method assumes that the tensor has at least two dimensions.
         * 
         * @attention The tensor will not be contiguous after transposition, consider using contiguous()
         *            if subsequent operations require contiguity.
         * 
         * @return: Transposed tensor
         * 
         * @throws: std::runtime_error if the tensor does not have at least two dimensions
         */
        if (shape.size() < 2) {
            throw std::runtime_error("Transpose requires at least two dimensions.");
        }

        // Create a permutation order for all dimensions
        std::vector<int> perm(shape.size());
        std::iota(perm.begin(), perm.end(), 0);  // Fill with 0, 1, 2, ..., n
        std::swap(perm[shape.size() - 2], perm[shape.size() - 1]);  // Swap the last two dimensions
        
        return this->permute(perm);
    }

    Tensor<T> reshape(std::initializer_list<int> shape) const {
        /**
         * Alternative method to call reshape with an initializer list
         * 
         * @param shape: New shape of the tensor
         * @return: Reshaped tensor
         * 
         * @throws: std::invalid_argument if the reshape size is not constant
        */
        return reshape(std::vector<int>(shape.begin(), shape.end()));
    }

    Tensor<T> reshape(std::vector<int> shape) const {
        /**
         * Returns a new tensor that is a reshaped view of the same underlying data
         * 
         * @param shape: New shape of the tensor
         * @return: Reshaped tensor
         * 
         * @throws: std::invalid_argument if the reshape size is not constant
         */
        if (std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) != 
            std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int>())) {
            throw std::invalid_argument("Reshape size must remain constant.");
        }

        Tensor<T> result(shape);
        result.data = data; // Share the same data

        return result;
    }

    Tensor<T> squeeze(int axis) const {
        /**
         * Removes a single-dimensional entry from the shape of the tensor at the specified axis.
         * 
         * @param axis: Axis to squeeze
         * @return: Squeezed tensor
         * 
         * @throws: std::invalid_argument if the axis is out of bounds
         */
        if (axis < 0 || axis >= shape.size()) {
            throw std::invalid_argument("Axis out of bounds.");
        }

        if (shape[axis] != 1) {
            return *this; // If the dimension is not 1, return the same tensor
        }

        std::vector<int> new_shape = shape;
        new_shape.erase(new_shape.begin() + axis);

        Tensor<T> result(new_shape);
        result.data = data; // Share the same data

        return result;
    }

    Tensor<T> unsqueeze(int axis) const {
        /**
         * Adds a single-dimensional entry to the shape of the tensor at the specified axis.
         * 
         * @param axis: Axis to unsqueeze
         * @return: Unsqueezed tensor
         * 
         * @throws: std::invalid_argument if the axis is out of bounds
         */
        if (axis < 0 || axis > shape.size()) {
            throw std::invalid_argument("Axis out of bounds.");
        }

        std::vector<int> new_shape = shape;
        new_shape.insert(new_shape.begin() + axis, 1);

        Tensor<T> result(new_shape);
        result.data = data; // Share the same data

        return result;
    }

    Tensor<T> contiguous() {
        /**
         * Returns a new tensor that is contiguous in memory.
         * 
         * Can improve efficiency of memory access following operations 
         * that may result in non-contiguous tensors, such as permute().
         * 
         * @return: Contiguous tensor
         */
        if (contiguousFlag) {
            return *this; // If already contiguous, return this tensor
        }
        contiguousFlag = true; // Set isContiguous to true

        int expected_size = 1;
        auto expected_strides = computeStrides(this->shape, expected_size);
        // Check if tensor is already contiguous
        if (this->strides == expected_strides && this->data.getSize() == expected_size) {
            return *this; // If already contiguous, return this tensor
        }

        // Create a new tensor with the same shape but default strides
        Tensor<T> result(this->shape);
        result.data.resize(expected_size);
        result.strides = expected_strides;

        // Use iterator to copy data in logical order
        auto it = begin();
        for (auto& value : result) {
            value = *it;
            ++it;
        }
        return result;
    }

    bool isContiguous() const {
        /**
         * Check if the tensor is contiguous in memory.
         * 
         * A tensor is contiguous if the strides match the expected strides for the shape.
         * 
         * @return: True if the tensor is contiguous, false otherwise
         */
        return contiguousFlag;
    }

    size_t size() const {
        return data.getSize();
    }
    
    std::vector<int> getShape() const {
        return shape;
    }

    d_type getDtype() const {
        return dtype;
    }

    void printData() const {
        /**
         * Print the tensor
        */
        printRecursive(0, 0);
        std::cout << "\n";
    }

    void printDims() const {
        /**
         * Print the dimensions of the tensor
        */
        std::cout << "Shape:  [";
        for (int i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << " ";
        }
        std::cout << "]\n";

        std::cout << "Stride: [";
        for (int i = 0; i < strides.size(); ++i) {
            std::cout << strides[i];
            if (i < strides.size() - 1) std::cout << " ";
        }
        std::cout << "]\n";
    }

    static Tensor<T> fromFile(const std::string& filePath) {
        /**
         * Load a tensor from a file
         * 
         * Currently only supports loading tensors from .npy files
         *
         * @param filePath: Path to the file
         * @return: Tensor loaded from the file
         *
         * @throws: std::runtime_error if the file format is not supported
         */
        std::string extension = filePath.substr(filePath.find_last_of(".") + 1);
        if (extension == "npy") {
            cnpy::NpyArray arr = cnpy::npy_load(filePath);
            std::vector<int> shape(arr.shape.begin(), arr.shape.end());

            d_type dtype;
            switch (arr.word_size) {
                case 4:
                    if (std::is_same<T, int32_t>::value)
                        dtype = d_type::I32;
                    else if (std::is_same<T, float>::value)
                        dtype = d_type::FP32;
                    else
                        throw std::runtime_error("Unsupported data type.");
                    break;
                case 8:
                    if (std::is_same<T, int64_t>::value)
                        dtype = d_type::I64;
                    else if (std::is_same<T, double>::value)
                        dtype = d_type::FP64;
                    else if (std::is_same<T, std::complex<float>>::value)
                        dtype = d_type::C64;
                    else
                        throw std::runtime_error("Unsupported data type.");
                    break;
                case 16:
                    if (std::is_same<T, std::complex<double>>::value)
                        dtype = d_type::C128;
                    else
                        throw std::runtime_error("Unsupported data type.");
                    break;
                default:
                    throw std::runtime_error("Unsupported data type.");
            }

            Tensor<T> tensor(shape);
            std::copy(arr.data<T>(), arr.data<T>() + tensor.size(), tensor.data.rawPtr());
            return tensor;
        } else {
            throw std::runtime_error("Unsupported file format. Supported formats: .npy");
        }
    }

    void toFile(const std::string& filePath) const {
        /**
         * Save a tensor to a file
         * 
         * Currently only supports saving tensors to .npy files
         *
         * @param filePath: Path to the file
         *
         * @throws: std::runtime_error if the file format is not supported
         */
        std::string extension = filePath.substr(filePath.find_last_of(".") + 1);
        if (extension == "npy") {
            std::vector<size_t> shape(this->shape.begin(), this->shape.end());
            cnpy::npy_save(filePath, data.rawPtr(), shape);
        } else {
            throw std::runtime_error("Unsupported file format. Supported formats: .npy");
        }
    }

    static Tensor<T> arange(T start, T stop, T step = 1) {
        /**
         * Create a tensor with values from start to stop with a specified step size
         * 
         * @param start: Start value
         * @param stop: Stop value
         * @param step: Step size, default is 1
         * @return: Tensor with values from start to stop
         * 
         * @throws: std::invalid_argument if the step is zero
         * @throws: std::invalid_argument if the start, stop, step combination is invalid
         */
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type.");
        if (step == 0) {
            throw std::invalid_argument("Step cannot be zero.");
        }
        if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
            throw std::invalid_argument("Invalid start, stop, step combination.");
        }
        int size = std::ceil((stop - start) / step);
        Tensor<T> result({size});
        for (int i = 0; i < size; ++i) {
            result[{i}] = start + i * step;
        }
        return result;
    }

    static Tensor<T> full(std::vector<int> shape, T value) {
        /**
         * Create a tensor of a specified shape filled with a value
         * 
         * @param shape: Shape of the tensor
         * @param value: Value to fill the tensor with
         * @return: Tensor filled with the specified value
         */
        Tensor<T> result(shape);
        std::fill(result.data.begin(), result.data.end(), value);
        return result;
    }

    static Tensor<T> full(std::initializer_list<int> shape, T value) {
        /**
         * Alternative method to call full with an initializer list
         * 
         * @param shape: Shape of the tensor
         * @param value: Value to fill the tensor with
         * @return: Tensor filled with the specified value
         */
        return full(std::vector<int>(shape.begin(), shape.end()), value);
    }

    static Tensor<T> sinusoid(std::vector<int> shape, T frequency = 1, T phase = 0, T amplitude = 1) {
        /**
         * Create a tensor of a specified shape filled with a sinusoidal wave
         * 
         * @TODO: Write unit tests for this method
         * 
         * @param shape: Shape of the tensor
         * @param frequency: Frequency of the sinusoidal wave
         * @param phase: Phase of the sinusoidal wave
         * @param amplitude: Amplitude of the sinusoidal wave
         * @return: Tensor filled with a sinusoidal wave
         */
        Tensor<T> result(shape);
        std::vector<int> idx(shape.size(), 0);
        int totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        for (auto it = result.begin(); it != result.end(); ++it) {
            T x = T(2 * M_PI) * frequency * T(std::inner_product(idx.begin(), idx.end(), result.strides.begin(), 0)) / T(totalSize) + phase;
            *it = amplitude * std::sin(x);
            for (int i = 0; i < shape.size(); ++i) {
                idx[i] += 1;
                if (idx[i] < shape[i]) {
                    break;
                }
                idx[i] = 0;
            }
        }
        return result;
    }

    static Tensor<T> sinusoid(std::initializer_list<int> shape, T frequency = 1, T phase = 0, T amplitude = 1) {
        /**
         * Alternative method to call sinusoid with an initializer list
         * 
         * @param shape: Shape of the tensor
         * @param frequency: Frequency of the sinusoidal wave
         * @param phase: Phase of the sinusoidal wave
         * @param amplitude: Amplitude of the sinusoidal wave
         * @return: Tensor filled with a sinusoidal wave
         */
        return sinusoid(std::vector<int>(shape.begin(), shape.end()), frequency, phase, amplitude);
    }

};

#endif // TENSOR_H