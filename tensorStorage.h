#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <cstdlib>
#include <cstring>
#include <iterator>
#include <new>
#include <memory>

template <typename T>
class TensorStorage {
    std::shared_ptr<T> data;
    size_t size;
    size_t alignment;

public:
    TensorStorage() : data(nullptr), size(0), alignment(0) {}

    TensorStorage(size_t size, size_t alignment = alignof(T)) : size(size), alignment(alignment) {
        /**
         * Initialize the data with the size and alignment
         * 
         * @param size: size of the data
         * @param alignment: alignment of the data
        */
        T* buffer = static_cast<T*>(std::aligned_alloc(alignment, size * sizeof(T)));
        if (!buffer) {
            throw std::bad_alloc();
        }
        data = std::shared_ptr<T>(buffer, std::free);
        std::memset(data.get(), 0, size * sizeof(T)); // zero-initialize the data
    }

    T &operator[](size_t index) {
        /**
         * Overload the [] operator to access the data
         * 
         * @param index: index of the data
         * @return: the data at the index
        */
        return data.get()[index];
    }

    const T &operator[](size_t index) const {
        /**
         * Overload the [] operator to access the data
         * 
         * @param index: index of the data
         * @return: the data at the index
        */
        return data.get()[index];
    }

    T *rawPtr() {
        /**
         * Return the raw pointer to the data
         * 
         * @return: the raw pointer to the data
        */
        return data.get();
    }

    const T *rawPtr() const {
        /**
         * Return the raw pointer to the data
         * 
         * @return: the raw pointer to the data
        */
        return data.get();
    }

    size_t getSize() const {
        /**
         * Return the size of the data
         * 
         * @return: the size of the data
        */
        return size;
    }

    void resize(size_t newSize) {
        /**
         * Resize the data
         * 
         * @param newSize: the new size of the data
        */
        if (newSize == size) {
            return;
        }
        T* existingData = data.get();
        T* buffer = static_cast<T*>(std::aligned_alloc(alignment, newSize * sizeof(T)));
        if (!buffer) { // ensure that the allocation was successful before proceeding
            throw std::bad_alloc();
        }
        auto deleter = [](T* ptr) { std::free(ptr); };
        std::unique_ptr<T, decltype(deleter)> newBuffer(buffer, deleter);
        std::memset(buffer, 0, newSize * sizeof(T)); // zero-initialize the new buffer
        if (existingData) {
            std::copy(existingData, existingData + std::min(size, newSize), buffer);
        }
        data = std::move(newBuffer);
        size = newSize;
    }

        class iterator {
            T* ptr;
        public:
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = T;
            using pointer = T*;
            using reference = T&;

            iterator(pointer ptr) : ptr(ptr) {}

            reference operator*() const { return *ptr; }
            pointer operator->() { return ptr; }

            // Increment and Decrement
            iterator& operator++() { ++ptr; return *this; }
            iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }
            iterator& operator--() { --ptr; return *this; }
            iterator operator--(int) { iterator tmp = *this; --(*this); return tmp; }

            // Arithmetic operations
            iterator& operator+=(difference_type n) { ptr += n; return *this; }
            iterator& operator-=(difference_type n) { ptr -= n; return *this; }
            iterator operator+(difference_type n) const { return iterator(ptr + n); }
            iterator operator-(difference_type n) const { return iterator(ptr - n); }
            difference_type operator-(const iterator& other) const { return ptr - other.ptr; }

            reference operator[](difference_type n) const { return *(ptr + n); }

            // Comparison operators
            bool operator< (const iterator& other) const { return ptr < other.ptr; }
            bool operator> (const iterator& other) const { return ptr > other.ptr; }
            bool operator<=(const iterator& other) const { return ptr <= other.ptr; }
            bool operator>=(const iterator& other) const { return ptr >= other.ptr; }
            bool operator==(const iterator& other) const { return ptr == other.ptr; }
            bool operator!=(const iterator& other) const { return ptr != other.ptr; }
        };

        iterator begin() const { return iterator(data.get()); }
        iterator end() const { return iterator(data.get() + size); }
};

#endif // TENSOR_STORAGE_H