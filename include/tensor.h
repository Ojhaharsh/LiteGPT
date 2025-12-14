#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <stdexcept>

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::vector<int>& strides);
    
    // Destructor
    ~Tensor();
    
    // Data access
    float* data() const { return data_.get(); }
    
    // Shape and stride information
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int ndim() const { return shape_.size(); }
    size_t total_elements() const { return total_elements_; }
    
    // Indexing operations
    float& operator[](const std::vector<int>& indices);
    const float& operator[](const std::vector<int>& indices) const;
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;
    
    // Element access with bounds checking
    float get(const std::vector<int>& indices) const;
    void set(const std::vector<int>& indices, float value);
    
    // Utility functions
    void fill(float value);
    void zero() { fill(0.0f); }
    void ones() { fill(1.0f); }
    void random();
    void mul(float scalar);
    
    // Memory management
    void reshape(const std::vector<int>& new_shape);
    void view(const std::vector<int>& new_shape);  // View without copying
    
    // Printing
    void print() const;
    
private:
    std::shared_ptr<float> data_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    size_t total_elements_;
    
    // Helper functions
    void compute_strides();
    size_t compute_index(const std::vector<int>& indices) const;
    void validate_indices(const std::vector<int>& indices) const;
};

#endif // TENSOR_H
