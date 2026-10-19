/*
 * Axis X5: pure operation descriptor validation. Backend preparation/submission
 * remain pending; this file currently implements validate_operation only.
 * Contract: enforce operations.h's mathematical shape/layout/type/constants and
 * access rules without touching a GPU. Repeated reads are deduplicated; an output
 * ID cannot also be read. Physical region overlap requires future binding checks.
 * Invalid descriptors -> invalid_argument, size overflow -> length_error,
 * metadata allocation failure propagates. No input metadata is retained.
 * Algorithm: transformer overlay, "Preparing the first execution recipe":
 * validate mode; visit the concrete equation; check IDs, dense nonempty layouts,
 * shape relations and constants; emit unique read/write roles for planning.
 * Backend selection and buffer binding happen later, before any execution.
 * Design and cost: O(sum of operand ranks + gather row count), bounded operand
 * metadata and no tensor allocation. Optimization: validate during preparation,
 * outside the eventual dispatch loop. No launch grid or model-name assumptions.
 * Unit tests exercise validate_operation's shape/access/error contracts with
 * ordinary metadata; kernel execution is integration work.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, validation implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): explicit inputs and no hidden execution state.
 * cpp-guidelines I.7 (#ri-post): output roles represent the checked equation.
 * cpp-guidelines ES.103 (#res-overflow): check size/position arithmetic first.
 * cpp-perf-guidelines GPU.2: validate dense logical addressing before selection.
 * Source: guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 */
#include "kernels/operations.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace dolittle::kernels {
namespace {

using tensors::ElementType;
using tensors::Layout;

void require(bool condition) {
    if (!condition) {
        throw std::invalid_argument("operation descriptor violates its contract");
    }
}

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::length_error("operation shape product overflows");
    }
    return a * b;
}

std::size_t trailing_elements(const Layout& layout) {
    std::size_t elements = 1;
    for (std::size_t i = 1; i < layout.shape.size(); ++i) {
        elements = checked_product(elements, layout.shape[i]);
    }
    return elements;
}

class DescriptorValidator {
public:
    DescriptorValidator(std::span<const Layout> layouts, ElementType storage)
        : layouts_(layouts), storage_(storage) {
        uses_.reserve(4);
    }

    void operator()(const Embedding& op) {
        const auto& ids = operand(op.tokens, OperandAccess::read, ElementType::uint32);
        const auto& weight = read(op.weight);
        const auto& output = write(op.output);
        require(ids.shape.size() == 1 && weight.shape.size() == 2 &&
                output.shape.size() == 2);
        require(output.shape[0] == ids.shape[0] &&
                output.shape[1] == weight.shape[1]);
    }

    void operator()(const Projection& op) {
        const auto& input = read(op.input);
        const auto& weight = read(op.weight);
        const auto& output = write(op.output);
        require(input.shape.size() >= 2 && output.shape.size() >= 2 &&
                weight.shape.size() == 2);
        require(input.shape[0] == output.shape[0] &&
                trailing_elements(input) == weight.shape[1] &&
                trailing_elements(output) == weight.shape[0]);
    }

    void operator()(const RmsNorm& op) {
        const auto& input = read(op.input);
        const auto& weight = read(op.weight);
        const auto& output = write(op.output);
        require(!input.shape.empty() && weight.shape.size() == 1);
        require(input.shape == output.shape && input.shape.back() == weight.shape[0]);
        require(std::isfinite(op.epsilon) && op.epsilon > 0);
    }

    void operator()(const Rotary& op) {
        const auto& input = read(op.input);
        const auto& output = write(op.output);
        require(input.shape.size() == 3 && input.shape == output.shape);
        require(input.shape[2] % 2 == 0 && std::isfinite(op.theta) && op.theta > 0);
        if (op.first_position > std::numeric_limits<std::size_t>::max() -
                                    (input.shape[0] - 1)) {
            throw std::length_error("rotary position range overflows");
        }
    }

    void operator()(const CausalAttention& op) {
        const auto& query = read(op.query);
        const auto& key = read(op.key);
        const auto& value = read(op.value);
        const auto& output = write(op.output);
        require(query.shape.size() == 3 && key.shape.size() == 3);
        require(key.shape == value.shape && query.shape == output.shape);
        require(query.shape[0] == key.shape[0] && query.shape[2] == key.shape[2] &&
                query.shape[1] % key.shape[1] == 0);
    }

    void operator()(const Add& op) { elementwise(op.left, op.right, op.output); }
    void operator()(const SiluMultiply& op) { elementwise(op.gate, op.up, op.output); }

    void operator()(const GatherRows& op) {
        const auto& input = read(op.input);
        const auto& output = write(op.output);
        require(!input.shape.empty() && output.shape.size() == input.shape.size());
        require(!op.rows.empty() && output.shape[0] == op.rows.size());
        for (std::size_t i = 1; i < input.shape.size(); ++i) {
            require(input.shape[i] == output.shape[i]);
        }
        for (std::size_t i = 0; i < op.rows.size(); ++i) {
            require(op.rows[i] < input.shape[0]);
            require(i == 0 || op.rows[i - 1] < op.rows[i]);
        }
    }

    std::vector<OperandUse> take_uses() { return std::move(uses_); }

private:
    const Layout& operand(TensorId id, OperandAccess access, ElementType type) {
        require(id.value < layouts_.size());
        const auto& layout = layouts_[id.value];
        require(layout.element_type == type);
        require(tensors::required_bytes(layout) != 0);
        auto expected_stride = tensors::element_bytes(type);
        for (auto i = layout.shape.size(); i != 0; --i) {
            require(layout.shape[i - 1] == 1 ||
                    layout.byte_strides[i - 1] == expected_stride);
            expected_stride = checked_product(expected_stride, layout.shape[i - 1]);
        }
        for (const auto& use : uses_) {
            if (use.tensor.value == id.value) {
                require(use.access == OperandAccess::read && access == OperandAccess::read);
                return layout;
            }
        }
        uses_.push_back({id, access});
        return layout;
    }

    const Layout& read(TensorId id) { return operand(id, OperandAccess::read, storage_); }
    const Layout& write(TensorId id) { return operand(id, OperandAccess::write, storage_); }

    void elementwise(TensorId a, TensorId b, TensorId output) {
        const auto& left = read(a);
        const auto& right = read(b);
        const auto& result = write(output);
        require(left.shape == right.shape && left.shape == result.shape);
    }

    std::span<const Layout> layouts_;
    ElementType storage_;
    std::vector<OperandUse> uses_;
};

}  // namespace

std::vector<OperandUse> validate_operation(
    const Operation& operation, std::span<const Layout> layouts,
    tensors::ComputationMode mode) {
    require((mode.activation_storage == ElementType::bfloat16 ||
             mode.activation_storage == ElementType::float32) &&
            mode.reduction_accumulation == ElementType::float32);
    require(!operation.valueless_by_exception());
    DescriptorValidator validator(layouts, mode.activation_storage);
    std::visit(validator, operation);
    return validator.take_uses();
}

}  // namespace dolittle::kernels
