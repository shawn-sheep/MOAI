#include "moai/openfhe/server_runtime.hpp"

#include "moai/openfhe/approximation_registry.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/evaluation_key_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

namespace moai::openfhe {
namespace {

using ContextImpl = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>;
using EvaluationKey = lbcrypto::EvalKey<lbcrypto::DCRTPoly>;
using MultiplicationKeyVector = std::vector<EvaluationKey>;
using AutomorphismKeyMap = std::map<uint32_t, EvaluationKey>;

bool ScalesMatch(double lhs, double rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs) || lhs <= 0.0 || rhs <= 0.0) {
        return false;
    }
    const double magnitude =
        std::max(1.0, std::max(std::abs(lhs), std::abs(rhs)));
    return std::abs(lhs - rhs) <= 1e-12 * magnitude;
}

void ValidateCiphertextMetadata(
    const CipherTensor& tensor,
    const char* operation) {
    if (tensor.empty() || tensor.packing.noise_scale_degree == 0 ||
        !std::isfinite(tensor.packing.scaling_factor) ||
        tensor.packing.scaling_factor <= 0.0) {
        throw std::invalid_argument(
            std::string(operation) + " received invalid packing metadata");
    }
    for (const auto& ciphertext : tensor.ciphertexts) {
        if (!ciphertext ||
            ciphertext->GetLevel() != tensor.packing.level ||
            ciphertext->GetNoiseScaleDeg() !=
                tensor.packing.noise_scale_degree ||
            !ScalesMatch(
                ciphertext->GetScalingFactor(),
                tensor.packing.scaling_factor)) {
            throw std::invalid_argument(
                std::string(operation) +
                " received stale ciphertext level/scale metadata");
        }
    }
}

void RefreshPackingMetadata(CipherTensor& tensor) {
    if (tensor.ciphertexts.empty()) {
        return;
    }
    tensor.packing.level = tensor.ciphertexts.front()->GetLevel();
    tensor.packing.noise_scale_degree =
        tensor.ciphertexts.front()->GetNoiseScaleDeg();
    tensor.packing.scaling_factor =
        tensor.ciphertexts.front()->GetScalingFactor();
    for (const auto& ciphertext : tensor.ciphertexts) {
        if (ciphertext->GetLevel() != tensor.packing.level ||
            ciphertext->GetNoiseScaleDeg() != tensor.packing.noise_scale_degree) {
            throw std::logic_error(
                "CipherTensor members do not share level/scale metadata");
        }
        if (!ScalesMatch(
                ciphertext->GetScalingFactor(),
                tensor.packing.scaling_factor)) {
            throw std::logic_error(
                "CipherTensor members do not share a scaling factor");
        }
    }
}

std::size_t BroadcastIndex(std::size_t index, std::size_t size) {
    return size == 1 ? 0 : index;
}

void ValidatePhysicalPacking(
    const CipherTensor& lhs,
    const CipherTensor& rhs,
    const char* operation) {
    if (lhs.empty() || rhs.empty() ||
        (rhs.size() != 1 && rhs.size() != lhs.size())) {
        throw std::invalid_argument(
            std::string(operation) +
            " requires a non-empty left tensor and broadcast-compatible right tensor");
    }
    if (lhs.packing.slot_count != rhs.packing.slot_count ||
        lhs.packing.active_slots != rhs.packing.active_slots ||
        lhs.packing.encoded_slots != rhs.packing.encoded_slots ||
        lhs.packing.batch_lanes != rhs.packing.batch_lanes) {
        throw std::invalid_argument(
            std::string(operation) + " physical packing contracts do not match");
    }
}

void ValidateLogicalTensorShape(
    const CipherTensor& tensor,
    const char* operation) {
    if (tensor.packing.logical_shape.size() != 2 ||
        tensor.packing.logical_shape[0] == 0 ||
        tensor.packing.logical_shape[1] == 0 ||
        (tensor.packing.layout != PackingLayout::kDiagonal &&
         tensor.packing.logical_shape[1] != tensor.size())) {
        throw std::invalid_argument(
            std::string(operation) +
            " received a CipherTensor with inconsistent logical shape metadata");
    }
}

bool LogicalFeatureShapesCompatible(
    const CipherTensor& lhs,
    const CipherTensor& rhs) {
    const bool exact_shape =
        lhs.packing.logical_shape == rhs.packing.logical_shape;
    const bool singleton_feature_broadcast =
        lhs.packing.logical_shape[0] == rhs.packing.logical_shape[0] &&
        lhs.packing.logical_shape[1] == lhs.size() &&
        rhs.packing.logical_shape[1] == 1 && rhs.size() == 1;
    return exact_shape || singleton_feature_broadcast;
}

void ValidateBinaryPacking(
    const CipherTensor& lhs,
    const CipherTensor& rhs,
    const char* operation) {
    ValidatePhysicalPacking(lhs, rhs, operation);
    ValidateLogicalTensorShape(lhs, operation);
    ValidateLogicalTensorShape(rhs, operation);
    if (lhs.packing.layout != rhs.packing.layout) {
        throw std::invalid_argument(
            std::string(operation) + " logical packing contracts do not match");
    }
    if (!LogicalFeatureShapesCompatible(lhs, rhs)) {
        throw std::invalid_argument(
            std::string(operation) +
            " requires identical logical shapes or a one-feature rhs broadcast");
    }
}

void ValidateMultiplicationPacking(
    const CipherTensor& lhs,
    const CipherTensor& rhs,
    const char* operation) {
    ValidatePhysicalPacking(lhs, rhs, operation);
    ValidateLogicalTensorShape(lhs, operation);
    ValidateLogicalTensorShape(rhs, operation);
    const bool matching_layout = lhs.packing.layout == rhs.packing.layout;
    const bool diagonal_column_singletons =
        lhs.packing.layout == PackingLayout::kDiagonal &&
        rhs.packing.layout == PackingLayout::kColumn &&
        lhs.size() == 1 && rhs.size() == 1;
    if ((!matching_layout && !diagonal_column_singletons) ||
        !LogicalFeatureShapesCompatible(lhs, rhs)) {
        throw std::invalid_argument(
            std::string(operation) +
            " logical packing contracts are not elementwise compatible");
    }
}

void ValidatePlainVectors(
    const CipherTensor& input,
    const std::vector<std::vector<double>>& plaintexts) {
    if (input.empty() ||
        (plaintexts.size() != 1 && plaintexts.size() != input.size())) {
        throw std::invalid_argument(
            "plaintext vector count must be one or match the ciphertext count");
    }
    for (const auto& values : plaintexts) {
        if (values.size() != input.packing.active_slots) {
            throw std::invalid_argument(
                "plaintext vector length must match PackingSpec.active_slots");
        }
        if (!std::all_of(values.begin(), values.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument(
                "plaintext vector contains a non-finite value");
        }
    }
}

bool SameEvaluationKeyPointers(
    const MultiplicationKeyVector& lhs,
    const MultiplicationKeyVector& rhs) {
    return lhs.size() == rhs.size() &&
        std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool SameEvaluationKeyPointers(
    const AutomorphismKeyMap& lhs,
    const AutomorphismKeyMap& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    return std::equal(
        lhs.begin(),
        lhs.end(),
        rhs.begin(),
        [](const auto& left, const auto& right) {
            return left.first == right.first && left.second == right.second;
        });
}

void ValidateEvaluationKey(
    const EvaluationKey& key,
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
    const std::string& key_tag,
    const char* key_kind) {
    if (!key || key->GetKeyTag() != key_tag ||
        key->GetCryptoContext().get() != context.get()) {
        throw std::invalid_argument(
            std::string("server ") + key_kind +
            " evaluation key has a mismatched tag or context");
    }
}

template <typename Value>
bool StrictlyIncreasing(const std::vector<Value>& values) {
    return std::adjacent_find(
        values.begin(),
        values.end(),
        [](const Value& lhs, const Value& rhs) { return lhs >= rhs; }) ==
        values.end();
}

void RequireInstalledEvaluationKeys(
    const std::string& key_tag,
    const MultiplicationKeyVector& multiplication_keys,
    const AutomorphismKeyMap& automorphism_keys,
    const char* operation) {
    const auto& all_multiplication_keys = ContextImpl::GetAllEvalMultKeys();
    const auto multiplication_it = all_multiplication_keys.find(key_tag);
    if (multiplication_it == all_multiplication_keys.end() ||
        !SameEvaluationKeyPointers(
            multiplication_it->second,
            multiplication_keys)) {
        throw std::logic_error(
            std::string(operation) +
            " evaluation-key registry multiplication container changed");
    }

    const auto& all_automorphism_keys =
        ContextImpl::GetAllEvalAutomorphismKeys();
    const auto automorphism_it = all_automorphism_keys.find(key_tag);
    if (automorphism_keys.empty()) {
        if (automorphism_it != all_automorphism_keys.end()) {
            throw std::logic_error(
                std::string(operation) +
                " evaluation-key registry automorphism container changed");
        }
        return;
    }
    if (automorphism_it == all_automorphism_keys.end() ||
        !automorphism_it->second ||
        !SameEvaluationKeyPointers(
            *automorphism_it->second,
            automorphism_keys)) {
        throw std::logic_error(
            std::string(operation) +
            " evaluation-key registry automorphism container changed");
    }
}
}  // namespace

ServerRuntime::ServerRuntime(
    ServerKeyBundle key_bundle,
    CryptoProfile profile)
    : key_bundle_(std::move(key_bundle)),
      context_(key_bundle_.context),
      profile_(std::move(profile)) {
    if (!context_) {
        throw std::invalid_argument("server context must not be null");
    }
    if (!key_bundle_.public_key ||
        key_bundle_.key_tag != key_bundle_.public_key->GetKeyTag() ||
        key_bundle_.key_tag.empty() ||
        key_bundle_.public_key->GetCryptoContext().get() != context_.get() ||
        key_bundle_.profile_parameter_sha256 != profile_.parameter_sha256 ||
        key_bundle_.multiplication_eval_keys.empty()) {
        throw std::invalid_argument("server evaluation-key bundle is incomplete");
    }
    ValidateCryptoProfile(profile_);
    ValidateCryptoContextMatchesProfile(context_, profile_);
    for (const auto& key : key_bundle_.multiplication_eval_keys) {
        ValidateEvaluationKey(
            key,
            context_,
            key_bundle_.key_tag,
            "multiplication");
    }
    for (const auto& [index, key] : key_bundle_.automorphism_eval_keys) {
        static_cast<void>(index);
        ValidateEvaluationKey(
            key,
            context_,
            key_bundle_.key_tag,
            "automorphism");
    }
    if (!StrictlyIncreasing(key_bundle_.rotation_indices) ||
        !StrictlyIncreasing(key_bundle_.bootstrap_required_indices)) {
        throw std::invalid_argument(
            "server evaluation-key indices must be sorted and unique");
    }
    for (const int32_t rotation : key_bundle_.rotation_indices) {
        const uint32_t automorphism_index =
            context_->FindAutomorphismIndex(
                static_cast<uint32_t>(rotation));
        if (key_bundle_.automorphism_eval_keys.count(
                automorphism_index) != 1) {
            throw std::invalid_argument(
                "server rotation index has no corresponding automorphism key");
        }
    }
    if (!key_bundle_.bootstrap_required_indices.empty() &&
        !profile_.bootstrap_enabled) {
        throw std::invalid_argument(
            "server bundle carries bootstrap keys for a non-bootstrap profile");
    }
    for (const uint32_t index : key_bundle_.bootstrap_required_indices) {
        if (key_bundle_.automorphism_eval_keys.count(index) != 1) {
            throw std::invalid_argument(
                "server bootstrap index has no corresponding automorphism key");
        }
    }

    std::unique_lock registry_lock(EvaluationKeyRegistryMutex());
    auto& all_multiplication_keys = ContextImpl::GetAllEvalMultKeys();
    const auto multiplication_it =
        all_multiplication_keys.find(key_bundle_.key_tag);
    if (multiplication_it != all_multiplication_keys.end() &&
        !SameEvaluationKeyPointers(
            multiplication_it->second,
            key_bundle_.multiplication_eval_keys)) {
        throw std::invalid_argument(
            "server key tag already has a different multiplication-key container");
    }
    auto& all_automorphism_keys =
        ContextImpl::GetAllEvalAutomorphismKeys();
    const auto automorphism_it =
        all_automorphism_keys.find(key_bundle_.key_tag);
    if (automorphism_it != all_automorphism_keys.end() &&
        (!automorphism_it->second ||
         !SameEvaluationKeyPointers(
             *automorphism_it->second,
             key_bundle_.automorphism_eval_keys))) {
        throw std::invalid_argument(
            "server key tag already has a different automorphism-key container");
    }
    if (key_bundle_.automorphism_eval_keys.empty() &&
        automorphism_it != all_automorphism_keys.end()) {
        throw std::invalid_argument(
            "server key tag already has an unexpected automorphism-key container");
    }

    bool inserted_multiplication_keys = false;
    bool inserted_automorphism_keys = false;
    try {
        if (multiplication_it == all_multiplication_keys.end()) {
            ContextImpl::InsertEvalMultKey(
                key_bundle_.multiplication_eval_keys,
                key_bundle_.key_tag);
            inserted_multiplication_keys = true;
        }
        if (!key_bundle_.automorphism_eval_keys.empty() &&
            automorphism_it == all_automorphism_keys.end()) {
            ContextImpl::InsertEvalAutomorphismKey(
                std::make_shared<AutomorphismKeyMap>(
                    key_bundle_.automorphism_eval_keys),
                key_bundle_.key_tag);
            inserted_automorphism_keys = true;
        }
        RequireInstalledEvaluationKeys(
            key_bundle_.key_tag,
            key_bundle_.multiplication_eval_keys,
            key_bundle_.automorphism_eval_keys,
            "server construction");
    }
    catch (...) {
        if (inserted_automorphism_keys) {
            ContextImpl::ClearEvalAutomorphismKeys(key_bundle_.key_tag);
        }
        if (inserted_multiplication_keys) {
            ContextImpl::ClearEvalMultKeys(key_bundle_.key_tag);
        }
        throw;
    }
    metrics_.multiplicative_depth = profile_.multiplicative_depth;
}

lbcrypto::Plaintext ServerRuntime::EncodeModelVector(
    const std::vector<double>& values,
    const PackingSpec& packing) const {
    if (values.size() != packing.active_slots ||
        packing.slot_count != profile_.slot_count ||
        packing.encoded_slots < packing.active_slots ||
        packing.encoded_slots > packing.slot_count) {
        throw std::invalid_argument("model vector does not match PackingSpec");
    }
    auto plaintext = context_->MakeCKKSPackedPlaintext(
        values,
        packing.noise_scale_degree,
        packing.level,
        nullptr,
        packing.encoded_slots);
    if (!ScalesMatch(
            plaintext->GetScalingFactor(),
            packing.scaling_factor)) {
        throw std::logic_error(
            "encoded model vector does not match PackingSpec scaling factor");
    }
    return plaintext;
}

lbcrypto::Plaintext ServerRuntime::EncodeSingleScaleModelVector(
    const std::vector<double>& values,
    const PackingSpec& double_scale_packing) const {
    if (values.size() != double_scale_packing.active_slots ||
        double_scale_packing.slot_count != profile_.slot_count ||
        double_scale_packing.encoded_slots <
            double_scale_packing.active_slots ||
        double_scale_packing.encoded_slots >
            double_scale_packing.slot_count ||
        double_scale_packing.noise_scale_degree != 2) {
        throw std::invalid_argument(
            "single-scale model vector requires double-scale input packing");
    }
    auto plaintext = context_->MakeCKKSPackedPlaintext(
        values,
        1,
        double_scale_packing.level,
        nullptr,
        double_scale_packing.encoded_slots);
    const double squared_plaintext_scale =
        plaintext->GetScalingFactor() * plaintext->GetScalingFactor();
    if (plaintext->GetNoiseScaleDeg() != 1 ||
        plaintext->GetLevel() != double_scale_packing.level ||
        !ScalesMatch(
            squared_plaintext_scale,
            double_scale_packing.scaling_factor)) {
        throw std::logic_error(
            "single-scale model vector does not match double-scale input");
    }
    return plaintext;
}

CipherTensor ServerRuntime::Rotate(
    const CipherTensor& input,
    int32_t index) {
    if (input.empty()) {
        throw std::invalid_argument("cannot rotate an empty CipherTensor");
    }
    if (std::find(
            key_bundle_.rotation_indices.begin(),
            key_bundle_.rotation_indices.end(),
            index) == key_bundle_.rotation_indices.end()) {
        throw std::invalid_argument("requested rotation key is not in the server bundle");
    }
    std::shared_lock registry_lock(EvaluationKeyRegistryMutex());
    RequireInstalledEvaluationKeys(
        key_bundle_.key_tag,
        key_bundle_.multiplication_eval_keys,
        key_bundle_.automorphism_eval_keys,
        "rotation");
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (const auto& ciphertext : input.ciphertexts) {
        result.ciphertexts.push_back(context_->EvalRotate(ciphertext, index));
    }
    metrics_.rotations += input.size();
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::MultiplyPlain(
    const CipherTensor& input,
    const std::vector<lbcrypto::Plaintext>& plaintexts) {
    if (input.empty() ||
        (plaintexts.size() != 1 && plaintexts.size() != input.size())) {
        throw std::invalid_argument(
            "plaintext count must be one or match the ciphertext count");
    }
    std::shared_lock registry_lock(EvaluationKeyRegistryMutex());
    RequireInstalledEvaluationKeys(
        key_bundle_.key_tag,
        key_bundle_.multiplication_eval_keys,
        key_bundle_.automorphism_eval_keys,
        "ciphertext-plaintext multiplication");
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        result.ciphertexts.push_back(
            context_->EvalMult(
                input.ciphertexts[i],
                plaintexts[BroadcastIndex(i, plaintexts.size())]));
    }
    metrics_.ct_pt_multiplications += input.size();
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::Multiply(
    const CipherTensor& lhs,
    const CipherTensor& rhs) {
    ValidateMultiplicationPacking(lhs, rhs, "ciphertext multiplication");
    std::shared_lock registry_lock(EvaluationKeyRegistryMutex());
    RequireInstalledEvaluationKeys(
        key_bundle_.key_tag,
        key_bundle_.multiplication_eval_keys,
        key_bundle_.automorphism_eval_keys,
        "ciphertext multiplication");
    CipherTensor result;
    result.packing = lhs.packing;
    result.ciphertexts.reserve(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result.ciphertexts.push_back(
            context_->EvalMult(
                lhs.ciphertexts[i],
                rhs.ciphertexts[BroadcastIndex(i, rhs.size())]));
    }
    metrics_.ct_ct_multiplications += lhs.size();
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::Add(
    const CipherTensor& lhs,
    const CipherTensor& rhs) {
    ValidateBinaryPacking(lhs, rhs, "ciphertext addition");
    CipherTensor result;
    result.packing = lhs.packing;
    result.ciphertexts.reserve(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result.ciphertexts.push_back(
            context_->EvalAdd(
                lhs.ciphertexts[i],
                rhs.ciphertexts[BroadcastIndex(i, rhs.size())]));
    }
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::Subtract(
    const CipherTensor& lhs,
    const CipherTensor& rhs) {
    ValidateBinaryPacking(lhs, rhs, "ciphertext subtraction");
    CipherTensor result;
    result.packing = lhs.packing;
    result.ciphertexts.reserve(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result.ciphertexts.push_back(
            context_->EvalSub(
                lhs.ciphertexts[i],
                rhs.ciphertexts[BroadcastIndex(i, rhs.size())]));
    }
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::AddPlain(
    const CipherTensor& input,
    const std::vector<std::vector<double>>& plaintexts) {
    ValidatePlainVectors(input, plaintexts);
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto plaintext = EncodeModelVector(
            plaintexts[BroadcastIndex(i, plaintexts.size())],
            input.packing);
        result.ciphertexts.push_back(
            context_->EvalAdd(input.ciphertexts[i], plaintext));
    }
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::SubtractPlain(
    const CipherTensor& input,
    const std::vector<std::vector<double>>& plaintexts) {
    ValidatePlainVectors(input, plaintexts);
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto plaintext = EncodeModelVector(
            plaintexts[BroadcastIndex(i, plaintexts.size())],
            input.packing);
        result.ciphertexts.push_back(
            context_->EvalSub(input.ciphertexts[i], plaintext));
    }
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::Sum(const CipherTensor& input) {
    if (input.empty()) {
        throw std::invalid_argument("cannot sum an empty CipherTensor");
    }
    if (input.packing.logical_shape.size() != 2 ||
        input.packing.logical_shape[0] == 0 ||
        input.packing.logical_shape[1] != input.size()) {
        throw std::invalid_argument(
            "sum requires a two-dimensional logical shape whose feature "
            "dimension matches the ciphertext count");
    }
    CipherTensor result;
    result.packing = input.packing;
    result.packing.logical_shape[1] = 1;
    result.ciphertexts.push_back(input.ciphertexts.front());
    for (std::size_t i = 1; i < input.size(); ++i) {
        result.ciphertexts.front() = context_->EvalAdd(
            result.ciphertexts.front(), input.ciphertexts[i]);
    }
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::SumSlots(const CipherTensor& input) {
    if (input.empty() || input.packing.layout != PackingLayout::kContiguous ||
        input.packing.batch_lanes != 1 ||
        input.packing.active_slots == 0 ||
        input.packing.active_slots != input.packing.encoded_slots ||
        input.packing.logical_shape != std::vector<std::size_t>{
            input.packing.active_slots,
            input.size()} ||
        (input.packing.active_slots & (input.packing.active_slots - 1)) != 0) {
        throw std::invalid_argument(
            "slot summation requires a non-empty power-of-two contiguous packing");
    }
    ValidateCiphertextMetadata(input, "slot summation");
    for (uint32_t step = 1; step < input.packing.active_slots; step <<= 1) {
        if (std::find(
                key_bundle_.rotation_indices.begin(),
                key_bundle_.rotation_indices.end(),
                static_cast<int32_t>(step)) ==
            key_bundle_.rotation_indices.end()) {
            throw std::invalid_argument(
                "slot summation rotation-key bundle is incomplete");
        }
    }
    auto result = input;
    for (uint32_t step = 1; step < input.packing.active_slots; step <<= 1) {
        result = Add(result, Rotate(result, static_cast<int32_t>(step)));
    }
    return result;
}

CipherTensor ServerRuntime::Rescale(const CipherTensor& input) {
    if (input.empty()) {
        throw std::invalid_argument("cannot rescale an empty CipherTensor");
    }
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (const auto& ciphertext : input.ciphertexts) {
        result.ciphertexts.push_back(context_->Rescale(ciphertext));
    }
    metrics_.rescale_operations += input.size();
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

uint32_t ServerRuntime::RemainingLevels(const CipherTensor& input) const {
    if (input.empty()) {
        throw std::invalid_argument(
            "cannot inspect remaining levels of an empty CipherTensor");
    }
    ValidateCiphertextMetadata(input, "remaining-level inspection");
    uint64_t maximum_used_levels = 0;
    for (const auto& ciphertext : input.ciphertexts) {
        const uint64_t scale_debt =
            ciphertext->GetNoiseScaleDeg() > 0
            ? ciphertext->GetNoiseScaleDeg() - 1
            : 0;
        maximum_used_levels = std::max(
            maximum_used_levels,
            static_cast<uint64_t>(ciphertext->GetLevel()) + scale_debt);
    }
    if (maximum_used_levels >= profile_.multiplicative_depth) {
        return 0;
    }
    return static_cast<uint32_t>(
        profile_.multiplicative_depth - maximum_used_levels);
}

void ServerRuntime::RequireEvaluationKeys(
    const std::vector<int32_t>& required_rotations,
    bool require_bootstrap) const {
    if (!StrictlyIncreasing(required_rotations)) {
        throw std::invalid_argument(
            "required evaluation-key rotations must be sorted and unique");
    }

    std::shared_lock registry_lock(EvaluationKeyRegistryMutex());
    RequireInstalledEvaluationKeys(
        key_bundle_.key_tag,
        key_bundle_.multiplication_eval_keys,
        key_bundle_.automorphism_eval_keys,
        "evaluation-key preflight");
    for (const int32_t rotation : required_rotations) {
        if (!std::binary_search(
                key_bundle_.rotation_indices.begin(),
                key_bundle_.rotation_indices.end(),
                rotation)) {
            throw std::logic_error(
                "evaluation-key preflight is missing a logical rotation");
        }
        const uint32_t automorphism_index =
            context_->FindAutomorphismIndex(
                static_cast<uint32_t>(rotation));
        if (key_bundle_.automorphism_eval_keys.count(
                automorphism_index) != 1) {
            throw std::logic_error(
                "evaluation-key preflight is missing a rotation automorphism key");
        }
    }

    if (!require_bootstrap) {
        return;
    }
    if (!profile_.bootstrap_enabled ||
        key_bundle_.bootstrap_required_indices.empty()) {
        throw std::logic_error(
            "evaluation-key preflight is missing bootstrap capability");
    }
    for (const uint32_t index : key_bundle_.bootstrap_required_indices) {
        if (key_bundle_.automorphism_eval_keys.count(index) != 1) {
            throw std::logic_error(
                "evaluation-key preflight is missing a bootstrap automorphism key");
        }
    }
}

void ServerRuntime::RequireUsableLevels(
    const CipherTensor& input,
    uint32_t required_levels,
    const std::string& operation) const {
    const uint32_t remaining = RemainingLevels(input);
    if (remaining < required_levels) {
        throw std::runtime_error(
            operation + " requires " + std::to_string(required_levels) +
            " usable levels but only " + std::to_string(remaining) +
            " remain");
    }
}

CipherTensor ServerRuntime::EvaluateChebyshev(
    const CipherTensor& input,
    const ApproximationContract& contract) {
    if (input.empty()) {
        throw std::invalid_argument(
            "cannot evaluate a polynomial on an empty CipherTensor");
    }
    if (contract.basis != "chebyshev" || contract.contract_id.empty() ||
        contract.coefficient_sha256.size() != 64 ||
        contract.degree == 0 || contract.required_depth == 0 ||
        contract.estimated_multiplications == 0 ||
        contract.coefficients.size() != contract.degree + 1 ||
        !std::isfinite(contract.interval.minimum) ||
        !std::isfinite(contract.interval.maximum) ||
        contract.interval.minimum >= contract.interval.maximum ||
        !std::all_of(
            contract.coefficients.begin(),
            contract.coefficients.end(),
            [](double value) { return std::isfinite(value); })) {
        throw std::invalid_argument("invalid Chebyshev approximation contract");
    }
    if (contract.coefficient_sha256 !=
        ComputeCoefficientSha256(contract.coefficients)) {
        throw std::invalid_argument(
            "Chebyshev approximation coefficient hash is stale");
    }
    RequireUsableLevels(input, contract.required_depth, contract.contract_id);
    std::shared_lock registry_lock(EvaluationKeyRegistryMutex());
    RequireInstalledEvaluationKeys(
        key_bundle_.key_tag,
        key_bundle_.multiplication_eval_keys,
        key_bundle_.automorphism_eval_keys,
        "Chebyshev evaluation");

    uint64_t input_used_levels = 0;
    for (const auto& ciphertext : input.ciphertexts) {
        const uint64_t scale_debt =
            ciphertext->GetNoiseScaleDeg() > 0
            ? ciphertext->GetNoiseScaleDeg() - 1
            : 0;
        input_used_levels = std::max(
            input_used_levels,
            static_cast<uint64_t>(ciphertext->GetLevel()) + scale_debt);
    }
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (const auto& ciphertext : input.ciphertexts) {
        result.ciphertexts.push_back(context_->EvalChebyshevSeries(
            ciphertext,
            contract.coefficients,
            contract.interval.minimum,
            contract.interval.maximum));
    }
    RefreshPackingMetadata(result);
    const uint64_t output_scale_debt =
        result.packing.noise_scale_degree > 0
        ? result.packing.noise_scale_degree - 1
        : 0;
    const uint64_t output_used_levels =
        static_cast<uint64_t>(result.packing.level) + output_scale_debt;
    if (output_used_levels < input_used_levels) {
        throw std::logic_error("Chebyshev evaluation moved to an invalid level");
    }
    const uint32_t consumed_depth =
        static_cast<uint32_t>(output_used_levels - input_used_levels);
    if (consumed_depth > contract.required_depth) {
        throw std::runtime_error(
            contract.contract_id + " consumed " +
            std::to_string(consumed_depth) +
            " levels, exceeding its declared depth " +
            std::to_string(contract.required_depth));
    }
    metrics_.chebyshev_evaluations += input.size();
    metrics_.estimated_polynomial_multiplications +=
        contract.estimated_multiplications * input.size();
    metrics_.max_polynomial_depth =
        std::max(metrics_.max_polynomial_depth, consumed_depth);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

void ServerRuntime::PrepareBootstrap() {
    if (!profile_.bootstrap_enabled) {
        throw std::logic_error("bootstrap is not enabled in this profile");
    }
    context_->EvalBootstrapPrecompute(profile_.bootstrap_slots);
    bootstrap_prepared_ = true;
}

CipherTensor ServerRuntime::Bootstrap(const CipherTensor& input) {
    if (!profile_.bootstrap_enabled) {
        throw std::logic_error("bootstrap is not enabled in this profile");
    }
    if (key_bundle_.bootstrap_required_indices.empty()) {
        throw std::logic_error("bootstrap key is unavailable");
    }
    if (input.empty()) {
        throw std::invalid_argument("cannot bootstrap an empty CipherTensor");
    }
    if (input.packing.encoded_slots != profile_.bootstrap_slots ||
        input.packing.active_slots > input.packing.encoded_slots) {
        throw std::invalid_argument(
            "ciphertext packing does not match the bootstrap slot profile");
    }
    ValidateCiphertextMetadata(input, "bootstrap");
    std::shared_lock registry_lock(EvaluationKeyRegistryMutex());
    RequireInstalledEvaluationKeys(
        key_bundle_.key_tag,
        key_bundle_.multiplication_eval_keys,
        key_bundle_.automorphism_eval_keys,
        "bootstrap");
    if (!bootstrap_prepared_) {
        PrepareBootstrap();
    }
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (const auto& ciphertext : input.ciphertexts) {
        result.ciphertexts.push_back(context_->EvalBootstrap(
            ciphertext,
            profile_.bootstrap_iterations,
            profile_.bootstrap_precision));
    }
    metrics_.bootstraps += input.size();
    metrics_.bootstrap_iterations +=
        input.size() * profile_.bootstrap_iterations;
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

}  // namespace moai::openfhe
