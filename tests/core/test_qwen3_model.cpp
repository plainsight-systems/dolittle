/*
 * Axis D3: isolated Qwen3 model-description and recipe-generation contracts.
 * Contract: use small configurations and inspect public owned layouts, semantic
 * parameter roles, observation sites and requested output shapes. Invalid
 * config, unsupported modes and selection errors must throw documented
 * exception types. No numerical execution, real checkpoints, recipe-validator
 * call, filesystem, device work, subprocesses or assertions about private
 * tensor IDs/op counts. Algorithm: docs/architecture/transformer-execution.md,
 * "One Qwen3 transformer block" and "First forward pass": construct varied
 * model descriptions, request complete recipes with/without selected logits,
 * inspect their public boundary metadata, then destroy caller/model metadata to
 * check ownership of returned public layouts and sites. Equal-valued adjacent
 * boundaries are verified numerically by GPU integration, without requiring
 * identical tensor IDs. Design and cost: bounded in-memory cases; linear
 * metadata inspection per fixture, except semantic-role uniqueness checks.
 * Overflow cases allocate no tensor payload. Numerical equivalence belongs to
 * independent GPU verification. Corpus consultation, implementation,
 * 2026-10-18: cpp-guidelines I.4 (#ri-typed), I.7 (#ri-post): compare semantic
 * contracts and strongly typed boundary identities, not a private instruction
 * schedule. cpp-guidelines ES.103 (#res-overflow): exercise public size/error
 * boundaries. cpp-perf-guidelines CDSA.32: returned preparation metadata owns
 * its layouts; returned public layouts/sites survive caller and model
 * destruction. Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

#include "models/qwen3/model.h"
#include "tests/support/check.h"

namespace {
namespace tensors = dolittle::tensors;
namespace models = dolittle::models;
namespace tests = dolittle::tests;
using models::qwen3::Config;
using models::qwen3::make_model;
using tensors::ElementType;
using tests::check;
using tests::expect_throws;

Config configuration() {
  return {3, 6, 4, 2, 2, 10, 17, {8}, 1e-6F, 10000.0, ElementType::bfloat16};
}

const models::ParameterSpec& role(const models::Model& model,
                                  const std::string& name) {
  const auto params = model.parameters();
  const auto found = std::find_if(
      params.begin(), params.end(),
      [&](const auto& parameter) { return parameter.role == name; });
  check(found != params.end(), "required semantic parameter role");
  return *found;
}

const models::ObservationValue& observation(
    const models::ForwardRecipe& recipe, models::ObservationBoundary boundary,
    std::size_t block) {
  const auto found =
      std::find_if(recipe.observations.begin(), recipe.observations.end(),
                   [&](const auto& value) {
                     return value.site.point.boundary == boundary &&
                            value.site.point.block.value == block;
                   });
  check(found != recipe.observations.end(), "every block boundary is present");
  return *found;
}

struct RecipeExtent {
  std::size_t length;
  std::size_t selected_rows;
};
void inspect_recipe(const Config& config, const models::ForwardRecipe& recipe,
                    RecipeExtent extent) {
  const auto [length, selected_rows] = extent;
  check(recipe.token_input.value < recipe.tensors.size(),
        "token layout exists");
  const auto& token = recipe.tensors.at(recipe.token_input.value);
  check(token.element_type == ElementType::uint32 &&
            token.shape == std::vector{length},
        "token input shape/type");
  check(recipe.observations.size() == config.block_count * 2,
        "exactly input and output sites for every block");
  for (std::size_t i = 0; i < config.block_count; ++i) {
    for (const auto boundary : {models::ObservationBoundary::block_input,
                                models::ObservationBoundary::block_output}) {
      const auto& value = observation(recipe, boundary, i);
      check(value.site.vector_width == config.residual_width &&
                value.site.storage_type == config.weight_type,
            "site metadata follows config");
      check(value.tensor.value < recipe.tensors.size(),
            "observation tensor exists");
      const auto& layout = recipe.tensors.at(value.tensor.value);
      check(layout.shape == std::vector{length, config.residual_width} &&
                layout.element_type == config.weight_type,
            "observation tensor shape/type");
    }
  }
  check(recipe.logits.has_value() == (selected_rows != 0),
        "requested logits presence");
  if (recipe.logits) {
    check(recipe.logits->value < recipe.tensors.size(), "logit tensor exists");
    check(recipe.tensors.at(recipe.logits->value).shape ==
              std::vector{selected_rows, config.vocabulary_size},
          "selected-logit shape");
  }
}

void descriptions_and_owned_recipes() {
  for (const auto type : {ElementType::bfloat16, ElementType::float32}) {
    auto config = configuration();
    config.weight_type = type;
    auto model = make_model(config);
    const tensors::ComputationMode mode{type, ElementType::float32};
    check(!model->identity().empty(), "versioned model identity");
    check(model->context_limit().value == config.context_limit.value &&
              model->vocabulary_size() == config.vocabulary_size,
          "owned configuration metadata");
    std::set<std::string> roles;
    for (const auto& parameter : model->parameters()) {
      check(roles.insert(parameter.role).second,
            "unique semantic parameter roles");
      check(parameter.layout.element_type == type, "parameter dtype");
    }
    check(role(*model, "token_embedding").layout.shape ==
              std::vector{config.vocabulary_size, config.residual_width},
          "embedding shape");
    check(role(*model, "vocabulary_projection").layout.shape ==
              std::vector{config.vocabulary_size, config.residual_width},
          "vocabulary shape");
    check(role(*model, "final_norm").layout.shape ==
              std::vector{config.residual_width},
          "final norm shape");
    for (std::size_t i = 0; i < config.block_count; ++i) {
      const auto prefix = "blocks." + std::to_string(i) + ".";
      check(role(*model, prefix + "query_projection").layout.shape ==
                std::vector{config.query_heads * config.head_width,
                            config.residual_width},
            "query projection width need not equal residual width");
      check(role(*model, prefix + "key_projection").layout.shape ==
                std::vector{config.kv_heads * config.head_width,
                            config.residual_width},
            "grouped key projection shape");
      check(role(*model, prefix + "attention_projection").layout.shape ==
                std::vector{config.residual_width,
                            config.query_heads * config.head_width},
            "attention output maps head width back to residual width");
    }
    const auto sites = model->observation_sites(mode);
    check(sites.size() == config.block_count * 2,
          "advertise all configured blocks");
    std::array selection{models::TokenPosition{0}, models::TokenPosition{3}};
    const auto recipe = model->full_forward({4}, mode, selection);
    selection.at(0).value = 99;
    const auto no_logits = model->full_forward({4}, mode, {});
    const auto expected_config = config;
    config.block_count = 1;
    config.residual_width = 2;
    model.reset();
    inspect_recipe(expected_config, recipe, {.length = 4, .selected_rows = 2});
    inspect_recipe(expected_config, no_logits,
                   {.length = 4, .selected_rows = 0});
  }
}

void invalid_configuration() {
  constexpr std::array dimensions{
      &Config::block_count,    &Config::residual_width, &Config::query_heads,
      &Config::kv_heads,       &Config::head_width,     &Config::mlp_width,
      &Config::vocabulary_size};
  for (auto member : dimensions) {
    auto config = configuration();
    config.*member = 0;
    expect_throws<std::invalid_argument>([&] { (void)make_model(config); });
  }
  for (const auto mutate : std::array<void (*)(Config&), 7>{
           [](Config& c) { c.context_limit.value = 0; },
           [](Config& c) { c.query_heads = 3; },
           [](Config& c) { c.head_width = 3; },
           [](Config& c) { c.rms_epsilon = 0; },
           [](Config& c) { c.rope_theta = -1; },
           [](Config& c) {
             c.rms_epsilon = std::numeric_limits<float>::quiet_NaN();
           },
           [](Config& c) { c.weight_type = ElementType::uint32; }}) {
    auto config = configuration();
    mutate(config);
    expect_throws<std::invalid_argument>([&] { (void)make_model(config); });
  }
  auto config = configuration();
  config.block_count = std::numeric_limits<std::size_t>::max();
  expect_throws<std::length_error>([&] { (void)make_model(config); });
  config = configuration();
  config.residual_width = std::numeric_limits<std::size_t>::max();
  expect_throws<std::length_error>([&] { (void)make_model(config); });
}

void invalid_requests() {
  const auto model = make_model(configuration());
  const tensors::ComputationMode mode{ElementType::bfloat16,
                                      ElementType::float32};
  for (const auto length : {std::size_t{0}, std::size_t{9}}) {
    expect_throws<std::invalid_argument>(
        [&] { (void)model->full_forward({length}, mode, {}); });
  }
  const std::array invalid_selections{
      std::array{models::TokenPosition{0}, models::TokenPosition{0}},
      std::array{models::TokenPosition{2}, models::TokenPosition{1}},
      std::array{models::TokenPosition{0}, models::TokenPosition{4}},
  };
  for (const auto& selection : invalid_selections) {
    expect_throws<std::invalid_argument>(
        [&] { (void)model->full_forward({4}, mode, selection); });
  }
  for (const auto bad_mode :
       {tensors::ComputationMode{ElementType::float32, ElementType::float32},
        tensors::ComputationMode{ElementType::bfloat16,
                                 ElementType::bfloat16}}) {
    expect_throws<std::invalid_argument>(
        [&] { (void)model->observation_sites(bad_mode); });
    expect_throws<std::invalid_argument>(
        [&] { (void)model->full_forward({4}, bad_mode, {}); });
  }
}

void context_and_vocabulary_boundaries() {
  auto config = configuration();
  config.block_count = 1;
  config.query_heads = config.kv_heads;
  const tensors::ComputationMode mode{ElementType::bfloat16,
                                      ElementType::float32};
  auto model = make_model(config);
  const std::array final_row{
      models::TokenPosition{config.context_limit.value - 1}};
  inspect_recipe(config,
                 model->full_forward(config.context_limit, mode, final_row),
                 {.length = config.context_limit.value, .selected_rows = 1});

  config.context_limit.value = std::numeric_limits<std::size_t>::max();
  model = make_model(config);
  expect_throws<std::length_error>(
      [&] { (void)model->full_forward(config.context_limit, mode, {}); });
  if constexpr (std::numeric_limits<std::size_t>::digits > 32) {
    config = configuration();
    config.vocabulary_size =
        std::size_t{std::numeric_limits<std::uint32_t>::max()} + 1;
    model = make_model(config);
    check(model->vocabulary_size() == config.vocabulary_size,
          "UINT32_MAX remains a representable final token ID");
    ++config.vocabulary_size;
    expect_throws<std::invalid_argument>([&] { (void)make_model(config); });
  }
}
}  // namespace

int main() try {
  descriptions_and_owned_recipes();
  invalid_configuration();
  invalid_requests();
  context_and_vocabulary_boundaries();
  const auto model = make_model(configuration());
  const tensors::ComputationMode mode{ElementType::bfloat16,
                                      ElementType::float32};
  auto recipe = model->incremental_forward({4}, mode);
  models::validate_incremental_recipe(*model, recipe, {4}, mode);
  auto missing = recipe;
  missing.cache_appends.pop_back();
  expect_throws<std::invalid_argument>(
      [&] { models::validate_incremental_recipe(*model, missing, {4}, mode); });
  auto wrong = recipe;
  wrong.cache_appends.front().value = wrong.cache_appends.front().rotated_key;
  expect_throws<std::invalid_argument>(
      [&] { models::validate_incremental_recipe(*model, wrong, {4}, mode); });
  auto invalid_tail = recipe;
  invalid_tail.logits_begin = 0;
  expect_throws<std::invalid_argument>([&] {
    models::validate_incremental_recipe(*model, invalid_tail, {4}, mode);
  });
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
