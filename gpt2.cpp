#include <unistd.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "cladtorch.hpp"
#include "llm/dataloader.h"
#include "llm/tokenizer.h"

using namespace cladtorch;

class GPT2Config {
public:
  int max_seq_len;
  int vocab_size;        // vocab size, e.g. 50257
  int padded_vocab_size; // padded to e.g. %128==0, 50304
  int num_layers;
  int num_heads;
  int channels;
};

class Block {
public:
  Tensor *ln1w = nullptr; // layernorm weights, (channels)
  Tensor *ln1b = nullptr; // layernorm biases, (channels)
  Tensor *qkvw = nullptr; // query, key, value weights, (3 * channels, channels)
  Tensor *qkvb = nullptr; // query, key, value biases, (3 * channels)
  Tensor *attprojw =
      nullptr; // attention projection weights, (channels, channels)
  Tensor *attprojb = nullptr; // attention projection biases, (channels)
  Tensor *ln2w = nullptr;     // layernorm weights, (channels)
  Tensor *ln2b = nullptr;     // layernorm biases, (channels)
  Tensor *fcw = nullptr; // fully connected weights, (4 * channels, channels)
  Tensor *fcb = nullptr; // fully connected biases, (4 * channels)
  Tensor *fcprojw =
      nullptr; // fully connected projection weights, (channels, 4 * channels)
  Tensor *fcprojb = nullptr; // fully connected projection biases, (channels)
};

class Embedding {
public:
  Tensor *wte = nullptr; // word token embeddings, (padded_vocab_size, channels)
  Tensor *wpe = nullptr; // word position embeddings, (max_seq_len, channels)
};

class LMHead {
public:
  Tensor *lnfw = nullptr; // layernorm weights, (channels)
  Tensor *lnfb = nullptr; // layernorm biases, (channels)
};

class GPT2 {
public:
  GPT2Config config;
  TensorContext *ctx = nullptr; // tensor memory context for the model

  // the weights of the model
  Embedding embedding;          // the embedding layer
  std::vector<Block> blocks;    // the transformer blocks
  LMHead lm_head;               // the language model head
  std::vector<Tensor *> params; // all the parameters of the model, the layout
                                // is the same as the checkpoint file
  int num_parameters = 0;

  int batch_size = 0; // the batch size (B) of current forward pass
  int seq_len = 0;    // the sequence length (T) of current forward pass

  Tensor *input = nullptr;     // the input tensor, (B, T)
  Tensor *input_pos = nullptr; // the input position tensor, (B, T)
  Tensor *target = nullptr;    // the target tensor, (B, T)

  Tensor *logits = nullptr; // the logits tensor, (B, T, padded_vocab_size)
  Tensor *probs = nullptr;  // the probs tensor, (B, T, padded_vocab_size)
  Tensor *losses = nullptr; // the losses tensor, (B, T)
  float mean_loss = 0.0f;

  // buffers for the AdamW optimizer
  std::vector<float> m_memory;
  std::vector<float> v_memory;

  ~GPT2() { delete ctx; }

  void zero_grad() { losses->zero_grad(); }
  void backward() {
    float dloss_mean = 1.0f / (batch_size * seq_len);
    losses->backward(true, dloss_mean);
  }
  void update(float learning_rate, float beta1, float beta2, float eps,
              float weight_decay, int t) {
    if (m_memory.empty()) {
      assert(num_parameters > 0);
      m_memory = std::vector<float>(num_parameters, 0.0f);
      v_memory = std::vector<float>(num_parameters, 0.0f);
    }

    int idx = 0;
    for (auto param : params) {
      auto weights = (float *)param->data();
      auto grads = (float *)param->grad()->data();

      for (int i = 0; i < param->num_elements(); i++) {
        auto w = weights[i], g = grads[i];
        // update the first moment (momentum)
        float m = beta1 * m_memory[idx] + (1.0f - beta1) * g;
        // update the second moment (RMSprop)
        float v = beta2 * v_memory[idx] + (1.0f - beta2) * g * g;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));
        // update
        m_memory[idx] = m;
        v_memory[idx] = v;
        weights[i] -=
            learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);

        idx++;
      }
    }
  }

  void forward(const int *inputs, const int *targets, int B, int T) {
    if (num_parameters == 0) {
      throw std::runtime_error("Model has not been initialized");
    }

    const auto &config = this->config;
    int HS = config.channels / config.num_heads;

    // First time forward pass, create the computation graph
    if (!input) {
      batch_size = B;
      seq_len = T;
      input = ctx->new_tensor({B, T}, TensorType::I32);

      // Create the position tensor using std::vector and iota/transform if
      // desired for complex logic
      std::vector<int> pos_p(B * T);
      for (int i = 0; i < B; ++i) {
        for (int j = 0; j < T; ++j) {
          pos_p[i * T + j] = j;
        }
      }
      input_pos = ctx->new_tensor({B, T}, TensorType::I32)->fill(pos_p);
      target = ctx->new_tensor({B, T}, TensorType::I32);

      Tensor *residual =
          &((*embedding.wte)[*input] + (*embedding.wpe)[*input_pos]);

      for (Block &block : blocks) {
        Tensor &ln1 = residual->norm() * *block.ln1w + *block.ln1b; // (B, T, C)
        Tensor &qkv = ln1.matmul(*block.qkvw) + *block.qkvb; // (B, T, 3 * C)
        const auto qkv_split = qkv.split(config.channels, 2);

        // Multi-head attention - using auto& for readability and efficiency
        auto &q =
            qkv_split[0]->view({B, T, config.num_heads, HS}).transpose(1, 2);
        auto &k =
            qkv_split[1]->view({B, T, config.num_heads, HS}).transpose(1, 2);
        auto &v =
            qkv_split[2]->view({B, T, config.num_heads, HS}).transpose(1, 2);
        Tensor *attn = &(q.matmul(k) * (1.0f / std::sqrt(HS))); // (B, NH, T, T)

        attn = &attn->softmax(true);             // Mask future tokens
        attn = &attn->matmul(v.transpose(2, 3)); // (B, NH, T, HS)
        attn =
            &attn->transpose(1, 2).view({B, T, config.channels}); // (B, T, C)
        Tensor &attn_proj =
            attn->matmul(*block.attprojw) + *block.attprojb; // (B, T, C)

        Tensor &residual2 = *residual + attn_proj;                  // (B, T, C)
        Tensor &ln2 = residual2.norm() * *block.ln2w + *block.ln2b; // (B, T, C)

        // Feed forward
        Tensor &fc = ln2.matmul(*block.fcw) + *block.fcb; // (B, T, 4 * C)
        Tensor &gelu = fc.gelu();                         // (B, T, 4 * C)
        Tensor &fc_proj =
            gelu.matmul(*block.fcprojw) + *block.fcprojb; // (B, T, C)
        residual = &(residual2 + fc_proj);                // (B, T, C)
      }

      Tensor &lnf =
          residual->norm() * *lm_head.lnfw + *lm_head.lnfb; // (B, T, C)
      logits = &lnf.matmul(*embedding.wte);                 // (B, T, Vp)
      probs = &logits->softmax(false, config.vocab_size);   // (B, T, Vp)
      losses = &probs->cross_entropy(*target);              // (B, T)

      ctx->print_layout();
      std::cout << "Computation Graph created successfully!" << std::endl;

    } else {
      // Input size validation remains similar, but slightly simplified
      // condition
      if (batch_size != B || seq_len != T) {
        if (targets != nullptr) {
          throw std::runtime_error("Dynamic batch size or sequence length not "
                                   "supported for training.");
        } else if (B > batch_size ||
                   T > seq_len) { // Combined else-if for clarity
          throw std::runtime_error("Input batch size or sequence length "
                                   "exceeds model graph dimensions.");
        }
        // Inference with potentially different but smaller batch/seq is
        // implicitly allowed now if no exception
      }
    }

    input->fill(inputs);

    if (targets) {
      target->fill(targets);
      losses->forward();

      float loss_sum = 0.0f;
      const float *loss_data = reinterpret_cast<float *>(losses->data());
      for (int i = 0; i < B * T; ++i) {
        loss_sum += loss_data[i];
      }
      mean_loss = loss_sum / (B * T);
    } else {
      probs->forward();
      mean_loss = -1.0f;
    }
  }
};

void gpt2_build_from_checkpoint(GPT2 *model,
                                const std::string &checkpoint_path) {
  FILE *model_file = fopen(checkpoint_path.c_str(), "rb");
  if (!model_file) {
    throw std::runtime_error("Could not open the model checkpoint file: " +
                             checkpoint_path);
  }
  std::unique_ptr<FILE, decltype(&fclose)> file_ptr(model_file, fclose);

  int model_header[256];
  size_t read_header_items = fread(model_header, sizeof(int), 256, model_file);
  if (read_header_items != 256) {
    throw std::runtime_error(
        "Failed to read model header from checkpoint file: " + checkpoint_path);
  }
  if (model_header[0] != 20240326) {
    throw std::runtime_error("Bad magic number in model checkpoint file: " +
                             checkpoint_path);
  }
  if (model_header[1] != 3) {
    throw std::runtime_error("Bad version number in model checkpoint file: " +
                             checkpoint_path);
  }

  // Read in hyperparameters
  auto &config = model->config;
  config.max_seq_len = model_header[2];
  config.vocab_size = model_header[3];
  config.num_layers = model_header[4];
  config.num_heads = model_header[5];
  config.channels = model_header[6];
  config.padded_vocab_size = model_header[7];

  // Print the hyperparameters - using structured binding for cleaner access
  std::cout << "[GPT-2]:\n"
            << "max_seq_len: " << config.max_seq_len << "\n"
            << "vocab_size: " << config.vocab_size << "\n"
            << "padded_vocab_size: " << config.padded_vocab_size << "\n"
            << "num_layers: " << config.num_layers << "\n"
            << "num_heads: " << config.num_heads << "\n"
            << "channels: " << config.channels << std::endl;

  if (model->ctx != nullptr) {
    throw std::runtime_error("Model context already initialized.");
  }
  // Initialize the parameter tensor sizes
  model->ctx = new TensorContext(8UL * 1024 * 1024 * 1024);
  auto &ctx = model->ctx;

  model->embedding.wte =
      ctx->new_tensor({config.padded_vocab_size, config.channels});
  model->embedding.wpe = ctx->new_tensor({config.max_seq_len, config.channels});
  model->params.insert(model->params.end(),
                       {model->embedding.wte, model->embedding.wpe});

  std::vector<std::vector<Tensor *>> block_params;
  // Reserve space for blocks to avoid reallocation
  model->blocks.reserve(config.num_layers);
  for (int l = 0; l < config.num_layers; ++l) {
    model->blocks.emplace_back(Block{
        .ln1w = ctx->new_tensor({config.channels}),
        .ln1b = ctx->new_tensor({config.channels}),
        .qkvw = ctx->new_tensor({3 * config.channels, config.channels}),
        .qkvb = ctx->new_tensor({3 * config.channels}),
        .attprojw = ctx->new_tensor({config.channels, config.channels}),
        .attprojb = ctx->new_tensor({config.channels}),
        .ln2w = ctx->new_tensor({config.channels}),
        .ln2b = ctx->new_tensor({config.channels}),
        .fcw = ctx->new_tensor({4 * config.channels, config.channels}),
        .fcb = ctx->new_tensor({4 * config.channels}),
        .fcprojw = ctx->new_tensor({config.channels, 4 * config.channels}),
        .fcprojb = ctx->new_tensor({config.channels})});
    // Get reference to the newly added block
    auto &block = model->blocks.back();
    block_params.push_back({block.ln1w, block.ln1b, block.qkvw, block.qkvb,
                            block.attprojw, block.attprojb, block.ln2w,
                            block.ln2b, block.fcw, block.fcb, block.fcprojw,
                            block.fcprojb});
  }

  // Parameter loading order is still as before
  for (size_t i = 0; i < block_params[0].size(); ++i) {
    for (int l = 0; l < config.num_layers; ++l) {
      model->params.push_back(block_params[l][i]);
    }
  }

  model->lm_head.lnfw = ctx->new_tensor({config.channels});
  model->lm_head.lnfb = ctx->new_tensor({config.channels});
  model->params.insert(model->params.end(),
                       {model->lm_head.lnfw, model->lm_head.lnfb});

  // Load the parameters
  model->num_parameters = 0;
  for (Tensor *t : model->params) {
    model->num_parameters += t->num_elements();
    size_t read_params_items =
        fread(t->data(), sizeof(float), t->num_elements(), model_file);
    if (read_params_items != t->num_elements()) {
      throw std::runtime_error(
          "Failed to read all parameters for tensor from checkpoint file: " +
          checkpoint_path);
    }
  }

  std::cout << "Number of Parameters: " << model->num_parameters << std::endl;

  ctx->print_layout();
  std::cout << "Checkpoint loaded successfully!" << std::endl;
}

uint32_t random_u32(uint64_t *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// random float32 in [0, 1)
float random_f32(uint64_t *state) {
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float *probs, int n, float coin) {
  // sample index from probs (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probs[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int main() {
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "llm_c/gpt2_124M.bin");

  // Build data loaders - using auto for type inference
  auto tiny_shakespeare_train =
      std::string("llm_c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin");
  auto tiny_shakespeare_val =
      std::string("llm_c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin");
  auto train_token = tiny_shakespeare_train;
  auto val_token = tiny_shakespeare_val;
  size_t batch_size = 4;
  size_t seq_len = 64;
  DataLoader train_loader, val_loader;
  dataloader_init(&train_loader, train_token.c_str(), batch_size, seq_len, 0, 1,
                  1);
  dataloader_init(&val_loader, val_token.c_str(), batch_size, seq_len, 0, 1, 0);
  std::cout << "train dataset num_batches: "
            << train_loader.num_tokens / (batch_size * seq_len) << std::endl;
  std::cout << "val dataset num_batches: "
            << val_loader.num_tokens / (batch_size * seq_len) << std::endl;
  int val_num_batches = 5;

  // Build the tokenizer
  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "llm_c/gpt2_tokenizer.bin");

  // Generation setup
  uint64_t rng_state = 1337;
  constexpr int gen_max_length = 64;
  std::vector<int> gen_tokens(batch_size * seq_len);

  // Training loop
  timespec start, end;
  for (int step = 0; step <= 40; ++step) {
    // Validation loss estimation
    if (step % 10 == 0) {
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; ++i) {
        dataloader_next_batch(&val_loader);
        model.forward(val_loader.inputs, val_loader.targets, batch_size,
                      seq_len);
        val_loss += model.mean_loss;
      }
      val_loss /= val_num_batches;
      std::cout << "val loss: " << val_loss << std::endl;
    }

    // Model inference and text generation
    if (step > 0 && step % 20 == 0) {
      std::fill(gen_tokens.begin(), gen_tokens.end(), tokenizer.eot_token);

      std::cout << "generating:\n---\n";
      for (int t = 1; t < gen_max_length; ++t) {
        model.forward(gen_tokens.data(), nullptr, batch_size, seq_len);
        float *probs = reinterpret_cast<float *>(model.probs->data()) +
                       (t - 1) * model.config.padded_vocab_size;
        float coin = random_f32(&rng_state);
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
        if (tokenizer.init_ok) {
          auto token_str = tokenizer_decode(&tokenizer, next_token);
          safe_printf(token_str);
        } else {
          std::cout << next_token << " ";
        }
        std::cout << std::flush;
      }
      std::cout << "\n---\n";
    }

    // Training step
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    model.forward(train_loader.inputs, train_loader.targets, batch_size,
                  seq_len);
    model.zero_grad();
    model.backward();
    model.update(1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    std::cout << "step " << step << " train Loss: " << model.mean_loss
              << " (took " << time_elapsed_s * 1000 << " ms)" << std::endl;
  }

  tokenizer_free(&tokenizer);
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  return 0;
}
