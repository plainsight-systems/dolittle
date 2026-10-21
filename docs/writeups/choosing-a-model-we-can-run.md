+++
title = 'Choosing a model we can run'
description = 'Why I chose the Qwen3 8B NLA pair, and how three models fit on a 24 GB GPU.'
date = '2026-10-16T08:00:00Z'
lastmod = '2026-10-21T00:00:00-07:00'
article_type = 'exploration'
+++

I wanted a model that generates thinking text and a natural language autoencoder
(NLA) trained to read its activations. I also wanted the experiment to run on consumer hardware. The
machine available for the work was an RTX 3090 Ti with 24 GB of VRAM, so the
pair had to fit there.

Two Qwen3 pairs were worth trying. Both support thinking, but they package
their NLA weights differently. That affects how much memory they need and how
the application switches between generating text and interpreting it.

## What we needed to run

There are three model roles in this experiment. The **target** generates the
thinking text and answer. The NLA’s **activation verbalizer** reads a vector
from inside the target and describes it in words. Its **activation
reconstructor** takes those words and predicts the vector they describe.
That last step lets us turn an edited explanation back into numbers we can
use to steer the target.

Before building the engine, we tried the published weights in Python with
PyTorch and Hugging Face Transformers. PyTorch runs the tensor operations;
Transformers supplies the model implementation and generation code. We added
a hook to the chosen transformer block to save its output at selected token
positions. For the 8B model, that saved the 4,096-value residual stream after
block 24’s attention and feed-forward updates had both been added.
We then passed the saved vector to the verbalizer and its description to the
reconstructor.

This let us check memory use, generation time and the descriptions themselves
before choosing which pair to implement in Dolittle.

## Two available pairs

The first experiment used Qwen3 4B with anicka’s
[verbalizer](https://huggingface.co/anicka/nla-qwen3-4b-universal-av-grpo) and
[reconstructor](https://huggingface.co/anicka/nla-qwen3-4b-universal-ar).
They use LoRA adapters, smaller sets of trained changes applied to the base
model. One loaded base can serve all three roles by switching adapters. This
pair can read activations at multiple layers.

The second used Qwen3 8B with
[marco-bazzani’s NLA pair](https://huggingface.co/marco-bazzani/Qwen3-8B-nla).
Its verbalizer has a full set of separately trained weights. Its reconstructor
has 28 transformer blocks followed by a layer that produces a 4,096-number
activation vector. The verbalizer and reconstructor were trained together to
reconstruct activations from block 24 of the target model.

The 4B pair can reuse one base model in memory. The 8B pair needs separate
weights for the target, verbalizer and reconstructor. That difference determines
how we run them on the GPU.

## One GPU, three models

Together, the 8B target and its two NLA components need about 42 GiB of BF16
weight storage. The PyTorch experiment fitted them into 24 GB of VRAM by loading
one role at a time: generate the response and save its activations, release
the target, then load the verbalizer or reconstructor when needed.

The saved text and activations remain available while models switch. We can
therefore read a response, select a token, and request an explanation without
keeping all three models on the GPU. The 8B weights stay at BF16 precision
throughout, without quantization.

## What the PyTorch experiments measured

Both pairs ran four prompts: marble arithmetic, tracking a key’s location,
writing a rabbit couplet, and a syllogism about cedar trees. The target models
used matching sampling settings and output budgets, each generating its own
response.

The table records those initial Python runs, using PEFT to load the 4B
adapters. Interpretation times exclude model loading; these are the measurements
used to choose the model, before the custom engine existed.

| Measurement | 4B pair | 8B pair |
| --- | --- | --- |
| Peak PyTorch allocation | 7.9 GiB | 15.4 GiB |
| Target generation | 27–30 tokens/s | About 30 tokens/s |
| Median interpretation generation | 10.3 seconds | 4.7 seconds |
| Thinking interpretations reaching the 160-token cap | 20 of 24 | 0 of 8 |
| Checkpoint download | About 8.4 GB | About 45 GB |
| Capture depths | Multiple | Block 24 |

The 8B verbalizer finished all eight thinking interpretations and four
prompt-end samples within the budget. Most of the 4B thinking interpretations
hit the limit. For an interactive demo, waiting for a complete explanation
made the 8B result attractive despite its larger weights.

All four 8B target generations used their 512-token budget while still
thinking. That budget was enough to collect activations from the thinking
stream, but too short to get their final answers.

## What the descriptions said

The 8B descriptions recognized broad tasks and some of the local structure.
A poem-editing activation was described as rejecting a phrase. A key-location
activation was described as resolving where the key was. They also introduced
names, countries and objects absent from the prompt.

In the [key example]({{< relref "/writing/two-views-of-a-thinking-model#when-the-views-disagree" >}}),
the NLA described the right kind of task while changing the story around it.
That mix of recognizable subject and invented detail gave us a reason to
compare several samples and token positions in the interface.

## Why I chose 8B

The 8B pair uses joint verbalizer/reconstructor training, its explanations
finished within the 160-token interpretation budget, and it fits the available GPU when the
roles run separately. Those were the reasons I chose it.

The cost is a larger download and time spent switching model weights. It also
gives us one trained capture layer, block 24. That is enough to run the
experiment: generate thinking, read an activation, edit its explanation and
see how the next output changes.

## From the Python experiment to Dolittle

*21 October 2026*

PyTorch already gave us access to the activations. Hooks can also replace a
block’s output, so we could have built the interaction around that Python
implementation. Building the inference engine was part of what I wanted to
explore, as it was with Charlotte and Serenity. I wanted to implement the
computation that makes the experiment work and give people a way to interact
with it.

Dolittle’s custom harness runs the published weights through a C++/CUDA
transformer. It handles loading, attention, the cache of earlier tokens, and
sampling the next token. Capture hooks save the activations for inspection;
a steering hook changes the selected activation before the next block runs.
The transformer computation stays on the GPU, while the CPU prepares requests,
encodes and decodes text, and passes results to the browser.

Owning that execution path lets us work directly on GPU scheduling, memory
reuse and model switching. The same engine runs the target and both NLA
components, one role at a time, to fit the 24 GB card.

PyTorch remains useful as an independent numerical reference: it produces
expected activations and outputs that we compare with the custom engine.
That is a separate job from the model-selection measurements above. It checks
whether our implementation performs the intended computation.

The application now gives thinking and the answer a shared default budget of
4,096 tokens. The [rhyming walkthrough]({{< relref "/writing/running-an-investigation" >}})
covers running the engine, inspecting an activation and trying an edit.
