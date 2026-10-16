+++
title = 'Choosing a model we can run'
description = 'Comparing two Qwen3/NLA pairs on a 3090 Ti, and why the 8B pair is the starting point for Dolittle.'
date = '2026-10-16T08:00:00Z'
article_type = 'exploration'
+++

The model choice has two parts. Dolittle needs a target that generates a
thinking stream, and an NLA trained to read activations from that particular
target. A popular model is only half a usable pairing.

I also want people to be able to run the result on capable consumer hardware.
The reference machine is an RTX 3090 Ti with 24 GB of VRAM. That is a useful
constraint: it gives the implementation a real machine to fit, and gives
anyone following along a concrete basis for judging their own hardware.

## Two available pairs

The first reference run used Qwen3 4B with anicka’s
[verbalizer](https://huggingface.co/anicka/nla-qwen3-4b-universal-av-grpo) and
[reconstructor](https://huggingface.co/anicka/nla-qwen3-4b-universal-ar).
They use LoRA adapters over the same base weights. One loaded base can serve
the target and both NLA roles in sequence, with the appropriate adapter
enabled for each role. This pair supports interpretations at multiple depths.

The second used Qwen3 8B with
[marco-bazzani’s NLA pair](https://huggingface.co/marco-bazzani/Qwen3-8B-nla).
Its verbalizer is a full fine-tune. Its reconstructor has 28 transformer blocks
and a separate linear output head that predicts a 4,096-dimensional vector.
The verbalizer and reconstructor were trained together using a reconstruction
reward. This NLA reads one target block, numbered 24 from zero.

Those are different execution contracts. Sharing the Qwen3 name does not let
us exchange their activations, adapters, or reconstruction conventions.

## One GPU, separate phases

Together, the 8B target, verbalizer, and reconstructor require about 42 GiB
of BF16 weight storage. They do not fit together in 24 GB of VRAM. The
reference run instead loaded each role separately: generate the target trace
and retain selected activations, release the target, load the verbalizer,
then run the reconstructor.

This fits the machine without quantization. It also makes weight loading part
of the eventual interaction design. Showing thinking text and interpretations
side by side does not require both to finish at the same time, but a visitor
still needs to understand what the application is working on.

## What the reference runs measured

Both runs used four prompts: marble arithmetic, tracking a key’s location,
writing a rabbit couplet, and evaluating a syllogism about cedar trees. The
target sampling settings and output caps were the same. Each model generated
its own trace, so the captured positions were not identical reasoning events.

| Measurement | 4B pair | 8B pair |
| --- | --- | --- |
| Peak PyTorch allocation | 7.9 GiB | 15.4 GiB |
| Target generation | 27–30 tokens/s | About 30 tokens/s |
| Median interpretation generation | 10.3 seconds | 4.7 seconds |
| Thinking interpretations reaching the 160-token cap | 20 of 24 | 0 of 8 |
| Checkpoint download | About 8.4 GB | About 45 GB |
| Capture depths | Multiple | Block 24 |

These are reference measurements using PyTorch, Transformers, and PEFT where
needed. They are not measurements of Dolittle’s future engine. The verbalizer
timings exclude weight loading, and the two pairs produce different lengths
of text through different execution paths. The result does not establish that
an 8B model is generally faster than a 4B model.

The 8B verbalizer completed all eight thinking interpretations and four
prompt-end controls within the cap. All four target generations reached
their 512-token limit while still thinking. That leaves us with useful
captures, but no answer-quality comparison from this short run.

## What the larger model did not settle

The 8B descriptions recognized broad tasks and sometimes the local structure
of the generated text. A poem-editing capture was described as rejecting a
phrase; a key-location capture was described as resolving where the key was.
They also substituted names, quantities, countries, and objects absent from
the visible context.

The [key example]({{< relref "/writing/two-views-of-a-thinking-model#when-the-views-disagree" >}})
shows the distinction. It is possible to recognize the kind of reasoning in
progress and still describe particulars that the transcript does not support.
These runs do not establish that either NLA is more semantically faithful.
Their raw reconstruction scores are also measured in different activation
spaces, so those numbers cannot serve as a shared accuracy scale.

## The choice for Dolittle

I selected the 8B pair. Its joint reconstruction training is closer to the
approach in Anthropic’s paper, its descriptions completed within the
reference limit, and the workload fits the GPU we have. The cost is a larger
download, separate model weights to manage, and a single supported capture
depth.

That gives the next stage a concrete starting point. The engine needs to
reproduce the target’s inference, capture the block 24 residual at the right
position, inject it into the verbalizer, and apply the reconstructor’s trained
head correctly. Those are implementation contracts we can verify. The
meaning of the resulting descriptions remains something to explore through
the application and the writing around it.
