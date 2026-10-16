+++
title = 'Two views of a thinking model'
description = 'What generated thinking tells us, what an activation verbalizer adds, and how to read the places where they disagree.'
date = '2026-10-16T09:00:00Z'
article_type = 'explanation'
+++

Dolittle started with [Anthropic’s paper on natural language autoencoders](https://transformer-circuits.pub/2026/nla/index.html).
I read it and thought this would be a fun experiment to build: run a model
that generates thinking text, interpret its internal activations with an NLA,
and put the two views beside each other in an interface people can explore.

The paper provides the approach. I want to understand it through implementation
and make it easy to interact with, following the same approach I took with
Charlotte and Serenity. Dolittle will attach the generated thinking and the
NLA descriptions to the same points in a generation so we can inspect them
together.

## The text the model generates

Thinking mode gives the model a trained way to generate a reasoning sequence
before its answer. Those are ordinary generated tokens with a particular role
in the conversation format. They become part of the context used to produce
later tokens, so the text participates in the continuing computation.

That does not make it a complete record of the computation. The model is also
processing numerical values through its layers. A readable sequence can be
useful without naming every association, intermediate result, or influence
represented in those values.

For Dolittle, I wanted a target with native thinking behavior. Qwen3 8B provides
that mode; its [model card](https://huggingface.co/Qwen/Qwen3-8B) documents the
thinking and non-thinking formats. A prompt asking an arbitrary model to show
its work would give us a different experiment.

## A description from an activation

An activation is a numerical value produced while the model processes its
input. The particular value Dolittle will capture is a vector from the
residual stream at one position in one transformer block. That position
matters: the state after consuming a token helps produce the next token.

An NLA adds two trained roles. The **activation verbalizer** receives the
captured vector and produces a description. The **activation reconstructor**
receives that description and predicts a vector. Training rewards the pair
for reconstructing the original activation through this text bottleneck.

The round trip is:

**Captured activation → text description → reconstructed activation.**

Reconstruction gives the training process an objective it can measure. It
does not turn every sentence in the description into a verified statement.
The [NLA article](https://www.anthropic.com/research/natural-language-autoencoders)
discusses incorrect descriptions and the need for independent corroboration.

The target model continues to generate its own text. The verbalizer is a
separate computation with different trained weights. Putting their outputs
beside each other makes the relationship easier to inspect, but the two
outputs still need to keep their identities.

## When the views disagree

One of the reference prompts follows a copper key. Mara puts it in a kitchen
drawer, then moves it to her coat pocket. The coat hangs by the door. At a
captured point in the 8B run, the generated thinking includes:

> The coat is hanging by the door, so the key is in the coat pocket, which is part of the coat.

The NLA description of that activation includes:

> the entire response resolves the key's location, final sentence restating the answer for the key's position tied to Rosa being inside the suitcase in Mexico City.

These are exact excerpts from one reference run, at zero-based block 24,
token position 426. The description recognizes that the response is resolving
a key’s location, but supplies a different person, container, and setting.

There are two kinds of claim to separate here. We can directly check a claim
about the transcript: it says coat pocket, and does not introduce Rosa or
Mexico City. A claim about what the numerical activation represents is harder
to check. The extra content could be an inaccurate interpretation or an
association not expressed in the thinking text. The comparison alone cannot
decide between them.

The same care applies in the other direction. An unexpected NLA description
is not evidence, by itself, that we have discovered hidden reasoning. A high
reconstruction score is not a truth score for its prose, either.

This is part of why I want both views available. A visitor should be able to
see what text the model had reached, which activation was captured, and what
the verbalizer said about it. The interface can preserve those distinctions
without requiring agreement between the descriptions.

## Building something we can inspect

The first implementation will use Qwen3 8B and the published
[marco-bazzani NLA pair](https://huggingface.co/marco-bazzani/Qwen3-8B-nla).
That NLA reads the output of block 24. The reference experiment has already
run the three roles on an RTX 3090 Ti, loading their weights in separate
phases. [The model comparison]({{< relref "/writing/choosing-a-model-we-can-run" >}})
explains the practical tradeoffs.

The next work is the direct C++/CUDA implementation and its browser interface.
Capturing the right vector, keeping it attached to the right text position,
and reproducing the published injection and reconstruction conventions are
concrete things we can check. Once those work, there will be a much more
useful way to explore the descriptions than reading them from a script’s
output.
