+++
title = 'Two views of a thinking model'
description = 'How an NLA turns the numbers inside a model into text we can read and edit.'
date = '2026-10-16T09:00:00Z'
lastmod = '2026-10-21T00:00:00-07:00'
article_type = 'explanation'
+++

A thinking model gives us text we can read, but it also processes vectors of
numbers at every layer. A natural language autoencoder, or NLA, translates one
of those vectors into a description. Put that description beside the thinking
text and we have two views of the same moment in generation.

[Anthropic’s NLA paper](https://transformer-circuits.pub/2026/nla/index.html)
uses poetry to explore what those descriptions contain. A model is asked to
finish a couplet whose first line ends in “grab it.” Before it writes the next
line, the NLA describes a plan involving “rabbit.” Changing the explanation
to “mouse” gives the researchers a way to steer the model toward a different
ending. Dolittle uses the same process with Qwen3 8B.

## The text the model generates

Qwen3 can generate a thinking sequence before its answer. Those thinking
tokens become part of the context for the tokens that follow. If it writes
out a calculation or considers a rhyme, that text can shape what it writes
next.

Inside the transformer, each token has a state vector that successive blocks
update. Attention combines information from the current and preceding token
positions and adds its result to that state. The block’s feed-forward network
then computes another update and adds it. This running state is the
**residual stream**.

Qwen’s feed-forward network uses **SwiGLU** gating. It projects the normalized
state along two paths, applies the SiLU activation function to one, and
multiplies the two results element by element. A final projection brings that
result back to the state’s width for the residual addition.

“Activations” is the broader name for intermediate values computed by the
model. Dolittle captures a specific one: the 4,096-value residual-stream vector
after block 24’s second addition, before block 25’s normalization. This gives
the NLA the accumulated block output, including the attention and feed-forward
updates. The gate values are intermediate steps within that computation.

## Turning an activation into words

The NLA’s **activation verbalizer** takes a captured vector and generates a
text description. Its **activation reconstructor** works in the other
direction, taking the description and predicting the original vector:

**Activation → text explanation → reconstructed activation.**

During training, the pair is rewarded for reconstructing the activation
accurately. This encourages the explanation to carry information from the
vector through the text and back again. The published
[Qwen3 8B NLA pair](https://huggingface.co/marco-bazzani/Qwen3-8B-nla)
was trained to read block 24, which is why Dolittle captures that block.

The thinking text comes from Qwen3’s own generation. The NLA explanation comes
from the verbalizer reading a saved activation. Keeping the two beside each
other lets us see what each describes at the same point.

## When the views disagree

In the initial PyTorch experiment, one prompt follows a copper key. Mara puts it in a kitchen
drawer, then moves it to her coat pocket. The coat hangs by the door. At
block 24, token position 426, the model’s thinking includes:

> The coat is hanging by the door, so the key is in the coat pocket, which is part of the coat.

The NLA description includes:

> the entire response resolves the key's location, final sentence restating the answer for the key's position tied to Rosa being inside the suitcase in Mexico City.

The NLA picks up the task of locating the key, but invents a different person,
container and setting. The model’s thinking follows the prompt’s coat pocket;
the explanation supplies a suitcase in Mexico City.

That gives us something specific to explore. We can generate another
explanation from the same activation, see which details recur, then move to a
nearby token and compare. Dolittle keeps each sample with the position it came
from, so those comparisons are a few clicks apart.

## Editing the explanation

The reconstructor makes the text useful for more than reading. We can edit an
explanation, reconstruct both versions, and subtract their vectors. The
difference gives us a direction to add to the model’s activation before it
continues generating.

In Anthropic’s rhyming example, the edit changes rabbit-related words to
mouse-related words. My Qwen runs produced varied descriptions, including
carrots, rabbits, invented classrooms and comments about the poem’s wording.
Rabbit was not a consistent result. The useful edit depends on what the
selected explanation actually says.

Dolittle runs an unsteered version and a steered version with the same
starting text and seed. We can read the resulting poems and the thinking that
led to them.

[Try the rhyming experiment]({{< relref "/writing/running-an-investigation" >}})
walks through the process. [The model comparison]({{< relref "/writing/choosing-a-model-we-can-run" >}})
explains why I chose the 8B pair and how its three model roles fit on one GPU.
