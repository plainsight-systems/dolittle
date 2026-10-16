+++
title = 'About Dolittle'
description = 'Why I want to put a model’s thinking text beside another view of its computation.'
+++

I read [Anthropic’s paper on natural language autoencoders](https://transformer-circuits.pub/2026/nla/index.html)
and thought this would be a fun experiment to build. It describes a way to
turn an internal activation into a natural language description. Putting
that beside a model’s generated thinking would give us two views to explore
in an interactive demo.

I want to put those two views beside each other and make them approachable.
Choose a point in the generation, see the thinking text available at that
point, and read the NLA’s interpretation of an activation captured there. The
places where they agree are interesting. So are the places where they do not.

## Understanding through implementation

Dolittle follows the reason I build these projects: writing the machinery gives
me something I can inspect and explain. The first engine is planned in C++ and CUDA,
running published model weights on Linux. The browser will provide the
interface, while the NVIDIA GPU does the inference. Users will build from
source through a Docker entry point for the Linux/NVIDIA build.

The selected target is Qwen3 8B with marco-bazzani’s trained NLA pair. Reference
runs on a 24 GB RTX 3090 Ti have established that the models can execute in
separate BF16 phases. The engine and live inference interface still need to be
built. There is no application launch command yet.

## Planned platform support

NVIDIA comes first, with CUDA on Linux and the 24 GB RTX 3090 Ti as the initial
verification target. Metal on macOS is planned after the NVIDIA implementation.
AMD support is undecided. The Metal build and hardware requirements will be
worked out in that phase; the measurements so far are from the NVIDIA
reference runs.

## Room to work through a question

The writing is part of the work. Some articles will explain a mechanism, some
will follow an experiment or a decision, and milestones will record a capability
once it works. A question does not need to wait for a milestone before it is
worth discussing.

The name comes from Doctor Dolittle, who learns to communicate with animals.
Here the attempt is to make an unfamiliar computational process easier to
inspect. Neither the thinking text nor the NLA description gives us an
unqualified account of everything happening inside the model.

## The wider workshop

I’m Andrew Hunter, a systems architect and builder. Dolittle is a
[Plainsight Systems](https://plainsight-systems.com/) project. The hub connects
the other builds and the questions that run between them; my
[personal site](https://andrewphunter.com/) holds my broader writing.

Begin with [two views of a thinking model]({{< relref "/writing/two-views-of-a-thinking-model" >}}),
or read [the model comparison]({{< relref "/writing/choosing-a-model-we-can-run" >}})
that led to the 8B choice.
