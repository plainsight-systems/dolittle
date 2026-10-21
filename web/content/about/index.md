+++
title = 'About Dolittle'
description = 'I read the NLA paper and wanted to build an experiment people could try for themselves.'
+++

I read [Anthropic’s paper on natural language autoencoders](https://transformer-circuits.pub/2026/nla/index.html)
and thought it would be a fun experiment to build. The paper turns a model’s
internal numbers into descriptions we can read. It also shows how editing
those descriptions can change what the model writes next.

Dolittle puts that process in a browser. You can follow Qwen3’s thinking,
select a token and read an NLA explanation of its activation. Generate more
samples, inspect a nearby token, or edit an explanation and compare the
original and steered versions of the response.

## Why build the engine?

Implementing the machinery is part of the interest for me, as it was with
Charlotte and Serenity. The initial Python experiment could already capture
activations and pass them to the NLA for description. Building our own engine
lets us explore the computation underneath: how attention runs on the GPU,
how we reuse memory between tokens, and where we capture or change an
activation. The browser makes that machinery something people can try.

Dolittle runs the published Qwen3 8B and
[marco-bazzani NLA weights](https://huggingface.co/marco-bazzani/Qwen3-8B-nla)
through a C++/CUDA engine. The GPU runs one model role at a time, which lets
the experiment fit on a 24 GB RTX 3090 Ti. The launcher builds from source in
Docker and downloads the models once.

## Where it runs

Linux/NVIDIA is available now. Native Metal on macOS is planned next, with a
native Windows/NVIDIA build also planned. AMD support is undecided. A browser
on a Mac can already connect to the Linux GPU host and run the interface.

[Try the rhyming experiment]({{< relref "/writing/running-an-investigation" >}})
covers setup, reading an explanation and steering the next line of a poem.

## Why Dolittle?

Doctor Dolittle learns to communicate with animals. The connection here is
turning an unfamiliar internal representation into language we can work with.
It seemed a fitting name for the experiment.

I’m Andrew Hunter, a systems architect and builder. Dolittle is a
[Plainsight Systems](https://plainsight-systems.com/) project, alongside
Charlotte, Serenity and Cradle. My [personal site](https://andrewphunter.com/)
holds my broader writing.
