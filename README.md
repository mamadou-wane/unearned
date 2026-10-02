# Unearned

Unearned is a planned experiment on reward hacking in code models trained with reinforcement learning (RL). A small C++ code model will be trained from scratch, fine-tuned on example solutions, and then trained further with rewards computed by running its code against tests. The experiment asks how much of that RL improvement is earned: how much still shows up on a separate, hardened evaluation with its own hidden tests and fuzzed inputs, which no training reward reads.

The main comparison will keep the prompts, sampling settings, and optimizer steps the same across arms and change only the reward. A naive reward scores compilation and the visible tests. A tests-augmented reward adds hidden tests and fuzzed inputs, separate from those in the hardened evaluation. A gated reward takes the tests-augmented reward and scales it by the verdict of structural and coverage checks from a planned C++ extension of [Skeptic](https://github.com/mamadou-wane/skeptic), a verifier for coding-agent reward hacking. The question is whether the gate produces more earned capability than either alternative.

## Status

Early development.

## Contributing and support

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. For questions, see [SUPPORT.md](SUPPORT.md).

## License

Original code and documentation in this repository are released under the [MIT License](LICENSE). Third-party source code, datasets, and model weights are not covered by that license. Each keeps its own license and terms, which will be noted when it is added.
