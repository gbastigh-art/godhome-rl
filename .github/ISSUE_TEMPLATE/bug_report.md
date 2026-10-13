name: Bug report

**What happened**

<!-- One or two sentences. -->

**What you expected**

<!-- What you expected instead. -->

**Reproduction**

```console
python -m godhome_rl <the command you ran>
```

**Environment**

Paste the output of:

```console
python -m godhome_rl doctor
python -m godhome_rl bench --boss "<the boss>" --episodes 2
```

Also useful:

* OS and Python version
* number of CPU cores
* if training: the last ~20 lines of `runs/<name>/metrics.csv`
* if training with `--verbose`: the matching `build/ports/harness-*.log`

**Checklist**

- [ ] I did not attach my Godhome installation or anything under `build/`
- [ ] I searched existing issues
- [ ] I can reproduce this from a fresh `python -m godhome_rl build`