# Drone flyby
Solution for task drone vision detection
## Training solution

All of solution for training and validation is referred to ```solution/```

## Quickstart

```cmd
pip install -r requirements.txt
```

Serve the baseline:

```cmd
python api.py
```

Then, in a second terminal, score it against the supplied scene:

```cmd
python local_evaluator.py
```

You now have a working endpoint and a number to improve. The baseline scores
about zero — it is edge detection with a fixed label, there to prove the
plumbing works, not to compete.

Check that the harness and the data agree with each other at any time:

```cmd
python local_evaluator.py --oracle
```

That feeds the ground truth in as predictions and should print `1.000`. If it
does, a low score is your model, not your setup.

### What is in this folder

| File | What it is |
|---|---|
| `api.py` | API with patching for the big image |
| `api_full.py` | Test api for full image. |
| `example.py` | The baseline detector and camera policy. **This is the file to replace.** |
| `dtos.py` | The request and response models, plus the protocol constants. |
| `utils.py` | Decoding, coordinate conversion, response validation, box drawing. |
| `local_evaluator.py` | Replays a scene through your endpoint and scores it. |
| `visualize.py` | Draws the ground truth onto the supplied frames. |
| `requirements.txt` | Dependencies. Loose pins, so they will not fight your detection stack. |
| `Dockerfile` | If you would rather containerise the server. |
| `solution` | Solution path. |


### Reference

Refer to this repo ```https://github.com/amboltio/Nordic-AI-Cup-2026/tree/main/drone-flyby```