# Flight 1 — comparison (one shared split; val = selection basis; trained test reads deferred)

| Contender | What it is | Train R² | Val R² | Test R² |
|---|---|---|---|---|
| Gradient boosting | the conventional tool, trained | 0.9123343713333352 | 0.6356559013459231 | deferred |
| PyTorch MLP | hand-built network, trained | 0.811007961730985 | 0.3193305738393898 | deferred |
| TabPFN | foundation model, context only, no training | — (no training) | 0.8123367964963065 | 0.8185785487036595 |
