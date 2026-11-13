# TimesFM implementation evidence — source ledger

Research/fetch date: 2026-11-12. Only primary technical sources are used.

| ID | Primary source | Local snapshot | Use |
|---|---|---|---|
| S1 | https://github.com/google-research/timesfm/blob/master/README.md | `raw/official/raw.githubusercontent.com_google-research_timesfm_master_README.md` | versions, install, 2.5/3.0 licensing boundary, unsupported-open-project caution |
| S2 | https://github.com/google-research/timesfm/blob/master/pyproject.toml | `raw/official/raw.githubusercontent.com_google-research_timesfm_master_pyproject.toml` | package 3.0.2, Python >=3.10, dependencies/extras |
| S3 | https://github.com/google-research/timesfm/blob/master/LICENSE | `raw/official/raw.githubusercontent.com_google-research_timesfm_master_LICENSE` | repository Apache-2.0 text |
| S4 | https://huggingface.co/google/timesfm-2.5-200m-pytorch | `raw/official/huggingface.co_google_timesfm-2.5-200m-pytorch_raw_main_README.md` | official 2.5 model card and output shapes |
| S5 | https://github.com/google-research/timesfm/blob/master/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py | `raw/official/raw.githubusercontent.com_google-research_timesfm_master_src_timesfm_timesfm_2p5_timesfm_2p5_torch.py` | actual 2.5 PyTorch load/compile/forecast behavior |
| S6 | https://github.com/google-research/timesfm/blob/master/src/timesfm/timesfm_2p5/timesfm_2p5_base.py | `raw/official/raw.githubusercontent.com_google-research_timesfm_master_src_timesfm_timesfm_2p5_timesfm_2p5_base.py` | actual XReg covariate API and validation |
| S7 | https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/references/data_preparation.md | local snapshot with same path pattern | input schema, NaN handling, covariate shapes |
| S8 | https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/references/system_requirements.md | local snapshot with same path pattern | official hardware estimates and troubleshooting |
| S9 | https://gml.noaa.gov/ccgg/trends/data.html | — | NOAA dataset landing page |
| S10 | https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv | `raw/official/noaa_co2_mm_mlo_2026-11-12.csv` | runnable monthly Mauna Loa CO2 dataset snapshot |
| S11 | https://arxiv.org/abs/2310.10688 | — | original TimesFM paper |

## Pinned artifacts

- Dataset snapshot SHA-256: `d895fb5b4a59fb5e4cd98e69b7667fe5bef595c6581f630852b12cb41c07f675`.
- Hugging Face model revision: `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`.
- `model.safetensors` SHA-256: `2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8` (925,181,104 bytes).
- `config.json` SHA-256: `cd3315b760d5cc7e278d7afdf41b897031ced888fc6c115bd9b3ac0ea2c47408`.
- All downloaded text/data snapshots were scanned with `untrusted_content_scan.py`: `risk=low`, zero findings. No downloaded scripts were executed.
