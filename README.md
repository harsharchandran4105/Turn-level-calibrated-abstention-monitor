# Calibrated Sequential Abstention for Multi-Turn LLM Defense

This repository contains the code, experiments, and thesis for an M.Tech research project extending the **TurnGate** response-aware monitor (Shen et al., 2026).

## 🎯 Key Contributions
1. **Confidence Calibration:** Applied Temperature Scaling to produce reliable confidence estimates for the TurnGate monitor.
2. **Selective Routing:** Introduced an abstention threshold ($\tau$) to route low-confidence predictions to a secondary mechanism.
3. **DMC Fallback:** Implemented a lightweight Discrete Markov Chain (DMC) to track sequential risk and catch ambiguous multi-turn attacks.

##  Key Findings
Our framework achieves a **+28.6% absolute improvement in Exact Turn Accuracy ($\phi_1$)** and a **95.9% reduction in Early Blocking** compared to the base TurnGate baseline. We also identify and analyze the "Markovian Utility Trap" (Benign Drift), revealing a fundamental safety-utility trade-off in sequential risk accumulation.

## 🛠️ How to Run
*(Instructions will be updated once the final pipeline is fully documented)*
1. Install dependencies: `pip install -r requirements.txt`
2. Place the MTID dataset in the `dataset/` folder.
3. Run evaluation: `python src/master_eval.py ...`

##  Thesis
The complete LaTeX thesis and presentation slides are available in the `thesis/` directory.

## 📜 Citation
If you find this work useful, please cite the base paper:
```bibtex
@article{shen2026turngate,
  title={One turn too late: Response-aware defense against hidden malicious intent in multi-turn dialogue},
  author={Shen, Xinjie and others},
  journal={arXiv preprint arXiv:2605.05630},
  year={2026}
}