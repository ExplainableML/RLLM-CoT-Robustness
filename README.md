<div align="center">
    
# Are Reasoning LLMs Robust to Interventions on their Chain-of-Thought? <br/> _ICLR 2026_
[![Paper](https://img.shields.io/badge/paper-OpenReview-8C1B13.svg)](https://openreview.net/forum?id=aQZIpELFwp)

Alexander von Recum <sup>1,2</sup> &#8198; Leander Girrbach<sup>1,3</sup> &#8198; Zeynep Akata<sup>1,3</sup>

<sup>1</sup>Helmholtz Munich &#8198; <sup>2</sup> Ludwig Maximilian University of Munich <sup>3</sup> Technical University of Munich, MCML
</div>

## Abstract
Reasoning LLMs (RLLMs) generate step-by-step chains of thought (CoTs) before giving an answer, which improves performance on complex tasks and makes reasoning transparent. But how robust are these reasoning traces to disruptions that occur _within_ them? To address this question, we introduce a controlled evaluation framework that perturbs a model’s own CoT at fixed timesteps. We design seven interventions (benign, neutral, and adversarial) and apply them to multiple open-weight RLLMs across Math, Science, and Logic tasks. Our results show that RLLMs are generally robust, reliably recovering from diverse perturbations, with robustness improving with model size and degrading when interventions occur early. However, robustness is not style-invariant: paraphrasing suppresses doubt-like expressions and reduces performance, while other interventions trigger doubt and support recovery. Recovery also carries a cost: neutral and adversarial noise can inflate CoT length by more than 200%, whereas paraphrasing shortens traces but harms accuracy. These findings provide new evidence on how RLLMs maintain reasoning integrity, identify doubt as a central recovery mechanism, and highlight trade-offs between robustness and efficiency that future training methods should address.
