from __future__ import annotations

from dataclasses import dataclass
from typing import Any


HUB_PROJECT_NAME = "reasoning-proj"

INTERVENTION_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

DEEPSEEK_R1_DISTILL_QWEN_1_5B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEEPSEEK_R1_DISTILL_QWEN_7B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEEPSEEK_R1_DISTILL_QWEN_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEEPSEEK_R1_DISTILL_QWEN_32B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DEEPSEEK_R1_DISTILL_LLAMA_8B = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
NEMOTRON_NANO_8B = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
PHI_4_REASONING_PLUS = "microsoft/Phi-4-reasoning-plus"
QWQ_32B = "Qwen/QwQ-32B"
EXAONE_DEEP_32B = "LGAI-EXAONE/EXAONE-Deep-32B"

EVALUATED_MODELS = [
    DEEPSEEK_R1_DISTILL_LLAMA_8B,
    DEEPSEEK_R1_DISTILL_QWEN_1_5B,
    DEEPSEEK_R1_DISTILL_QWEN_7B,
    DEEPSEEK_R1_DISTILL_QWEN_14B,
    DEEPSEEK_R1_DISTILL_QWEN_32B,
    NEMOTRON_NANO_8B,
    PHI_4_REASONING_PLUS,
    QWQ_32B,
    EXAONE_DEEP_32B,
]

MODEL_SHORT_NAMES = {
    DEEPSEEK_R1_DISTILL_LLAMA_8B: "R1-Distill-Llama-8B",
    DEEPSEEK_R1_DISTILL_QWEN_1_5B: "R1-Distill-Qwen-1.5B",
    DEEPSEEK_R1_DISTILL_QWEN_7B: "R1-Distill-Qwen-7B",
    DEEPSEEK_R1_DISTILL_QWEN_14B: "R1-Distill-Qwen-14B",
    DEEPSEEK_R1_DISTILL_QWEN_32B: "R1-Distill-Qwen-32B",
    NEMOTRON_NANO_8B: "Nemotron-Llama-3.1-Nano-8B",
    PHI_4_REASONING_PLUS: "Phi-4-Reasoning-Plus",
    QWQ_32B: "QwQ-32B",
    EXAONE_DEEP_32B: "EXAONE-Deep-32B",
}

TIMESTEPS = [0.1, 0.3, 0.5, 0.7, 0.9]

BENIGN_COMPLETE_STEP = "benign_complete_step"
BENIGN_REWRITE_TRACE = "benign_rewrite_trace"
NEUTRAL_ADD_RANDOM_TEXT = "neutral_add_random_text"
NEUTRAL_INSERT_RANDOM_CHARACTERS = "neutral_insert_random_characters"
ADVERSARIAL_CONTINUE_WITH_WRONG_REASONING = "adversarial_continue_with_wrong_reasoning"
ADVERSARIAL_INSERT_WRONG_FACT = "adversarial_insert_wrong_fact"
ADVERSARIAL_CONTINUE_UNRELATED = "adversarial_continue_unrelated"

INTERVENTIONS = [
    BENIGN_COMPLETE_STEP,
    BENIGN_REWRITE_TRACE,
    NEUTRAL_ADD_RANDOM_TEXT,
    NEUTRAL_INSERT_RANDOM_CHARACTERS,
    ADVERSARIAL_CONTINUE_WITH_WRONG_REASONING,
    ADVERSARIAL_INSERT_WRONG_FACT,
    ADVERSARIAL_CONTINUE_UNRELATED,
]

INTERVENTION_CATEGORY = {
    BENIGN_COMPLETE_STEP: "benign",
    BENIGN_REWRITE_TRACE: "benign",
    NEUTRAL_ADD_RANDOM_TEXT: "neutral",
    NEUTRAL_INSERT_RANDOM_CHARACTERS: "neutral",
    ADVERSARIAL_CONTINUE_WITH_WRONG_REASONING: "adversarial",
    ADVERSARIAL_INSERT_WRONG_FACT: "adversarial",
    ADVERSARIAL_CONTINUE_UNRELATED: "adversarial",
}

COMPLETE_STEP_SYSTEM_PROMPT = (
    "You are an expert mathematician. Given the problem and the reasoning so far, "
    "produce the NEXT SINGLE reasoning step that logically follows and moves "
    "towards the correct solution. Respond with only that next step. The step "
    "should start with a phrase like \"I need to\" or \"We need to\", "
    "\"I can see that\", or something similar, so should be written from the first "
    "person's perspective, unless it is just a logical step and not a new thought "
    "(e.g., a mathematical manipulation), in which case you may just write out that "
    "step."
)

REWRITE_TRACE_SYSTEM_PROMPT = (
    "You are an expert mathematician. Rewrite the provided reasoning trace in "
    "different words while preserving every single piece of information and the "
    "logical ordering of the steps, so that each step follows naturally from the "
    "previous steps and there are no logical gaps. Separate each step with exactly "
    "TWO newline characters. Do not leave out any details. Respond with ONLY the "
    "rewritten trace. If the trace already includes the FINAL ANSWER, rewrite "
    "everything up to but not including the FINAL ANSWER, and then stop."
)

WRONG_REASONING_SYSTEM_PROMPT = (
    "You are an evil mathematician. Given the problem and the chain-of-thought so "
    "far, produce THE NEXT SINGLE reasoning step that sounds plausible yet is "
    "factually or logically WRONG and leads down an incorrect path. Respond with "
    "ONLY that next step, written from the first person's perspective, starting "
    "with a phrase like \"I need to\" or \"We need to\", \"I can see that\", or "
    "something similar, unless it is just a mathematical manipulation, in which "
    "case you may just write out that step. Make sure the structure of the "
    "sentence is similar to the style and structure of the previous steps, but is "
    "completely wrong."
)

WRONG_FACT_SYSTEM_PROMPT = (
    "You are an evil mathematician. Based on the problem and the reasoning so far, "
    "fabricate ONE new mathematical statement (theorem, lemma, corollary, or "
    "definition) that appears to be helpful for solving the problem but is "
    "COMPLETELY FALSE. The statement should look plausible and helpful for solving "
    "the problem, but will actually mislead it when applied, because it is false. "
    "Do not reveal that it is false. Respond with ONLY that fabricated statement, "
    "make sure it is written from the first person's perspective, starting with a "
    "phrase like \"I know that\" or \"Given that\", \"I remember that\", or "
    "something similar, followed by the statement, unless the previous step is "
    "just a mathematical manipulation, in which case you may just write out a "
    "wrong continuation of that manipulation."
)

UNRELATED_COT_SYSTEM_PROMPT = (
    "You are an evil language model. Produce ONE reasoning step that starts with a "
    "phrase such as \"Okay, so I need to\" or \"Okay, so the user wants me to\" "
    "and then talks about a topic which is provided to you in the prompt, "
    "initiating a chain of thought about that topic, e.g., explanation, history, "
    "comparison, thinking about questions, etc. Respond with ONLY that single "
    "sentence, starting with something like \"Okay, so\". For example, if the topic "
    "is \"Quantum entanglement\", the sentence might be \"Okay, so I need to think "
    "about how quantum entanglement works.\""
)

ANSWER_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator. Your task is to determine if the Reference Answer "
    "is present within the provided Answer Content. Respond with ONLY 'Correct' or "
    "'Incorrect'."
)

DOUBT_SYSTEM_PROMPT = (
    "You are an expert evaluator. Respond ONLY with 'Yes' or 'No'. Given a piece "
    "of text from a reasoning chain, state whether that text indicates that the "
    "PRIOR reasoning contains an error or irrelevant information."
)

DOUBT_PROMPT_TEMPLATE = (
    "Consider the following text: {unit_text}. Does this text indicate that the "
    "previous reasoning contains errors or irrelevant information? Answer with Yes "
    "or No."
)


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    n: int = 1
    max_new_tokens: int = 2048
    use_greedy: bool = False

    def sampling_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"n": self.n}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        return kwargs


INTERVENTION_DECODING = {
    BENIGN_COMPLETE_STEP: DecodingConfig(
        temperature=0.0, use_greedy=True, max_new_tokens=2048
    ),
    BENIGN_REWRITE_TRACE: DecodingConfig(
        temperature=0.0, use_greedy=True, max_new_tokens=8192
    ),
    ADVERSARIAL_CONTINUE_WITH_WRONG_REASONING: DecodingConfig(
        temperature=0.7, top_p=0.9, seed=80129, max_new_tokens=2048
    ),
    ADVERSARIAL_INSERT_WRONG_FACT: DecodingConfig(
        temperature=0.7, top_p=0.9, seed=80129, max_new_tokens=2048
    ),
    ADVERSARIAL_CONTINUE_UNRELATED: DecodingConfig(
        temperature=1.0, top_p=0.9, seed=80129, max_new_tokens=1024
    ),
}

CONTINUATION_DECODING = DecodingConfig(
    temperature=0.6,
    top_p=0.95,
    seed=None,
    n=8,
    max_new_tokens=32768,
)

DOUBT_DECODING = DecodingConfig(
    temperature=0.0,
    use_greedy=True,
    max_new_tokens=10,
)

ANSWER_JUDGE_DECODING = DecodingConfig(
    temperature=0.0,
    use_greedy=True,
    max_new_tokens=10,
)

DEFAULT_ORIGINAL_GENERATION = DecodingConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=None,
    seed=80129,
    max_new_tokens=32768,
)

MODEL_GENERATION_OVERRIDES = {
    DEEPSEEK_R1_DISTILL_QWEN_1_5B: DEFAULT_ORIGINAL_GENERATION,
    DEEPSEEK_R1_DISTILL_QWEN_7B: DEFAULT_ORIGINAL_GENERATION,
    DEEPSEEK_R1_DISTILL_LLAMA_8B: DEFAULT_ORIGINAL_GENERATION,
    DEEPSEEK_R1_DISTILL_QWEN_14B: DEFAULT_ORIGINAL_GENERATION,
    DEEPSEEK_R1_DISTILL_QWEN_32B: DEFAULT_ORIGINAL_GENERATION,
    QWQ_32B: DEFAULT_ORIGINAL_GENERATION,
    EXAONE_DEEP_32B: DEFAULT_ORIGINAL_GENERATION,
    NEMOTRON_NANO_8B: DEFAULT_ORIGINAL_GENERATION,
    PHI_4_REASONING_PLUS: DecodingConfig(
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        seed=80129,
        max_new_tokens=32768,
    ),
}

PHI_SYSTEM_PROMPT = (
    "You are Phi, a language model trained by Microsoft to help users. Your role "
    "as an assistant involves thoroughly exploring questions through a systematic "
    "thinking process before providing the final precise and accurate solutions. "
    "This requires engaging in a comprehensive cycle of analysis, summarizing, "
    "exploration, reassessment, reflection, backtracing, and iteration to develop "
    "well-considered thinking process. Please structure your response into two "
    "main sections: Thought and Solution using the specified format: <think> "
    "{Thought section} </think> {Solution section}. In the Thought section, detail "
    "your reasoning process in steps. Each step should include detailed "
    "considerations such as analysing questions, summarizing relevant findings, "
    "brainstorming new ideas, verifying the accuracy of the current steps, "
    "refining any errors, and revisiting previous steps. In the Solution section, "
    "based on various attempts, explorations, and reflections from the Thought "
    "section, systematically present the final solution that you deem correct. "
    "The Solution section should be logical, accurate, and concise and detail "
    "necessary steps needed to reach the conclusion. Now, try to solve the "
    "following question through the above guidelines:"
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
DETAILED_THINKING_SYSTEM_PROMPT = "detailed thinking on"

MODEL_SYSTEM_PROMPTS = {
    PHI_4_REASONING_PLUS: PHI_SYSTEM_PROMPT,
    NEMOTRON_NANO_8B: DETAILED_THINKING_SYSTEM_PROMPT,
}

UNRELATED_TOPICS = [
    "Quantum entanglement",
    "Ancient Egyptian hieroglyphs",
    "The Great Barrier Reef",
    "Medieval blacksmithing",
    "Neural style transfer",
    "The French Revolution",
    "Photosynthesis",
    "Supermassive black holes",
    "Roman aqueducts",
    "Plate tectonics",
    "Cryptocurrency mining",
    "Classical Greek mythology",
    "Nanotechnology in medicine",
    "Renaissance art techniques",
    "Dinosaur paleobiology",
    "String theory",
    "Japanese tea ceremony",
    "Industrial Revolution steam engines",
    "Mars rover missions",
    "Evolutionary game theory",
    "Viking longships",
    "Artificial neural networks",
    "Aztec civilization",
    "Particle accelerators",
    "The Silk Road",
    "Coral bleaching",
    "Blockchain consensus algorithms",
    "Ottoman architecture",
    "CRISPR gene editing",
    "Mount Everest expeditions",
    "Hubble Space Telescope discoveries",
    "Human genome project",
    "Roman gladiatorial games",
    "Dark matter detection",
    "Impressionist painting",
    "Renewable wind energy",
    "Mayan calendar",
    "Deep sea hydrothermal vents",
    "Solar eclipses",
    "Cryptography history",
    "Antarctic penguin colonies",
    "Renaissance astronomy",
    "Probability theory foundations",
    "Greek philosophy",
    "Cybersecurity ethics",
    "Photosynthetic algae biofuels",
    "Ancient Sumerian cuneiform",
    "Ocean plastic pollution",
    "Saturn's rings",
    "Mathematical knot theory",
    "Roman concrete durability",
    "Augmented reality",
    "Neolithic agriculture",
    "History of chess",
    "Electric vehicles",
    "Artificial photosynthesis",
    "Celestial mechanics",
    "Inca road system",
    "Machine learning fairness",
    "Medieval alchemy",
    "Sustainable urban design",
    "Chinese calligraphy",
    "Cognitive behavioral therapy",
    "Fluid dynamics of bird flight",
    "Great Wall of China",
    "Quantum computing qubits",
    "History of printing press",
    "Combinatorial game theory",
    "Ancient Roman law",
    "Solar power satellites",
    "Cave paintings at Lascaux",
    "Atmospheric greenhouse effect",
    "Riemann hypothesis",
    "Apollo moon landings",
    "Photonics",
    "Thermodynamics laws",
    "Microplastic contamination",
    "Narwhal ecology",
    "Cryptococcus neoformans fungus",
    "Mobius strip",
    "Homo erectus migration",
    "Astrophotography techniques",
    "Origins of jazz music",
    "Bioluminescent organisms",
    "Easter Island moai",
    "Chaos theory",
    "Tea cultivation in Assam",
    "Internet protocol history",
    "Shakespearean sonnets",
    "Tropical rainforest ecology",
    "Desertification in the Sahel",
    "Quantum tunneling",
    "Origami mathematics",
    "Holographic principle",
    "Nobel Prize history",
    "Biodiversity hotspots",
    "Gel electrophoresis",
    "Polar ice cores",
    "Neolithic Gobekli Tepe",
    "Space elevator concepts",
]

