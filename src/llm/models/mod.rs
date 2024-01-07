// source: https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized/main.rs
use serde::Deserialize;
use std::str::FromStr;

#[derive(Default, Deserialize, Clone, Debug, Copy, PartialEq, Eq)]
pub enum Models {
    #[serde(rename = "7b")]
    L7b,
    #[serde(rename = "13b")]
    L13b,
    #[serde(rename = "70b")]
    L70b,
    #[serde(rename = "7b-chat")]
    L7bChat,
    #[serde(rename = "13b-chat")]
    L13bChat,
    #[serde(rename = "70b-chat")]
    L70bChat,
    #[serde(rename = "7b-code")]
    L7bCode,
    #[serde(rename = "13b-code")]
    L13bCode,
    #[serde(rename = "32b-code")]
    L34bCode,
    #[serde(rename = "7b-leo")]
    Leo7b,
    #[serde(rename = "13b-leo")]
    Leo13b,
    #[serde(rename = "7b-mistral")]
    Mistral7b,
    #[serde(rename = "7b-mistral-instruct")]
    Mistral7bInstruct,
    #[serde(rename = "7b-zephyr-a")]
    Zephyr7bAlpha,
    #[serde(rename = "7b-zephyr-b")]
    Zephyr7bBeta,
    #[default]
    #[serde(rename = "7b-open-chat-3.5")]
    OpenChat35,
    #[serde(rename = "7b-starling-a")]
    Starling7bAlpha,
    #[serde(rename = "mixtral")]
    Mixtral,
    #[serde(rename = "mixtral-instruct")]
    MixtralInstruct,

    #[serde(rename = "phi-hermes")]
    PhiHermes,
    #[serde(rename = "phi-v1")]
    PhiV1,
    #[serde(rename = "phi-v1.5")]
    PhiV1_5,
    #[serde(rename = "phi-v2")]
    PhiV2,
}

#[derive(Deserialize)]
struct StringEnumHelper {
    model: Models,
}

impl FromStr for Models {
    type Err = serde_yaml::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let helper: StringEnumHelper = serde_yaml::from_str(&format!("model: {}", s))?;
        Ok(helper.model)
    }
}

impl Models {
    pub fn is_mistral(&self) -> bool {
        match self {
            Self::OpenChat35
            | Self::Starling7bAlpha
            | Self::Zephyr7bAlpha
            | Self::Zephyr7bBeta
            | Self::Mixtral
            | Self::MixtralInstruct
            | Self::Mistral7b
            | Self::Mistral7bInstruct => true,
            _ => false,
        }
    }

    pub fn is_zephyr(&self) -> bool {
        match self {
            Self::Zephyr7bAlpha | Self::Zephyr7bBeta => true,
            _ => false,
        }
    }

    pub fn is_open_chat(&self) -> bool {
        match self {
            Self::OpenChat35 | Self::Starling7bAlpha => true,
            _ => false,
        }
    }

    pub fn is_phi(&self) -> bool {
        match self {
            Self::PhiHermes | Self::PhiV1 | Self::PhiV1_5 | Self::PhiV2 => true,
            _ => false,
        }
    }

    pub fn tokenizer_repo(&self) -> &'static str {
        match self {
            Models::L7b
            | Models::L13b
            | Models::L70b
            | Models::L7bChat
            | Models::L13bChat
            | Models::L70bChat
            | Models::L7bCode
            | Models::L13bCode
            | Models::L34bCode => "hf-internal-testing/llama-tokenizer",
            Models::Leo7b => "LeoLM/leo-hessianai-7b",
            Models::Leo13b => "LeoLM/leo-hessianai-13b",
            Models::Mixtral => "mistralai/Mixtral-8x7B-v0.1",
            Models::MixtralInstruct => "mistralai/Mixtral-8x7B-Instruct-v0.1",
            Models::Mistral7b
            | Models::Mistral7bInstruct
            | Models::Zephyr7bAlpha
            | Models::Zephyr7bBeta => "mistralai/Mistral-7B-v0.1",
            Models::OpenChat35 => "openchat/openchat_3.5",
            Models::Starling7bAlpha => "berkeley-nest/Starling-LM-7B-alpha",
            Models::PhiV1 => "microsoft/phi-1",
            Models::PhiV1_5 => "microsoft/phi-1.5",
            Models::PhiV2 => "microsoft/phi-2",
            Models::PhiHermes => "lmz/candle-quantized-phi",
        }
    }

    pub fn repo_path(&self) -> (&str, &str) {
        match self {
            Models::L7b => ("TheBloke/Llama-2-7B-GGML", "llama-2-7b.ggmlv3.q4_0.bin"),
            Models::L13b => ("TheBloke/Llama-2-13B-GGML", "llama-2-13b.ggmlv3.q4_0.bin"),
            Models::L70b => ("TheBloke/Llama-2-70B-GGML", "llama-2-70b.ggmlv3.q4_0.bin"),
            Models::L7bChat => (
                "TheBloke/Llama-2-7B-Chat-GGML",
                "llama-2-7b-chat.ggmlv3.q4_0.bin",
            ),
            Models::L13bChat => (
                "TheBloke/Llama-2-13B-Chat-GGML",
                "llama-2-13b-chat.ggmlv3.q4_0.bin",
            ),
            Models::L70bChat => (
                "TheBloke/Llama-2-70B-Chat-GGML",
                "llama-2-70b-chat.ggmlv3.q4_0.bin",
            ),
            Models::L7bCode => ("TheBloke/CodeLlama-7B-GGUF", "codellama-7b.Q8_0.gguf"),
            Models::L13bCode => ("TheBloke/CodeLlama-13B-GGUF", "codellama-13b.Q8_0.gguf"),
            Models::L34bCode => ("TheBloke/CodeLlama-34B-GGUF", "codellama-34b.Q8_0.gguf"),
            Models::Leo7b => (
                "TheBloke/leo-hessianai-7B-GGUF",
                "leo-hessianai-7b.Q4_K_M.gguf",
            ),
            Models::Leo13b => (
                "TheBloke/leo-hessianai-13B-GGUF",
                "leo-hessianai-13b.Q4_K_M.gguf",
            ),
            Models::Mixtral => (
                "TheBloke/Mixtral-8x7B-v0.1-GGUF",
                "mixtral-8x7b-v0.1.Q4_K_M.gguf",
            ),
            Models::MixtralInstruct => (
                "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            ),
            Models::Mistral7b => (
                "TheBloke/Mistral-7B-v0.1-GGUF",
                "mistral-7b-v0.1.Q4_K_S.gguf",
            ),
            Models::Mistral7bInstruct => (
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                "mistral-7b-instruct-v0.1.Q4_K_S.gguf",
            ),
            Models::Zephyr7bAlpha => (
                "TheBloke/zephyr-7B-alpha-GGUF",
                "zephyr-7b-alpha.Q4_K_M.gguf",
            ),
            Models::Zephyr7bBeta => ("TheBloke/zephyr-7B-beta-GGUF", "zephyr-7b-beta.Q4_K_M.gguf"),
            Models::OpenChat35 => ("TheBloke/openchat_3.5-GGUF", "openchat_3.5.Q4_K_M.gguf"),
            Models::Starling7bAlpha => (
                "TheBloke/Starling-LM-7B-alpha-GGUF",
                "starling-lm-7b-alpha.Q4_K_M.gguf",
            ),
            Models::PhiV1 => ("lmz/candle-quantized-phi", "model-v1-q4k.gguf"),
            Models::PhiV1_5 => ("lmz/candle-quantized-phi", "model-q4k.gguf"),
            Models::PhiV2 => ("lmz/candle-quantized-phi", "model-v2-q4k.gguf"),
            Models::PhiHermes => ("lmz/candle-quantized-phi", "model-phi-hermes-1_3B-q4k.gguf"),
        }
    }
}
